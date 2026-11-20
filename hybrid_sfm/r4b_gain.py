"""R4b: a cascade whose escalation score predicts the *gain* from escalating, using a cheap probe of the cheap pair.

Stage 1   classical + VGGT reconstruct (as R4).
Stage 1.5 both get a short 3DGS probe (EARLY steps, ~7 s at 250). A router on the 32 geometric features plus these
          probe results picks a method (classical, VGGT, or VGGT+BA without probing it).
Stage 2   escalate: VGGT+BA reconstructs; all candidates are probed to 1,000 steps (the early probes continue, so
          their cost is not paid twice); the best probe wins.

Escalation score = a random forest's prediction of gain = loss(stage-1.5 choice) - loss(stage-2 choice), which can be
negative: on some chunks the probe picks worse. R4 scored "how bad will this chunk be", which is a different question.

Everything is out-of-scene and nested. For each held-out scene s:
  - stage-1.5 routers fitted on the other 6 scenes give the test choices; an inner leave-one-scene-out over those 6
    gives out-of-scene choices for the training chunks, from which the gain targets are computed;
  - the gain model fitted on the 6 scenes scores the test chunks; an inner leave-one-scene-out gives calibration
    scores for conformal risk control (same risk as R4: probe skipped while the stage-1.5 choice is > DELTA off).

Usage:
    python -m hybrid_sfm.r4b_gain docs/research/results/r3/features.csv docs/research/results/r2/r2_labels.jsonl \
        docs/research/results/r3b/r3b_probes.jsonl --out docs/research/results/r4/r4b_results.md
"""
import argparse
import json
import os
import sys

import numpy as np

from .r3_router import METHODS, fit_forest, load_features, load_probes, load_targets, probe_router, probe_time
from .r4_cascade import ALPHAS, DELTA, PROBE_STEP, crc_lambda

CHEAP = ("classical", "vggt")


def early_features(F, names, P, step):
    """Geometric features + the early probe of the cheap pair (NaN if a method can't be probed: low coverage)."""
    new = [f"early_{m}_psnr" for m in CHEAP] + ["early_diff", "early_max", "early_n"]
    F2 = {}
    for c, x in F.items():
        ps = [P.get(c, {}).get(m, {}).get("psnr", {}).get(step, np.nan) for m in CHEAP]
        ok = [p for p in ps if not np.isnan(p)]
        F2[c] = np.concatenate([x, ps, [ps[0] - ps[1] if len(ok) == 2 else np.nan, max(ok) if ok else np.nan, len(ok)]])
    return F2, names + new


def _route(T, F2, names2, train, test, seeds):
    preds = [fit_forest(T, F2, names2, train, None, sd)[0](test) for sd in seeds]
    return {c: METHODS[int(np.argmin(np.mean([p[c] for p in preds], 0)))] for c in test}


def _gain_model(X, y, seed):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5, random_state=seed,
                                 n_jobs=-1).fit(X, y)


def evaluate(T, F, meta, names, P, early=250, seeds=(0, 1), n_splits=1000):
    chunks = sorted(c for c in T if c in F and c in P)
    scenes = sorted({meta[c]["scene"] for c in chunks})
    F2, names2 = early_features(F, names, P, early)
    stage2 = probe_router(P, PROBE_STEP)
    choice2 = {c: stage2(c) for c in chunks}
    L = {c: T[c]["loss"] for c in chunks}
    t = {c: {m: float(np.nan_to_num(v)) for m, v in T[c]["time"].items()} for c in chunks}
    best = {c: min(L[c].values()) for c in chunks}
    cost1 = lambda c, ch: t[c]["classical"] + t[c]["vggt"] + probe_time(P, c, early, CHEAP) + (t[c]["vggt_ba"] if ch == "vggt_ba" else 0.0)
    cost2 = lambda c: t[c]["classical"] + t[c]["vggt"] + t[c]["vggt_ba"] + probe_time(P, c, PROBE_STEP)
    gain = lambda c, ch: L[c][ch] - L[c][choice2[c]]
    X = lambda cs: np.array([F2[c] for c in cs])

    choice1, score, cal = {}, {}, {}
    for s in scenes:
        train = [c for c in chunks if meta[c]["scene"] != s]
        test = [c for c in chunks if meta[c]["scene"] == s]
        choice1.update(_route(T, F2, names2, train, test, seeds))
        inner_choice = {}
        for s2 in scenes:                                       # out-of-scene stage-1.5 choices for the training chunks
            if s2 != s:
                tr2 = [c for c in train if meta[c]["scene"] != s2]
                inner_choice.update(_route(T, F2, names2, tr2, [c for c in train if meta[c]["scene"] == s2], seeds))
        y = np.array([gain(c, inner_choice[c]) for c in train])
        gm = [_gain_model(X(train), y, sd) for sd in seeds]
        for c, v in zip(test, np.mean([g.predict(X(test)) for g in gm], 0)):
            score[c] = float(v)
        cal[s] = {}
        for s2 in scenes:                                       # out-of-scene gain scores for calibration
            if s2 == s:
                continue
            tr2 = [c for c in train if meta[c]["scene"] != s2]
            te2 = [c for c in train if meta[c]["scene"] == s2]
            y2 = np.array([gain(c, inner_choice[c]) for c in tr2])
            pv = np.mean([_gain_model(X(tr2), y2, sd).predict(X(te2)) for sd in seeds], 0)
            for c, v in zip(te2, pv):
                cal[s][c] = dict(score=float(v), bad=L[c][inner_choice[c]] - best[c] > DELTA)

    def run(cs, lam_of_scene):
        loss, time, esc, miss = [], [], [], []
        for c in cs:
            e = score[c] >= lam_of_scene(meta[c]["scene"])
            ch = choice2[c] if e else choice1[c]
            loss.append(L[c][ch]); time.append(cost2(c) if e else cost1(c, choice1[c])); esc.append(e)
            miss.append((not e) and L[c][choice1[c]] - best[c] > DELTA)
        return dict(loss=float(np.mean(loss)), time=float(np.mean(time)), esc=float(np.mean(esc)), risk=float(np.mean(miss)),
                    per_scene_risk={s: float(np.mean([m for c, m in zip(cs, miss) if meta[c]["scene"] == s])) for s in scenes})

    curve = []
    for q in np.linspace(0, 1, 11):
        thr = {s: (np.inf if q == 0 else -np.inf if q == 1 else float(np.quantile([v["score"] for v in cal[s].values()], 1 - q)))
               for s in scenes}
        curve.append(dict(q=float(q), **{k: v for k, v in run(chunks, thr.get).items() if k != "per_scene_risk"}))
    crc = {}
    for a in ALPHAS:
        lam = {s: crc_lambda([v["score"] for v in cal[s].values()], [v["bad"] for v in cal[s].values()], a) for s in scenes}
        r = run(chunks, lam.get)
        r["scenes_over"] = sum(v > a for v in r.pop("per_scene_risk").values())
        crc[a] = r
    rng = np.random.default_rng(0)
    exch = {}
    for a in ALPHAS:
        risks = []
        for _ in range(n_splits):
            perm = rng.permutation(len(chunks))
            cb, te = [chunks[i] for i in perm[: len(chunks) // 2]], [chunks[i] for i in perm[len(chunks) // 2:]]
            lam = crc_lambda([score[c] for c in cb], [L[c][choice1[c]] - best[c] > DELTA for c in cb], a)
            risks.append(run(te, lambda s: lam)["risk"])
        exch[a] = dict(mean_risk=float(np.mean(risks)), p_over=float(np.mean(np.array(risks) > a)))
    gains = np.array([gain(c, choice1[c]) for c in chunks])
    from scipy.stats import spearmanr
    base = dict(stage15=dict(loss=float(np.mean([L[c][choice1[c]] for c in chunks])),
                             time=float(np.mean([cost1(c, choice1[c]) for c in chunks]))),
                stage2=dict(loss=float(np.mean([L[c][choice2[c]] for c in chunks])), time=float(np.mean([cost2(c) for c in chunks]))),
                oracle=float(np.mean(list(best.values()))),
                helps=int((gains > 1e-9).sum()), hurts=int((gains < -1e-9).sum()),
                score_vs_gain_rho=float(spearmanr([score[c] for c in chunks], gains)[0]),
                stage15_bad=float(np.mean([L[c][choice1[c]] - best[c] > DELTA for c in chunks])))
    return dict(n=len(chunks), early=early, base=base, curve=curve, crc=crc, exch=exch,
                choice1=choice1, score=score)


def report(Es, r4):
    L = ["# R4b: escalation on predicted gain, after a cheap early probe", "",
         "Stage 1.5 adds an early 3DGS probe of classical and VGGT to the geometric router; the escalation score predicts "
         "the gain from escalating (which can be negative). Same leave-one-scene-out protocol, risk and conformal "
         "calibration as R4 ([r4_results.md](r4_results.md)); all scores nested and out-of-scene.", ""]
    L += ["| Policy | Escalated | Mean loss | Mean time | Realized risk | Scenes over α |", "|---|---|---|---|---|---|"]
    if r4:
        L.append(f"| R4 stage 1 only (geometric router) | 0% | {r4['base']['stage1']['loss']:.2f} | {r4['base']['stage1']['time']:.0f} s | {r4['base']['stage1_bad']:.2f} | – |")
        for a in (0.1, 0.2):
            r = r4["crc"]["pred_loss"][str(a)] if str(a) in r4["crc"]["pred_loss"] else r4["crc"]["pred_loss"][a]
            L.append(f"| R4 cascade, α = {a} (predicted loss) | {100 * r['esc']:.0f}% | {r['loss']:.2f} | {r['time']:.0f} s | {r['risk']:.3f} | {r['scenes_over']}/7 |")
    for E in Es:
        b = E["base"]
        L.append(f"| **R4b stage 1.5 only (early probe @{E['early']})** | 0% | {b['stage15']['loss']:.2f} | {b['stage15']['time']:.0f} s | {b['stage15_bad']:.2f} | – |")
        for a, r in E["crc"].items():
            L.append(f"| R4b @{E['early']}, α = {a} (predicted gain) | {100 * r['esc']:.0f}% | {r['loss']:.2f} | {r['time']:.0f} s | {r['risk']:.3f} | {r['scenes_over']}/7 |")
    b = Es[0]["base"]
    L.append(f"| always escalate | 100% | {b['stage2']['loss']:.2f} | {b['stage2']['time']:.0f} s | 0 | – |")
    L.append(f"| oracle | – | {b['oracle']:.2f} | – | – | – |")
    for E in Es:
        b = E["base"]
        L += ["", f"**Early probe @{E['early']}:** escalation helps on {b['helps']} chunks and hurts on {b['hurts']}; "
              f"predicted vs true gain Spearman ρ = {b['score_vs_gain_rho']:.2f}.", "",
              "| Escalated (target) | Mean loss | Mean time | Risk |", "|---|---|---|---|"]
        for p in E["curve"]:
            L.append(f"| {100 * p['q']:.0f}% | {p['loss']:.2f} dB | {p['time']:.0f} s | {p['risk']:.3f} |")
        L += ["", "Exchangeable check (1,000 random 50/50 splits): " + "; ".join(
            f"α = {a}: mean risk {r['mean_risk']:.3f}, {100 * r['p_over']:.0f}% of splits over" for a, r in E["exch"].items()) + "."]
    return "\n".join(L) + "\n"


def plot(Es, r4, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    if r4:
        for k, col, ls, lab in [("pred_loss", "#9ca3af", "-", "R4: predicted loss"), ("hindsight", "#16a34a", "-.", "R4: hindsight (not deployable)")]:
            cv = r4["curve"][k]
            ax.plot([p["time"] for p in cv], [p["loss"] for p in cv], ls, color=col, marker="o", ms=3, label=lab)
    for E, col in zip(Es, ["#2563eb", "#7c3aed"]):
        cv = E["curve"]
        ax.plot([p["time"] for p in cv], [p["loss"] for p in cv], "-", color=col, marker="o", ms=3,
                label=f"R4b: early probe @{E['early']} + predicted gain")
        for a in (0.1, 0.2):
            r = E["crc"][a]
            ax.scatter([r["time"]], [r["loss"]], marker="*", s=90, color="#f59e0b", zorder=6)
            ax.annotate(f"α={a}", (r["time"], r["loss"]), textcoords="offset points", xytext=(5, -10), fontsize=8)
    ax.axhline(Es[0]["base"]["oracle"], color="#111827", lw=0.8, ls="--")
    ax.set_xlabel("mean decision time per chunk (s): reconstructions + probe training")
    ax.set_ylabel("mean twin loss vs official poses (dB)")
    ax.set_title("Escalating on predicted gain (leave-one-scene-out, 105 chunks)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("labels")
    ap.add_argument("probes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--early", type=int, nargs="+", default=[250, 500])
    ap.add_argument("--r4", help="r4_results.json, to compare")
    a = ap.parse_args()
    F, meta, names = load_features(a.features)
    T, P = load_targets([a.labels]), load_probes(a.probes)
    Es = [evaluate(T, F, meta, names, P, early=e) for e in a.early]
    r4 = json.load(open(a.r4)) if a.r4 else None
    if r4:                                      # json keys are strings
        r4["crc"] = {k: {float(x): v for x, v in d.items()} for k, d in r4["crc"].items()}
    d = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(d, exist_ok=True)
    plot(Es, r4, os.path.join(d, "r4b_cost_quality.png"))
    md = report(Es, r4) + "\n![R4b cost-quality curve](r4b_cost_quality.png)\n"
    open(a.out, "w", encoding="utf-8").write(md)
    json.dump([{k: v for k, v in E.items()} for E in Es], open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1, default=float)
    print(md)


if __name__ == "__main__":
    main()

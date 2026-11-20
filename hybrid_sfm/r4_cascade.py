"""R4: cost-aware cascade routing with conformal risk control.

Stage 1 (always): classical + VGGT run; the R3 geometric router (random forests on 32 features) picks a method.
Stage 2 (only if escalated): VGGT+BA also runs and every usable candidate gets a short 3DGS probe (R3b);
the best probe wins. A score s(c) from stage 1 decides escalation: escalate when s(c) >= lambda.

Scores (higher = more reason to escalate):
  pred_loss   the stage-1 forest's predicted loss of its own choice (a chunk predicted to go badly)
  margin      minus the gap between the best and second-best predicted loss (an uncertain choice)
  random      a fixed random number per chunk (control: escalation that ignores the chunk)
  hindsight   the true gain from escalating (not deployable: the best any score could do)

Conformal risk control (Angelopoulos et al., 2022) picks lambda on calibration chunks so that, for a new
exchangeable chunk, E[L(lambda)] <= alpha, with
    L_i(lambda) = 1[ s_i < lambda  and  regret of the stage-1 choice on chunk i > delta ]
"skipped the probe although the cheap choice was more than delta dB worse than the best method".
L_i is monotone (non-increasing as lambda falls), as CRC requires. The bound is
    (n / (n + 1)) * Rhat_n(lambda) + 1 / (n + 1) <= alpha     (loss bounded by 1).

Every score comes from a model that never saw the chunk's scene: for a held-out scene, forests are fitted on the
other 6 scenes, and calibration scores for those 6 come from an inner leave-one-scene-out (nested).

Usage:
    python -m hybrid_sfm.r4_cascade docs/research/results/r3/features.csv docs/research/results/r2/r2_labels.jsonl \
        docs/research/results/r3b/r3b_probes.jsonl --out docs/research/results/r4/r4_results.md
"""
import argparse
import json
import os
import sys

import numpy as np

from .r3_router import METHODS, fit_forest, load_features, load_probes, load_targets, probe_router, probe_time

PROBE_STEP = 1000
DELTA = 0.5                       # dB: a decision within DELTA of the best method counts as fine
ALPHAS = (0.05, 0.1, 0.2, 0.3)
SCORES = ("pred_loss", "margin", "random", "hindsight")


# ------------------------------------------------------------------ stage 1 predictions (nested, out-of-scene)
def _forest_pred(T, F, names, train, test, seeds):
    preds = []
    for seed in seeds:
        p, _ = fit_forest(T, F, names, train, None, seed)
        preds.append(p(test))
    return {c: np.mean([pp[c] for pp in preds], 0) for c in test}


def stage1(T, F, meta, names, chunks, seeds=(0, 1)):
    """outer[c]: predicted losses for chunk c from forests that never saw c's scene.
    inner[s][c]: for each held-out scene s, predictions for the chunks of the other scenes, each from forests that
    saw neither s nor the chunk's own scene (calibration scores for s)."""
    scenes = sorted({meta[c]["scene"] for c in chunks})
    outer, inner = {}, {}
    for s in scenes:
        train = [c for c in chunks if meta[c]["scene"] != s]
        outer.update(_forest_pred(T, F, names, train, [c for c in chunks if meta[c]["scene"] == s], seeds))
        inner[s] = {}
        for s2 in scenes:
            if s2 == s:
                continue
            tr2 = [c for c in train if meta[c]["scene"] != s2]
            inner[s].update(_forest_pred(T, F, names, tr2, [c for c in train if meta[c]["scene"] == s2], seeds))
    return outer, inner


# ------------------------------------------------------------------ per-chunk quantities
def chunk_table(T, P, pred, chunks, seed=0):
    """Per chunk: stage-1 choice, its loss and time; escalated choice, loss and time; regret; scores."""
    rng = np.random.default_rng(seed)
    rnd = {c: rng.uniform() for c in sorted(chunks)}
    probe = probe_router(P, PROBE_STEP)
    rows = {}
    for c in chunks:
        L = T[c]["loss"]
        t = {m: float(np.nan_to_num(v)) for m, v in T[c]["time"].items()}     # a missing wall time counts as 0
        pl = pred[c]
        ch1 = METHODS[int(np.argmin(pl))]
        best = min(L.values())
        t1 = t["classical"] + t["vggt"] + (t["vggt_ba"] if ch1 == "vggt_ba" else 0.0)
        ch2 = probe(c)
        t2 = t["classical"] + t["vggt"] + t["vggt_ba"] + probe_time(P, c, PROBE_STEP)
        srt = np.sort(pl)
        rows[c] = dict(choice1=ch1, loss1=L[ch1], time1=t1, choice2=ch2, loss2=L[ch2], time2=t2,
                       regret1=L[ch1] - best, oracle=best,
                       score=dict(pred_loss=float(pl.min()), margin=float(-(srt[1] - srt[0])), random=rnd[c],
                                  hindsight=float(L[ch1] - L[ch2])))
    return rows


def policy(rows, chunks, score, lam):
    """Mean loss, mean time, escalation rate and risk of 'escalate iff score >= lam'."""
    esc = np.array([rows[c]["score"][score] >= lam for c in chunks])
    loss = np.array([rows[c]["loss2"] if e else rows[c]["loss1"] for c, e in zip(chunks, esc)])
    time = np.array([rows[c]["time2"] if e else rows[c]["time1"] for c, e in zip(chunks, esc)])
    miss = np.array([(not e) and rows[c]["regret1"] > DELTA for c, e in zip(chunks, esc)])
    return dict(loss=loss, time=time, esc=esc, miss=miss)


# ------------------------------------------------------------------ conformal risk control
def crc_lambda(cal_scores, cal_bad, alpha):
    """Largest lambda (least escalation) with (n/(n+1)) Rhat(lambda) + 1/(n+1) <= alpha. -inf = escalate everything."""
    s, bad = np.asarray(cal_scores, float), np.asarray(cal_bad, bool)
    n = len(s)
    for lam in np.concatenate([[np.inf], np.sort(np.unique(s))[::-1]]):
        risk = np.mean((s < lam) & bad)
        if n / (n + 1) * risk + 1 / (n + 1) <= alpha:
            return float(lam)
    return -np.inf


def evaluate(T, F, meta, names, P, seeds=(0, 1), n_splits=1000):
    chunks = sorted(c for c in T if c in F and c in P)
    scenes = sorted({meta[c]["scene"] for c in chunks})
    outer, inner = stage1(T, F, meta, names, chunks, seeds)
    rows = chunk_table(T, P, outer, chunks)
    cal_rows = {s: chunk_table(T, P, inner[s], list(inner[s])) for s in scenes}

    # 1. cost-quality curve: escalate a target fraction q, threshold = calibration quantile (per held-out scene)
    curve = {k: [] for k in SCORES}
    for k in SCORES:
        for q in np.linspace(0, 1, 11):
            agg = dict(loss=[], time=[], esc=[], miss=[])
            for s in scenes:
                test = [c for c in chunks if meta[c]["scene"] == s]
                cal = [cal_rows[s][c]["score"][k] for c in cal_rows[s]] if k != "hindsight" else \
                      [rows[c]["score"][k] for c in test]                       # hindsight: ranks the test chunks themselves
                lam = np.inf if q == 0 else (-np.inf if q == 1 else float(np.quantile(cal, 1 - q)))
                r = policy(rows, test, k, lam)
                for key in agg:
                    agg[key] += list(r[key])
            curve[k].append(dict(q=float(q), esc=float(np.mean(agg["esc"])), loss=float(np.mean(agg["loss"])),
                                 time=float(np.mean(agg["time"])), risk=float(np.mean(agg["miss"]))))

    # 2. conformal risk control, leave-one-scene-out (calibrate on the other 6 scenes: tests shift across scenes)
    crc = {k: {} for k in ("pred_loss", "margin", "random")}
    for k in crc:
        for a in ALPHAS:
            agg = dict(loss=[], time=[], esc=[], miss=[]); per_scene = {}
            for s in scenes:
                cr = cal_rows[s]
                lam = crc_lambda([cr[c]["score"][k] for c in cr], [cr[c]["regret1"] > DELTA for c in cr], a)
                test = [c for c in chunks if meta[c]["scene"] == s]
                r = policy(rows, test, k, lam)
                per_scene[s] = float(np.mean(r["miss"]))
                for key in agg:
                    agg[key] += list(r[key])
            crc[k][a] = dict(risk=float(np.mean(agg["miss"])), loss=float(np.mean(agg["loss"])),
                             time=float(np.mean(agg["time"])), esc=float(np.mean(agg["esc"])), per_scene=per_scene,
                             scenes_over=sum(v > a for v in per_scene.values()))

    # 3. exchangeable check: random 50/50 splits of the 105 chunks (out-of-scene scores), calibrate on one half
    rng = np.random.default_rng(0)
    exch = {}
    for a in ALPHAS:
        risks, escs = [], []
        for _ in range(n_splits):
            perm = rng.permutation(len(chunks))
            cal, test = [chunks[i] for i in perm[: len(chunks) // 2]], [chunks[i] for i in perm[len(chunks) // 2:]]
            lam = crc_lambda([rows[c]["score"]["pred_loss"] for c in cal], [rows[c]["regret1"] > DELTA for c in cal], a)
            r = policy(rows, test, "pred_loss", lam)
            risks.append(np.mean(r["miss"])); escs.append(np.mean(r["esc"]))
        exch[a] = dict(mean_risk=float(np.mean(risks)), p_over=float(np.mean(np.array(risks) > a)),
                       esc=float(np.mean(escs)))

    base = dict(stage1=dict(loss=float(np.mean([rows[c]["loss1"] for c in chunks])),
                            time=float(np.mean([rows[c]["time1"] for c in chunks]))),
                probe_all=dict(loss=float(np.mean([rows[c]["loss2"] for c in chunks])),
                               time=float(np.mean([rows[c]["time2"] for c in chunks]))),
                always_vggt=dict(loss=float(np.mean([T[c]["loss"]["vggt"] for c in chunks])),
                                 time=float(np.mean([T[c]["time"]["vggt"] for c in chunks]))),
                oracle=dict(loss=float(np.mean([rows[c]["oracle"] for c in chunks]))),
                stage1_bad=float(np.mean([rows[c]["regret1"] > DELTA for c in chunks])),
                escalation_hurts=int(sum(rows[c]["loss2"] > rows[c]["loss1"] + 1e-9 for c in chunks)),
                escalation_helps=int(sum(rows[c]["loss2"] < rows[c]["loss1"] - 1e-9 for c in chunks)))
    return dict(n=len(chunks), scenes=scenes, base=base, curve=curve, crc=crc, exch=exch,
                rows={c: {k: v for k, v in r.items()} for c, r in rows.items()})


# ------------------------------------------------------------------ report
def plot(E, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    style = dict(pred_loss=("#2563eb", "-", "predicted loss (ours)"), margin=("#7c3aed", "--", "decision margin"),
                 random=("#9ca3af", ":", "random escalation"), hindsight=("#16a34a", "-.", "hindsight (not deployable)"))
    for k in SCORES:
        col, ls, lab = style[k]
        cv = E["curve"][k]
        ax.plot([p["time"] for p in cv], [p["loss"] for p in cv], ls, color=col, marker="o", ms=3, label=lab)
    b = E["base"]
    ax.scatter([b["always_vggt"]["time"]], [b["always_vggt"]["loss"]], color="#dc2626", zorder=5)
    ax.annotate("always VGGT", (b["always_vggt"]["time"], b["always_vggt"]["loss"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axhline(b["oracle"]["loss"], color="#111827", lw=0.8, ls="--")
    ax.annotate("oracle", (ax.get_xlim()[0], b["oracle"]["loss"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    for a in (0.1, 0.2):
        r = E["crc"]["pred_loss"][a]
        ax.scatter([r["time"]], [r["loss"]], marker="*", s=90, color="#f59e0b", zorder=6)
        ax.annotate(f"CRC α={a}", (r["time"], r["loss"]), textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel("mean decision time per chunk (s): reconstructions + probe training")
    ax.set_ylabel("mean twin loss vs official poses (dB)")
    ax.set_title("Cost-aware routing, leave-one-scene-out (105 chunks)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def report(E, fig_name):
    b = E["base"]
    L = ["# R4: cost-aware cascade with conformal risk control", "",
         f"**{E['n']} chunks, {len(E['scenes'])} scenes, leave-one-scene-out.** Stage 1 always runs classical + VGGT "
         f"and the R3 geometric router picks a method. Escalation adds VGGT+BA and a {PROBE_STEP}-step 3DGS probe of "
         f"every usable candidate (R3b), and the best probe wins. Loss = dB below the official poses (3k-step 3DGS). "
         f"Time = reconstructions + probe training, mean per chunk.", "",
         "| Reference policy | Mean loss | Mean time |", "|---|---|---|",
         f"| always VGGT | {b['always_vggt']['loss']:.2f} | {b['always_vggt']['time']:.0f} s |",
         f"| stage 1 only (never escalate) | {b['stage1']['loss']:.2f} | {b['stage1']['time']:.0f} s |",
         f"| always escalate (probe everything) | {b['probe_all']['loss']:.2f} | {b['probe_all']['time']:.0f} s |",
         f"| oracle | {b['oracle']['loss']:.2f} | – |", "",
         f"Stage 1's choice is more than {DELTA} dB worse than the best method on {100 * b['stage1_bad']:.0f}% of chunks. "
         f"Escalating changes the outcome for the better on {b['escalation_helps']} chunks and for the worse on "
         f"{b['escalation_hurts']}.", "",
         f"![cost-quality curve]({fig_name})", "",
         "**Cost–quality curve** (escalate a target share of chunks; the threshold is that quantile of the calibration "
         "scores from the other 6 scenes):", "",
         "| Escalated (target) | " + " | ".join(f"{k}: loss / time" for k in SCORES) + " |",
         "|---|" + "---|" * len(SCORES)]
    for i in range(len(E["curve"]["pred_loss"])):
        L.append(f"| {100 * E['curve']['pred_loss'][i]['q']:.0f}% | " + " | ".join(
            f"{E['curve'][k][i]['loss']:.2f} dB / {E['curve'][k][i]['time']:.0f} s" for k in SCORES) + " |")
    L += ["", f"**Conformal risk control** (risk = share of chunks where the probe was skipped although stage 1's choice "
          f"was > {DELTA} dB worse than the best method). λ calibrated on the other 6 scenes, tested on the held-out "
          f"scene; the guarantee assumes exchangeable chunks, which scene shift violates, so this is the hard test:", "",
          "| Score | α | Realized risk | Scenes over α | Escalated | Mean loss | Mean time |", "|---|---|---|---|---|---|---|"]
    for k, d in E["crc"].items():
        for a, r in d.items():
            L.append(f"| {k} | {a} | {r['risk']:.3f} | {r['scenes_over']}/{len(E['scenes'])} | {100 * r['esc']:.0f}% | "
                     f"{r['loss']:.2f} | {r['time']:.0f} s |")
    L += ["", "**Exchangeable check** (predicted-loss score; 1,000 random 50/50 splits of the 105 chunks; scores are "
          "out-of-scene; calibrate on one half, test on the other):", "",
          "| α | Mean realized risk | Splits over α | Escalated |", "|---|---|---|---|"]
    for a, r in E["exch"].items():
        L.append(f"| {a} | {r['mean_risk']:.3f} | {100 * r['p_over']:.0f}% | {100 * r['esc']:.0f}% |")
    return "\n".join(L) + "\n"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("labels")
    ap.add_argument("probes")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    F, meta, names = load_features(a.features)
    E = evaluate(load_targets([a.labels]), F, meta, names, load_probes(a.probes))
    d = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(d, exist_ok=True)
    fig = "r4_cost_quality.png"
    plot(E, os.path.join(d, fig))
    md = report(E, fig)
    open(a.out, "w", encoding="utf-8").write(md)
    json.dump({k: v for k, v in E.items()}, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1, default=float)
    print(md)


if __name__ == "__main__":
    main()

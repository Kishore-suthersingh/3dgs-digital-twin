"""R3: learned per-chunk routing between classical, VGGT and VGGT+BA, evaluated leave-one-scene-out.

Targets come from R2: loss_m = PSNR(official poses) - PSNR(method m), 3k-step 3DGS, FAIL_LOSS when m gives no
usable twin. Features come from r3_features (no reference poses). Every router is fitted on 6 scenes and scored
on the 7th, so nothing about a test scene is seen in training.

Routers:
  always <m>                fixed choice
  best fixed (train)        the fixed choice with the lowest training loss
  kind rule (metadata)      best method per chunk kind on the training scenes. Uses how the chunk was built,
                            which a deployed system does not know: an optimistic heuristic, not a baseline we beat fairly
  stump: <feature>          one threshold on one feature, a method on each side, fitted on training scenes
  best stump (train)        the stump whose feature does best on the training scenes
  learned (...)             one random forest per method predicts its loss; route to the lowest prediction
  pose-error router (GT)    picks the usable method with the best AUC@3 against the reference poses. Not
                            deployable (needs ground truth); tests whether even perfect pose error is enough (H1)
  oracle                    best method per chunk (upper bound)

Usage:
    python -m hybrid_sfm.r3_router docs/research/results/r3/features.csv docs/research/results/r2/r2_labels.jsonl \
        --out docs/research/results/r3/r3_results.md
"""
import argparse
import csv
import json
import sys
import os

import numpy as np

from .r2_summary import FAIL_LOSS, load

METHODS = ["classical", "vggt", "vggt_ba"]
GROUPS = dict(conf="VGGT confidence", geo="VGGT camera layout", foc="VGGT focal", mv="multi-view depth consistency",
              cls="classical stats", agree="VGGT vs classical agreement", n="chunk size")
VGGT_ONLY = ("conf", "geo", "foc", "mv", "n")
STUMP_FEATURES = ["cls_registered", "conf_mean", "mv_consistent", "agree_auc3", "geo_nn_baseline"]
NEAR_OPT_DB = 0.5          # "near-oracle": within this many dB of the best method
TIE_DB = 0.3               # pairwise ranking ignores pairs closer than this (label noise, R2 fidelity check)


# ------------------------------------------------------------------ data
def load_features(path):
    rows = list(csv.DictReader(open(path)))
    names = [k for k in rows[0] if k not in ("chunk", "scene", "kind")]
    F = {r["chunk"]: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for k in names]) for r in rows}
    meta = {r["chunk"]: dict(scene=r["scene"], kind=r["kind"]) for r in rows}
    return F, meta, names


def load_targets(label_paths, steps=3000):
    """{chunk: dict(loss={m: dB}, time={m: s}, pose={m: dict}, ref_psnr)} for chunks with a reference label."""
    recs = load(label_paths)
    by = {}
    for r in recs:
        if r["variant"] == "own" and r["steps"] == steps:
            by.setdefault(r["chunk"], {})[r["method"]] = r
    out = {}
    for c, d in by.items():
        ref = d.get("ref")
        if not ref or ref["status"] != "ok" or not ref.get("gs"):
            continue
        T = dict(loss={}, time={}, pose={}, ref_psnr=ref["gs"]["psnr"])
        for m in METHODS:
            r = d.get(m)
            ok = r is not None and r["status"] == "ok" and r.get("gs")
            T["loss"][m] = ref["gs"]["psnr"] - r["gs"]["psnr"] if ok else FAIL_LOSS
            T["time"][m] = ((r or {}).get("recon") or {}).get("wall_s", np.nan)
            T["pose"][m] = dict((r or {}).get("pose") or {}, focal_err=(r or {}).get("focal_err")) if ok else None
        out[c] = T
    return out


def load_probes(path):
    """R3b probes: {chunk: {method: {"status", "psnr": {step: dB}, "train_s": {step: s}}}} (later lines win)."""
    P = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        pr = r.get("probe") or {}
        P.setdefault(r["chunk"], {})[r["method"]] = dict(
            status=r["status"], psnr={int(k): v["psnr"] for k, v in pr.items()},
            train_s={int(k): v.get("train_s") or 0.0 for k, v in pr.items()})
    return P


def probe_steps(P):
    return sorted({s for d in P.values() for r in d.values() for s in r["psnr"]})


def add_probe_features(F, names, P, step):
    """Per method: probe PSNR at `step` and its gap to the chunk's best probe (NaN when the method can't be probed)."""
    new = [f"probe_{m}_psnr" for m in METHODS] + [f"probe_{m}_gap" for m in METHODS]
    F2 = {}
    for c, x in F.items():
        ps = [P.get(c, {}).get(m, {}).get("psnr", {}).get(step, np.nan) for m in METHODS]
        best = np.nanmax(ps) if not np.all(np.isnan(ps)) else np.nan
        F2[c] = np.concatenate([x, ps, [p - best for p in ps]])
    return F2, names + new


def probe_router(P, step, candidates=METHODS):
    """Highest probe PSNR among the candidates that could be probed; VGGT (never fails in R2) if none."""
    def choose(c):
        ok = {m: P[c][m]["psnr"][step] for m in candidates if m in P.get(c, {}) and step in P[c][m]["psnr"]}
        return max(ok, key=ok.get) if ok else "vggt"
    return choose


def probe_time(P, c, step, candidates=METHODS):
    return sum(P[c][m]["train_s"].get(step, 0.0) for m in candidates if m in P.get(c, {}))


# ------------------------------------------------------------------ routers
def _losses(T, chunks):
    return np.array([[T[c]["loss"][m] for m in METHODS] for c in chunks])


def fit_fixed(T, train):
    return METHODS[int(_losses(T, train).mean(0).argmin())]


def fit_kind(T, meta, train):
    best, fallback = {}, fit_fixed(T, train)
    for k in {meta[c]["kind"] for c in train}:
        cs = [c for c in train if meta[c]["kind"] == k]
        best[k] = METHODS[int(_losses(T, cs).mean(0).argmin())]
    return lambda c: best.get(meta[c]["kind"], fallback)


def fit_stump(T, F, names, train, feature):
    """Threshold t on one feature; method a if x <= t (or missing), b otherwise. Minimises mean training loss."""
    j = names.index(feature)
    x = np.array([F[c][j] for c in train])
    L = _losses(T, train)
    best = (L.mean(0).min(), np.inf, int(L.mean(0).argmin()), int(L.mean(0).argmin()))
    for t in np.unique(x[~np.isnan(x)]):
        lo = ~(x > t)                                       # NaN goes left
        if lo.all() or not lo.any():
            continue
        a, b = int(L[lo].mean(0).argmin()), int(L[~lo].mean(0).argmin())
        score = (L[lo, a].sum() + L[~lo, b].sum()) / len(train)
        if score < best[0] - 1e-12:
            best = (score, t, a, b)
    _, t, a, b = best
    return (lambda c: METHODS[a] if not (F[c][j] > t) else METHODS[b]), best[0]


def fit_forest(T, F, names, train, groups=None, seed=0, model="rf"):
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    cols = [i for i, n in enumerate(names) if groups is None or n.split("_")[0] in groups]
    X = np.array([F[c][cols] for c in train])
    models = []
    for k, m in enumerate(METHODS):
        y = np.array([T[c]["loss"][m] for c in train])
        if model == "rf":
            reg = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.5, random_state=seed, n_jobs=-1)
        else:
            reg = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=5,
                                                random_state=seed)
        models.append(reg.fit(X, y))
    imp = np.mean([getattr(r, "feature_importances_", np.zeros(len(cols))) for r in models], 0)
    def pred(cs):                                           # {chunk: predicted loss per method}, batched
        X = np.array([F[c][cols] for c in cs])
        P = np.stack([r.predict(X) for r in models], 1)
        return dict(zip(cs, P))
    return pred, dict(zip([names[i] for i in cols], imp))


def pose_router(T, metric="AUC@3", higher_better=True):
    def choose(c):
        ok = {m: p for m, p in T[c]["pose"].items() if p is not None and p.get(metric) is not None}
        if not ok:
            return "vggt"
        return (max if higher_better else min)(ok, key=lambda m: ok[m][metric])
    return choose


# ------------------------------------------------------------------ evaluation
def route_time(T, c, choice, probes):
    """Pose time of a routing decision: the probes it needs, plus the chosen method if it isn't one of them."""
    t = T[c]["time"]
    return sum(t[p] for p in probes) + (t[choice] if choice not in probes else 0.0)


def evaluate(T, F, meta, names, seeds=(0, 1, 2, 3, 4), n_boot=2000, P=None):
    chunks = sorted(c for c in T if c in F and (P is None or c in P))
    scenes = sorted({meta[c]["scene"] for c in chunks})
    choice = {}                    # router -> {chunk: method}
    probes = {}                    # router -> probes it always runs
    pred_loss = {}                 # chunk -> out-of-fold predicted losses (learned, all features)
    importance = []
    for m in METHODS:
        choice[f"always {m}"] = {c: m for c in chunks}; probes[f"always {m}"] = ()
    choice["oracle"] = {c: min(METHODS, key=T[c]["loss"].get) for c in chunks}; probes["oracle"] = ()
    for metric, hb in [("AUC@3", True), ("AUC@30", True), ("rot_err_median", False)]:
        r = pose_router(T, metric, hb)
        name = f"pose-error router: {metric} (GT, not deployable)"
        choice[name] = {c: r(c) for c in chunks}; probes[name] = tuple(METHODS)
    extra_t = {}                   # router -> {chunk: probe training seconds}
    steps = probe_steps(P) if P else []
    for st in steps:
        for label, cand in [("all methods", METHODS), ("classical + VGGT", ("classical", "vggt"))]:
            name = f"probe@{st}: best probe PSNR ({label})"
            r = probe_router(P, st, cand)
            choice[name] = {c: r(c) for c in chunks}; probes[name] = tuple(cand)
            extra_t[name] = {c: probe_time(P, c, st, cand) for c in chunks}
    learned = {"learned: all features": (None, "rf", ("classical", "vggt")),
               "learned: VGGT probe only": (VGGT_ONLY, "rf", ("vggt",)),
               "learned: all features, gradient boosting": (None, "hgb", ("classical", "vggt"))}
    for st in steps:
        learned[f"learned: all features + probe@{st}"] = (None, "rf", tuple(METHODS), st)
    for g in GROUPS:
        learned[f"ablation: without {GROUPS[g]}"] = (tuple(k for k in GROUPS if k != g), "rf", ("classical", "vggt"))
    for name in ["best fixed (train)", "kind rule (metadata)", "best stump (train)"] + [f"stump: {f}" for f in STUMP_FEATURES] + list(learned):
        choice[name] = {}
    stump_pick = []
    for s in scenes:
        train = [c for c in chunks if meta[c]["scene"] != s]
        test = [c for c in chunks if meta[c]["scene"] == s]
        fx = fit_fixed(T, train)
        kind = fit_kind(T, meta, train)
        stumps = {f: fit_stump(T, F, names, train, f) for f in names}
        bf = min(stumps, key=lambda f: stumps[f][1]); stump_pick.append(bf)
        for c in test:
            choice["best fixed (train)"][c] = fx
            choice["kind rule (metadata)"][c] = kind(c)
            choice["best stump (train)"][c] = stumps[bf][0](c)
            for f in STUMP_FEATURES:
                choice[f"stump: {f}"][c] = stumps[f][0](c)
        for name, spec in learned.items():
            groups, model = spec[0], spec[1]
            Fx, nx = add_probe_features(F, names, P, spec[3]) if len(spec) > 3 else (F, names)
            preds = []
            for seed in (seeds if model == "rf" else seeds[:1]):
                p, imp = fit_forest(T, Fx, nx, train, groups, seed, model)
                preds.append(p(test))
                if name == "learned: all features":
                    importance.append(imp)
            for c in test:
                pl = np.mean([pp[c] for pp in preds], 0)
                choice[name][c] = METHODS[int(pl.argmin())]
                if name == "learned: all features":
                    pred_loss[c] = pl
    for name, spec in learned.items():
        probes[name] = spec[2]
        if len(spec) > 3:
            extra_t[name] = {c: probe_time(P, c, spec[3]) for c in chunks}
    for name in ["best fixed (train)", "kind rule (metadata)"]:
        probes[name] = ()
    for name in ["best stump (train)"] + [f"stump: {f}" for f in STUMP_FEATURES]:
        probes[name] = ("classical", "vggt")

    L = {c: T[c]["loss"] for c in chunks}
    oracle = np.array([min(L[c].values()) for c in chunks])
    base = np.array([L[c]["vggt"] for c in chunks])          # best fixed policy on the full set (R2)
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(chunks), (n_boot, len(chunks)))
    res = {}
    for name, ch in choice.items():
        l = np.array([L[c][ch[c]] for c in chunks])
        d = l - base
        lo, hi = np.percentile(d[boot].mean(1), [2.5, 97.5])
        res[name] = dict(mean=float(l.mean()), median=float(np.median(l)), regret=float((l - oracle).mean()),
                         near=float((l - oracle <= NEAR_OPT_DB).mean()), fails=int((l >= FAIL_LOSS).sum()),
                         gap_closed=float((base.mean() - l.mean()) / (base.mean() - oracle.mean())),
                         vs_vggt=(float(d.mean()), float(lo), float(hi)),
                         time=float(np.nanmean([route_time(T, c, ch[c], probes[name]) + extra_t.get(name, {}).get(c, 0.0)
                                                for c in chunks])),
                         probe_s=float(np.mean([extra_t.get(name, {}).get(c, 0.0) for c in chunks])),
                         picks={m: sum(ch[c] == m for c in chunks) for m in METHODS},
                         per_scene={s: float(np.mean([L[c][ch[c]] for c in chunks if meta[c]["scene"] == s])) for s in scenes})
    imp = {k: float(np.mean([d[k] for d in importance])) for k in importance[0]}
    grp = {g: sum(v for k, v in imp.items() if k.split("_")[0] == g) for g in GROUPS}
    pair = pairwise_ranking(T, chunks, pred_loss, P)
    return dict(n=len(chunks), scenes=scenes, results=res, importance=imp, group_importance=grp, pairwise=pair,
                stump_pick=stump_pick, choice=choice)


def pairwise_ranking(T, chunks, pred_loss, P=None):
    """Within a chunk, for each pair of usable methods whose PSNR differs by >= TIE_DB: does the score order them right?"""
    scores = {"AUC@3 (GT)": lambda c, m: -T[c]["pose"][m]["AUC@3"],
              "AUC@30 (GT)": lambda c, m: -T[c]["pose"][m]["AUC@30"],
              "median rotation error (GT)": lambda c, m: T[c]["pose"][m]["rot_err_median"],
              "ATE (GT)": lambda c, m: T[c]["pose"][m]["ATE_%"],
              "focal error (GT)": lambda c, m: abs(T[c]["pose"][m]["focal_err"] or 0),
              "learned predicted loss (no GT)": lambda c, m: pred_loss[c][METHODS.index(m)]}
    for st in (probe_steps(P) if P else []):
        scores[f"probe PSNR at {st} steps (no GT)"] = (lambda st: lambda c, m: -P[c][m]["psnr"].get(st, -np.inf)
                                                        if m in P.get(c, {}) else np.inf)(st)
    out = {}
    for name, sc in scores.items():
        right = n = 0
        for c in chunks:
            ok = [m for m in METHODS if T[c]["pose"][m] is not None]
            for a in range(len(ok)):
                for b in range(a + 1, len(ok)):
                    la, lb = T[c]["loss"][ok[a]], T[c]["loss"][ok[b]]
                    if abs(la - lb) < TIE_DB or c not in pred_loss:
                        continue
                    sa, sb = sc(c, ok[a]), sc(c, ok[b])
                    if sa == sb:
                        right += 0.5
                    else:
                        right += (sa < sb) == (la < lb)
                    n += 1
        out[name] = (right / max(n, 1), n)
    return out


# ------------------------------------------------------------------ report
def report(E):
    R = E["results"]
    L = [f"# R3 router results", "",
         f"**{E['n']} chunks, {len(E['scenes'])} scenes, leave-one-scene-out** (every router is fitted on 6 scenes and "
         f"scored on the 7th). Loss = dB below the official poses (3k-step 3DGS, {FAIL_LOSS:.0f} dB for no usable twin). "
         f"*Gap closed* = share of the always-VGGT → oracle gap recovered. *vs always VGGT* = mean loss difference "
         f"with a paired bootstrap 95% CI (negative = better). Pose time includes the probes a router needs "
         f"(classical + VGGT forward for the full router) and, for probe routers, the probes' 3DGS training time.", "",
         "| Router | Mean loss | Median | Regret | Near-oracle (≤0.5 dB) | Unusable | Gap closed | vs always VGGT (95% CI) | Pose time | Picks c / v / ba |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    order = ["always classical", "always vggt", "always vggt_ba", "best fixed (train)", "kind rule (metadata)",
             "best stump (train)"] + [f"stump: {f}" for f in STUMP_FEATURES] + \
            ["learned: VGGT probe only", "learned: all features", "learned: all features, gradient boosting"] + \
            [k for k in R if k.startswith("probe@")] + [k for k in R if "+ probe@" in k] +             [k for k in R if k.startswith("pose-error")] + ["oracle"]
    for k in order:
        r = R[k]
        d, lo, hi = r["vs_vggt"]
        name = f"**{k}**" if k == "learned: all features" else k
        L.append(f"| {name} | {r['mean']:.2f} | {r['median']:.2f} | {r['regret']:.2f} | {100 * r['near']:.0f}% | {r['fails']} | "
                 f"{100 * r['gap_closed']:.0f}% | {d:+.2f} [{lo:+.2f}, {hi:+.2f}] | {r['time']:.0f} s | "
                 f"{' / '.join(str(r['picks'][m]) for m in METHODS)} |")
    L += ["", f"Best stump picked per fold: {', '.join(sorted(set(E['stump_pick'])))}.", "",
          "**Per held-out scene** (mean loss, dB):", "",
          "| Scene | always classical | always VGGT | learned: all features | oracle |", "|---|---|---|---|---|"]
    for s in E["scenes"]:
        L.append(f"| {s} | " + " | ".join(f"{R[k]['per_scene'][s]:.2f}" for k in
                                         ["always classical", "always vggt", "learned: all features", "oracle"]) + " |")
    L += ["", "**Ablation** (learned router without one feature group):", "",
          "| Without | Mean loss | Gap closed |", "|---|---|---|"]
    full = R["learned: all features"]
    L.append(f"| (nothing, full router) | {full['mean']:.2f} | {100 * full['gap_closed']:.0f}% |")
    for g, label in GROUPS.items():
        r = R[f"ablation: without {label}"]
        L.append(f"| {label} | {r['mean']:.2f} | {100 * r['gap_closed']:.0f}% |")
    L += ["", "**Feature importance** (random forest impurity, averaged over folds, methods and seeds; by group):", "",
          "| Group | Share |", "|---|---|"]
    for g, v in sorted(E["group_importance"].items(), key=lambda x: -x[1]):
        L.append(f"| {GROUPS[g]} | {100 * v:.0f}% |")
    top = sorted(E["importance"].items(), key=lambda x: -x[1])[:8]
    L += ["", "Top features: " + ", ".join(f"`{k}` {100 * v:.1f}%" for k, v in top) + ".", "",
          f"**Within-chunk ranking** (pairs of usable methods on the same chunk whose PSNR differs by ≥ {TIE_DB} dB: "
          "how often does the score order them correctly? 50% = chance):", "",
          "| Score | Correct | Pairs |", "|---|---|---|"]
    for k, (acc, n) in E["pairwise"].items():
        L.append(f"| {k} | {100 * acc:.0f}% | {n} |")
    return "\n".join(L) + "\n"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")          # Windows consoles: the report has ≥ and →
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--probes", help="R3b r3b_probes.jsonl: adds probe routers and probe features")
    a = ap.parse_args()
    F, meta, names = load_features(a.features)
    T = load_targets(a.labels)
    E = evaluate(T, F, meta, names, P=load_probes(a.probes) if a.probes else None)
    md = report(E)
    open(a.out, "w", encoding="utf-8").write(md)
    slim = {k: v for k, v in E.items() if k != "choice"}
    json.dump(dict(slim, choice=E["choice"]), open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1, default=float)
    print(md)


if __name__ == "__main__":
    main()

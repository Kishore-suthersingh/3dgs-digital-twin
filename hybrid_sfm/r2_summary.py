"""Summarise R2 labels (JSON lines written by notebook 05) into r2_summary.md.

One record per (chunk, method, variant, steps):
    chunk, scene, kind, n, method ('ref' | 'classical' | 'vggt' | 'vggt_ba'), variant ('own' | 'refpts'), steps,
    status ('ok' | 'recon_failed' | 'low_coverage' | 'train_failed'), recon {wall_s, ...}, pose {...} | None,
    focal_err, gs {psnr, ssim, lpips, num_GS, train_s} | None
Later records replace earlier ones with the same key, so several runs' files can simply be concatenated.

Routing loss of method m on chunk c: L = PSNR_ref(c) - PSNR_m(c) (dB below the official poses), or FAIL_LOSS if m
produced no usable twin. Usage:
    python -m hybrid_sfm.r2_summary labels.jsonl [more.jsonl ...] [--out r2_summary.md] [--steps 3000 --long 7000]
"""
import argparse
import json
from collections import defaultdict

import numpy as np

METHODS = ["classical", "vggt", "vggt_ba"]
FAIL_LOSS = 10.0
POSE_KEYS = ["AUC@30", "AUC@5", "AUC@3", "RRA@5", "rot_err_median", "ATE_%", "focal_err"]


def load(paths):
    recs = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    recs[(r["chunk"], r["method"], r.get("variant", "own"), r["steps"])] = r
    return list(recs.values())


def rankdata(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    for v in np.unique(x):                      # average ranks for ties
        m = x == v
        ranks[m] = ranks[m].mean()
    return ranks


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan"), int(ok.sum())
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return float("nan"), int(ok.sum())
    return float(np.corrcoef(ra, rb)[0, 1]), int(ok.sum())


def psnr(r):
    return r["gs"]["psnr"] if r and r.get("status") == "ok" and r.get("gs") else None


def summarize(records, steps=3000, long_steps=7000, fail_loss=FAIL_LOSS):
    idx = {(r["chunk"], r["method"], r.get("variant", "own"), r["steps"]): r for r in records}
    chunks = sorted({r["chunk"] for r in records})
    meta = {r["chunk"]: r for r in records}
    get = lambda c, m, v="own", s=steps: idx.get((c, m, v, s))
    L = ["# R2 label summary", ""]
    res = dict(n_chunks=len(chunks))

    # ---- 1. what ran
    L += [f"**{len(chunks)} chunks** from {len({meta[c]['scene'] for c in chunks})} scenes; 3DGS labels at {steps} steps.", "",
          "| Method | Reconstructed | ≥ 90% registered | 3DGS trained | Mean pose time |", "|---|---|---|---|---|"]
    for m in ["ref"] + METHODS:
        rs = [get(c, m) for c in chunks]
        rs = [r for r in rs if r]
        rec_ok = [r for r in rs if r["status"] != "recon_failed"]
        cov = [r for r in rec_ok if r["status"] != "low_coverage"]
        tr = [r for r in rs if r["status"] == "ok"]
        t = [r["recon"].get("wall_s", np.nan) for r in rec_ok if r.get("recon")]
        L.append(f"| {m} | {len(rec_ok)}/{len(rs)} | {len(cov)}/{len(rs)} | {len(tr)}/{len(rs)} | "
                 + ("–" if m == "ref" or not t else f"{np.nanmean(t):.0f} s") + " |")

    # ---- 2. losses and routing baselines
    loss, cost = defaultdict(dict), defaultdict(dict)
    for c in chunks:
        p_ref = psnr(get(c, "ref"))
        if p_ref is None:
            continue
        for m in METHODS:
            r = get(c, m)
            if r is None:
                continue
            p = psnr(r)
            loss[c][m] = fail_loss if p is None else p_ref - p
            cost[c][m] = (r.get("recon") or {}).get("wall_s", np.nan)
    full = [c for c in loss if all(m in loss[c] for m in METHODS)]
    res["n_labelled"] = len(full)
    L += ["", f"**Routing on {len(full)} chunks with all labels** (loss = dB below the official poses; a method "
          f"with no usable twin costs {fail_loss:.0f} dB):", "",
          "| Policy | Mean loss | Median loss | Failures | Mean pose time | Best on |", "|---|---|---|---|---|---|"]
    wins = {m: sum(1 for c in full if min(loss[c], key=loss[c].get) == m) for m in METHODS}
    pol = {}
    for m in METHODS:
        l = [loss[c][m] for c in full]
        pol[f"always {m}"] = dict(mean=float(np.mean(l)) if l else np.nan, median=float(np.median(l)) if l else np.nan,
                                  fails=sum(x >= fail_loss for x in l), time=float(np.nanmean([cost[c][m] for c in full])) if full else np.nan,
                                  wins=wins[m])
    lo = [min(loss[c].values()) for c in full]
    pol["oracle (best per chunk)"] = dict(mean=float(np.mean(lo)) if lo else np.nan, median=float(np.median(lo)) if lo else np.nan,
                                          fails=sum(x >= fail_loss for x in lo),
                                          time=float(np.nanmean([cost[c][min(loss[c], key=loss[c].get)] for c in full])) if full else np.nan,
                                          wins=None)
    for k, v in pol.items():
        L.append(f"| {k} | {v['mean']:.2f} dB | {v['median']:.2f} dB | {v['fails']} | {v['time']:.0f} s | "
                 + ("–" if v["wins"] is None else f"{v['wins']}/{len(full)}") + " |")
    res["policies"] = pol

    # ---- 3. by chunk kind
    kinds = sorted({meta[c]["kind"] for c in full})
    L += ["", "**Mean loss by chunk kind** (window = consecutive images, stride = spread over the capture, random):", "",
          "| Kind | Chunks | " + " | ".join(METHODS) + " | Oracle |", "|---|---|" + "---|" * (len(METHODS) + 1)]
    for k in kinds:
        cs = [c for c in full if meta[c]["kind"] == k]
        L.append(f"| {k} | {len(cs)} | " + " | ".join(f"{np.mean([loss[c][m] for c in cs]):.2f}" for m in METHODS)
                 + f" | {np.mean([min(loss[c].values()) for c in cs]):.2f} |")

    # ---- 4. H1 preview: do pose metrics predict the twin's loss?
    rows = [(c, m) for c in loss for m in loss[c] if loss[c][m] < fail_loss and (get(c, m) or {}).get("pose")]
    L += ["", f"**Do pose metrics predict the twin's loss?** Spearman ρ over {len(rows)} trained (chunk, method) pairs. "
          "A perfect predictor has |ρ| = 1; for accuracy metrics (AUC, RRA) the expected sign is negative:", "",
          "| Pose metric | ρ with loss | n |", "|---|---|---|"]
    res["pose_vs_loss"] = {}
    for k in POSE_KEYS:
        vals = [(get(c, m)["focal_err"] if k == "focal_err" else get(c, m)["pose"].get(k)) for c, m in rows]
        rho, n = spearman([np.nan if v is None else v for v in vals], [loss[c][m] for c, m in rows])
        res["pose_vs_loss"][k] = rho
        L.append(f"| {k} | {rho:+.2f} | {n} |")

    # ---- 5. fidelity: short vs long recipe
    pairs = [(psnr(get(c, m, "own", steps)), psnr(get(c, m, "own", long_steps)))
             for c in chunks for m in ["ref"] + METHODS]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    rho, n = spearman([a for a, _ in pairs], [b for _, b in pairs])
    same_best, n_cmp = 0, 0
    for c in chunks:
        a = {m: psnr(get(c, m, "own", steps)) for m in METHODS}
        b = {m: psnr(get(c, m, "own", long_steps)) for m in METHODS}
        if all(v is not None for v in list(a.values()) + list(b.values())):
            n_cmp += 1
            same_best += max(a, key=a.get) == max(b, key=b.get)
    res["fidelity"] = dict(rho=rho, n=n, same_best=same_best, n_chunks=n_cmp)
    L += ["", f"**Label fidelity ({steps} vs {long_steps} steps):** Spearman ρ = {rho:.2f} over {n} runs; "
          f"same best method on {same_best}/{n_cmp} chunks."]

    # ---- 6. initialisation control
    L += ["", "**Initialisation control** (same chunks, each method trained from its own points vs from the reference "
          "points moved into its frame; mean loss in dB):", "", "| Method | Chunks | Own points | Reference points | Difference |",
          "|---|---|---|---|---|"]
    res["init_control"] = {}
    for m in METHODS:
        cs = [c for c in loss if psnr(get(c, m, "refpts")) is not None and psnr(get(c, m)) is not None]
        if not cs:
            continue
        own = np.mean([psnr(get(c, "ref")) - psnr(get(c, m)) for c in cs])
        ref = np.mean([psnr(get(c, "ref")) - psnr(get(c, m, "refpts")) for c in cs])
        res["init_control"][m] = dict(n=len(cs), own=float(own), refpts=float(ref))
        L.append(f"| {m} | {len(cs)} | {own:.2f} | {ref:.2f} | {own - ref:+.2f} |")
    return "\n".join(L) + "\n", res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--out", default="r2_summary.md")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--long", type=int, default=7000)
    a = ap.parse_args()
    md, res = summarize(load(a.labels), a.steps, a.long)
    open(a.out, "w", encoding="utf-8").write(md)
    json.dump(res, open(a.out.replace(".md", ".json"), "w"), indent=1, default=float)
    print(md)


if __name__ == "__main__":
    main()

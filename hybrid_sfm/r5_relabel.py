"""R5.1: R2-format labels from the independent scores, so R3/R3b/R4 can be rerun unchanged.

Each record of r5_scores.jsonl becomes an R2 label (variant 'own', 3000 steps) whose gs.psnr is the chosen
independent score; poses, timings and statuses come from R2's labels.

    python -m hybrid_sfm.r5_relabel r5_scores.jsonl docs/research/results/r2/r2_labels.jsonl --score psnr_refined --out labels_refined.jsonl
"""
import argparse
import json

from .r2_summary import load


def relabel(r5_records, r2_records, score="psnr_refined"):
    r2 = {(r["chunk"], r["method"]): r for r in r2_records if r["variant"] == "own" and r["steps"] == 3000}
    out = []
    for r in r5_records:
        base = r2.get((r["chunk"], r["method"]))
        if base is None:
            continue
        rec = dict(base, status=r["status"], gs=None)
        if r["status"] == "ok" and r.get("indep") and score in r["indep"]:
            rec["gs"] = dict(psnr=r["indep"][score], ssim=r["indep"].get(score.replace("psnr", "ssim")), own_pose_psnr=r["gs"]["psnr"])
        elif r["status"] == "ok":
            rec["status"] = "eval_failed"
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("r5_scores")
    ap.add_argument("r2_labels")
    ap.add_argument("--score", default="psnr_refined", choices=["psnr_refined", "psnr_aligned", "psnr_own_pose"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r5 = [json.loads(l) for l in open(a.r5_scores) if l.strip()]
    out = relabel(r5, load([a.r2_labels]), a.score)
    with open(a.out, "w") as f:
        f.write("\n".join(json.dumps(r, default=float) for r in out) + "\n")
    print(f"wrote {len(out)} labels ({a.score}) to {a.out}")


if __name__ == "__main__":
    main()

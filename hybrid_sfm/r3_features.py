"""R3 router features, per chunk, from what exists before choosing a reconstructor (no reference poses).

Probes: one VGGT forward pass (saved by R2 as r2_pred/<chunk>.npz) and the classical reconstruction (ALIKED +
LightGlue -> COLMAP, the cheapest method at ~34 s). VGGT+BA is the expensive option the router decides on, so
nothing from its output is used.

Feature groups (prefix):
  n_        chunk size
  conf_     VGGT depth confidence (distribution, per-frame minimum)
  geo_      VGGT camera layout: baseline / depth, angle to the nearest view, angular span
  foc_      VGGT focal length: f / W and its spread across frames (one physical camera -> should be constant)
  mv_       multi-view depth consistency: VGGT depth of frame i, unprojected and reprojected into its nearest
            views, against their VGGT depth (overlap and fraction within 5%)
  cls_      classical: registered fraction, sub-models, matched-pair fraction, keypoints, track length, reprojection
  agree_    VGGT vs classical: pose agreement (AUC@3/@5, median rotation difference, ATE) and focal difference

Usage:
    python -m hybrid_sfm.r3_features "results (1)/r2" "results (3)/r2" --out docs/research/results/r3/features.csv
"""
import argparse
import csv
import json
import os

import numpy as np

from . import colmap_io as cio
from . import pose_metrics as pm


def _angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


def vggt_features(z, k_nn=2, rel_tol=0.05, max_pix=4000, seed=0):
    """z: dict-like with extrinsic (S,3,4) world->cam, intrinsic (S,3,3) at `resolution`, depth/conf (S,h,w) subsampled."""
    E, K = np.asarray(z["extrinsic"], np.float64), np.asarray(z["intrinsic"], np.float64)
    D, C = np.asarray(z["depth"], np.float32), np.asarray(z["conf"], np.float32)
    if D.ndim == 4:
        D = D[..., 0]
    S, h, w = D.shape
    W, H = [int(v) for v in z["resolution"]] if "resolution" in z else (518, 518)
    sx, sy = W / w, H / h                                   # subsampled pixel (u, v) <-> full-res (u*sx, v*sy)
    f = {}
    c = C.reshape(S, -1)
    f["conf_mean"], f["conf_median"] = float(c.mean()), float(np.median(c))
    f["conf_p10"], f["conf_p90"] = float(np.percentile(c, 10)), float(np.percentile(c, 90))
    f["conf_frac_gt2"] = float((c > 2).mean())
    per_frame = c.mean(1)
    f["conf_frame_min"], f["conf_frame_cv"] = float(per_frame.min()), float(per_frame.std() / max(per_frame.mean(), 1e-9))

    R, t = E[:, :, :3], E[:, :, 3]
    centres = -np.einsum("sji,sj->si", R, t)                # C = -R^T t
    axes = R[:, 2, :]                                       # optical axis in world = third row of R
    depth_med = float(np.median(D[D > 0])) if (D > 0).any() else 1.0
    dist = np.linalg.norm(centres[:, None] - centres[None], axis=2)
    np.fill_diagonal(dist, np.inf)
    ang = np.array([[_angle_deg(axes[i], axes[j]) if i != j else np.inf for j in range(S)] for i in range(S)])
    f["geo_nn_baseline"] = float(np.median(dist.min(1)) / depth_med)
    f["geo_max_nn_baseline"] = float(dist.min(1).max() / depth_med)
    f["geo_nn_angle_med"], f["geo_nn_angle_max"] = float(np.median(ang.min(1))), float(ang.min(1).max())
    f["geo_angle_span"] = float(np.max(np.where(np.isinf(ang), 0, ang)))
    f["geo_extent"] = float(np.max(np.where(np.isinf(dist), 0, dist)) / depth_med)

    fw = K[:, 0, 0] / W
    f["foc_f_over_w"], f["foc_cv"] = float(np.median(fw)), float(fw.std() / max(fw.mean(), 1e-9))

    # multi-view depth consistency, on the more confident half of the pixels
    rng = np.random.default_rng(seed)
    vv, uu = np.mgrid[0:h, 0:w]
    pix = np.stack([uu.ravel() * sx, vv.ravel() * sy, np.ones(h * w)], 0)            # full-res homogeneous pixels
    thr = np.median(C)
    overlap, consistent, frame_cons = [], [], []
    for i in range(S):
        keep = np.flatnonzero((C[i].ravel() >= thr) & (D[i].ravel() > 0))
        if len(keep) == 0:
            continue
        keep = rng.choice(keep, min(max_pix, len(keep)), replace=False)
        Xc = (np.linalg.inv(K[i]) @ pix[:, keep]) * D[i].ravel()[keep]               # camera i coordinates
        Xw = R[i].T @ (Xc - t[i][:, None])
        cons_i = []
        for j in np.argsort(dist[i])[:k_nn]:
            Xj = R[j] @ Xw + t[j][:, None]
            zj = Xj[2]
            uvj = (K[j] @ Xj)[:2] / np.where(zj > 1e-9, zj, np.nan)
            uj, vj = np.round(uvj[0] / sx).astype(float), np.round(uvj[1] / sy).astype(float)
            inside = (zj > 1e-9) & (uj >= 0) & (uj < w) & (vj >= 0) & (vj < h)
            overlap.append(inside.mean())
            if inside.sum() < 20:
                cons_i.append(0.0); continue
            dj = D[j][vj[inside].astype(int), uj[inside].astype(int)]
            ok = np.abs(zj[inside] - dj) / np.maximum(dj, 1e-9) < rel_tol
            cons_i.append(float(ok.mean()))
        consistent += cons_i
        frame_cons.append(np.mean(cons_i))
    f["mv_overlap"] = float(np.mean(overlap)) if overlap else 0.0
    f["mv_consistent"] = float(np.mean(consistent)) if consistent else 0.0
    f["mv_frame_min"] = float(np.min(frame_cons)) if frame_cons else 0.0
    return f


def classical_features(info, model, n_images):
    c = (info or {}).get("classical", {}) if info else {}
    f = dict(cls_registered=0.0, cls_models=float(c.get("models", 0)),
             cls_pair_frac=c.get("pairs_with_matches", 0) / max(n_images * (n_images - 1) / 2, 1),
             cls_kpts=float(c.get("mean_kpts", np.nan)), cls_points_per_image=0.0,
             cls_track_len=np.nan, cls_reproj=np.nan, cls_sift_fallback=float(c.get("features", "aliked") == "sift"))
    if model is not None and model["images"]:
        f["cls_registered"] = len(model["images"]) / n_images
        pts = list(model["points"].values())
        f["cls_points_per_image"] = len(pts) / n_images
        if pts:
            f["cls_track_len"] = float(np.mean([len(p["image_ids"]) for p in pts]))
            f["cls_reproj"] = float(np.mean([p["error"] for p in pts]))
    return f


def agreement_features(vggt_model, cls_model):
    f = {k: np.nan for k in ["agree_auc3", "agree_auc5", "agree_rot_med", "agree_ate", "agree_focal"]}
    if vggt_model is None or cls_model is None:
        return f
    V, Cp = cio.poses_by_name(vggt_model), cio.poses_by_name(cls_model)
    common = sorted(set(V) & set(Cp))
    if len(common) < 3:
        return f
    e = pm.pose_errors(V, Cp, names=common)                  # classical as the (unknown-quality) reference
    f.update(agree_auc3=e["AUC@3"], agree_auc5=e["AUC@5"], agree_rot_med=e["rot_err_median"], agree_ate=e["ATE_%"])
    try:
        f["agree_focal"] = abs(pm.focal_error(vggt_model, cls_model, common))
    except Exception:
        pass
    return f


def _read(d):
    try:
        return cio.read_model(d) if d and os.path.isdir(d) else None
    except Exception:
        return None


def chunk_features(r2_dir, chunk):
    """All features for one chunk of an R2 output folder (r2_pred/<chunk>.npz, r2_models/<chunk>/<method>)."""
    with np.load(os.path.join(r2_dir, "r2_pred", f"{chunk['id']}.npz")) as z:
        f = vggt_features({k: z[k] for k in z.files})
    md = os.path.join(r2_dir, "r2_models", chunk["id"])
    info = json.load(open(os.path.join(md, "classical", "info.json"))) if os.path.exists(os.path.join(md, "classical", "info.json")) else None
    cls = _read(cio.find_model_dir(os.path.join(md, "classical")) if os.path.isdir(os.path.join(md, "classical")) else None)
    vgg = _read(os.path.join(md, "vggt", "sparse"))
    f.update(classical_features(info, cls, chunk["n"]))
    f.update(agreement_features(vgg, cls))
    f["n_images"] = float(chunk["n"])
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("r2_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for d in a.r2_dirs:
        for c in json.load(open(os.path.join(d, "r2_chunks.json"))):
            if not os.path.exists(os.path.join(d, "r2_pred", f"{c['id']}.npz")) or any(r["chunk"] == c["id"] for r in rows):
                continue
            rows.append(dict(chunk=c["id"], scene=c["scene"], kind=c["kind"], **chunk_features(d, c)))
            print(f"{len(rows):>3} {c['id']}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    keys = list(rows[0])
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, keys)
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", a.out, len(rows), "chunks,", len(keys) - 3, "features")


if __name__ == "__main__":
    main()

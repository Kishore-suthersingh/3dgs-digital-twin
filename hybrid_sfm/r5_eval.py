"""R5.1: independent scoring of a 3DGS twin trained on a method's poses.

R2's label renders the held-out views at the method's *own* estimated poses. That is deployable, but it shares a
measurement with the R3b probe, and it never checks the twin against the true cameras. This script scores a trained
twin the way pose-free novel-view-synthesis papers do:

  1. aligned:  a Sim(3) from the reference cameras to the twin's own *training* cameras (Umeyama on camera centres,
               in gsplat's normalized training frame), applied to the held-out reference cameras; rendered with the
               reference intrinsics (the real camera), compared with the real photo;
  2. refined:  starting from (1), each held-out pose is refined by test-time optimisation (Gaussians frozen, Adam on
               an L1 photometric loss), the standard protocol of NoPe-NeRF, CF-3DGS and others. This removes the
               alignment error and measures what the twin can render.

Runs in gsplat's examples/ folder (imports its COLMAP Parser so the split and the normalisation match training):
    python r5_eval.py --data_dir <method scene> --ref_model <ref sparse dir> --ckpt <ckpt.pt> --out <json>
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from hybrid_sfm import colmap_io as cio, pose_metrics as pm          # in the repo
except ImportError:                                                      # copied next to colmap_io.py (notebook)
    import colmap_io as cio
    import pose_metrics as pm


# ------------------------------------------------------------------ pose math (numpy, tested locally)
def view_in_frame(R_c, t_c, s, R, t):
    """Reference world->camera (R_c, t_c), and a Sim(3) x_frame = s R x_ref + t.
    Returns world->camera in the frame, with camera coordinates scaled by s (projection unchanged)."""
    R2 = R_c @ R.T
    return R2, s * t_c - R2 @ t


def intrinsics(cam, width):
    """Pinhole K of a COLMAP camera, scaled to an image `width` pixels wide (distortion terms are ignored)."""
    p, m = np.asarray(cam["params"], float), cam["model"]
    if m in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        fx = fy = p[0]; cx, cy = p[1], p[2]
    else:                                                             # PINHOLE, OPENCV, FULL_OPENCV, ...
        fx, fy, cx, cy = p[:4]
    k = width / cam["width"]
    return np.array([[fx * k, 0, cx * k], [0, fy * k, cy * k], [0, 0, 1.0]])


def align_ref_to_frame(ref_poses, frame_c2w, names):
    """Sim(3) taking reference camera centres onto the twin's training camera centres (frame_c2w: {name: 4x4})."""
    common = [n for n in names if n in ref_poses and n in frame_c2w]
    if len(common) < 3:
        raise ValueError(f"only {len(common)} shared training cameras")
    A = np.array([pm.centre(*ref_poses[n]) for n in common])
    B = np.array([frame_c2w[n][:3, 3] for n in common])
    return pm.umeyama(A, B)


# ------------------------------------------------------------------ rendering (GPU)
def _so3_exp(w):
    import torch
    th = torch.sqrt((w * w).sum() + 1e-12)                  # smooth at w = 0 (the optimisation starts there)
    k = w / th
    Kx = torch.zeros(3, 3, device=w.device, dtype=w.dtype)
    Kx[0, 1], Kx[0, 2], Kx[1, 0], Kx[1, 2], Kx[2, 0], Kx[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, device=w.device, dtype=w.dtype) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="the method's scene folder the twin was trained on")
    ap.add_argument("--ref_model", required=True, help="reference COLMAP model of the same chunk")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test_every", type=int, default=8)
    ap.add_argument("--tto_iters", type=int, default=300)
    ap.add_argument("--tto_lr", type=float, default=2e-3)
    a = ap.parse_args()

    import imageio.v2 as imageio
    import torch
    from gsplat.rendering import rasterization
    from datasets.colmap import Parser                                  # gsplat examples/datasets
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

    dev = "cuda"
    parser = Parser(data_dir=a.data_dir, factor=1, normalize=True, test_every=a.test_every)
    names = list(parser.image_names)
    idx = np.arange(len(names))
    test_i, train_i = idx[idx % a.test_every == 0], idx[idx % a.test_every != 0]
    c2w = {n: parser.camtoworlds[i] for i, n in enumerate(names)}

    ref = cio.read_model(a.ref_model)
    ref_poses = cio.poses_by_name(ref)
    ref_cam = {im["name"]: ref["cameras"][im["camera_id"]] for im in ref["images"].values()}
    s, R, t = align_ref_to_frame(ref_poses, c2w, [names[i] for i in train_i])

    sp = torch.load(a.ckpt, map_location=dev)["splats"]
    means, quats = sp["means"], torch.nn.functional.normalize(sp["quats"], dim=-1)
    scales, opac = torch.exp(sp["scales"]), torch.sigmoid(sp["opacities"])
    colors = torch.cat([sp["sh0"], sp["shN"]], 1)
    sh_degree = int(round(np.sqrt(colors.shape[1]))) - 1

    def render(viewmat, K, W, H):
        out, _, _ = rasterization(means, quats, scales, opac, colors, viewmat[None], K[None], W, H, sh_degree=sh_degree)
        return out[0].clamp(0, 1)

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(dev)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)
    score = lambda img, gt: (float(psnr(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])),
                             float(ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])))
    per = []
    for i in test_i:
        n = names[i]
        if n not in ref_poses:
            continue
        gt = torch.from_numpy(imageio.imread(parser.image_paths[i])[..., :3].astype(np.float32) / 255).to(dev)
        H, W = gt.shape[:2]
        R2, t2 = view_in_frame(*ref_poses[n], s, R, t)
        V0 = torch.eye(4, device=dev)
        V0[:3, :3], V0[:3, 3] = torch.tensor(R2, device=dev), torch.tensor(t2, device=dev)
        K = torch.tensor(intrinsics(ref_cam[n], W), device=dev, dtype=torch.float32)
        with torch.no_grad():
            p0, s0 = score(render(V0, K, W, H), gt)
        own = torch.linalg.inv(torch.tensor(c2w[n], device=dev, dtype=torch.float32))           # the method's own pose
        Kown = torch.tensor(parser.Ks_dict[parser.camera_ids[i]], device=dev, dtype=torch.float32)
        with torch.no_grad():
            p_own, _ = score(render(own, Kown, W, H), gt)
        w = torch.zeros(3, device=dev, requires_grad=True)
        u = torch.zeros(3, device=dev, requires_grad=True)
        opt = torch.optim.Adam([w, u], lr=a.tto_lr)
        for _ in range(a.tto_iters):
            D = torch.eye(4, device=dev)
            D = D.clone()
            D[:3, :3] = _so3_exp(w)
            D[:3, 3] = u
            img = render(D @ V0, K, W, H)
            loss = (img - gt).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            D = torch.eye(4, device=dev); D[:3, :3] = _so3_exp(w); D[:3, 3] = u
            p1, s1 = score(render(D @ V0, K, W, H), gt)
        per.append(dict(name=n, psnr_aligned=p0, ssim_aligned=s0, psnr_refined=p1, ssim_refined=s1, psnr_own_pose=p_own,
                        tto_rot_deg=float(np.degrees(torch.linalg.norm(w).item())), tto_trans=float(torch.linalg.norm(u).item())))
    res = dict(n_test=len(per), sim3_scale=float(s),
               **{k: float(np.mean([p[k] for p in per])) for k in ("psnr_aligned", "ssim_aligned", "psnr_refined",
                                                                  "ssim_refined", "psnr_own_pose")} if per else {},
               per_image=per)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_image"}))


if __name__ == "__main__":
    main()

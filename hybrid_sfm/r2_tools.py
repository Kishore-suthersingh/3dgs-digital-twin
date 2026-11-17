"""R2 helpers on COLMAP models (dicts from colmap_io), numpy only."""
import numpy as np

from . import colmap_io as cio
from . import pose_metrics as pm


def align_to_ref(pred_model, ref_model, names=None):
    """Sim(3) (s, R, t) with ref ≈ s R pred + t, from the camera centres both models share."""
    P, Q = cio.poses_by_name(pred_model), cio.poses_by_name(ref_model)
    common = sorted(n for n in (names if names is not None else P) if n in P and n in Q)
    if len(common) < 3:
        raise ValueError(f"only {len(common)} shared images, need 3 for a Sim(3)")
    C_p = np.array([pm.centre(*P[n]) for n in common])
    C_q = np.array([pm.centre(*Q[n]) for n in common])
    return pm.umeyama(C_p, C_q)


def swap_points(pred_model, ref_model, names=None):
    """The method's cameras with the reference's 3D points moved into the method's frame.

    Controlled initialisation for R2: every method then starts 3DGS from the same point cloud, so a PSNR gap
    comes from poses and intrinsics only (in R1, VGGT started from 100k points and VGGT+BA from 8.9k).
    Tracks are dropped (gsplat only uses them for depth supervision, which R2 doesn't use).
    """
    s, R, t = align_to_ref(pred_model, ref_model, names)
    pts = {}
    for pid, p in ref_model["points"].items():
        x = R.T @ (np.asarray(p["xyz"], float) - t) / s          # ref frame -> method frame
        pts[pid] = dict(xyz=x, rgb=p["rgb"], error=p["error"], image_ids=np.zeros(0, np.int32),
                        point2D_idxs=np.zeros(0, np.int32))
    images = {}
    for iid, im in pred_model["images"].items():
        images[iid] = dict(im, point3D_ids=np.full(len(im["point3D_ids"]), -1, np.int64))
    return dict(cameras=pred_model["cameras"], images=images, points=pts)


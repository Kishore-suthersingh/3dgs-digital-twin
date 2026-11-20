"""Tests for R5.1 pose math (numpy only; rendering needs a GPU). Run: python hybrid_sfm/tests/test_r5.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from hybrid_sfm import r5_eval as r5  # noqa: E402
import test_r1 as T  # noqa: E402


def _c2w(R, t):
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R.T, -R.T @ t
    return M


def test_reference_view_lands_on_the_twins_own_camera():
    """Perfect poses in another frame (as gsplat's normalisation does): the moved reference test view must equal
    the twin's own camera for that image, rotation exactly and translation exactly (camera coords scaled by s)."""
    ref = T.look_at_poses(16)
    s0, R0, t0 = 0.23, T.random_rot(), np.array([0.4, -1.0, 2.0])
    frame = T.sim3_transform(ref, s0, R0, t0)
    names = sorted(ref)
    train = [n for i, n in enumerate(names) if i % 8]
    s, R, t = r5.align_ref_to_frame(ref, {n: _c2w(*frame[n]) for n in names}, train)
    assert abs(s - s0) < 1e-9 and np.allclose(R, R0, atol=1e-9) and np.allclose(t, t0, atol=1e-9)
    for n in names[::8]:                                     # the held-out views
        R2, t2 = r5.view_in_frame(*ref[n], s, R, t)
        assert np.allclose(R2, frame[n][0], atol=1e-9) and np.allclose(t2, frame[n][1], atol=1e-9)


def test_noisy_training_cameras_give_a_close_alignment():
    ref = T.look_at_poses(24)
    frame = T.sim3_transform(ref, 1.7, T.random_rot(), np.ones(3))
    rng = np.random.default_rng(0)
    c2w = {n: _c2w(*p) for n, p in frame.items()}
    for M in c2w.values():
        M[:3, 3] += rng.normal(scale=0.01, size=3)
    s, R, t = r5.align_ref_to_frame(ref, c2w, sorted(ref))
    assert abs(s - 1.7) < 0.05


def test_intrinsics_scale_to_the_image():
    cam = dict(model="PINHOLE", width=4000, height=3000, params=np.array([3000.0, 3100.0, 2000.0, 1500.0]))
    K = r5.intrinsics(cam, 500)
    assert np.allclose(K, [[375, 0, 250], [0, 387.5, 187.5], [0, 0, 1]])
    K2 = r5.intrinsics(dict(model="SIMPLE_RADIAL", width=1000, height=800, params=np.array([900.0, 500.0, 400.0, 0.01])), 250)
    assert np.allclose(K2, [[225, 0, 125], [0, 225, 100], [0, 0, 1]])


def test_relabel_uses_the_independent_score():
    from hybrid_sfm import r5_relabel as rl
    from test_r2 import _rec
    r2 = [_rec("g_c0", "ref", 25.0), _rec("g_c0", "vggt", 21.0), _rec("g_c0", "vggt_ba", None, status="recon_failed")]
    r5s = [dict(chunk="g_c0", method="ref", status="ok", gs=dict(psnr=25.0), indep=dict(psnr_refined=23.0, ssim_refined=0.7)),
           dict(chunk="g_c0", method="vggt", status="ok", gs=dict(psnr=21.0), indep=dict(psnr_refined=22.5)),
           dict(chunk="g_c0", method="vggt_ba", status="recon_failed", gs=None, indep=None)]
    out = {r["method"]: r for r in rl.relabel(r5s, r2)}
    assert out["ref"]["gs"]["psnr"] == 23.0 and out["vggt"]["gs"]["psnr"] == 22.5 and out["vggt"]["gs"]["own_pose_psnr"] == 21.0
    assert out["vggt_ba"]["status"] == "recon_failed" and out["vggt"]["pose"] == r2[1]["pose"]


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)

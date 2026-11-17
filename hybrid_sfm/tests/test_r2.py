"""Tests for the R2 parts of hybrid_sfm (numpy only, plus a stub of VGGT's demo_colmap). Run: python hybrid_sfm/tests/test_r2.py"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from hybrid_sfm import chunks as ch  # noqa: E402
from hybrid_sfm import colmap_io as cio  # noqa: E402
from hybrid_sfm import pose_metrics as pm  # noqa: E402
from hybrid_sfm import r2_summary as rs  # noqa: E402
from hybrid_sfm import r2_tools as rt  # noqa: E402
import test_r1 as T  # noqa: E402


def test_chunks_deterministic_and_valid():
    names = [f"DSC{i:05d}.JPG" for i in range(185)]
    a, b = ch.make_chunks("garden", names), ch.make_chunks("garden", list(reversed(names)))
    assert a == b and len(a) == 15 and len({c["id"] for c in a}) == 15
    for c in a:
        assert len(c["names"]) == c["n"] == len(set(c["names"])) and set(c["names"]) <= set(names)
        assert c["names"] == sorted(c["names"]) and c["n"] <= 40
    w = [c for c in a if c["kind"] == "window" and c["n"] == 24]
    assert all(names.index(c["names"][-1]) - names.index(c["names"][0]) == 23 for c in w)   # consecutive
    assert ch.make_chunks("bicycle", names)[-1]["names"] != a[-1]["names"]                    # seed depends on scene
    small = ch.make_chunks("tiny", names[:20])
    assert all(c["n"] <= 20 for c in small)


def test_tight_auc_and_focal():
    ref = T.look_at_poses(20)
    m = pm.pose_errors(ref, ref)
    assert m["AUC@3"] == m["AUC@5"] == m["AUC@30"] == 1.0 and m["RRA@1"] == 1.0
    noisy = {}
    rng = np.random.default_rng(3)
    for n, (R, t) in ref.items():                     # ~0.5° rotation noise: invisible at 30°, visible at 3°
        q = cio.rotmat2qvec(R) + rng.normal(scale=0.004, size=4)
        noisy[n] = (cio.qvec2rotmat(q), t)
    e = pm.pose_errors(noisy, ref)
    assert e["AUC@30"] > 0.95 and e["AUC@3"] < e["AUC@5"] < e["AUC@30"]
    a, b = T.synthetic_model(ref, 10), T.synthetic_model(ref, 10)
    b["cameras"][1] = dict(b["cameras"][1], params=b["cameras"][1]["params"] * np.array([1.1, 1.1, 1, 1]))
    assert abs(pm.focal_error(b, a) - 0.1) < 1e-9 and pm.focal_error(a, a) == 0
    big = dict(a, cameras={1: dict(a["cameras"][1], width=a["cameras"][1]["width"] * 4,
                                   params=a["cameras"][1]["params"] * 4)})
    assert abs(pm.focal_error(big, a)) < 1e-12                                   # f / W: resolution-independent


def test_swap_points_projects_identically():
    ref_poses = T.look_at_poses(12)
    ref = T.synthetic_model(ref_poses, 100)
    s0, R0, t0 = 0.37, T.random_rot(), np.array([2.0, -1.0, 0.5])
    pred = T.synthetic_model(T.sim3_transform(ref_poses, s0, R0, t0), 30)       # same cameras, another world frame
    out = rt.swap_points(pred, ref)
    assert len(out["points"]) == 100 and all(len(p["image_ids"]) == 0 for p in out["points"].values())
    assert all((im["point3D_ids"] == -1).all() for im in out["images"].values())
    P, Q = cio.poses_by_name(out), cio.poses_by_name(ref)
    for pid in list(ref["points"])[:20]:                                       # each point lands at the same pixel
        for n in list(Q)[:4]:
            (Rq, tq), (Rp, tp) = Q[n], P[n]
            xq, xp = Rq @ ref["points"][pid]["xyz"] + tq, Rp @ out["points"][pid]["xyz"] + tp
            assert np.allclose(xq[:2] / xq[2], xp[:2] / xp[2], atol=1e-9)
    with tempfile.TemporaryDirectory() as d:
        cio.write_model_bin(out, d)
        assert len(cio.read_model(d)["points"]) == 100


def _rec(chunk, method, psnr=None, status="ok", steps=3000, variant="own", kind="window", auc=0.9, wall=10.0):
    return dict(chunk=chunk, scene=chunk.split("_")[0], kind=kind, n=24, method=method, variant=variant, steps=steps,
                status=status, recon=dict(wall_s=wall), focal_err=0.01,
                pose=None if method == "ref" else {k: auc for k in ["AUC@30", "AUC@5", "AUC@3", "RRA@5"]} | dict(rot_err_median=1 - auc, **{"ATE_%": 1.0}),
                gs=None if psnr is None else dict(psnr=psnr, ssim=0.5, lpips=0.4, num_GS=1000, train_s=60))


def test_summary():
    R = []
    for i in range(6):
        c, kind = f"garden_c{i}", ("window" if i < 3 else "stride")
        R += [_rec(c, "ref", 25.0), _rec(c, "classical", 24.5 if kind == "window" else None,
                                          "ok" if kind == "window" else "low_coverage", kind=kind, auc=0.99, wall=60),
              _rec(c, "vggt", 22.0 + 0.3 * i, kind=kind, auc=0.95 + 0.005 * i, wall=40),
              _rec(c, "vggt_ba", 23.0, kind=kind, auc=0.97, wall=90)]
        if i < 2:
            R += [_rec(c, m, p + 1, steps=7000) for m, p in [("ref", 25), ("classical", 24.5), ("vggt", 22), ("vggt_ba", 23)]]
            R += [_rec(c, "vggt", 23.5, variant="refpts")]
    R.append(_rec("garden_c0", "vggt", 22.0))                                  # duplicate key: last one wins
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "l.jsonl")
        open(p, "w").write("\n".join(json.dumps(r) for r in R))
        recs = rs.load([p])
    md, res = rs.summarize(recs)
    assert res["n_labelled"] == 6
    pol = res["policies"]
    assert abs(pol["always classical"]["mean"] - (3 * 0.5 + 3 * 10) / 6) < 1e-9 and pol["always classical"]["fails"] == 3
    assert abs(pol["oracle (best per chunk)"]["mean"] - (0.5 * 3 + 2.0 + 1.8 + 1.5) / 6) < 1e-9
    assert res["fidelity"]["n_chunks"] == 2 and res["fidelity"]["same_best"] == 2 and res["fidelity"]["rho"] > 0.99
    assert res["init_control"]["vggt"]["n"] == 2
    assert res["pose_vs_loss"]["AUC@30"] < 0                                   # better poses -> lower loss here
    assert "| window | 3 |" in md and "| stride | 3 |" in md


STUB = '''
import argparse, numpy as np, torch
def run_VGGT(model, images, dtype, resolution=518):
    S = images.shape[0]
    return (np.zeros((S, 3, 4), np.float32), np.tile(np.eye(3, dtype=np.float32), (S, 1, 1)),
            np.ones((S, 518, 518, 1), np.float32), torch.full((S, 518, 518), 3.0))
def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--scene_dir"); p.add_argument("--use_ba", action="store_true")
    p.add_argument("--fine_tracking", action="store_true", default=True); return p.parse_args()
class M(torch.nn.Module): pass
def demo_fn(args):
    import glob, os
    assert args.fine_tracking is False
    run_VGGT(M(), torch.zeros(len(glob.glob(os.path.join(args.scene_dir, "images", "*"))), 3, 518, 518), None)
'''


def test_vggt_wrapper_saves_predictions():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  (skipped: no torch)"); return
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "demo_colmap.py"), "w").write(STUB)
        os.makedirs(os.path.join(d, "scene", "images"))
        for k in range(3):
            open(os.path.join(d, "scene", "images", f"im{k}.jpg"), "w").close()
        wrapper = os.path.join(os.path.dirname(HERE), "vggt_colmap_lowmem.py")
        out = os.path.join(d, "pred.npz")
        r = subprocess.run([sys.executable, wrapper, "--scene_dir", os.path.join(d, "scene"), "--no_fine_tracking",
                            "--save_pred", out], cwd=d, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": d})
        assert r.returncode == 0, r.stderr[-2000:]
        with np.load(out) as z:
            assert z["depth"].shape == (3, 130, 130) and z["conf"].dtype == np.float16
            assert sorted(z["names"]) == ["im0.jpg", "im1.jpg", "im2.jpg"]


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)

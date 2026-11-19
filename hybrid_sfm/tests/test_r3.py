"""Tests for R3 (router features and leave-one-scene-out evaluation). Run: python hybrid_sfm/tests/test_r3.py"""
import csv
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from hybrid_sfm import r3_features as rf  # noqa: E402
from hybrid_sfm import r3_router as rr  # noqa: E402
import test_r1 as T  # noqa: E402
from test_r2 import _rec  # noqa: E402


def _plane_scene(S=6, res=518, sub=4, slope=0.2):
    """Cameras side by side at z = -5 looking along +z at the plane z = slope * x: exact depth maps."""
    f = 400.0
    K = np.array([[f, 0, res / 2], [0, f, res / 2], [0, 0, 1]])
    E, D = [], []
    h = w = res // sub + (res % sub > 0)
    vv, uu = np.mgrid[0:h, 0:w]
    rays = np.linalg.inv(K) @ np.stack([uu.ravel() * sub, vv.ravel() * sub, np.ones(h * w)])
    for i in range(S):
        C = np.array([0.3 * i, 0.0, -5.0])
        E.append(np.hstack([np.eye(3), -C[:, None]]))
        lam = (slope * C[0] - C[2]) / (1 - slope * rays[0])           # ray (x, y, 1) * lam hits the plane
        D.append(lam.reshape(h, w))
    return dict(extrinsic=np.array(E), intrinsic=np.tile(K, (S, 1, 1)), depth=np.array(D, np.float32),
                conf=np.full((S, h, w), 3.0, np.float32), resolution=np.array([res, res]))


def test_multiview_consistency():
    z = _plane_scene()
    f = rf.vggt_features(z)
    assert f["mv_consistent"] > 0.97 and f["mv_overlap"] > 0.5, f
    assert abs(f["geo_nn_baseline"] - 0.3 / np.median(z["depth"])) < 1e-6 and f["foc_cv"] < 1e-9
    noisy = dict(z, depth=(z["depth"] * np.random.default_rng(0).uniform(0.7, 1.3, z["depth"].shape)).astype(np.float32))
    assert rf.vggt_features(noisy)["mv_consistent"] < 0.6


def test_classical_and_agreement_features():
    poses = T.look_at_poses(10)
    m = T.synthetic_model(poses, 50)
    info = dict(classical=dict(models=1, pairs_with_matches=30, mean_kpts=1000.0, features="aliked+lightglue"))
    f = rf.classical_features(info, m, 10)
    assert f["cls_registered"] == 1.0 and abs(f["cls_pair_frac"] - 30 / 45) < 1e-12 and f["cls_track_len"] > 0
    assert rf.classical_features(None, None, 10)["cls_registered"] == 0.0
    a = rf.agreement_features(T.synthetic_model(T.sim3_transform(poses, 2.0, T.random_rot(), np.ones(3)), 20), m)
    assert a["agree_auc3"] > 0.999 and a["agree_rot_med"] < 1e-6 and a["agree_focal"] < 1e-9
    assert np.isnan(rf.agreement_features(None, m)["agree_auc3"])


def test_router_learns_a_feature_rule():
    """Synthetic: classical is best when feature x is high, VGGT when it is low; VGGT+BA always fails."""
    rng = np.random.default_rng(0)
    rows, recs = [], []
    for s in range(4):
        for i in range(12):
            c, x = f"sc{s}_c{i}", rng.uniform()
            rows.append(dict(chunk=c, scene=f"sc{s}", kind="window", cls_registered=x, conf_mean=rng.uniform(),
                             mv_consistent=rng.uniform(), agree_auc3=rng.uniform(), geo_nn_baseline=rng.uniform()))
            lc, lv = (0.5, 3.0) if x > 0.5 else (6.0, 2.0)
            recs += [_rec(c, "ref", 25.0), _rec(c, "classical", 25.0 - lc), _rec(c, "vggt", 25.0 - lv),
                     _rec(c, "vggt_ba", None, status="recon_failed")]
            for r in recs[-4:]:
                r["scene"] = f"sc{s}"
    with tempfile.TemporaryDirectory() as d:
        fp, lp = os.path.join(d, "f.csv"), os.path.join(d, "l.jsonl")
        with open(fp, "w", newline="") as fh:
            wr = csv.DictWriter(fh, list(rows[0])); wr.writeheader(); wr.writerows(rows)
        open(lp, "w").write("\n".join(json.dumps(r) for r in recs))
        F, meta, names = rr.load_features(fp)
        Tg = rr.load_targets([lp])
    E = rr.evaluate(Tg, F, meta, names, seeds=(0,), n_boot=200)
    R = E["results"]
    assert R["oracle"]["regret"] == 0 and R["always vggt_ba"]["fails"] == 48
    assert R["stump: cls_registered"]["regret"] < 0.3 and R["learned: all features"]["regret"] < 0.5
    assert R["learned: all features"]["mean"] < R["always vggt"]["mean"] - 0.3
    assert R["learned: all features"]["picks"]["vggt_ba"] == 0
    assert E["pairwise"]["learned predicted loss (no GT)"][0] > 0.8
    md = rr.report(E)
    assert "leave-one-scene-out" in md and "| sc0 |" in md


def test_probe_routers():
    """Probes that rank methods like the labels: the probe router reaches the oracle; time includes probe training."""
    rng = np.random.default_rng(1)
    rows, recs, probes = [], [], []
    for s in range(3):
        for i in range(8):
            c = f"sc{s}_c{i}"
            rows.append(dict(chunk=c, scene=f"sc{s}", kind="window", cls_registered=rng.uniform(), conf_mean=rng.uniform(),
                             mv_consistent=rng.uniform(), agree_auc3=rng.uniform(), geo_nn_baseline=rng.uniform()))
            ps = dict(classical=rng.uniform(20, 26), vggt=rng.uniform(20, 26))
            recs += [_rec(c, "ref", 27.0), _rec(c, "classical", ps["classical"]), _rec(c, "vggt", ps["vggt"]),
                     _rec(c, "vggt_ba", None, status="recon_failed")]
            for r in recs[-4:]:
                r["scene"] = f"sc{s}"
            for m, v in ps.items():                      # probe = label - 3 dB + small noise that keeps the order
                probes.append(dict(chunk=c, method=m, status="ok", probe={"500": dict(psnr=v - 3, train_s=12.0)}))
            probes.append(dict(chunk=c, method="vggt_ba", status="recon_failed", probe=None))
    with tempfile.TemporaryDirectory() as d:
        fp, lp, pp = (os.path.join(d, x) for x in ("f.csv", "l.jsonl", "p.jsonl"))
        with open(fp, "w", newline="") as fh:
            wr = csv.DictWriter(fh, list(rows[0])); wr.writeheader(); wr.writerows(rows)
        open(lp, "w").write("\n".join(json.dumps(r) for r in recs))
        open(pp, "w").write("\n".join(json.dumps(r) for r in probes))
        F, meta, names = rr.load_features(fp)
        Tg, P = rr.load_targets([lp]), rr.load_probes(pp)
    E = rr.evaluate(Tg, F, meta, names, seeds=(0,), n_boot=100, P=P)
    R = E["results"]
    k = "probe@500: best probe PSNR (all methods)"
    assert R[k]["regret"] < 1e-9 and abs(R[k]["probe_s"] - 24.0) < 1e-9
    assert R[k]["time"] > R["always vggt"]["time"]
    assert "learned: all features + probe@500" in R and E["pairwise"]["probe PSNR at 500 steps (no GT)"][0] == 1.0
    assert "probe@500" in rr.report(E)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)

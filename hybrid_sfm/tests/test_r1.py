"""Tests for hybrid_sfm.colmap_io and hybrid_sfm.pose_metrics (numpy only). Run: python -m pytest hybrid_sfm/tests -q"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hybrid_sfm import colmap_io as cio  # noqa: E402
from hybrid_sfm import pose_metrics as pm  # noqa: E402

rng = np.random.default_rng(0)


def random_rot():
    q = rng.normal(size=4)
    return cio.qvec2rotmat(q / np.linalg.norm(q))


def look_at_poses(n=20, radius=4.0):
    """Cameras on a circle, looking at the origin (world-to-camera)."""
    poses = {}
    for k in range(n):
        a = 2 * np.pi * k / n
        C = np.array([radius * np.cos(a), radius * np.sin(a), 1.0 + 0.3 * np.sin(3 * a)])
        z = -C / np.linalg.norm(C)
        x = np.cross([0, 0, 1.0], z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        poses[f"img_{k:03d}.jpg"] = (R, -R @ C)
    return poses


def synthetic_model(poses, n_points=200):
    cams = {1: dict(model="PINHOLE", width=1297, height=840, params=np.array([1000.0, 1001.0, 648.5, 420.0]))}
    images, points = {}, {}
    names = sorted(poses)
    for i, n in enumerate(names, start=1):
        R, t = poses[n]
        images[i] = dict(qvec=cio.rotmat2qvec(R), tvec=t, camera_id=1, name=n,
                         xys=np.zeros((0, 2)), point3D_ids=np.zeros(0, np.int64))
    obs = {i: [] for i in images}
    for pid in range(1, n_points + 1):
        seen = rng.choice(sorted(images), size=rng.integers(2, 6), replace=False)
        idxs = []
        for iid in seen:
            obs[iid].append(pid); idxs.append(len(obs[iid]) - 1)
        points[pid] = dict(xyz=rng.normal(size=3), rgb=rng.integers(0, 255, 3).astype(np.uint8), error=0.5,
                           image_ids=np.array(seen, np.int32), point2D_idxs=np.array(idxs, np.int32))
    for iid, pids in obs.items():
        images[iid]["point3D_ids"] = np.array(pids, np.int64)
        images[iid]["xys"] = rng.uniform(0, 800, (len(pids), 2))
    return dict(cameras=cams, images=images, points=points)


def test_quaternion_roundtrip():
    for _ in range(50):
        R = random_rot()
        assert np.allclose(cio.qvec2rotmat(cio.rotmat2qvec(R)), R, atol=1e-9)


def test_binary_roundtrip_and_subset():
    poses = look_at_poses()
    model = synthetic_model(poses)
    with tempfile.TemporaryDirectory() as d:
        cio.write_model_bin(model, os.path.join(d, "sparse", "0"))
        back = cio.read_model(d)                                       # resolves sparse/0
        assert back["cameras"][1]["model"] == "PINHOLE" and np.allclose(back["cameras"][1]["params"], model["cameras"][1]["params"])
        assert len(back["images"]) == 20 and len(back["points"]) == 200
        for iid, im in model["images"].items():
            b = back["images"][iid]
            assert b["name"] == im["name"] and np.allclose(b["tvec"], im["tvec"])
            assert np.array_equal(b["point3D_ids"], im["point3D_ids"]) and np.allclose(b["xys"], im["xys"])
        for pid, p in model["points"].items():
            assert np.array_equal(back["points"][pid]["image_ids"], p["image_ids"])
        pp = cio.poses_by_name(back)
        for n, (R, t) in poses.items():
            assert np.allclose(pp[n][0], R, atol=1e-9) and np.allclose(pp[n][1], t)

        keep = sorted(poses)[::3]
        sub = cio.subset_model(back, keep)
        assert sorted(im["name"] for im in sub["images"].values()) == keep
        kept_ids = set(sub["images"])
        for p in sub["points"].values():                               # every track entry points at a kept image
            assert set(p["image_ids"]) <= kept_ids and len(p["image_ids"]) >= 2
        for im in sub["images"].values():                              # every observation points at a kept point or -1
            assert all(i == -1 or i in sub["points"] for i in im["point3D_ids"])
        cio.write_model_bin(sub, os.path.join(d, "sub"))
        assert len(cio.read_model(os.path.join(d, "sub"))["images"]) == len(keep)


def test_text_reader():
    poses = look_at_poses(5)
    model = synthetic_model(poses, 10)
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "cameras.txt"), "w") as f:
            f.write("# comment\n1 PINHOLE 1297 840 1000 1001 648.5 420\n")
        with open(os.path.join(d, "images.txt"), "w") as f:
            f.write("# header\n")
            for iid, im in sorted(model["images"].items()):
                f.write(f"{iid} {' '.join(map(str, im['qvec']))} {' '.join(map(str, im['tvec']))} 1 {im['name']}\n")
                f.write(" ".join(f"{x} {y} {p}" for (x, y), p in zip(im["xys"], im["point3D_ids"])) + "\n")
        with open(os.path.join(d, "points3D.txt"), "w") as f:
            for pid, p in model["points"].items():
                tr = " ".join(f"{a} {b}" for a, b in zip(p["image_ids"], p["point2D_idxs"]))
                f.write(f"{pid} {' '.join(map(str, p['xyz']))} {' '.join(map(str, p['rgb']))} 0.5 {tr}\n")
        back = cio.read_model(d)
        assert len(back["images"]) == 5 and len(back["points"]) == 10
        assert np.allclose(cio.poses_by_name(back)["img_002.jpg"][1], poses["img_002.jpg"][1])


def sim3_transform(poses, s, R0, t0):
    """Same cameras in a world moved by x' = s R0 x + t0."""
    out = {}
    for n, (R, t) in poses.items():
        C = pm.centre(R, t)
        R2 = R @ R0.T
        out[n] = (R2, -R2 @ (s * R0 @ C + t0))
    return out


def test_metrics_perfect_under_sim3():
    ref = look_at_poses(25)
    pred = sim3_transform(ref, 3.7, random_rot(), np.array([5.0, -2.0, 1.0]))
    m = pm.pose_errors(pred, ref)
    assert m["registered"] == 1.0 and m["AUC@30"] > 0.999 and m["RRA@5"] == 1.0 and m["RTA@5"] == 1.0
    assert m["ATE_%"] < 1e-6


def test_metrics_noise_and_missing():
    ref = look_at_poses(25)
    noisy = {}
    for n, (R, t) in ref.items():
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        ang = np.radians(10)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        dR = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
        noisy[n] = (dR @ R, t + rng.normal(scale=0.3, size=3))
    m = pm.pose_errors(noisy, ref)
    assert 0 < m["AUC@30"] < 0.9 and m["ATE_%"] > 1
    dropped = {n: p for i, (n, p) in enumerate(sorted(ref.items())) if i % 5}
    m2 = pm.pose_errors(sim3_transform(dropped, 2.0, random_rot(), np.zeros(3)), ref)
    assert m2["n_registered"] == 20 and abs(m2["registered"] - 0.8) < 1e-9
    assert abs(m2["AUC@30"] - 190 / 300) < 0.01          # only pairs with both images registered count as correct
    assert m2["ATE_%"] < 1e-6


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)

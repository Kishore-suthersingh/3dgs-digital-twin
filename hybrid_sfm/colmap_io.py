"""Minimal COLMAP model I/O (binary and text), numpy only.

Written for the research track so pose evaluation and subsetting don't depend on a particular pycolmap
version (gsplat's examples install a pycolmap *fork* that shadows the official package; VGGT's
demo_colmap.py needs official pycolmap 3.10).

Model = dict with
    cameras: {camera_id: dict(model, width, height, params)}
    images:  {image_id: dict(qvec (w,x,y,z), tvec, camera_id, name, xys (P,2), point3D_ids (P,))}
    points:  {point3D_id: dict(xyz, rgb, error, image_ids, point2D_idxs)}
Poses are world-to-camera (COLMAP / OpenCV convention): x_cam = R(q) @ x_world + t.
"""
import os
import struct

import numpy as np

CAMERA_MODELS = {  # id: (name, number of params)
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
    4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}
MODEL_ID = {name: i for i, (name, _) in CAMERA_MODELS.items()}


def _read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


# ------------------------------------------------------------------ binary
def read_cameras_bin(path):
    cams = {}
    with open(path, "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            cid, mid, w, h = _read(f, "iiQQ")
            name, n = CAMERA_MODELS[mid]
            cams[cid] = dict(model=name, width=w, height=h, params=np.array(_read(f, "d" * n)))
    return cams


def read_images_bin(path):
    imgs = {}
    with open(path, "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            iid, = _read(f, "i")
            q = np.array(_read(f, "dddd"))
            t = np.array(_read(f, "ddd"))
            cid, = _read(f, "i")
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            n, = _read(f, "Q")
            raw = np.frombuffer(f.read(24 * n), dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<i8")]))
            imgs[iid] = dict(qvec=q, tvec=t, camera_id=cid, name=name.decode(),
                             xys=np.stack([raw["x"], raw["y"]], 1), point3D_ids=raw["id"].copy())
    return imgs


def read_points_bin(path):
    pts = {}
    with open(path, "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            pid, = _read(f, "Q")
            xyz = np.array(_read(f, "ddd"))
            rgb = np.array(_read(f, "BBB"), np.uint8)
            err, n = _read(f, "dQ")
            tr = np.frombuffer(f.read(8 * n), dtype="<i4").reshape(-1, 2)
            pts[pid] = dict(xyz=xyz, rgb=rgb, error=err, image_ids=tr[:, 0].copy(), point2D_idxs=tr[:, 1].copy())
    return pts


def write_model_bin(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cameras.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(model["cameras"])))
        for cid, c in sorted(model["cameras"].items()):
            f.write(struct.pack("<iiQQ", cid, MODEL_ID[c["model"]], int(c["width"]), int(c["height"])))
            f.write(struct.pack("<" + "d" * len(c["params"]), *map(float, c["params"])))
    with open(os.path.join(out_dir, "images.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(model["images"])))
        for iid, im in sorted(model["images"].items()):
            f.write(struct.pack("<i4d3di", iid, *map(float, im["qvec"]), *map(float, im["tvec"]), im["camera_id"]))
            f.write(im["name"].encode() + b"\x00")
            xys, ids = np.asarray(im["xys"], float).reshape(-1, 2), np.asarray(im["point3D_ids"], np.int64)
            f.write(struct.pack("<Q", len(ids)))
            rec = np.empty(len(ids), dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<i8")]))
            rec["x"], rec["y"], rec["id"] = xys[:, 0], xys[:, 1], ids
            f.write(rec.tobytes())
    with open(os.path.join(out_dir, "points3D.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(model["points"])))
        for pid, p in sorted(model["points"].items()):
            f.write(struct.pack("<Q3d3Bd", pid, *map(float, p["xyz"]), *map(int, p["rgb"]), float(p["error"])))
            tr = np.stack([p["image_ids"], p["point2D_idxs"]], 1).astype("<i4")
            f.write(struct.pack("<Q", len(tr)) + tr.tobytes())


# ------------------------------------------------------------------ text (read only)
def _data_lines(path):
    with open(path) as f:
        return [l.rstrip("\n") for l in f if not l.startswith("#")]


def read_model_txt(d):
    cams, imgs, pts = {}, {}, {}
    for l in _data_lines(os.path.join(d, "cameras.txt")):
        if not l.strip():
            continue
        v = l.split()
        cams[int(v[0])] = dict(model=v[1], width=int(v[2]), height=int(v[3]), params=np.array(v[4:], float))
    lines = _data_lines(os.path.join(d, "images.txt"))   # pairs: header line, observations line (may be empty)
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        v = lines[i].split()
        obs = np.array(lines[i + 1].split(), float).reshape(-1, 3) if i + 1 < len(lines) else np.zeros((0, 3))
        imgs[int(v[0])] = dict(qvec=np.array(v[1:5], float), tvec=np.array(v[5:8], float), camera_id=int(v[8]),
                               name=" ".join(v[9:]), xys=obs[:, :2], point3D_ids=obs[:, 2].astype(np.int64))
        i += 2
    for l in _data_lines(os.path.join(d, "points3D.txt")):
        if not l.strip():
            continue
        v = l.split()
        tr = np.array(v[8:], int).reshape(-1, 2)
        pts[int(v[0])] = dict(xyz=np.array(v[1:4], float), rgb=np.array(v[4:7], np.uint8), error=float(v[7]),
                              image_ids=tr[:, 0], point2D_idxs=tr[:, 1])
    return dict(cameras=cams, images=imgs, points=pts)


def find_model_dir(scene_dir):
    """gsplat's rule: sparse/0 if it exists, else sparse."""
    for d in (os.path.join(scene_dir, "sparse", "0"), os.path.join(scene_dir, "sparse")):
        if os.path.exists(os.path.join(d, "images.bin")) or os.path.exists(os.path.join(d, "images.txt")):
            return d
    raise FileNotFoundError(f"no COLMAP model under {scene_dir}/sparse")


def read_model(d):
    if os.path.isdir(os.path.join(d, "sparse")):
        d = find_model_dir(d)
    if os.path.exists(os.path.join(d, "images.bin")):
        return dict(cameras=read_cameras_bin(os.path.join(d, "cameras.bin")),
                    images=read_images_bin(os.path.join(d, "images.bin")),
                    points=read_points_bin(os.path.join(d, "points3D.bin")))
    return read_model_txt(d)


# ------------------------------------------------------------------ helpers
def qvec2rotmat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rotmat2qvec(R):
    K = np.array([[R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
                  [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
                  [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0],
                  [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], R[0, 0] + R[1, 1] + R[2, 2]]]) / 3
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    return q if q[0] >= 0 else -q


def poses_by_name(model):
    """{image name: (R (3,3), t (3,))}, world-to-camera."""
    return {im["name"]: (qvec2rotmat(im["qvec"]), np.asarray(im["tvec"], float)) for im in model["images"].values()}


def subset_model(model, keep_names, min_track=2):
    """Keep only images in keep_names; keep 3D points still seen by >= min_track of them.

    gsplat looks every image in the model up in images/ and every track entry up in the model's images,
    so a subset needs both trimmed. Dropped points are set to -1 in the images' observations.
    """
    keep_names = set(keep_names)
    images = {i: dict(im) for i, im in model["images"].items() if im["name"] in keep_names}
    missing = keep_names - {im["name"] for im in images.values()}
    if missing:
        raise KeyError(f"{len(missing)} names not in the model, e.g. {sorted(missing)[:3]}")
    kept_ids = np.array(sorted(images))
    points = {}
    for pid, p in model["points"].items():
        m = np.isin(p["image_ids"], kept_ids)
        if m.sum() >= min_track:
            points[pid] = dict(p, image_ids=p["image_ids"][m], point2D_idxs=p["point2D_idxs"][m])
    for im in images.values():
        ids = np.asarray(im["point3D_ids"], np.int64).copy()
        ids[(ids >= 0) & ~np.isin(ids, np.fromiter(points.keys(), np.int64, len(points)))] = -1
        im["point3D_ids"] = ids
    cams = {c: model["cameras"][c] for c in {im["camera_id"] for im in images.values()}}
    return dict(cameras=cams, images=images, points=points)

"""R2 classical reconstructor: ALIKED + LightGlue matches (GPU) -> COLMAP verification and incremental mapping.

R1's COLMAP baseline (SIFT, exhaustive CPU matching) took 16-19 min for 32 images, far too slow for 100+ chunks.
This is the hloc-style pipeline from the proposal: learned features and matcher on the GPU, then COLMAP's own
geometric verification (pycolmap.verify_matches) and incremental mapper. One shared PINHOLE camera, as in R1.
Falls back to R1's SIFT pipeline if LightGlue can't be loaded (recorded in classical_info.json).

Usage (VGGT environment: pycolmap 3.10, torch, LightGlue):
    python run_classical.py <scene_dir> [--max_kpts 2048]      # reads <scene_dir>/images, writes <scene_dir>/sparse/0
"""
import argparse
import itertools
import json
import os
import shutil
import sqlite3
import time

import numpy as np
import pycolmap

MAX_IMAGE_ID = 2147483647          # COLMAP's pair-id encoding


def pair_id(i, j):
    return i * MAX_IMAGE_ID + j if i < j else j * MAX_IMAGE_ID + i


def extract_and_match(image_dir, names, max_kpts=2048):
    """ALIKED keypoints at native resolution, LightGlue matches for every pair. Returns (kpts, matches)."""
    import torch
    from lightglue import ALIKED, LightGlue
    from lightglue.utils import load_image, rbd
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = ALIKED(max_num_keypoints=max_kpts).eval().to(dev)
    matcher = LightGlue(features="aliked").eval().to(dev)
    feats, kpts, matches = {}, {}, {}
    with torch.inference_mode():
        for n in names:
            feats[n] = extractor.extract(load_image(os.path.join(image_dir, n)).to(dev), resize=None)
            kpts[n] = rbd(feats[n])["keypoints"].cpu().numpy().astype(np.float32)
        for a, b in itertools.combinations(names, 2):
            out = rbd(matcher({"image0": feats[a], "image1": feats[b]}))
            matches[(a, b)] = out["matches"].cpu().numpy().astype(np.uint32)
    return kpts, matches


_SCHEMA = [   # COLMAP 3.x tables (as in hloc / COLMAP's database.py); only used if pycolmap can't create the file
    "CREATE TABLE IF NOT EXISTS cameras (camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, model INTEGER NOT NULL, "
    "width INTEGER NOT NULL, height INTEGER NOT NULL, params BLOB, prior_focal_length INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS images (image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, name TEXT NOT NULL UNIQUE, "
    "camera_id INTEGER NOT NULL, CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < 2147483647), "
    "FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))",
    "CREATE TABLE IF NOT EXISTS keypoints (image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, "
    "cols INTEGER NOT NULL, data BLOB, FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)",
    "CREATE TABLE IF NOT EXISTS descriptors (image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, "
    "cols INTEGER NOT NULL, data BLOB, FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)",
    "CREATE TABLE IF NOT EXISTS matches (pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, "
    "cols INTEGER NOT NULL, data BLOB)",
    "CREATE TABLE IF NOT EXISTS two_view_geometries (pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL, "
    "cols INTEGER NOT NULL, data BLOB, config INTEGER NOT NULL, F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB)",
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)",
]


def create_db(db_path):
    """An empty COLMAP database in this pycolmap's own schema (4.x adds rig/frame tables)."""
    for make in (lambda: pycolmap.Database.open(db_path), lambda: pycolmap.Database(db_path)):
        try:
            db = make()
            if hasattr(db, "close"):
                db.close()
            if os.path.exists(db_path):
                return "pycolmap"
        except (TypeError, AttributeError, ValueError, RuntimeError):
            pass
    con = sqlite3.connect(db_path)
    for q in _SCHEMA:
        con.execute(q)
    con.commit()
    con.close()
    return "sql"


def import_images(db_path, image_dir, names):
    """Cameras + images rows: one shared PINHOLE camera (focal prior 1.2 x max side, refined by BA)."""
    create_db(db_path)
    ro = pycolmap.ImageReaderOptions()
    ro.camera_model = "PINHOLE"
    pycolmap.import_images(db_path, image_dir, pycolmap.CameraMode.SINGLE, names, ro)   # positional: 3.10 and 4.x name args differently


def write_features(db_path, kpts, matches, min_matches=15):
    """Keypoints and raw matches into a database made by import_images (COLMAP's pixel-centre convention)."""
    con = sqlite3.connect(db_path)
    ids = {name: iid for iid, name in con.execute("SELECT image_id, name FROM images")}
    for n, k in kpts.items():
        k = np.ascontiguousarray(k + 0.5, np.float32)          # (0, 0) = centre of the top-left pixel -> COLMAP's corner origin
        con.execute("INSERT INTO keypoints(image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                    (ids[n], k.shape[0], 2, k.tobytes()))
    pairs = []
    for (a, b), m in matches.items():
        if len(m) < min_matches:
            continue
        m = np.asarray(m, np.uint32).reshape(-1, 2)
        if ids[a] > ids[b]:
            m = m[:, ::-1]
        m = np.ascontiguousarray(m)
        con.execute("INSERT INTO matches(pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                    (pair_id(ids[a], ids[b]), m.shape[0], 2, m.tobytes()))
        pairs.append((a, b))
    con.commit()
    con.close()
    return pairs


def verify_and_map(db_path, image_dir, pairs, work):
    pairs_txt = os.path.join(work, "pairs.txt")
    with open(pairs_txt, "w") as f:
        f.write("\n".join(f"{a} {b}" for a, b in pairs) + "\n")
    pycolmap.verify_matches(db_path, pairs_txt)
    return pycolmap.incremental_mapping(db_path, image_dir, os.path.join(work, "mapping"))


def sift_pipeline(db_path, image_dir, names, work):
    """R1's COLMAP baseline (fallback)."""
    ro = pycolmap.ImageReaderOptions()
    ro.camera_model = "PINHOLE"
    try:
        pycolmap.extract_features(db_path, image_dir, image_list=names, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro)
    except TypeError:
        pycolmap.extract_features(db_path, image_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro)
    pycolmap.match_exhaustive(db_path)
    return pycolmap.incremental_mapping(db_path, image_dir, os.path.join(work, "mapping"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--max_kpts", type=int, default=2048)
    ap.add_argument("--sift", action="store_true", help="skip LightGlue, use R1's SIFT pipeline")
    a = ap.parse_args()
    image_dir, work = os.path.join(a.scene, "images"), os.path.join(a.scene, "classical_work")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    db = os.path.join(work, "database.db")
    names = sorted(os.listdir(image_dir))
    info = dict(n_images=len(names), pycolmap=pycolmap.__version__)
    t0 = time.time()
    kpts = None
    if not a.sift:
        try:
            kpts, matches = extract_and_match(image_dir, names, a.max_kpts)
        except Exception as e:          # no LightGlue, weights not downloadable, ...: R1's SIFT pipeline instead
            print("ALIKED + LightGlue unavailable, using SIFT:", repr(e)[:300])
            info["fallback_reason"] = repr(e)[:300]
    if kpts is not None:
        info.update(features="aliked+lightglue", match_s=round(time.time() - t0, 1),
                    mean_kpts=float(np.mean([len(k) for k in kpts.values()])))
        t1 = time.time()
        import_images(db, image_dir, names)
        pairs = write_features(db, kpts, matches)
        info["pairs_with_matches"] = len(pairs)
        recs = verify_and_map(db, image_dir, pairs, work)
    else:
        t1 = time.time()
        info.update(features="sift", match_s=0.0)
        create_db(db)
        recs = sift_pipeline(db, image_dir, names, work)
    info["map_s"] = round(time.time() - t1, 1)
    info["total_s"] = round(time.time() - t0, 1)
    if not recs:
        json.dump(dict(info, registered=0), open(os.path.join(a.scene, "classical_info.json"), "w"), indent=1)
        raise SystemExit("classical: no reconstruction")
    best = max(recs.values(), key=lambda r: r.num_reg_images())
    out = os.path.join(a.scene, "sparse", "0")
    os.makedirs(out, exist_ok=True)
    best.write(out)
    info.update(models=len(recs), registered=best.num_reg_images(), points=best.num_points3D())
    json.dump(info, open(os.path.join(a.scene, "classical_info.json"), "w"), indent=1)
    print(info)


if __name__ == "__main__":
    main()

"""Classical baseline: COLMAP (SIFT + exhaustive matching + incremental mapping) through pycolmap.

One shared PINHOLE camera, so the model matches undistorted inputs (Mip-NeRF 360 images_4 are already
undistorted; a distortion model would make gsplat crop the images and break PSNR comparisons, F1.3).
Usage (VGGT environment, official pycolmap 3.10):
    python run_colmap.py <scene_dir>        # reads <scene_dir>/images, writes <scene_dir>/sparse/0
"""
import json
import os
import shutil
import sys
import time

import pycolmap


def main():
    scene = sys.argv[1]
    images, work = os.path.join(scene, "images"), os.path.join(scene, "colmap_work")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    db = os.path.join(work, "database.db")
    t0 = time.time()
    try:
        pycolmap.extract_features(db, images, camera_mode=pycolmap.CameraMode.SINGLE, camera_model="PINHOLE")
    except TypeError:                                   # older/newer signature: camera model via reader options
        ro = pycolmap.ImageReaderOptions()
        ro.camera_model = "PINHOLE"
        pycolmap.extract_features(db, images, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro)
    t1 = time.time()
    pycolmap.match_exhaustive(db)
    t2 = time.time()
    recs = pycolmap.incremental_mapping(db, images, os.path.join(work, "mapping"))
    t3 = time.time()
    if not recs:
        raise SystemExit("COLMAP: no reconstruction")
    best = max(recs.values(), key=lambda r: r.num_reg_images())
    out = os.path.join(scene, "sparse", "0")
    os.makedirs(out, exist_ok=True)
    best.write(out)
    info = dict(models=len(recs), registered=best.num_reg_images(), points=best.num_points3D(),
                extract_s=round(t1 - t0, 1), match_s=round(t2 - t1, 1), map_s=round(t3 - t2, 1),
                total_s=round(t3 - t0, 1), pycolmap=pycolmap.__version__)
    json.dump(info, open(os.path.join(scene, "colmap_info.json"), "w"), indent=1)
    print(info)


if __name__ == "__main__":
    main()

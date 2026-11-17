"""VGGT's demo_colmap.py with less GPU memory, plus an optional dump of VGGT's raw predictions.

demo_colmap.py keeps the 1B-parameter model (~5 GB in fp32) on the GPU while it predicts tracks for bundle
adjustment, which runs out of memory at 32 images on a T4. This wrapper moves the model to the CPU as soon as
the forward pass is done. Everything else is VGGT's own code and arguments.
Extra flags:
    --no_fine_tracking   demo_colmap's --fine_tracking defaults to True and can't be turned off there
    --save_pred X.npz    save VGGT's cameras, depth and depth confidence (1/4 resolution, fp16) for R3's signals
Run from the vggt repo folder (or with it on PYTHONPATH):
    python vggt_colmap_lowmem.py --scene_dir X [--use_ba] [--max_query_pts 2048 --query_frame_num 5] [--no_fine_tracking]
"""
import glob
import os
import sys

import numpy as np
import torch

import demo_colmap as dc

_run_vggt = dc.run_VGGT
SAVE_PRED, SCENE_DIR = None, None


def _np(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=np.float32)


def save_pred(out, path, scene_dir):
    extrinsic, intrinsic, depth, conf = (_np(x) for x in out[:4])
    depth, conf = depth.reshape(depth.shape[0], depth.shape[1], depth.shape[2]), conf.reshape(conf.shape[0], conf.shape[1], conf.shape[2])
    names = [os.path.basename(p) for p in glob.glob(os.path.join(scene_dir, "images", "*"))]   # demo_colmap's own order
    np.savez_compressed(path, names=np.array(names), extrinsic=extrinsic.astype(np.float32),
                        intrinsic=intrinsic.astype(np.float32), depth=depth[:, ::4, ::4].astype(np.float16),
                        conf=conf[:, ::4, ::4].astype(np.float16), resolution=np.array(depth.shape[1:3]))
    print(f"saved VGGT predictions for {len(names)} images to {path}")


def _run_then_free(model, *args, **kwargs):
    out = _run_vggt(model, *args, **kwargs)
    model.cpu()
    torch.cuda.empty_cache()
    if SAVE_PRED:
        try:
            save_pred(out, SAVE_PRED, SCENE_DIR)
        except Exception as e:          # never lose the reconstruction over the extra file
            print("could not save VGGT predictions:", repr(e))
    return out


dc.run_VGGT = _run_then_free

if __name__ == "__main__":
    no_fine = "--no_fine_tracking" in sys.argv
    if no_fine:
        sys.argv.remove("--no_fine_tracking")
    if "--save_pred" in sys.argv:
        i = sys.argv.index("--save_pred")
        SAVE_PRED = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    args = dc.parse_args()
    SCENE_DIR = args.scene_dir
    if no_fine:
        args.fine_tracking = False
    with torch.no_grad():
        dc.demo_fn(args)

"""VGGT's demo_colmap.py with less GPU memory, for VGGT + BA on a 16 GB T4.

demo_colmap.py keeps the 1B-parameter model (~5 GB in fp32) on the GPU while it predicts tracks for bundle
adjustment, which runs out of memory at 32 images on a T4. This wrapper moves the model to the CPU as soon as
the forward pass is done. Everything else is VGGT's own code and arguments.
Extra flag: --no_fine_tracking (demo_colmap's --fine_tracking defaults to True and can't be turned off there).
Run from the vggt repo folder (or with it on PYTHONPATH):
    python vggt_colmap_lowmem.py --scene_dir X --use_ba [--max_query_pts 2048 --query_frame_num 5] [--no_fine_tracking]
"""
import sys

import torch

import demo_colmap as dc

_run_vggt = dc.run_VGGT


def _run_then_free(model, *args, **kwargs):
    out = _run_vggt(model, *args, **kwargs)
    model.cpu()
    torch.cuda.empty_cache()
    return out


dc.run_VGGT = _run_then_free

if __name__ == "__main__":
    no_fine = "--no_fine_tracking" in sys.argv
    if no_fine:
        sys.argv.remove("--no_fine_tracking")
    args = dc.parse_args()
    if no_fine:
        args.fine_tracking = False
    with torch.no_grad():
        dc.demo_fn(args)

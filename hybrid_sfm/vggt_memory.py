"""Peak GPU memory and time of one VGGT pass vs number of images (R1: how big can a chunk be on this GPU?).

Runs the same branches as VGGT's demo_colmap.py (aggregator + camera head + depth head) at 518x518.
Usage (inside the VGGT environment, with the vggt repo on PYTHONPATH):
    python vggt_memory.py <image_dir> <out.json> 8 16 24 32 48 64
Stops at the first size that runs out of memory.
"""
import glob
import json
import sys
import time

import torch
import torch.nn.functional as F


def main():
    image_dir, out = sys.argv[1], sys.argv[2]
    sizes = [int(s) for s in sys.argv[3:]] or [8, 16, 24, 32, 48, 64]
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    dev = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url("https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"))
    model = model.eval().to(dev)
    weights_gb = torch.cuda.memory_allocated() / 1e9

    paths = sorted(glob.glob(f"{image_dir}/*"))
    rows = []
    for n in sizes:
        chosen = [paths[i % len(paths)] for i in range(n)]
        images = load_and_preprocess_images(chosen, mode="pad").to(dev)
        images = F.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False)
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t = time.time()
        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    tokens, ps_idx = model.aggregator(images[None])
                pose_enc = model.camera_head(tokens)[-1]
                pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                model.depth_head(tokens, images[None], ps_idx)
            torch.cuda.synchronize()
            row = dict(images=n, peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                       seconds=round(time.time() - t, 2), status="ok")
        except torch.cuda.OutOfMemoryError:
            row = dict(images=n, peak_gb=None, seconds=None, status="out of memory")
        print(row, flush=True)
        rows.append(row)
        del images
        torch.cuda.empty_cache()
        if row["status"] != "ok":
            break
    json.dump(dict(gpu=torch.cuda.get_device_name(0), dtype=str(dtype), weights_gb=round(weights_gb, 2),
                   total_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), rows=rows),
              open(out, "w"), indent=1)


if __name__ == "__main__":
    main()

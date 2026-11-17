"""R2 chunk sampler: deterministic image subsets of one scene with varied difficulty.

A chunk is what the router decides on. To get labels where each reconstructor sometimes wins and sometimes
fails, chunks vary in size and in baseline:
- window: n consecutive images (images are sorted by name, i.e. capture order): narrow baseline, partial view;
- stride: n images evenly spaced over the whole capture: wide baseline, sparse 360° coverage (R1's bicycle
  setting, where COLMAP registered 11 of 32);
- random: n images drawn at random: irregular overlap.
15 chunks per scene by default (6 window, 5 stride, 4 random); sizes stay <= 40 so one VGGT pass fits a T4 (R1).
"""
import hashlib

import numpy as np

DEFAULT_SPECS = ([("window", 24, k) for k in range(4)] + [("window", 40, k) for k in range(2)]
                 + [("stride", n, 0) for n in (12, 16, 24, 32, 40)]
                 + [("random", n, k) for k, n in enumerate((16, 24, 32, 40))])


def _seed(*parts):
    return int(hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def make_chunks(scene, names, specs=DEFAULT_SPECS):
    """[{id, scene, kind, n, k, names}] for one scene. Same input -> same chunks, on any machine."""
    names = sorted(names)
    N = len(names)
    out = []
    for kind, n, k in specs:
        n = min(n, N)
        if kind == "window":
            n_windows = sum(1 for s in specs if s[0] == "window" and s[1] == n)
            start = int(round(k * (N - n) / max(n_windows, 1))) if N > n else 0
            idx = np.arange(start, start + n)
        elif kind == "stride":
            idx = np.unique(np.linspace(0, N - 1, n).round().astype(int))
        elif kind == "random":
            idx = np.sort(np.random.default_rng(_seed(scene, kind, n, k)).choice(N, n, replace=False))
        else:
            raise ValueError(kind)
        out.append(dict(id=f"{scene}_{kind}{n}_{k}", scene=scene, kind=kind, n=int(len(idx)), k=k,
                        names=[names[i] for i in idx]))
    return out

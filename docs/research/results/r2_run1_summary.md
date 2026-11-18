# R2 label summary

**60 chunks** from 4 scenes; 3DGS labels at 3000 steps.

| Method | Reconstructed | ≥ 90% registered | 3DGS trained | Mean pose time |
|---|---|---|---|---|
| ref | 60/60 | 60/60 | 60/60 | – |
| classical | 60/60 | 47/60 | 47/60 | 32 s |
| vggt | 60/60 | 60/60 | 60/60 | 75 s |
| vggt_ba | 43/60 | 43/60 | 43/60 | 107 s |

**Routing on 60 chunks with all labels** (loss = dB below the official poses; a method with no usable twin costs 10 dB):

| Policy | Mean loss | Median loss | Failures | Mean pose time | Best on |
|---|---|---|---|---|---|
| always classical | 3.10 dB | 1.52 dB | 13 | 32 s | 35/60 |
| always vggt | 2.84 dB | 2.77 dB | 0 | 75 s | 9/60 |
| always vggt_ba | 4.13 dB | 2.34 dB | 17 | 107 s | 16/60 |
| oracle (best per chunk) | 1.16 dB | 1.23 dB | 0 | 60 s | – |

**Mean loss by chunk kind** (window = consecutive images, stride = spread over the capture, random):

| Kind | Chunks | classical | vggt | vggt_ba | Oracle |
|---|---|---|---|---|---|
| random | 16 | 4.11 | 2.60 | 4.55 | 1.42 |
| stride | 20 | 4.01 | 2.39 | 3.96 | 0.95 |
| window | 24 | 1.66 | 3.38 | 4.00 | 1.17 |

**Do pose metrics predict the twin's loss?** Spearman ρ over 150 trained (chunk, method) pairs. A perfect predictor has |ρ| = 1; for accuracy metrics (AUC, RRA) the expected sign is negative:

| Pose metric | ρ with loss | n |
|---|---|---|
| AUC@30 | -0.14 | 150 |
| AUC@5 | -0.18 | 150 |
| AUC@3 | -0.21 | 150 |
| RRA@5 | +0.09 | 150 |
| rot_err_median | +0.18 | 150 |
| ATE_% | +0.11 | 150 |
| focal_err | +0.30 | 150 |

**Label fidelity (3000 vs 7000 steps):** Spearman ρ = 0.95 over 28 runs; same best method on 3/4 chunks.

**Initialisation control** (same chunks, each method trained from its own points vs from the reference points moved into its frame; mean loss in dB):

| Method | Chunks | Own points | Reference points | Difference |
|---|---|---|---|---|
| classical | 6 | 1.28 | 0.50 | +0.78 |
| vggt | 8 | 3.02 | 1.86 | +1.16 |
| vggt_ba | 6 | 1.99 | 0.52 | +1.47 |


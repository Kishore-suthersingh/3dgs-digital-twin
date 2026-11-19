# R2 label summary

**105 chunks** from 7 scenes; 3DGS labels at 3000 steps.

| Method | Reconstructed | ≥ 90% registered | 3DGS trained | Mean pose time |
|---|---|---|---|---|
| ref | 105/105 | 105/105 | 105/105 | – |
| classical | 105/105 | 86/105 | 86/105 | 34 s |
| vggt | 105/105 | 105/105 | 105/105 | 76 s |
| vggt_ba | 74/105 | 74/105 | 74/105 | 112 s |

**Routing on 105 chunks with all labels** (loss = dB below the official poses; a method with no usable twin costs 10 dB):

| Policy | Mean loss | Median loss | Failures | Mean pose time | Best on |
|---|---|---|---|---|---|
| always classical | 3.11 dB | 1.76 dB | 19 | 34 s | 49/105 |
| always vggt | 2.53 dB | 2.32 dB | 0 | 76 s | 21/105 |
| always vggt_ba | 4.17 dB | 2.23 dB | 31 | 110 s | 35/105 |
| oracle (best per chunk) | 1.18 dB | 1.12 dB | 0 | 71 s | – |

**Mean loss by chunk kind** (window = consecutive images, stride = spread over the capture, random):

| Kind | Chunks | classical | vggt | vggt_ba | Oracle |
|---|---|---|---|---|---|
| random | 28 | 3.43 | 2.33 | 4.31 | 1.30 |
| stride | 35 | 4.33 | 2.02 | 3.88 | 1.01 |
| window | 42 | 1.88 | 3.10 | 4.33 | 1.24 |

**Do pose metrics predict the twin's loss?** Spearman ρ over 265 trained (chunk, method) pairs. A perfect predictor has |ρ| = 1; for accuracy metrics (AUC, RRA) the expected sign is negative:

| Pose metric | ρ with loss | n |
|---|---|---|
| AUC@30 | -0.14 | 265 |
| AUC@5 | -0.16 | 265 |
| AUC@3 | -0.18 | 265 |
| RRA@5 | -0.01 | 265 |
| rot_err_median | +0.18 | 265 |
| ATE_% | +0.13 | 265 |
| focal_err | +0.26 | 265 |

**Label fidelity (3000 vs 7000 steps):** Spearman ρ = 0.97 over 51 runs; same best method on 5/9 chunks.

**Initialisation control** (same chunks, each method trained from its own points vs from the reference points moved into its frame; mean loss in dB):

| Method | Chunks | Own points | Reference points | Difference |
|---|---|---|---|---|
| classical | 10 | 1.49 | 0.45 | +1.05 |
| vggt | 14 | 2.52 | 1.82 | +0.70 |
| vggt_ba | 12 | 1.78 | 0.64 | +1.14 |

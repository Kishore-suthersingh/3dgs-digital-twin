# R3 router results

**105 chunks, 7 scenes, leave-one-scene-out** (every router is fitted on 6 scenes and scored on the 7th). Loss = dB below the official poses (3k-step 3DGS, 10 dB for no usable twin). *Gap closed* = share of the always-VGGT → oracle gap recovered. *vs always VGGT* = mean loss difference with a paired bootstrap 95% CI (negative = better). Pose time includes the probes a router needs (classical + VGGT forward for the full router) and, for probe routers, the probes' 3DGS training time.

| Router | Mean loss | Median | Regret | Near-oracle (≤0.5 dB) | Unusable | Gap closed | vs always VGGT (95% CI) | Pose time | Picks c / v / ba |
|---|---|---|---|---|---|---|---|---|---|
| always classical | 3.11 | 1.76 | 1.93 | 59% | 19 | -42% | +0.57 [-0.13, +1.31] | 34 s | 105 / 0 / 0 |
| always vggt | 2.53 | 2.32 | 1.35 | 31% | 0 | 0% | +0.00 [+0.00, +0.00] | 76 s | 0 / 105 / 0 |
| always vggt_ba | 4.17 | 2.23 | 2.99 | 40% | 31 | -121% | +1.64 [+0.94, +2.41] | 110 s | 0 / 0 / 105 |
| best fixed (train) | 2.53 | 2.32 | 1.35 | 31% | 0 | 0% | +0.00 [+0.00, +0.00] | 76 s | 0 / 105 / 0 |
| kind rule (metadata) | 2.04 | 1.88 | 0.86 | 50% | 1 | 36% | -0.49 [-0.79, -0.23] | 61 s | 42 / 63 / 0 |
| best stump (train) | 1.80 | 1.56 | 0.62 | 66% | 1 | 54% | -0.73 [-1.03, -0.41] | 110 s | 83 / 22 / 0 |
| stump: cls_registered | 1.68 | 1.55 | 0.50 | 70% | 0 | 63% | -0.86 [-1.17, -0.56] | 110 s | 86 / 19 / 0 |
| stump: conf_mean | 3.11 | 2.63 | 1.93 | 24% | 6 | -43% | +0.58 [+0.22, +1.03] | 110 s | 30 / 75 / 0 |
| stump: mv_consistent | 2.27 | 1.99 | 1.09 | 50% | 1 | 20% | -0.27 [-0.53, +0.04] | 110 s | 60 / 45 / 0 |
| stump: agree_auc3 | 2.55 | 1.99 | 1.37 | 45% | 4 | -1% | +0.02 [-0.32, +0.41] | 110 s | 51 / 54 / 0 |
| stump: geo_nn_baseline | 2.30 | 1.85 | 1.12 | 50% | 5 | 17% | -0.24 [-0.66, +0.25] | 110 s | 61 / 44 / 0 |
| learned: VGGT probe only | 2.15 | 1.82 | 0.97 | 55% | 0 | 28% | -0.38 [-0.57, -0.19] | 113 s | 32 / 51 / 22 |
| **learned: all features** | 1.65 | 1.50 | 0.46 | 71% | 0 | 66% | -0.89 [-1.16, -0.64] | 122 s | 60 / 34 / 11 |
| learned: all features, gradient boosting | 1.60 | 1.46 | 0.42 | 76% | 0 | 69% | -0.93 [-1.22, -0.64] | 133 s | 58 / 26 / 21 |
| probe@250: best probe PSNR (all methods) | 1.59 | 1.48 | 0.41 | 70% | 0 | 70% | -0.94 [-1.23, -0.68] | 229 s | 50 / 51 / 4 |
| probe@250: best probe PSNR (classical + VGGT) | 1.64 | 1.50 | 0.46 | 70% | 0 | 66% | -0.89 [-1.18, -0.64] | 116 s | 50 / 55 / 0 |
| probe@500: best probe PSNR (all methods) | 1.47 | 1.36 | 0.29 | 80% | 0 | 79% | -1.06 [-1.34, -0.81] | 243 s | 56 / 34 / 15 |
| probe@500: best probe PSNR (classical + VGGT) | 1.59 | 1.48 | 0.41 | 74% | 0 | 70% | -0.94 [-1.22, -0.69] | 126 s | 61 / 44 / 0 |
| probe@1000: best probe PSNR (all methods) | 1.42 | 1.29 | 0.24 | 83% | 0 | 83% | -1.12 [-1.40, -0.85] | 264 s | 47 / 35 / 23 |
| probe@1000: best probe PSNR (classical + VGGT) | 1.58 | 1.48 | 0.40 | 75% | 0 | 70% | -0.95 [-1.23, -0.70] | 142 s | 54 / 51 / 0 |
| learned: all features + probe@250 | 1.49 | 1.29 | 0.31 | 82% | 0 | 77% | -1.05 [-1.34, -0.75] | 229 s | 56 / 12 / 37 |
| learned: all features + probe@500 | 1.51 | 1.29 | 0.33 | 82% | 0 | 76% | -1.03 [-1.31, -0.73] | 243 s | 56 / 11 / 38 |
| learned: all features + probe@1000 | 1.50 | 1.26 | 0.32 | 80% | 0 | 77% | -1.04 [-1.33, -0.74] | 264 s | 53 / 13 / 39 |
| pose-error router: AUC@3 (GT, not deployable) | 1.60 | 1.56 | 0.42 | 70% | 0 | 69% | -0.94 [-1.21, -0.69] | 220 s | 65 / 11 / 29 |
| pose-error router: AUC@30 (GT, not deployable) | 1.61 | 1.48 | 0.43 | 71% | 0 | 69% | -0.93 [-1.20, -0.68] | 220 s | 68 / 20 / 17 |
| pose-error router: rot_err_median (GT, not deployable) | 1.55 | 1.56 | 0.37 | 74% | 0 | 73% | -0.99 [-1.25, -0.74] | 220 s | 55 / 10 / 40 |
| oracle | 1.18 | 1.12 | 0.00 | 100% | 0 | 100% | -1.35 [-1.62, -1.12] | 71 s | 49 / 21 / 35 |

Best stump picked per fold: cls_points_per_image, cls_registered.

**Per held-out scene** (mean loss, dB):

| Scene | always classical | always VGGT | learned: all features | oracle |
|---|---|---|---|---|
| bicycle | 3.43 | 3.05 | 0.89 | 0.52 |
| bonsai | 3.65 | 3.23 | 2.54 | 2.03 |
| counter | 3.08 | 1.23 | 0.89 | 0.64 |
| garden | 1.25 | 2.60 | 1.34 | 1.23 |
| kitchen | 2.70 | 3.76 | 3.17 | 2.43 |
| room | 3.60 | 1.39 | 1.41 | 0.54 |
| stump | 4.06 | 2.47 | 1.28 | 0.87 |

**Ablation** (learned router without one feature group):

| Without | Mean loss | Gap closed |
|---|---|---|
| (nothing, full router) | 1.65 | 66% |
| VGGT confidence | 1.75 | 58% |
| VGGT camera layout | 1.69 | 63% |
| VGGT focal | 1.65 | 66% |
| multi-view depth consistency | 1.67 | 63% |
| classical stats | 2.31 | 17% |
| VGGT vs classical agreement | 1.68 | 63% |
| chunk size | 1.68 | 63% |

**Feature importance** (random forest impurity, averaged over folds, methods and seeds; by group):

| Group | Share |
|---|---|
| classical stats | 36% |
| VGGT confidence | 19% |
| multi-view depth consistency | 13% |
| VGGT camera layout | 12% |
| chunk size | 9% |
| VGGT vs classical agreement | 8% |
| VGGT focal | 4% |

Top features: `cls_registered` 17.3%, `n_images` 9.2%, `cls_points_per_image` 7.4%, `cls_track_len` 6.1%, `mv_frame_min` 6.0%, `conf_frame_cv` 5.8%, `mv_overlap` 3.7%, `conf_frac_gt2` 3.5%.

**Within-chunk ranking** (pairs of usable methods on the same chunk whose PSNR differs by ≥ 0.3 dB: how often does the score order them correctly? 50% = chance):

| Score | Correct | Pairs |
|---|---|---|
| AUC@3 (GT) | 73% | 182 |
| AUC@30 (GT) | 72% | 182 |
| median rotation error (GT) | 75% | 182 |
| ATE (GT) | 73% | 182 |
| focal error (GT) | 69% | 182 |
| learned predicted loss (no GT) | 62% | 182 |
| probe PSNR at 250 steps (no GT) | 58% | 182 |
| probe PSNR at 500 steps (no GT) | 76% | 182 |
| probe PSNR at 1000 steps (no GT) | 80% | 182 |

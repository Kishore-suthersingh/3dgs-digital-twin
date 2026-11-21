# R3 router results

**105 chunks, 7 scenes, leave-one-scene-out** (every router is fitted on 6 scenes and scored on the 7th). Loss = dB below the official poses (3k-step 3DGS, 10 dB for no usable twin). *Gap closed* = share of the always-VGGT → oracle gap recovered. *vs always VGGT* = mean loss difference with a paired bootstrap 95% CI (negative = better). Pose time includes the probes a router needs (classical + VGGT forward for the full router) and, for probe routers, the probes' 3DGS training time.

| Router | Mean loss | Median | Regret | Near-oracle (≤0.5 dB) | Unusable | Gap closed | vs always VGGT (95% CI) | Pose time | Picks c / v / ba |
|---|---|---|---|---|---|---|---|---|---|
| always classical | 3.31 | 1.52 | 2.50 | 58% | 23 | -108% | +1.30 [+0.45, +2.19] | 34 s | 105 / 0 / 0 |
| always vggt | 2.02 | 1.64 | 1.20 | 36% | 0 | 0% | +0.00 [+0.00, +0.00] | 76 s | 0 / 105 / 0 |
| always vggt_ba | 4.04 | 2.01 | 3.23 | 37% | 31 | -169% | +2.02 [+1.31, +2.82] | 110 s | 0 / 0 / 105 |
| best fixed (train) | 2.02 | 1.64 | 1.20 | 36% | 0 | 0% | +0.00 [+0.00, +0.00] | 76 s | 0 / 105 / 0 |
| kind rule (metadata) | 1.80 | 1.32 | 0.98 | 58% | 3 | 18% | -0.22 [-0.69, +0.38] | 61 s | 42 / 63 / 0 |
| best stump (train) | 2.38 | 1.59 | 1.56 | 57% | 7 | -30% | +0.36 [-0.23, +1.05] | 110 s | 74 / 31 / 0 |
| stump: cls_registered | 2.05 | 1.50 | 1.23 | 61% | 4 | -2% | +0.03 [-0.50, +0.70] | 110 s | 76 / 29 / 0 |
| stump: conf_mean | 2.90 | 1.94 | 2.09 | 27% | 8 | -74% | +0.89 [+0.34, +1.54] | 110 s | 17 / 88 / 0 |
| stump: mv_consistent | 2.19 | 1.64 | 1.37 | 40% | 2 | -14% | +0.17 [-0.20, +0.73] | 110 s | 12 / 93 / 0 |
| stump: agree_auc3 | 2.13 | 1.57 | 1.32 | 48% | 4 | -9% | +0.11 [-0.22, +0.51] | 110 s | 37 / 68 / 0 |
| stump: geo_nn_baseline | 2.73 | 1.90 | 1.91 | 30% | 6 | -59% | +0.71 [+0.21, +1.37] | 110 s | 16 / 89 / 0 |
| learned: VGGT probe only | 2.10 | 1.51 | 1.28 | 51% | 2 | -7% | +0.08 [-0.32, +0.65] | 93 s | 26 / 71 / 8 |
| **learned: all features** | 1.52 | 1.12 | 0.71 | 65% | 1 | 41% | -0.50 [-0.76, -0.24] | 117 s | 45 / 52 / 8 |
| learned: all features, gradient boosting | 1.42 | 1.22 | 0.61 | 66% | 0 | 49% | -0.59 [-0.84, -0.37] | 120 s | 52 / 42 / 11 |
| probe@250: best probe PSNR (all methods) | 1.18 | 0.97 | 0.36 | 79% | 0 | 70% | -0.84 [-1.10, -0.60] | 229 s | 50 / 51 / 4 |
| probe@250: best probe PSNR (classical + VGGT) | 1.24 | 1.01 | 0.43 | 75% | 0 | 65% | -0.78 [-1.03, -0.53] | 116 s | 50 / 55 / 0 |
| probe@500: best probe PSNR (all methods) | 1.16 | 0.92 | 0.35 | 82% | 0 | 71% | -0.86 [-1.12, -0.60] | 243 s | 56 / 34 / 15 |
| probe@500: best probe PSNR (classical + VGGT) | 1.39 | 1.03 | 0.57 | 76% | 1 | 53% | -0.63 [-0.95, -0.30] | 126 s | 61 / 44 / 0 |
| probe@1000: best probe PSNR (all methods) | 1.10 | 0.80 | 0.28 | 86% | 0 | 77% | -0.92 [-1.18, -0.67] | 264 s | 47 / 35 / 23 |
| probe@1000: best probe PSNR (classical + VGGT) | 1.28 | 0.90 | 0.46 | 75% | 0 | 62% | -0.74 [-0.99, -0.51] | 142 s | 54 / 51 / 0 |
| learned: all features + probe@250 | 1.21 | 1.01 | 0.40 | 75% | 0 | 67% | -0.81 [-1.07, -0.55] | 229 s | 52 / 20 / 33 |
| learned: all features + probe@500 | 1.24 | 1.08 | 0.43 | 74% | 0 | 65% | -0.78 [-1.04, -0.52] | 243 s | 50 / 22 / 33 |
| learned: all features + probe@1000 | 1.11 | 0.97 | 0.29 | 83% | 0 | 76% | -0.91 [-1.17, -0.65] | 264 s | 48 / 19 / 38 |
| pose-error router: AUC@3 (GT, not deployable) | 1.24 | 1.06 | 0.42 | 74% | 0 | 65% | -0.78 [-1.05, -0.53] | 220 s | 65 / 11 / 29 |
| pose-error router: AUC@30 (GT, not deployable) | 1.25 | 1.09 | 0.43 | 71% | 0 | 64% | -0.77 [-1.04, -0.51] | 220 s | 68 / 20 / 17 |
| pose-error router: rot_err_median (GT, not deployable) | 1.21 | 1.19 | 0.39 | 76% | 0 | 68% | -0.81 [-1.08, -0.56] | 220 s | 55 / 10 / 40 |
| oracle | 0.82 | 0.65 | 0.00 | 100% | 0 | 100% | -1.20 [-1.43, -0.99] | 68 s | 51 / 26 / 28 |

Best stump picked per fold: cls_pair_frac, cls_points_per_image, cls_registered.

**Per held-out scene** (mean loss, dB):

| Scene | always classical | always VGGT | learned: all features | oracle |
|---|---|---|---|---|
| bicycle | 3.46 | 2.63 | 1.03 | 0.61 |
| bonsai | 3.94 | 3.87 | 3.69 | 2.14 |
| counter | 2.93 | 0.95 | 1.05 | 0.46 |
| garden | 1.32 | 2.34 | 1.35 | 1.24 |
| kitchen | 1.85 | 1.93 | 1.95 | 0.69 |
| room | 5.71 | 0.58 | 0.54 | -0.01 |
| stump | 4.00 | 1.83 | 1.04 | 0.59 |

**Ablation** (learned router without one feature group):

| Without | Mean loss | Gap closed |
|---|---|---|
| (nothing, full router) | 1.52 | 41% |
| VGGT confidence | 1.68 | 28% |
| VGGT camera layout | 1.55 | 39% |
| VGGT focal | 1.52 | 41% |
| multi-view depth consistency | 1.53 | 41% |
| classical stats | 1.87 | 12% |
| VGGT vs classical agreement | 1.88 | 12% |
| chunk size | 1.53 | 41% |

**Feature importance** (random forest impurity, averaged over folds, methods and seeds; by group):

| Group | Share |
|---|---|
| classical stats | 28% |
| VGGT confidence | 18% |
| VGGT vs classical agreement | 17% |
| VGGT camera layout | 16% |
| multi-view depth consistency | 9% |
| chunk size | 9% |
| VGGT focal | 4% |

Top features: `cls_registered` 10.7%, `n_images` 8.7%, `cls_points_per_image` 7.7%, `conf_frame_cv` 7.0%, `agree_ate` 5.4%, `mv_frame_min` 4.5%, `cls_track_len` 4.1%, `geo_max_nn_baseline` 4.1%.

**Within-chunk ranking** (pairs of usable methods on the same chunk whose PSNR differs by ≥ 0.3 dB: how often does the score order them correctly? 50% = chance):

| Score | Correct | Pairs |
|---|---|---|
| AUC@3 (GT) | 70% | 182 |
| AUC@30 (GT) | 66% | 182 |
| median rotation error (GT) | 74% | 182 |
| ATE (GT) | 68% | 182 |
| focal error (GT) | 68% | 182 |
| learned predicted loss (no GT) | 54% | 182 |
| probe PSNR at 250 steps (no GT) | 66% | 182 |
| probe PSNR at 500 steps (no GT) | 75% | 182 |
| probe PSNR at 1000 steps (no GT) | 83% | 182 |

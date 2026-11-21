# R5.1: independent scoring of the twins

**Question.** R2's label scores each twin at the method's *own* held-out poses, the same measurement the R3b probe makes after 1,000 steps. Do R2–R4's conclusions survive a label that never uses the method's poses at test time?

**Protocol** (`hybrid_sfm/r5_eval.py`, notebook `07_r5_independent.ipynb`, one Kaggle T4 run). Same 105 chunks, same 3k-step twins (retrained from R2's reconstructions). For each held-out view:

1. **aligned:** a Sim(3) maps the reference cameras onto the twin's own *training* cameras (Umeyama on centres). The held-out reference camera is rendered with the reference intrinsics.
2. **refined:** starting from (1), 200 steps of test-time pose optimisation, with the Gaussians frozen (the NoPe-NeRF / CF-3DGS protocol).

Label = PSNR_ref − PSNR_method under (2), with 10 dB when there is no usable twin. Files:
- scores: [r5_scores.jsonl](r5_scores.jsonl)
- labels in R2 format: [r5_labels_refined.jsonl](r5_labels_refined.jsonl) and [r5_labels_aligned.jsonl](r5_labels_aligned.jsonl)
- reruns: [r3_indep.md](r3_indep.md), [r3b_indep.md](r3b_indep.md), [r4_indep.md](r4_indep.md)

## 1. How the labels differ

| | value |
|---|---|
| (chunk, method) pairs scored | 265 (+105 reference twins) |
| Spearman ρ, own-pose loss vs independent loss | 0.79 (Pearson 0.65) |
| median abs. difference | 0.21 dB (mean 0.75) |
| pairs differing by more than 1 dB | 50 / 265 |
| same best method per chunk | 91 / 105 |
| cost of picking R2's oracle, scored on independent labels | 0.10 dB mean (max 4.5) |
| reference twin: aligned vs own pose | within 0.03 dB (checks the alignment) |
| aligned before refinement, mean extra loss | 2.3 dB (pose error alone) |

Agreement is per scene: garden ρ = 0.99, bicycle 0.95, stump 0.94, counter 0.90, bonsai 0.80, room 0.71, kitchen 0.42.

The disagreements are of two kinds:

- **Own-pose scoring hid real failures.** Classical on room and kitchen sometimes scores 3–5 dB at its own poses but 10–18 dB independently. The Sim(3) residual is huge, and refinement would need 15–33° of rotation. Some cameras were registered wrongly, yet the twin is self-consistent at its own poses. 4 pairs flip to "unusable".
- **Own-pose scoring over-penalised noisy held-out poses.** On kitchen, after refinement, the twin itself is about as good as the reference twin, or slightly better.

## 2. Do the conclusions hold?

Leave-one-scene-out over 105 chunks, every router exactly as in R3/R3b, with only the label changed:

| Router | Regret, own-pose labels (R3b) | Regret, independent labels | Gap closed (indep.) | Unusable (indep.) |
|---|---|---|---|---|
| always VGGT | 1.35 | 1.20 | 0% | 0 |
| always classical | 1.93 | 2.50 | −108% | 23 |
| best one-threshold rule (stump) | 0.62 | 1.56 | −30% | 7 |
| learned router, 32 features (RF) | 0.46 | 0.71 | 41% | 1 |
| ground-truth pose router, AUC@3 (not deployable) | 0.42 | 0.42 | 65% | 0 |
| **probe@1000, best probe PSNR** | **0.24** | **0.28** | **77%** | **0** |
| learned + probe@1000 | 0.32 | 0.29 | 76% | 0 |

Paired bootstrap, mean loss difference with 95% CI (negative = probe better):

| | own-pose labels | independent labels |
|---|---|---|
| probe@1000 vs best stump | −0.39 [−0.60, −0.19] | **−1.28 [−1.92, −0.74]** |
| probe@1000 vs learned router | −0.23 [−0.39, −0.08] | **−0.43 [−0.71, −0.18]** |
| probe@1000 vs always VGGT | — | −0.92 [−1.18, −0.67] |

Within-chunk pairwise ranking (ties < 0.3 dB ignored, 182 pairs):

| Signal | own-pose | independent |
|---|---|---|
| probe PSNR @1000 (no ground truth) | 80% | **83%** |
| probe PSNR @500 | 76% | 75% |
| median rotation error (ground truth) | 75% | 74% |
| AUC@3 (ground truth) | 73% | 70% |
| learned predicted loss | 62% | 54% |

Conformal cascade (R4), realized risk on held-out scenes:

| α | own-pose labels | independent labels | loss / time (indep.) |
|---|---|---|---|
| 0.05 | 0.048 | 0.057 | 1.15 dB / 228 s |
| 0.1 | 0.095 | 0.114 | 1.20 dB / 200 s |
| 0.2 | 0.190 | 0.190 | 1.30 dB / 167 s |
| 0.3 | 0.267 | 0.267 | 1.37 dB / 146 s |

(Probing everything costs 264 s and reaches 1.10 dB; the oracle reaches 0.82 dB.)

## 3. Reading

1. **The main claim holds and gets stronger.** The 1,000-step probe is still the best deployable router (77% of the gap) and ranks methods better than ground-truth pose error does (83% vs 70–74%). The circularity concern is answered: the probe was not winning because it shares a measurement with the label.
2. **Cheap geometric rules were partly an artefact of the own-pose label.** The one-threshold rule falls from 63% of the gap to −30%, worse than always-VGGT. It favours classical, whose wrong registrations the own-pose label could not see. The learned feature router also drops, from 66% to 41%. This makes the paper's case sharper: pose-level signals miss failures that only the rendered twin exposes.
3. **Conformal control is marginal, not exact, across scenes.** The realized risk slightly exceeds α at 0.05 and 0.1 (0.057, 0.114). Scenes are not exchangeable, and per-scene coverage fails on 2–3 of 7. The paper should state the guarantee as marginal over exchangeable chunks and report this gap as is.
4. **For the paper, the independent label becomes the primary label.** Own-pose results go in an appendix, as a robustness check.

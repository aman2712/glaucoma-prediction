# EfficientNet held-out evaluation: development gains did not establish external superiority

Job 300953 completed both partitions successfully on 15 November 2026. Completion binds frozen protocol SHA-256 `ad35dc9783e997a2d2faeab32d778fda35db3444a9ff04ce7b42e6ea580ec434`. Full protocol, per-seed/ensemble predictions and model checkpoints remain in the HPC archive under `hpc_experiments/runs/efficientnet_heldout_20261115/`.

The E2 and E0 ensembles each use original 808/200-trained checkpoints for seeds 2440–2442, averaged in logit space. These are not CV-fold models. Validation-only thresholds were frozen before inference: E2 0.0231444 and E0 0.0168631. The historical single-seed Baseline B uses its original 0.0078666 threshold and saved scores. No retraining, threshold retuning or best-seed selection occurred during evaluation.

## Full classification cohorts

| Dataset | Model | n / positives | AUROC | AP | Sensitivity | Specificity | Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| CHAKSU | Historical B, single ResNet | 335 / 51 | 0.7921 | 0.4078 | 86.27% | 55.63% | 0.1230 |
| CHAKSU | Matched E0, three ResNets | 335 / 51 | 0.8112 | 0.4691 | 78.43% | 60.92% | 0.1097 |
| CHAKSU | E2, three EfficientNets | 335 / 51 | 0.8418 | 0.5305 | 86.27% | 61.62% | 0.1098 |
| PAPILA | Historical B, single ResNet | 420 / 87 | 0.7362 | 0.4348 | 59.77% | 75.68% | 0.1939 |
| PAPILA | Matched E0, three ResNets | 420 / 87 | 0.7360 | 0.4424 | 29.89% | 92.79% | 0.1987 |
| PAPILA | E2, three EfficientNets | 420 / 87 | 0.7268 | 0.4319 | 63.22% | 70.57% | 0.1914 |

E2 AUROC 95% grouped-bootstrap intervals: CHAKSU **0.7807–0.8933**, PAPILA **0.6493–0.7997**. All intervals are in [confidence_intervals.csv](confidence_intervals.csv), and all point metrics/confusion counts are in [metrics.csv](metrics.csv).

## Paired architecture comparison

| AUROC difference | CHAKSU estimate [95% CI] | PAPILA estimate [95% CI] |
|---|---:|---:|
| E2 minus matched E0 | +0.0307 [-0.0123, +0.0723] | -0.0092 [-0.0863, +0.0640] |
| E2 minus historical B | +0.0497 [+0.0074, +0.0928] | -0.0094 [-0.0897, +0.0632] |

Intervals use 2,000 paired resamples of whole analysis groups (314 CHAKSU; 210 PAPILA), conditional on fitted models and thresholds. Both primary matched-E0 intervals include zero. The historical CHAKSU gain excludes zero, but that comparison also changes the number of ensemble members, so it does not isolate architecture effects. No external AUROC improvement was demonstrated.

On PAPILA E2 detects 55/87 positives, misses 32, and flags 98/333 negatives. Relative to historical B, this is three additional detected positives and 17 additional false positives. The validation 90% sensitivity target does not transfer. Higher sensitivity than E0 comes with lower specificity; it is not evidence of dominance at all operating points.

## Measurement baseline A: descriptive comparison

Baseline A scores were not located for this evaluation, so its published common-split results are transcribed from `docs/baseline_a.md`, not recomputed. Its cohort is segmentation-eligible; [baseline_A_eligible_subset.csv](baseline_A_eligible_subset.csv) reports a post-hoc descriptive restriction of the already frozen image predictions to that predefined eligibility policy, without new inference or tuning.

| Cohort | Measurement A, published AUROC | E2 on matching eligibility policy | Historical B on matching eligibility policy |
|---|---:|---:|---:|
| CHAKSU: 326 / 50 positives | 0.943 [0.898–0.976] | 0.8421 | 0.7929 |
| PAPILA: 418 / 85 positives | 0.731 [0.663–0.791] | 0.7275 | 0.7351 |

A's published intervals use image-level resampling. No paired E2–A interval can be calculated without A's actual aligned per-image scores. Threshold criteria and training procedures also differ. Anatomy remains numerically stronger on CHAKSU, whose suspicion labels track cup morphology; PAPILA values are similar and do not establish comparative clinical utility.

## Evidence-supported conclusion

EfficientNet improved development discrimination across grouped resplits and has a positive CHAKSU held-out point difference, but the matched held-out interval includes zero and the gain did not transfer to PAPILA. Improving internal architecture performance alone did not resolve cross-dataset generalization. Reference-standard differences, acquisition differences, calibration and incomplete patient linkage remain plausible contributors; no causal attribution was established.

Both test cohorts had been inspected in earlier baselines/P01. This is **reused-test exploratory evaluation**, not an unexposed confirmation of the development-selected architecture. Do not tune on these results and then present another evaluation on the same cohorts as independent evidence. No prospective clinical/workflow validation was performed.

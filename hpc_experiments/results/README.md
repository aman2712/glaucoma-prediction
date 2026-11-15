# Main result: EfficientNet-B0 versus ResNet-18

**Held-out evaluation is now complete:** [test comparison](efficientnet_heldout_20261115/README.md). E2's development advantage did not establish superiority on either held-out cohort against matched E0; PAPILA AUROC was slightly lower. Read the development findings below together with that result.

The strongest completed experiment is the grouped development cross-validation study confirming the architecture comparison. The six-experiment screening study identified EfficientNet-B0 (E2) as the leading candidate; the subsequent grouped study checked whether that advantage persisted under different development splits.

## Six-experiment screening

Three seeds per method, same original 808/200 training/validation cohort. Reported values are means across seeds; AUROC SD describes training-seed variation on that same validation sample.

| Experiment | Change | Validation AUROC mean ± SD | Mean AP |
|---|---|---:|---:|
| E0 | Image-only ResNet-18 reference | 0.8162 ± 0.0169 | 0.4580 |
| E1 | ResNet-18 final-block-only fine-tuning | 0.7516 ± 0.0193 | 0.3434 |
| E2 | Image-only EfficientNet-B0 | **0.8594 ± 0.0064** | **0.5153** |
| E3 | ResNet-18 + auxiliary CDR supervision | 0.8157 ± 0.0187 | 0.4578 |
| E4 | E3 + predicted-CDR fusion | 0.8180 ± 0.0204 | 0.4459 |
| E5 | E4 + stronger colour augmentation | 0.8238 ± 0.0143 | 0.4659 |

Evidence: [comparison](six_experiments_20261114/validation_comparison.csv), [experiment log](six_experiments_20261114/experiment_log.csv), [paired seed differences](six_experiments_20261114/paired_seed_differences.csv), and [selected runs](six_experiments_20261114/selected_runs.json). Earliest completed runs were chosen by timestamp; duplicate E4 reruns are preserved on HPC and excluded, without choosing based on scores.

## Grouped split confirmation — primary finding

All 30 fits and outer evaluations completed: two architectures × five outer folds × three seeds. Only the original 1,008 eligible CHAKSU development images were used. Known analysis groups stay together. Within each outer training pool, a separate grouped inner selection subset chooses the checkpoint and threshold; outer-held images do not influence those decisions. Frozen design and exact role counts: [SPLIT_CONFIRMATION.md](../SPLIT_CONFIRMATION.md).

The primary comparison uses a mean-logit three-seed ensemble within each fold, then averages AUROC differences across folds. It is not the original single-split mean over individual seed models.

| Outer fold | E0 AUROC | E2 AUROC | Difference |
|---|---:|---:|---:|
| 1 | 0.8378 | 0.8998 | +0.0620 |
| 2 | 0.8270 | 0.8377 | +0.0107 |
| 3 | 0.8077 | 0.8470 | +0.0393 |
| 4 | 0.8205 | 0.8558 | +0.0353 |
| 5 | 0.8965 | 0.9002 | +0.0038 |
| Mean | **0.8379** | **0.8681** | **+0.0302** |

The paired mean AUROC difference is **+0.03022**, with a **95% grouped-bootstrap interval of +0.00627 to +0.05343**, from 2,000 valid resamples. The predefined positive-evidence rule was met. E2 has higher AUROC in all five ensemble fold comparisons and 13/15 individual fold/seed pairs. Mean ensemble AP is 0.5791 for E2 versus 0.5019 for E0. Individual-model threshold results show a trade-off: E2 generally has higher sensitivity but somewhat lower specificity; it does not dominate every operating point.

Evidence: [primary result](cv_confirmation_20261115/primary.json), [ensemble fold metrics](cv_confirmation_20261115/ensemble_fold_metrics.csv), and [all 30 seed/fold metrics](cv_confirmation_20261115/per_seed_fold_metrics.csv). HPC pilot job 299421, remaining array 299422, aggregate job 299423. The result was independently checked against the saved fold table before publication.

## Interpretation and limits

EfficientNet-B0 is the leading architecture under the specified development-selection rule, and its advantage persists under this grouped resplitting procedure. Anatomical auxiliary supervision/fusion has not demonstrated a reliable additional benefit in the tested ResNet-18 designs. These are controlled findings, not evidence that anatomy can never help.

E2 was chosen after examining earlier development results, so the CV is a **post-selection development sensitivity analysis**, not independent external confirmation. Patient identity is incomplete in CHAKSU; grouping handles known entities only. The bootstrap interval conditions on fitted models and does not capture uncertainty from repeating the entire search/training process. Neither these results nor the larger validation AUROC establish calibration, external sensitivity, clinical safety, or prospective workflow utility. No CHAKSU-test or PAPILA inference was performed in this CV study. Previously observed test results must be disclosed if those cohorts are reused later.

Preserve all seeds, folds, negative experiments, configurations and checkpoints in the approved HPC archive. Do not select the best seed. Next, freeze an agreed final prediction/threshold protocol and coordinate evaluation with the team; new independent data would be needed for an unexposed external confirmation.

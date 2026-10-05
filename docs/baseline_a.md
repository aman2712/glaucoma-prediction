# Baseline A: Measurement-Only Glaucoma Classifier

## Question
Do optic disc and cup measurements predicted by a segmentation network classify glaucoma,
and does the classifier hold up on a different camera and hospital (PAPILA)?

## Method
1. **Segment** disc and cup with a U-Net trained on CHAKSU train (512 px, STAPLE expert masks).
2. **Measure** from predicted masks: vertical CDR, horizontal CDR, area CDR, disc area, cup area.
3. **Classify** with logistic regression on the three ratios (vertical, horizontal, area CDR) only, with no image features. Absolute areas are excluded because they depend on each camera's field of view; an ablation on training cross-validation checks this, and checks a gradient-boosting model.
4. **Oracle** (reference only): the same classifier on measurements from expert masks. The gap to Baseline A is the cost of segmentation error.

## Data and splits
Frozen by `script/prepare_data.py` (seed 2440); see `DATA_PREPROCESSING.md`.

| Split | Role | Images (segmentation) |
|---|---|---|
| CHAKSU train | fit U-Net and classifier | 787 |
| CHAKSU validation | tune, choose threshold | 200 |
| CHAKSU test | in-domain test, used once | 327 |
| PAPILA | external test, used once | 486 (418 labelled) |

PAPILA "suspect" eyes are excluded from classification. PAPILA is never used for training or tuning.

## Evaluation protocol
- The regularisation strength C is chosen by 5-fold cross-validation on the training split (duplicate groups kept in one fold), because validation has only 27 glaucoma cases. The decision threshold is chosen on validation.
- The threshold is chosen once (Youden's J on validation) and applied unchanged to both test sets.
- Test sets are evaluated once, after everything is frozen.
- 95% confidence intervals: bootstrap, 2,000 resamples, over images.

## Metrics
**Classification (primary: AUC)**

| Metric | Why |
|---|---|
| AUC-ROC | threshold-free ranking quality; standard for glaucoma screening |
| Sensitivity, specificity | at the validation-chosen threshold; clinical meaning |
| Sensitivity at 95% specificity | screening-style operating point |
| AUPRC | positives are rare (about 13%), so AUC alone can be optimistic |
| Balanced accuracy | single-number summary robust to imbalance |

Plain accuracy is reported but not used for conclusions (always predicting "healthy" scores about 87%).

**Segmentation**: Dice and IoU for disc and cup; vertical CDR mean absolute error and Pearson correlation against expert masks (as in the REFUGE challenge).

Comparisons with the other baselines use the same splits and a paired bootstrap of the AUC difference.

## Limitations (to state in results)
- Splits are provisional: 1,304 CHAKSU images lack patient IDs, so patient overlap across splits cannot be excluded.
- Some groups have very few glaucoma cases (CHAKSU Bosch test: 1; Forus test: 2), so no per-camera claims.
- Glaucoma prevalence differs between CHAKSU and PAPILA, which affects precision-style metrics.

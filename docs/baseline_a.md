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
| CHAKSU train | fit U-Net; classifier development | 787 |
| CHAKSU validation | pick U-Net epoch; classifier development | 200 |
| CHAKSU test | in-domain test, used once | 327 |
| PAPILA | external test, used once | 486 (418 labelled) |

PAPILA "suspect" eyes are excluded from classification. PAPILA is never used for training or tuning.

## Evaluation protocol
- The U-Net is trained on train; the validation split picks its best epoch.
- The classifier uses all development images (train and validation), because validation alone has only 27 glaucoma cases. The regularisation strength C is chosen by 5-fold cross-validation (duplicate groups kept in one fold). The decision threshold is Youden's J on the pooled out-of-fold scores. The final classifier is then fitted on all development images.
- Everything is frozen and committed (`docs/frozen_baseline_a.json`: code version and file hashes) before any test image is touched.
- The test sets are evaluated once, by `final_evaluation.py`, which refuses to run on a changed pipeline or to run twice.
- 95% confidence intervals: bootstrap, 2,000 resamples, over images.

## Reporting rules (fixed before the test)
- CHAKSU test and PAPILA are reported separately and never pooled.
- PAPILA "suspect" eyes are excluded from the main result. Counting them as glaucoma is reported only as a labelled sensitivity analysis.
- Reference points on the same images: the oracle (expert masks) and vertical CDR alone with no model.
- Segmentation quality (Dice, CDR agreement) is reported on both test sets.
- Comparisons with the other baselines use per-image scores (`runs/final/scores.csv`) and a paired bootstrap of the AUC difference.
- No setting is changed after the test results are seen.

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

## Results
Frozen pipeline, run once on each test set. 95% confidence intervals in brackets.

| | CHAKSU test (327 images, 50 glaucoma) | PAPILA (418 images, 85 glaucoma) |
|---|---|---|
| **AUC, Baseline A** | **0.934** (0.884-0.972) | **0.718** (0.647-0.790) |
| AUC, expert masks (oracle) | 0.984 (0.973-0.993) | 0.799 (0.738-0.856) |
| AUC, vertical CDR alone, no model | 0.937 (0.895-0.970) | 0.722 (0.644-0.798) |
| Sensitivity / specificity | 0.78 / 0.91 | 0.40 / 0.94 |
| Sensitivity at 95% specificity | 0.74 | 0.40 |
| AUPRC | 0.829 | 0.559 |
| Dice, disc / cup | 0.976 / 0.916 | 0.836 / 0.472 |

If PAPILA suspect eyes count as glaucoma: AUC 0.658 (0.600-0.715); oracle 0.742.

- On the same population as training, vertical CDR alone predicts glaucoma well. The classifier adds nothing over it.
- On a new camera and hospital, performance drops sharply, and the threshold set on CHAKSU misses most glaucoma.
- Even perfect outlines only reach 0.80 on PAPILA, so segmentation errors are not the whole story.

## Why PAPILA failed (post-hoc)
Found after seeing the test results; nothing was changed because of it.

- **Cups are drawn differently.** The average expert vertical CDR is 0.36 on PAPILA and 0.55 on CHAKSU. PAPILA's glaucoma eyes (0.52) sit below CHAKSU's healthy average (0.53 versus 0.65 for CHAKSU glaucoma). A threshold learned on CHAKSU therefore calls most PAPILA glaucoma healthy.
- **The model draws CHAKSU-style cups.** On PAPILA its cups are about twice as large as the experts' (median 2.2x) and predicted vertical CDR averages 0.51 against 0.36.
- **Some images get no outline.** No disc was found in 20 of 486 PAPILA images and no cup in 40.
- **Images differ in scale.** PAPILA discs are about 1.8x larger in pixels than CHAKSU's.
- **Labels mean different things.** CHAKSU's glaucoma label follows expert votes that track CDR closely. PAPILA's is a clinical diagnosis.

## Limitations
- Splits are provisional: 1,304 CHAKSU images lack patient IDs, so patient overlap across splits cannot be excluded.
- CHAKSU glaucoma labels follow cup-to-disc ratio closely, so CHAKSU results are optimistic for a CDR-based method.
- The segmentation model saw only CHAKSU cameras, and PAPILA outlines follow a different convention.
- Training-set measurements come from images the segmentation model learned from, so they are slightly cleaner than on new images.
- Few glaucoma cases in some groups (CHAKSU Bosch test: 1; Forus test: 2), so no per-camera claims.
- Glaucoma prevalence differs between CHAKSU (13%) and PAPILA (20%), which affects precision-style metrics.

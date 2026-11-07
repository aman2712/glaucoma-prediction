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
The common split of record (`splits/`, shared with the image-only baseline): the data pipeline's frozen split (`data_pipeline/prepare_data.py`, seed 2440, see `data_pipeline/DATA_PREPROCESSING.md`) plus a precautionary grouping of possible repeat patients. Compared with the pipeline split it moves four images: two are quarantined (one CHAKSU train, one CHAKSU test) and two validation images move to training.

| Split | Role | Images (segmentation) |
|---|---|---|
| CHAKSU train | fit U-Net; classifier development | 787 |
| CHAKSU validation | pick U-Net epoch; classifier development | 199 |
| CHAKSU test | in-domain test, used once | 326 |
| PAPILA | external test, used once | 486 (418 labelled) |

PAPILA "suspect" eyes are excluded from classification. PAPILA is never used for training or tuning.

## Evaluation protocol
- The U-Net is trained on train; the validation split picks its best epoch.
- The classifier uses all development images (train and validation), because validation alone has only 27 glaucoma cases. The regularisation strength C is chosen by 5-fold cross-validation (images of one analysis group, meaning the same patient, a duplicate or a possible repeat patient, kept in one fold). The decision threshold is Youden's J on the pooled out-of-fold scores. The final classifier is then fitted on all development images.
- Everything is frozen and committed (`docs/frozen_baseline_a_common_split.json`: code version and file hashes, including the split file) before any test image is touched.
- The freeze records the code version and the hash of each file. Later work changed `src/` for other models, so the code check (`freeze.verify`) passes only at the frozen commit 55ff4b4; the file hashes (`freeze.verify_files`) still pass and are what later experiments rely on.
- The test sets are evaluated once, by `final_evaluation.py`, which refuses to run on a changed pipeline or to run twice.
- 95% confidence intervals: bootstrap, 2,000 resamples, over images.

## Reporting rules (fixed before the test)
- CHAKSU test and PAPILA are reported separately and never pooled.
- PAPILA "suspect" eyes are excluded from the main result. Counting them as glaucoma is reported only as a labelled sensitivity analysis.
- Reference points on the same images: the oracle (expert masks) and vertical CDR alone with no model.
- Segmentation quality (Dice, CDR agreement) is reported on both test sets.
- Comparisons with the other baselines use per-image scores (`runs/final/common_split/scores.csv`) and a paired bootstrap of the AUC difference.
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
Numbers of record: the pipeline frozen on the common split (`docs/frozen_baseline_a_common_split.json`) and run once on each test set. 95% confidence intervals in brackets.

| | CHAKSU test (326 images, 50 glaucoma) | PAPILA (418 images, 85 glaucoma) |
|---|---|---|
| **AUC, Baseline A** | **0.943** (0.898-0.976) | **0.731** (0.663-0.791) |
| AUC, expert masks (oracle) | 0.984 (0.973-0.993) | 0.799 (0.738-0.856) |
| AUC, vertical CDR alone, no model | 0.948 (0.909-0.979) | 0.762 (0.694-0.824) |
| Sensitivity / specificity | 0.80 / 0.92 | 0.59 / 0.78 |
| Sensitivity at 95% specificity | 0.74 | 0.35 |
| AUPRC | 0.838 | 0.524 |
| Dice, disc / cup | 0.977 / 0.914 | 0.800 / 0.458 |

If PAPILA suspect eyes count as glaucoma: AUC 0.656 (0.600-0.708); oracle 0.742 (0.689-0.789).

- On the same population as training, vertical CDR alone predicts glaucoma well. The classifier adds nothing over it.
- On a new camera and hospital, ranking performance drops sharply. Even perfect outlines only reach 0.80 on PAPILA, so segmentation errors are not the whole story.
- The sensitivity and specificity on PAPILA depend on the particular trained segmentation network (see the earlier run below), so read the AUC, not the operating point.

### Earlier run on the pipeline split
The first version of this pipeline used the data pipeline's split without the precautionary grouping. The common split was adopted so that every model shares one split. The U-Net was retrained and the pipeline re-frozen and re-evaluated with **no setting changed** (the segmentation config differs only in its run name). The test sets had already been evaluated once, by the earlier version, so this is a re-evaluation, not a first look; both are reported (experiment log, E05 and E10).

| | CHAKSU test (327 images) | PAPILA (418 images) |
|---|---|---|
| AUC, Baseline A | 0.934 (0.884-0.972) | 0.718 (0.647-0.790) |
| AUC, expert masks (oracle) | 0.984 (0.973-0.993) | 0.799 (0.738-0.856) |
| Sensitivity / specificity | 0.78 / 0.91 | 0.40 / 0.94 |
| Dice, disc / cup | 0.976 / 0.916 | 0.836 / 0.472 |

All AUCs of the two runs agree within the earlier intervals. The PAPILA sensitivity and specificity do not (0.40 / 0.94 earlier, 0.59 / 0.78 now), although the expert-mask results are identical. The cause is the retrained U-Net: on PAPILA it draws larger cups (mean predicted vertical CDR 0.51 earlier, 0.56 now) and finds no outline in more images (disc missing in 20 earlier, 36 now), while on CHAKSU test its predictions barely changed (0.552, 0.548). A threshold chosen on CHAKSU therefore sits at a different place in the PAPILA score distribution (13% of PAPILA scores above it earlier, 30% now). The ranking (AUC) is stable; the operating point is not.

**What the threshold is, and why the same number gives different results.** The classifier gives each eye a score between 0 and 1. The frozen threshold (0.568 for predicted masks) is a fixed cutoff: an eye scoring at or above it is called glaucoma, below it healthy. It was chosen on development data only (Youden's J, the cutoff that best balances sensitivity and specificity), and never re-tuned on test data. Sensitivity is the share of glaucoma eyes flagged; specificity is the share of healthy eyes cleared. The number 0.568 did not change between the two runs, but the two U-Nets give different scores to the same PAPILA eyes. The retrained one draws larger cups, so every ratio and score rises and more eyes (sick and healthy) cross the same cutoff: sensitivity rose from 0.40 to 0.59 and specificity fell from 0.94 to 0.78. AUC ignores the cutoff, so it barely moved. A threshold chosen on CHAKSU therefore does not transfer reliably to PAPILA, and the sensitivity and specificity on PAPILA should not be quoted without this caveat.

## Why PAPILA failed (post-hoc)
Found after seeing the test results; nothing was changed because of it.

- **Cups are drawn differently.** The average expert vertical CDR is 0.36 on PAPILA and 0.55 on CHAKSU. PAPILA's glaucoma eyes average 0.52, about the same as CHAKSU's healthy eyes (0.53); CHAKSU's glaucoma eyes average 0.67. A threshold learned on CHAKSU therefore calls many PAPILA glaucoma eyes healthy.
- **The model draws CHAKSU-style cups.** On PAPILA its cups are more than twice as large as the experts' (median 2.3x; 2.2x in the earlier run) and predicted vertical CDR averages 0.56 against 0.36.
- **Some images get no outline.** No disc was found in 36 of 486 PAPILA images and no cup in 46 (20 and 40 in the earlier run).
- **Images differ in scale.** PAPILA discs are about 1.8x larger in pixels than CHAKSU's.
- **Labels mean different things.** CHAKSU's glaucoma label follows expert votes that track CDR closely. PAPILA's is a clinical diagnosis.

## Limitations
- Splits are provisional: 1,304 CHAKSU images lack patient IDs, so patient overlap across splits cannot be excluded.
- CHAKSU glaucoma labels follow cup-to-disc ratio closely, so CHAKSU results are optimistic for a CDR-based method.
- The segmentation model saw only CHAKSU cameras, and PAPILA outlines follow a different convention.
- Training-set measurements come from images the segmentation model learned from, so they are slightly cleaner than on new images.
- Few glaucoma cases in some groups (CHAKSU Bosch test: 1; Forus test: 2), so no per-camera claims.
- Glaucoma prevalence differs between CHAKSU (13%) and PAPILA (20%), which affects precision-style metrics.
- One trained segmentation network per version. Retraining changed the PAPILA operating point a lot while leaving the AUC alone, so training variability is not covered by the confidence intervals (they only resample images). Running several seeds would show how much the PAPILA operating point varies; planned for the combined-model experiments.

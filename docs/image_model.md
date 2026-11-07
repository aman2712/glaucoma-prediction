# Baseline B: Image-Only Glaucoma Classifier

## Question
How well does a pretrained CNN that sees only the fundus photograph detect glaucoma, and does it hold up on a different camera and hospital (PAPILA)? This is the reference for the comparison with the measurement-only Baseline A (`docs/baseline_a.md`) and the combined model.

Only the RGB photograph enters the model. Expert masks, measurements, expert votes, camera IDs and clinical covariates do not. This is a research baseline, not a validated diagnostic system.

## Method
- **Input:** the full photograph, square-padded and resized to 224 x 224 by the data pipeline, ImageNet-normalised. Training adds mild brightness, contrast and saturation jitter. No crops, flips or rotations.
- **Model:** ImageNet-pretrained ResNet-18 with a one-logit head (ResNet-50 and EfficientNet-B0 are selectable in the config). Unweighted binary cross-entropy; class imbalance is handled by ranking metrics and the threshold, not by reweighting.
- **Training:** two head-only epochs, then the whole network is fine-tuned with BatchNorm statistics held fixed. AdamW, head learning rate 1e-4, backbone 1e-5, weight decay 1e-4, batch size 16, seed 2440, at most 25 epochs, early stopping after 6 fine-tuning epochs without gain.
- **Model choice:** the fine-tuning epoch with the best validation AUROC (ties broken by validation loss).
- **Threshold:** the highest-specificity threshold that reaches 90% sensitivity on CHAKSU validation. The score is an uncalibrated suspicion score, not a clinical probability.

All settings are in `configs/image_model.yaml`; the code is in `src/glaucoma_project/image_model/`.

## Data and splits
The model uses the pipeline's frozen split (`DATA_PREPROCESSING.md` in `data_pipeline/`) with one extra precaution, the **effective split** (`data/effective_splits.py`). CHAKSU Bosch images named `P<number>_Image<n>` may show the same patient in different releases, which the pipeline cannot confirm. They are grouped by P-number and merged with the known groups. A group spanning releases is quarantined; a group spanning train and validation moves to train. The rule uses no image content and no model result.

This is the common split of record (`splits/`), also used by Baseline A. Compared with the pipeline's own split it moves four images: two are quarantined and two validation images move to training.

| Split | Dataset | Images | Negative | Positive |
|---|---|---:|---:|---:|
| Training | CHAKSU | 808 | 699 | 109 |
| Validation | CHAKSU | 200 | 172 | 28 |
| Held-out in-domain test | CHAKSU | 335 | 284 | 51 |
| External test | PAPILA | 420 | 333 | 87 |

CHAKSU positives are expert-majority **glaucoma suspect** labels, not confirmed diagnoses. PAPILA uses clinical labels and excludes its 68 suspect images. This difference limits cross-dataset interpretation. Known groups and duplicates never cross splits, but unknown CHAKSU patient linkage means the results cannot be called strictly patient-disjoint.

Before reading any image, `data/audit.py` checks that the export is complete and unaltered (hashes), that each split file equals the eligible rows of `all.csv`, that datasets and labels are as expected, and that no known patient, pixel hash or group crosses splits.

## Evaluation protocol
- Development (`image_model/develop`) reads only train and validation images.
- `image_model/final` freezes the checkpoint, common split and threshold (SHA-256 in `frozen_protocol.json`), checks them again, then evaluates CHAKSU test and PAPILA **once**. It refuses to run twice and refuses a changed checkpoint.
- 95% confidence intervals come from 1,000 bootstrap resamples of whole patient groups (314 CHAKSU and 210 PAPILA groups). They describe sampling noise for this checkpoint and threshold, not variation from retraining, and cannot account for unrecorded patient overlap.
- CHAKSU and PAPILA are reported separately and never pooled.
- Saved evidence for the recorded run is in `docs/image_model_evidence/`.

## Results
The recorded run: ResNet-18, selected epoch 8 of 14 (early stopping), threshold 0.0078666. Intervals in brackets.

| Metric | CHAKSU validation | CHAKSU test | PAPILA (external) |
|---|---:|---:|---:|
| Images / positives | 200 / 28 | 335 / 51 | 420 / 87 |
| AUROC | 0.839 | 0.792 (0.722-0.851) | 0.736 (0.664-0.804) |
| Average precision | 0.487 | 0.408 (0.291-0.539) | 0.435 (0.313-0.564) |
| Sensitivity | 92.9% | 86.3% (75.7-95.1) | 59.8% (47.6-71.1) |
| Specificity | 54.7% | 55.6% (49.0-61.9) | 75.7% (70.3-80.8) |
| Balanced accuracy | 73.8% | 71.0% (64.7-76.4) | 67.7% (61.5-74.1) |
| Brier score | 0.103 | 0.123 (0.094-0.153) | 0.194 (0.144-0.245) |

Sources: `validation_metrics.csv` and `final_evaluation/*_confidence_intervals.csv` in `docs/image_model_evidence/`. Validation estimates chose the epoch and threshold, so they are not independent evidence, and no validation intervals were saved.

- **A useful ranking signal that does not transfer fully.** AUROC fell from 0.839 (validation) to 0.792 (CHAKSU test) and 0.736 (PAPILA).
- **The sensitivity target did not transfer.** The threshold found 44 of 51 CHAKSU test positives but only 52 of 87 PAPILA positives (35 missed). At the default threshold 0.5, PAPILA had no positive predictions at all.
- **Many false alarms.** At the screening threshold, 78 of the 172 validation negatives were flagged (specificity 54.7%).
- **Overfitting.** Training loss kept falling (0.143 at epoch 8, 0.015 at epoch 14) while validation loss rose (0.375, then 0.565); validation AUROC peaked at epoch 8 (0.839, against 0.801 at epoch 14). See `learning_curves.png`.
- **Poorly calibrated.** Validation Brier 0.103, ECE 0.083; the mean score (7.0%) is about half the observed prevalence (14%).
- **Cameras (validation).** Bosch AUROC 0.983 (4 positives), Remidio 0.849 (21), Forus 0.548 (3). Bosch and Forus have too few positives for conclusions.
- **Differences in dataset, prevalence and label definition** prevent attributing the PAPILA drop to camera shift alone.

## Limitations
- Incomplete patient linkage: unrecognised repeat CHAKSU patients could cross splits.
- Label shift: CHAKSU expert-majority suspicion versus PAPILA clinical diagnosis.
- Small camera subgroups, one seed and one selected model; no measure of training variability.
- No prospective, clinician-reader or workflow validation.
- The completed baseline must not be tuned against the test results now seen; later changes are separate experiments, with test reuse disclosed.

## Reproduction
```bash
python -m glaucoma_project.image_model.develop                 # audit, train, validation report -> runs/image_model/<run_name>
python -m glaucoma_project.image_model.final --run-dir runs/image_model/resnet18 --check-only
python -m glaucoma_project.image_model.final --run-dir runs/image_model/resnet18   # freeze + one-shot test evaluation
```
Each run needs a new `run_name` in the config. On a GPU cluster set `CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic kernels. Exact replication across devices and library versions is not guaranteed.

The recorded run was trained on a GPU (RTX 5000 Ada, torch 2.9.1+cu126, Slurm jobs 219631 and 219927; `environment.json`) with the original standalone notebook implementation, which has since been ported into `src/`. On synthetic data with fixed seeds, the port reproduced the original's training history, checkpoint weights, predictions and metrics exactly on CPU, and on the real prepared data its audit and effective split are identical. The recorded checkpoint is kept in approved team storage (not in Git); it was not re-evaluated with the port.

Do not commit prepared images, checkpoints, image-level predictions or figures that contain fundus photographs.

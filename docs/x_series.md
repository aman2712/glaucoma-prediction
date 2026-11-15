# The X-series: seven attempts to improve cross-camera glaucoma prediction

## Question
The cup-to-disc ratios (vertical, horizontal, area) measured from predicted disc and cup masks rank glaucoma well on CHAKSU (AUC 0.943) but poorly on PAPILA, a different camera and hospital (AUC 0.731). A photograph-only network reaches 0.81 and 0.76. The X-series tried seven principled changes, chosen from the cross-dataset glaucoma literature, to close that gap. The design, the rules and each experiment's settings were written in [`experiment_log.md`](experiment_log.md) ("X-series design" and the clarifications) before the experiment ran.

## Rules, in short
- One attempt per experiment, in order X1 to X7. Every result is logged, including nulls.
- **Development gate.** An image or segmentation variant first had to beat its baseline on Bosch and Forus development images (198 images, 35 glaucoma) after training on Remidio only. This stands in for "a new camera". A variant that failed the gate was never scored on a test set.
- A variant that passed was frozen (code version and file hashes) and scored once on CHAKSU test and PAPILA, separately. PAPILA labels were never used for tuning.
- An improvement is claimed only if the 95% interval of a paired bootstrap difference (2,000 resamples over patient groups) excludes 0.
- Both test sets had been used by earlier experiments and the series makes 8 primary test comparisons (X1, X5, X6, X7, two test sets each), so every win is exploratory. Only independent data would confirm it.

## Results

| | What it changed | Gate (development) | CHAKSU test | PAPILA | Verdict |
|---|---|---|---|---|---|
| X1 | z-score the ratios within each dataset | passed (balanced accuracy 0.882 against 0.860) | AUC 0.945, unchanged | balanced accuracy 0.649 against 0.702 (-0.053, -0.111 to -0.001); AUC +0.005 (-0.002 to +0.013) | Rejected: it moves the operating point the wrong way, AUC cannot change |
| X2 | photograph cropped around the predicted disc | AUC 0.804 against 0.805 | not scored | not scored | Dropped at the gate |
| X3 | stronger colour augmentation and per-image normalisation | AUC 0.776 against 0.805 | not scored | not scored | Dropped at the gate |
| X4 | DINOv2 backbone instead of ResNet-18 | AUC 0.680 against 0.805 | not scored | not scored | Dropped at the gate |
| **X5** | **segmentation network trained with scale and colour augmentation, plus batch-norm adaptation** | Dice 0.936 against 0.932 | AUC 0.932 against 0.943 (-0.011, -0.040 to +0.015) | **AUC 0.823 against 0.731 (+0.092, +0.036 to +0.151)** | **Improvement on PAPILA (exploratory)** |
| X6 | one network that segments and classifies | AUC 0.831 against 0.805 | AUC 0.854 against 0.813 (+0.040, -0.009 to +0.090) | AUC 0.653 against 0.764 (-0.111, -0.181 to -0.039) | Rejected: worse on PAPILA |
| X7 | P01b blend using the X5 ratios | not applicable | AUC 0.929 against 0.932 for X5 alone (-0.003, -0.012 to +0.005) | AUC 0.825 against 0.823 (+0.002, -0.006 to +0.011) | No improvement over the X5 ratios alone |

Evidence tables for every row are in [`x_evidence/`](x_evidence/), and the notebook `notebooks/07_x_series.ipynb` shows them. Each experiment's frozen record is `frozen_x{n}.json`.

## What the results say
- **The weak link was the segmentation step.** The one change that helped was making the disc and cup network robust to a new camera. On PAPILA its disc Dice rose from 0.800 to 0.926, its cup Dice from 0.458 to 0.535, and the vertical CDR error fell from 0.215 to 0.172. The classifier on top was unchanged.
- **The augmentation did the work.** Batch-norm adaptation to the new photographs made no measurable difference (PAPILA disc Dice 0.926 with it, 0.923 without; the development gate agreed). X5 is domain generalisation by augmentation, not domain adaptation.
- **Better inputs to the photograph model did not transfer.** Cropping, stronger colour augmentation and a foundation-model backbone did not help on development cameras.
- **A gate pass is not a result.** X6 passed the gate and was clearly worse on PAPILA. The gate has 35 positives and CHAKSU's label convention, so it can rule a variant out but cannot promise a gain.
- **The image score adds nothing once the ratios are good.** The P01b blend equalled the old ratios alone, and the X7 blend equals the X5 ratios alone.
- **The operating point is still wrong for PAPILA.** At the threshold chosen on development data, X5 flags 51% of glaucoma eyes with 98% specificity. Ranking improved but the cutoff does not transfer.

## Limits
- One trained segmentation network for X5. The intervals cover test-image sampling, not training variability.
- Both test sets were used before. The PAPILA gain needs independent data.
- X5's PAPILA AUC (0.823) is above the expert-mask reference (0.799). This is not explained and should not be read as better than experts.
- The X7 blend uses P01b's image scores. A cross-fitted X6 image score was not made, and X6 did not improve the image model anyway.
- X5's batch-norm adaptation used unlabelled PAPILA photographs, which is label-free adaptation, though the gain does not depend on it.
- Comparisons with published PAPILA numbers are not like for like (different label handling and training data), so no claim to beat them is made.

## Disclosed deviations
- X2: six spurious tiny disc boxes in PAPILA were replaced by whole images before any X2 training (experiment log, X2 clarification).
- X7: the ratio job first crashed while writing its hash file (a relative manifest path), after writing the ratios. The code was fixed and the ratios regenerated; the two ratio files are byte-identical. No label or test score was involved.
- X6: the plan said to agree it with Indira first. It was run on the project owner's instruction. Indira's branch uses its own protocol, so X6 does not replace her work.

## Reproduce
Code: `data/segmentation.py` (augmentation), `segmentation/adapt.py`, `segmentation/robust_study.py` and `robust_final.py` (X5), `segmentation/multitask.py` and `combined_model/multitask_study.py` (X6), `combined_model/ratio_blend_study.py` (X7), `combined_model/variant_study.py` (X2 to X4), `classification/ratio_study.py` (X1). Job scripts and the order of commands are in [`hpc/README.md`](../hpc/README.md). All tests run without patient data.

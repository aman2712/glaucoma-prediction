# Combined model (P01): photograph plus measurements

## Question
Does giving a CNN the three automatically measured cup-to-disc ratios in addition to the photograph improve glaucoma prediction over the photograph alone? Design and decision rule: `docs/experiment_log.md` ("P01 design"), fixed before any run.

## Method
- **Arms.** *Image-only*: ResNet-18, ImageNet-pretrained, the Baseline B recipe (`configs/p01_image_only.yaml`). *Combined*: the same network and recipe; the 512 pooled image features are joined to the three ratios (vertical, horizontal, area CDR, standardised with training statistics) before the single-logit layer (`configs/p01_combined.yaml`). Nothing else differs.
- **Ratios** come from the frozen Baseline A U-Net applied to every classification image, never from expert masks. They agree exactly with Baseline A's own predicted measurements on the 1,730 images they share. 30 PAPILA images have no detected disc and get the training median.
- **Seeds.** 5 per arm (2440-2444), 10 runs in all. An arm's score is the mean logit over its seeds.
- **Data.** The common split (`splits/`). Training 808, validation 200 (epoch selection only), CHAKSU test 335 and PAPILA 420, each scored once after freezing.
- **Comparison.** AUC difference, combined minus image-only, paired bootstrap over `analysis_group`s (2,000 resamples). An improvement is claimed only if the 95% interval excludes 0.
- **Compute.** University HPC lab, one RTX 5000 Ada per run, torch 2.11.0+cu128. Code, hashes and library versions are recorded (`docs/frozen_p01.json`, each `run.json`).

## Results
Test AUC of the seed-averaged scores, with 95% intervals (patient-group bootstrap).

| | CHAKSU test (335 images, 51 positive) | PAPILA (420, 87 positive) |
|---|---:|---:|
| Image-only | 0.813 (0.745-0.874) | 0.764 (0.695-0.828) |
| Combined | 0.816 (0.749-0.876) | 0.770 (0.703-0.833) |
| **Combined minus image-only** | **+0.003 (-0.005 to +0.011)** | **+0.006 (-0.002 to +0.015)** |
| Measurements only (Baseline A, same images) | 0.943 | 0.731 |
| Combined minus measurements-only | -0.129 (-0.197 to -0.059) | +0.039 (-0.033 to +0.111) |

Source: `docs/p01_evidence/` (`primary.csv`, `secondary.csv`, `per_seed.csv`, `arm_metrics.csv`, `no_disc.csv`, `development_runs.csv`). The notebook `notebooks/05_combined_model.ipynb` reads these tables, shows the frozen record and draws the figures; it needs no patient data.

- **The pre-registered question is answered "not shown".** Both intervals include 0. The combined arm is ahead in 4 of 5 seeds on each test set, but the spread across seeds (up to 0.02) is larger than the average difference.
- **On CHAKSU the photograph adds nothing to the measurements, and costs a lot.** Three ratios in a logistic regression (0.943) beat both CNN arms (0.81) by about 0.13, a difference whose interval excludes 0. CHAKSU labels follow cup size closely, so this favours the ratios.
- **On PAPILA the CNN arms are somewhat ahead of the ratios (+0.03 to +0.04), but not beyond sampling noise.**
- **The network barely uses the ratios.** Both arms drive the training loss to near 0 (about 0.01) from the photograph alone, so the extra inputs add little to what it learns. This is an explanation consistent with the numbers, not something this experiment tested.
- **Threshold metrics** (descriptive, threshold from the seed-averaged validation scores): PAPILA sensitivity 0.39 (image-only) and 0.46 (combined), specificity 0.91 and 0.89, with overlapping intervals. They do not transfer between the datasets, as in Baseline A.
- **Images with no detected disc** (30 PAPILA images, 9 positive): 0.67 (image-only) and 0.65 (combined), too few to conclude anything.
- **Image-only reproduces Baseline B.** Seed 2440 reaches validation AUROC 0.838 against 0.839 earlier. Averaging five seeds lifts the test AUC above the single model's 0.792 and 0.736.

## Limitations
- This tests one way of combining the inputs: joined at the last layer of a network trained end to end. A null result is "not shown for this design", not "measurements cannot help". The two-step blend (P01b, below) does not let the network fit the labels first.
- CHAKSU labels follow cup-to-disc ratio, so CHAKSU is not a fair test of whether measurements add information; PAPILA is the more informative test, with 87 positives and intervals of about 0.13 width.
- The ratios of training images come from images the U-Net learned from, so they are slightly cleaner than on new images.
- Intervals describe sampling of test images; the seed spread is reported separately.
- A first scoring attempt failed on a software bug before any metric was computed; the runs were frozen again after the fix (experiment log, E13).
- Later work (P01b) extended shared modules (`image_model/train.py`, `runtime.py`), so the code check of `docs/frozen_p01.json` (`freeze.verify`) passes only at its frozen commit `02becaa`. The hashes of the checkpoints and inputs (`freeze.verify_files`) still pass and are what the results rely on.

## Two-step blend (P01b)
Design fixed before any model was trained (`docs/experiment_log.md`, "P01b design"; result E14). Five grouped folds of the 1,008 development images; for each of 5 seeds, 5 ResNet-18 networks (10 epochs, the P01 image-only recipe) each score the fold they did not train on. The out-of-fold image score is the mean logit over the seeds. A logistic regression (the Baseline A routine, C = 1.0 by grouped cross-validation) then combines that score with the three ratios. Test images get the mean logit of all 25 networks. Frozen at commit 31fd3c8 (`docs/frozen_p01b.json`), test sets scored once.

| | CHAKSU test (335 images, 51 positive) | PAPILA (420, 87 positive) |
|---|---:|---:|
| Image score (25 networks) | 0.837 | 0.779 |
| Blend (image score + ratios) | 0.945 | 0.749 |
| **Blend minus image score** | **+0.108 (+0.040 to +0.175)** | **-0.030 (-0.101 to +0.040)** |
| Blend minus measurements-only (Baseline A) | +0.001 (-0.004 to +0.007) | +0.014 (+0.004 to +0.024) |

Source: `docs/p01b_evidence/`; notebook `notebooks/06_two_step_blend.ipynb`.

- **CHAKSU: improvement over the image score, but it is the ratios.** The blend (0.945) equals the three ratios alone (0.943), and CHAKSU labels follow cup size closely.
- **PAPILA: improvement not shown.** The blend is not better than the image score on a new camera and hospital (-0.030, interval includes 0).
- **The image adds little on top of the ratios.** Blend minus measurements-only is +0.001 on CHAKSU and +0.014 on PAPILA (interval just above 0, a small gain).
- **Development diagnostic** (before freezing): image score out-of-fold AUC 0.854, blend cross-validated AUC 0.957.
- **Caveats.** The test sets were used by earlier experiments, so this is exploratory. Development image scores come from single networks and test scores average 25, so their spread differs (stated in the design). A different image score, such as one that adapts to a new camera, is not tested here.

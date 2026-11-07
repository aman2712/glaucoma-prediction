# Anatomy-informed glaucoma prescreening from fundus photographs

Does adding automatically estimated optic-disc and optic-cup measurements improve glaucoma
prediction compared with using the photograph alone? ENG 2440 capstone project of Indira
Duisembayeva, Md Aman Khan and Yomna Elnikety. This is a research prototype, not a diagnostic system.

Data: **CHAKSU** for development and in-domain testing, **PAPILA** as a fully external test
(not committed to this repository, see [`data_pipeline/README.md`](data_pipeline/README.md)).

## Models

| Model | Sees | Description | Code |
|---|---|---|---|
| Baseline A | measurements from predicted disc/cup masks | [`docs/baseline_a.md`](docs/baseline_a.md) | `segmentation/`, `measurements/`, `classification/` |
| Baseline B | the photograph only | [`docs/image_model.md`](docs/image_model.md) | `image_model/` |
| Combined | both | [`docs/combined_model.md`](docs/combined_model.md) | `combined_model/`, `hpc/` |

## Layout

```
data_pipeline/   data preparation: inventory, masks, frozen splits (own README)
src/glaucoma_project/
  data/          audit of the prepared export, common split, datasets and loaders
  segmentation/  disc/cup U-Net
  measurements/  masks -> cup-to-disc ratios
  classification/  logistic regression on measurements
  image_model/   image-only CNN, optionally with measurements: train, freeze, final evaluation
  combined_model/  P01: image-only vs combined, several seeds: freeze, score, analyse
  evaluation/    metrics, bootstrap, paired comparison
configs/         settings of each model (YAML)
hpc/             job scripts and notes for the university HPC lab
tests/           unit tests (no patient data needed)
notebooks/       one driver notebook per model, plus data exploration
docs/            protocols, results, and saved evidence
runs/ data/      local only (git-ignored): outputs and datasets
```

## Reproduce the main results

Needs Python 3.11 or newer. Step 2 needs the datasets, which are not in the repository.

1. **Environment and tests** (about 90 tests, none needs patient data):
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt && pip install -e .
   pytest
   ```
2. **Data.** Download CHAKSU and PAPILA and run the preparation steps in [`data_pipeline/README.md`](data_pipeline/README.md); the models read the result from `runs/prep`. Then check that your copy matches the split of record in `splits/`:
   ```bash
   python -m glaucoma_project.data.common_split --check
   ```
3. **Baseline A** (measurements only). Train the U-Net, then run `notebooks/03_baseline_a.ipynb`, which freezes the pipeline and evaluates the test sets once:
   ```bash
   python -m glaucoma_project.segmentation.train --config configs/segmentation.yaml
   ```
4. **Baseline B** (image only):
   ```bash
   python -m glaucoma_project.image_model.develop
   python -m glaucoma_project.image_model.final --run-dir runs/image_model/resnet18   # freeze + one-shot test evaluation
   ```
5. **Combined study (P01)**: ratios for every image, 10 trainings (2 arms x 5 seeds), freeze, one-shot scoring, analysis. The commands, including the job script for the university GPU cluster, are in [`hpc/README.md`](hpc/README.md) the design is in [`docs/experiment_log.md`](docs/experiment_log.md), and `notebooks/05_combined_model.ipynb` shows the results from the saved tables.

Every run writes its configuration, git commit and metrics to its own folder under `runs/`. Test sets are evaluated once, after the model and threshold are frozen; the code refuses a second evaluation.

| Result to compare | CHAKSU test AUC | PAPILA AUC | Where |
|---|---:|---:|---|
| Baseline A | 0.943 | 0.731 | [`docs/baseline_a.md`](docs/baseline_a.md) |
| Baseline B | 0.792 | 0.736 | [`docs/image_model.md`](docs/image_model.md) |
| Combined minus image-only | +0.003 | +0.006 | [`docs/combined_model.md`](docs/combined_model.md) |

Retraining on other hardware gives slightly different numbers. On the cluster GPU the image-only baseline (seed 2440) reproduced a validation AUC of 0.838 against 0.839 on the first run.

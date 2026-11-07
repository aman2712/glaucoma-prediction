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
| Combined | both | planned | |

## Layout

```
data_pipeline/   data preparation: inventory, masks, frozen splits (own README)
src/glaucoma_project/
  data/          audit of the prepared export, effective split, datasets and loaders
  segmentation/  disc/cup U-Net
  measurements/  masks -> cup-to-disc ratios
  classification/  logistic regression on measurements
  image_model/   image-only CNN: train, freeze, final evaluation
  evaluation/    metrics, bootstrap, paired comparison
configs/         settings of each model (YAML)
tests/           unit tests (no patient data needed)
notebooks/       one driver notebook per model, plus data exploration
docs/            protocols, results, and saved evidence
runs/ data/      local only (git-ignored): outputs and datasets
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

Prepare the data first (see [`data_pipeline/README.md`](data_pipeline/README.md)); the models read it from `runs/prep`.

## Run

```bash
python -m glaucoma_project.segmentation.train          # Baseline A: U-Net, then notebooks/03_baseline_a.ipynb
python -m glaucoma_project.image_model.develop         # Baseline B: audit, train, validation report
python -m glaucoma_project.image_model.final --run-dir runs/image_model/resnet18   # freeze + one-shot test evaluation
pytest                                                  # all unit tests
```

Every run writes its configuration, git commit and metrics to its own folder under `runs/`.
Test sets are evaluated once, after the model and threshold are frozen.

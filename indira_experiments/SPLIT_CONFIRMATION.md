# E0 versus E2: grouped split confirmation

Started 15 November 2026 after E2 was selected from E0–E5 on the original validation cohort. This is a development sensitivity analysis, not independent confirmation of the model-selection result. Previously inspected CHAKSU test and PAPILA images are not opened.

## Frozen design

- Pool only the original eligible CHAKSU train and validation partitions: 1,008 images and 137 positive labels. Retain original partition metadata in the study archive.
- Five outer StratifiedGroupKFold splits, grouping by analysis_group, seed 73115. All images in a known group stay together. Every eligible development image is held out once.
- Within each outer training pool, select the first split of five grouped stratified inner folds, seed 83115 + outer fold index. This inner selection subset chooses checkpoint epoch and threshold. It never supplies training gradients.
- E0 and E2 use the identical assignments, seeds 2440/2441/2442, and their existing training recipes. Thirty fits total. Do not choose a winning seed, adapt hyperparameters, or reroll folds after scores are observed.
- Restore the inner-selected checkpoint and evaluate outer-held images only after selection. The outer images cannot affect early stopping or threshold choice.
- Primary: mean outer-fold AUROC difference, E2 minus E0, with a mean-logit three-seed ensemble within each fold. Report the per-seed and per-fold metrics too. This estimand differs from the original three-seed mean on one validation split.
- Prespecified positive-evidence rule: primary difference >0 and the 95% paired analysis-group bootstrap interval entirely above zero. Run 2,000 within-fold group resamples. These intervals condition on fitted models; they do not measure uncertainty from repeating the entire training and selection process.
- No statistical or clinical claim of independent superiority follows from development CV after architecture selection. Unknown CHAKSU patient linkage and small inner positive counts remain limitations.

## Frozen role counts

| Outer fold | Fit images / positives | Inner selection images / positives | Outer-held images / positives |
|---|---:|---:|---:|
| 0 | 647 / 93 | 160 / 14 | 201 / 30 |
| 1 | 647 / 85 | 160 / 24 | 201 / 28 |
| 2 | 645 / 89 | 161 / 22 | 202 / 26 |
| 3 | 646 / 93 | 161 / 22 | 201 / 22 |
| 4 | 643 / 86 | 162 / 20 | 203 / 31 |

The split algorithm approximates class balance under group constraints; it does not promise identical prevalence. These assignments were frozen before training and are not changed to improve scores.

## Execution

The study is stored on HPC under `indira_experiments/runs/split_confirmation_20261115/`. It contains protocol.json, assignments.csv, development.csv and fold_counts.csv. The protocol binds assignment, development-table and training-source hashes; source edits invalidate it.

Task mapping: `fold = task // 6`, arm E0 for offsets 0–2 and E2 for offsets 3–5, seeds 2440–2442. Pilot task 0 was submitted first to verify training plus outer evaluation. Tasks 1–29 are submitted after pilot success, with bounded concurrency. Do not submit task 0 again.

```bash
source .venv-indira/bin/activate
squeue -u "$USER"
```

After all 30 tasks report outer completion:

```bash
python indira_experiments/split_confirmation.py summarize \
  --study indira_experiments/runs/split_confirmation_20261115
```

This produces `comparison/per_seed_fold_metrics.csv`, `ensemble_fold_metrics.csv` and `primary.json`. It refuses incomplete studies and existing comparison output. For each training task, `status.json` records inner training completion only; `outer_completion.json` is required to establish the task completed the outer evaluation. Preserve failed runs and inspect logs before retrying; do not delete completed evidence.

All image-level artifacts/checkpoints remain in ignored run storage. Publish only aggregate tables and documented findings on the experiment branch. No final-test scoring is part of this workflow.

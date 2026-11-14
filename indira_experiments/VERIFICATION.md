# Software verification — 14 November 2026

Environment: existing local Python 3.13.5, torch 2.9.1, torchvision 0.24.1. This is not a clean HPC reproduction.

- Ten artificial-data checks cover all six architecture/configuration paths, paired initialization, batch-normalization freezing, partial fine-tuning, mask geometry, missing-target loss, rejection of test datasets, manifest tampering, path containment, completed-run summaries, duplicate seed rejection, and saved-checkpoint inference.
- Two short artificial runs exercise head warm-up and fine-tuning, checkpoint selection/reloading, metrics, logs, plots, and anatomy MAE. Random initialization/pixels are explicitly marked synthetic and excluded from real experiment tables.
- Current common manifest SHA-256 verified; development count is 808 train and 200 validation.
- Old and current manifests agree on all 1,833 image identities, split assignments, analysis groups, binary labels, eligibility, image/mask paths, cameras and patient IDs. Original pixel-hash metadata differs; the experiment package uses the current common manifest and records actual prepared-image content hashes.
- Slurm script syntax checked with `bash -n`.
- The real local Mac training-image copy contains 808 zero-byte image files. No real-data training has been performed locally; use the intact HPC prepared export and preflight it before submitting.

No model improvement, clinical validity, or independent held-out confirmation is claimed by these software checks.

# AI-use and contribution log

## 2026-11-14 — experiment framework preparation

- Purpose: implement a development-only controlled study of image classification, anatomy supervision, predicted-ratio fusion and photometric augmentation.
- AI assistance: generated initial model/runner, immutable split audit, experiment plan, Slurm array, aggregate reporting tool, software checks and documentation.
- Human responsibilities: review code and medical assumptions, verify prepared masks/labels, submit jobs within HPC policy, inspect failures, perform/obtain clinical error review, and interpret actual results. The student must understand the auxiliary loss, validation selection, missing-target handling and inference pathway for the viva.
- Evidence status: local checks use artificial pixels/masks. Real glaucoma experiments and performance improvements are not claimed until actual HPC runs complete.
- Data limitations retained: incomplete patient linkage, small validation sample, prior held-out test exposure and uncalibrated risk scores.
- Freeze this plan before running the new study. Log subsequent config changes, rationale, failed runs and test exposure rather than overwriting history.

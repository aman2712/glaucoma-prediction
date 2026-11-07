# Running on the HPC lab

How the P01 experiment is run on the MBZUAI lab cluster. The rules of the lab apply: compute only inside
an allocation (`salloc` or `sbatch`), never on the login node; no sudo; release the allocation and log out
when finished.

## One-time setup
1. Log in: `ssh <user>@login-student-lab.mbzu.ae`, then start `tmux` so the session survives a disconnect.
2. Get an allocation: `salloc -N1 -n12 --mem=24G` (add the GPU option the lab documents; `sinfo` lists what exists).
3. Clone the repository into the work directory (`/l/users/<user>/`, 3 TB; home is 1 TB) and check out the branch.
4. `bash hpc/setup_env.sh` creates the `glaucoma` conda environment and prints whether the GPU is visible.
5. Copy the prepared data and the inputs from your computer, then check the copy:
   `rsync -av runs/prep/ <user>@login-student-lab.mbzu.ae:<repo>/runs/prep/`, the same for `runs/p01/measurements.*`,
   `runs/final/common_split/scores.csv`, then `python -m glaucoma_project.data.common_split --check`
   and compare the SHA-256 of `runs/p01/measurements.csv` with `runs/p01/measurements.json`.

## P01 runs
1. Smoke test first, in the allocation: `python -m glaucoma_project.image_model.develop --config configs/p01_image_only.yaml --seed 2440`
   with `--runs-dir runs/p01_smoke` instead, and note the time per epoch to confirm the job time limit in
   `hpc/p01_train.sbatch`. Delete the smoke folder afterwards; it is not part of the study.
2. `sbatch hpc/p01_train.sbatch` runs all 10 (arm, seed) combinations; `squeue -u $USER` shows progress.
3. Freeze, then score the test sets once: `python -m glaucoma_project.combined_model.study freeze`,
   then `... study score`, then `python -m glaucoma_project.combined_model.analysis --scores runs/p01/final --out runs/p01/analysis`.
   Commit `docs/frozen_p01.json`; it records the commit, the checkpoint hashes and the inputs.

All 10 runs use the same GPU type. Each run folder records its commit, library versions and GPU in `run.json`.

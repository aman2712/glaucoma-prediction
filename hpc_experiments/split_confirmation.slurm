#!/bin/bash
#SBATCH --job-name=indira-cv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/indira_cv_%A_%a.out
#SBATCH --error=logs/indira_cv_%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv-indira/bin/activate
export MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python -u hpc_experiments/split_confirmation.py run \
  --study hpc_experiments/runs/split_confirmation_20261115 \
  --task "$SLURM_ARRAY_TASK_ID" --artifacts artifacts

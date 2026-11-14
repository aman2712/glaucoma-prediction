#!/bin/bash
#SBATCH --job-name=indira-models
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/indira_%A_%a.out
#SBATCH --error=logs/indira_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv-indira/bin/activate
export MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# The first three IDs are the main controlled study. Others are optional.
experiments=(E0 E3 E4 E1 E2 E5)
seeds=(2440 2441 2442)
task="${SLURM_ARRAY_TASK_ID:?Submit with --array}"
if ((task < 0 || task >= 18)); then
  echo 'Array index must be between 0 and 17' >&2
  exit 2
fi
experiment="${experiments[$((task / 3))]}"
seed="${seeds[$((task % 3))]}"

python -u hpc_experiments/train.py \
  --experiment "$experiment" --seed "$seed" \
  --manifest splits/all.csv \
  --artifacts artifacts \
  --output hpc_experiments/runs \
  --device cuda

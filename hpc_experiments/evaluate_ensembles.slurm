#!/bin/bash
#SBATCH --job-name=efficientnet-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/efficientnet_test_%j.out
#SBATCH --error=logs/efficientnet_test_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv-indira/bin/activate
export MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python -u hpc_experiments/evaluate_ensembles.py evaluate \
  --out hpc_experiments/runs/efficientnet_heldout_20261115 --artifacts artifacts

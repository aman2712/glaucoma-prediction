#!/bin/bash
# P01b training: one array task per seed (5 in all); each trains the 5 fold networks of its seed.
# Submit from the repository root:   sbatch hpc/p01b_train.sbatch
#SBATCH --job-name=p01b
#SBATCH --array=0-4%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=runs/p01b/slurm/%x_%A_%a.out
set -euo pipefail
mkdir -p runs/p01b/slurm
source /apps/local/anaconda3/conda_init.sh
conda activate glaucoma
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # reproducible GPU matrix kernels

SEED=$(python -m glaucoma_project.combined_model.blend_study job "$SLURM_ARRAY_TASK_ID")
echo "commit $(git rev-parse HEAD) | seed $SEED | $(hostname)"
python -m glaucoma_project.combined_model.blend_study train --seed "$SEED"

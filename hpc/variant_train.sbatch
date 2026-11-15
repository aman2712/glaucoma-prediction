#!/bin/bash
# Full trainings of an image-variant study (5 seeds), after the gate passed. Submit from the repository root:
#   sbatch --export=ALL,STUDY=configs/x2.yaml hpc/variant_train.sbatch
#SBATCH --job-name=variant
#SBATCH --array=0-4%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
: "${STUDY:?set STUDY to the study config, for example configs/x2.yaml}"
mkdir -p runs/slurm
source /apps/local/anaconda3/conda_init.sh
conda activate glaucoma
export CUBLAS_WORKSPACE_CONFIG=:4096:8

SEED=$(python -m glaucoma_project.combined_model.variant_study "$STUDY" job "$SLURM_ARRAY_TASK_ID")
echo "commit $(git rev-parse HEAD) | $STUDY | seed $SEED | $(hostname)"
python -m glaucoma_project.combined_model.variant_study "$STUDY" train --seed "$SEED"

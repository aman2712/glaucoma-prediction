#!/bin/bash
# Development gate of a segmentation study (X5): 6 trainings (variant and baseline, 3 seeds each) on the
# Remidio camera only. Submit from the repository root with the study config:
#   sbatch --export=ALL,STUDY=configs/x5.yaml hpc/segmentation_gate.sbatch
#SBATCH --job-name=seg-gate
#SBATCH --array=0-5%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
: "${STUDY:?set STUDY to the study config, for example configs/x5.yaml}"
mkdir -p runs/slurm
source /apps/local/anaconda3/conda_init.sh
conda activate glaucoma
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # reproducible GPU matrix kernels

read -r ARM SEED < <(python -m glaucoma_project.segmentation.robust_study "$STUDY" gate-job "$SLURM_ARRAY_TASK_ID")
echo "commit $(git rev-parse HEAD) | $STUDY | gate $ARM seed $SEED | $(hostname)"
python -m glaucoma_project.segmentation.robust_study "$STUDY" gate-run "$ARM" --seed "$SEED"

#!/bin/bash
# X5 final stage, after the gate passed: train the robust U-Net on CHAKSU train, then fit the classifier.
#   sbatch --export=ALL,STUDY=configs/x5.yaml hpc/segmentation_final.sbatch
#SBATCH --job-name=seg-final
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm/%x_%j.out
set -euo pipefail
: "${STUDY:?set STUDY to the study config, for example configs/x5.yaml}"
mkdir -p runs/slurm
source /apps/local/anaconda3/conda_init.sh
conda activate glaucoma
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "commit $(git rev-parse HEAD) | $STUDY | $(hostname)"
python -m glaucoma_project.segmentation.robust_final "$STUDY" train
python -m glaucoma_project.segmentation.robust_final "$STUDY" fit-classifier

#!/bin/bash
# Development gate of X6: 3 trainings (seeds 2440 to 2442) of the variant on the Remidio camera only. The baseline arm is the P01 image-only recipe; its gate scores are copied from runs/x2/gate.
#   sbatch --export=ALL,STUDY=configs/x6.yaml hpc/multitask_gate.sbatch
#SBATCH --job-name=mt-gate
#SBATCH --array=0-2%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
: "${STUDY:?set STUDY to the study config, for example configs/x6.yaml}"
mkdir -p runs/slurm
source /apps/local/anaconda3/conda_init.sh
conda activate glaucoma
export CUBLAS_WORKSPACE_CONFIG=:4096:8

read -r ARM SEED < <(python -m glaucoma_project.combined_model.variant_study "$STUDY" gate-job "$SLURM_ARRAY_TASK_ID")
echo "commit $(git rev-parse HEAD) | $STUDY | gate $ARM seed $SEED | $(hostname)"
python -m glaucoma_project.combined_model.multitask_study "$STUDY" gate-run --seed "$SEED"

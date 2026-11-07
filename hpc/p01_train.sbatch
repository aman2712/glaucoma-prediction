#!/bin/bash
# P01 development runs: one array task per (arm, seed), 10 in all. Submit from the repository root:
#   sbatch hpc/p01_train.sbatch
# The %2 keeps at most two tasks running at once, so the shared lab is not monopolised.
#SBATCH --job-name=p01
#SBATCH --array=0-9%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=runs/p01/slurm/%x_%A_%a.out
set -euo pipefail
mkdir -p runs/p01/slurm
source /apps/local/anaconda3/conda_init.sh
conda activate glaucoma
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # reproducible GPU matrix kernels

read -r CONFIG SEED < <(python -m glaucoma_project.combined_model.study job "$SLURM_ARRAY_TASK_ID")
MEASUREMENTS=()
grep -q measurement_columns "$CONFIG" && MEASUREMENTS=(--measurements runs/p01/measurements.csv)
echo "commit $(git rev-parse HEAD) | $CONFIG | seed $SEED | $(hostname)"
python -m glaucoma_project.image_model.develop --config "$CONFIG" --seed "$SEED" \
    --runs-dir runs/p01 "${MEASUREMENTS[@]}"

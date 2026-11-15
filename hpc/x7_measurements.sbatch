#!/bin/bash
# X7 input: the three ratios of every classification image from the frozen X5 network, with batch-norm
# statistics re-estimated on each split. Needs docs/frozen_x5.json and the X5 weights.
#   sbatch hpc/x7_measurements.sbatch
#SBATCH --job-name=x7-ratios
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm/%x_%j.out
set -euo pipefail
mkdir -p runs/slurm
source /apps/local/anaconda3/conda_init.sh
conda activate glaucoma

echo "commit $(git rev-parse HEAD) | $(hostname)"
python -m glaucoma_project.measurements.classification_inputs --manifest docs/frozen_x5.json --adapt --out runs/x5/measurements.csv

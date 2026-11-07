#!/bin/bash
# One-time: create the conda environment for this project on the HPC lab (no sudo needed).
# Run inside an allocation (salloc), never on the login node.
set -euo pipefail
source /apps/local/anaconda3/conda_init.sh
conda create -y -n glaucoma python=3.12
conda activate glaucoma
pip install -r requirements.txt
pip install -e .
python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name() if torch.cuda.is_available() else "none")
PY

#!/bin/bash
# One-time: create the conda environment for this project on the HPC lab (no sudo needed).
# Run inside an allocation (salloc), never on the login node.
set -euo pipefail
source /apps/local/anaconda3/conda_init.sh
conda create -y -n glaucoma python=3.12
conda activate glaucoma
# The lab's GPU driver supports CUDA 12.8, so take the matching PyTorch build (the default one needs 13).
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name() if torch.cuda.is_available() else "none")
PY

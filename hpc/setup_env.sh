#!/bin/bash
# One-time: create the conda environment for this project on the HPC lab (no sudo needed).
# Run inside an allocation (salloc), never on the login node.
set -euo pipefail
source /apps/local/anaconda3/conda_init.sh
conda create -y -n glaucoma python=3.12
conda activate glaucoma
# The lab's GPU driver supports CUDA 12.8 and the newest CUDA 12.8 build of PyTorch is 2.11, so the
# HPC uses torch 2.11.0 / torchvision 0.26.0 instead of the versions in requirements.txt. Every run records
# its library versions, and all P01 runs use this one environment.
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
grep -v -E '^(torch|torchvision)==' requirements.txt > "${TMPDIR:-/tmp}/requirements_no_torch.txt"
pip install -r "${TMPDIR:-/tmp}/requirements_no_torch.txt"
pip install -e .
python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name() if torch.cuda.is_available() else "none")
PY

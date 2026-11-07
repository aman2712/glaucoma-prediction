"""Device choice and random seeding shared by every training script."""
import random
import subprocess

import numpy as np
import torch

from glaucoma_project import REPO_ROOT


def pick_device() -> torch.device:
    """CUDA if available, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch. ``deterministic`` also asks cuDNN and PyTorch for
    reproducible kernels, at some cost in speed (warns when a kernel has no such version)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def git_commit() -> str:
    """Hash of the current commit, recorded with every run."""
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                          text=True).stdout.strip()

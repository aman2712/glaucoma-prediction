"""Re-estimate batch-norm statistics on unlabelled images of a new camera or hospital (AdaBN)."""
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def adapt_batchnorm(model: nn.Module, loader: DataLoader, device: torch.device) -> nn.Module:
    """A copy of ``model`` whose batch-norm running statistics are the cumulative average over one
    pass of ``loader``. Weights are unchanged, and only the photographs (``batch["image"]``) are read.
    The model that was passed in is not modified."""
    adapted = copy.deepcopy(model).to(device).eval()
    norms = [m for m in adapted.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for norm in norms:
        norm.reset_running_stats()
        norm.momentum = None  # cumulative average instead of a moving one
        norm.train()
    seen = 0
    for batch in loader:
        adapted(batch["image"].to(device))
        seen += len(batch["image"])
    if not seen:
        raise ValueError("Cannot adapt batch-norm statistics: the loader gave no images")
    return adapted.eval()

import torch
from torch import nn
from torch.nn import functional as F


class DiceBCELoss(nn.Module):
    """Binary cross-entropy plus soft Dice loss, averaged over disc and cup channels."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        intersection = (prob * target).sum(dim=(2, 3))
        total = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * intersection + self.eps) / (total + self.eps)
        return F.binary_cross_entropy_with_logits(logits, target) + (1 - dice).mean()

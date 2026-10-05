"""Segmentation overlap metrics for binary masks shaped (N, C, H, W)."""
import torch


def dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dice per image and channel, shape (N, C). Two empty masks score 1."""
    pred, target = pred.float(), target.float()
    intersection = (pred * target).sum(dim=(2, 3))
    return (2 * intersection + eps) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps)


def iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Intersection over union per image and channel, shape (N, C)."""
    pred, target = pred.float(), target.float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    return (intersection + eps) / (union + eps)

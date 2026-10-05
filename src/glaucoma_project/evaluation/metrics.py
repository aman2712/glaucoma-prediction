"""Segmentation overlap metrics for binary masks shaped (N, C, H, W), and measurement agreement."""
import pandas as pd
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


def agreement(predicted, expert) -> dict:
    """Mean absolute error and Pearson correlation between two measurement series (NaNs dropped)."""
    pair = pd.concat([predicted, expert], axis=1, keys=["p", "e"]).dropna()
    return {"mae": float((pair.p - pair.e).abs().mean()), "pearson": float(pair.p.corr(pair.e)),
            "n": len(pair)}

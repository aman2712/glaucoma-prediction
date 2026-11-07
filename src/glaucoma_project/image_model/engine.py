"""One pass over a data loader: training or evaluation."""
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

MAX_GRAD_NORM = 5.0


def set_training_mode(model: nn.Module, head_only: bool = False) -> None:
    """Train mode, except that BatchNorm layers always stay in eval mode (keeping the pretrained
    statistics stable on a small, imbalanced dataset) and the whole model does when only the
    head is trained."""
    model.eval() if head_only else model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer=None,
              head_only: bool = False) -> tuple[float, pd.DataFrame]:
    """Mean BCE loss and per-image probabilities over the loader; trains if given an optimizer.

    The loss is unweighted: class imbalance is handled by ranking metrics and the threshold,
    not by reweighting, so probabilities reflect the training prevalence.
    """
    training = optimizer is not None
    if training:
        set_training_mode(model, head_only)
    else:
        model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss, labels, probabilities, image_ids = 0.0, [], [], []
    with torch.set_grad_enabled(training):
        for images, targets, ids in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images).reshape(-1)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
            total_loss += loss.item() * len(targets)
            labels.extend(targets.int().tolist())
            probabilities.extend(torch.sigmoid(logits).detach().cpu().tolist())
            image_ids.extend(ids)
    if not labels:
        raise ValueError("Empty data loader")
    return total_loss / len(labels), pd.DataFrame(
        {"image_id": image_ids, "label": labels, "probability": probabilities})

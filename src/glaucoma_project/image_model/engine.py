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


def _forward(model: nn.Module, inputs, device: torch.device) -> torch.Tensor:
    """Logits for a batch: images alone, or images with measurements when the loader supplies both."""
    if isinstance(inputs, (list, tuple)):
        return model(*(part.to(device) for part in inputs)).reshape(-1)
    return model(inputs.to(device)).reshape(-1)


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer=None,
              head_only: bool = False) -> tuple[float, pd.DataFrame]:
    """Mean BCE loss and per-image logits and probabilities over the loader; trains if given an optimizer.

    The loss is unweighted: class imbalance is handled by ranking metrics and the threshold,
    not by reweighting, so probabilities reflect the training prevalence.
    """
    training = optimizer is not None
    if training:
        set_training_mode(model, head_only)
    else:
        model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss, labels, logit_values, image_ids = 0.0, [], [], []
    with torch.set_grad_enabled(training):
        for inputs, targets, ids in loader:
            targets = targets.to(device)
            logits = _forward(model, inputs, device)
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
            logit_values.extend(logits.detach().cpu().double().tolist())
            image_ids.extend(ids)
    if not labels:
        raise ValueError("Empty data loader")
    logit_values = torch.tensor(logit_values, dtype=torch.float64)
    return total_loss / len(labels), pd.DataFrame(
        {"image_id": image_ids, "label": labels, "probability": torch.sigmoid(logit_values).tolist(),
         "logit": logit_values.tolist()})

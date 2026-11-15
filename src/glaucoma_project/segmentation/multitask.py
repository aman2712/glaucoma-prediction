"""A network that segments the disc and cup and classifies glaucoma from the same photograph (experiment X6)."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader

from glaucoma_project.segmentation.losses import DiceBCELoss
from glaucoma_project.segmentation.model import DiscCupUNet

_segmentation_loss = DiceBCELoss()


def joint_loss(mask_logits: torch.Tensor, glaucoma_logit: torch.Tensor, mask: torch.Tensor,
               label: torch.Tensor) -> torch.Tensor:
    """The Baseline A segmentation loss plus the glaucoma cross-entropy, with equal weights."""
    return _segmentation_loss(mask_logits, mask) + F.binary_cross_entropy_with_logits(glaucoma_logit.reshape(-1), label)


@torch.no_grad()
def predict_logits(model: DiscCupUNet, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    """Glaucoma logit of every image of a loader of photographs, with its label."""
    model.eval()
    frames = []
    for batch in loader:
        _, logit = model(batch["image"].to(device))
        frames.append(pd.DataFrame({"image_id": list(batch["image_id"]), "label": batch["label"].numpy().astype(int),
                                    "logit": logit.reshape(-1).cpu().numpy()}))
    return pd.concat(frames, ignore_index=True)


def fit_multitask(model: DiscCupUNet, train_loader: DataLoader, validation_loader: DataLoader, run_dir: Path,
                  config: dict, device: torch.device) -> pd.DataFrame:
    """Train for ``config["epochs"]`` with the joint loss. ``best.pt`` holds the weights of the epoch with the
    highest validation AUROC of the glaucoma logit (an earlier epoch wins a tie); ``history.csv`` is written
    every epoch. Returns the history."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    history, best = [], -1.0
    for epoch in range(1, config["epochs"] + 1):
        start, losses = time.time(), []
        model.train()
        for batch in train_loader:
            mask_logits, logit = model(batch["image"].to(device))
            loss = joint_loss(mask_logits, logit, batch["mask"].to(device), batch["label"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        scores = predict_logits(model, validation_loader, device)
        auroc = float(roc_auc_score(scores.label, scores.logit))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_auroc": auroc,
                        "seconds": round(time.time() - start, 1)})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if auroc > best:
            best = auroc
            torch.save(model.state_dict(), run_dir / "best.pt")
        print(history[-1], flush=True)
    return pd.DataFrame(history)

"""Train the image-only model: a few head-only warm-up epochs, then fine-tuning of everything."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from glaucoma_project.data.classification import make_loader
from glaucoma_project.image_model.engine import run_epoch
from glaucoma_project.image_model.model import build_model, head_parameters, restore_model
from glaucoma_project.runtime import set_seed

TIE_TOLERANCE = 1e-8


def _optimizer(model: torch.nn.Module, head_only: bool, config: dict) -> torch.optim.AdamW:
    """AdamW over the head, plus the backbone (at a smaller learning rate) once it is unfrozen."""
    head = head_parameters(model)
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    for parameter in backbone:
        parameter.requires_grad_(not head_only)
    groups = [{"params": head, "lr": config["head_lr"]}]
    if not head_only:
        groups.append({"params": backbone, "lr": config["backbone_lr"]})
    return torch.optim.AdamW(groups, weight_decay=config["weight_decay"])


def _improves(auroc: float, loss: float, best: tuple[float, float]) -> bool:
    """Higher validation AUROC wins; a tie is broken by lower validation loss."""
    best_auroc, best_loss = best
    return auroc > best_auroc + TIE_TOLERANCE or (abs(auroc - best_auroc) <= TIE_TOLERANCE and loss < best_loss)


def fit(train_rows: pd.DataFrame, val_rows: pd.DataFrame, root: Path, run_dir: Path, config: dict,
        device: torch.device, measurement_scaler: dict | None = None) -> tuple[torch.nn.Module, dict, pd.DataFrame]:
    """Train and return the best fine-tuned model, its checkpoint dict and the epoch history.

    With ``config["measurement_columns"]`` the rows must already hold the scaled measurements, and
    ``measurement_scaler`` (saved in the checkpoint) records how they were scaled.

    The best epoch is chosen only among fine-tuning epochs, by validation AUROC. Writes
    ``history.csv`` after every epoch and ``best.pt`` whenever the best epoch improves, and
    stops after ``patience`` fine-tuning epochs without improvement.
    """
    if (run_dir / "best.pt").exists():
        raise FileExistsError("A checkpoint already exists; use a new run directory for a new experiment")
    if config["epochs"] <= config["warmup_epochs"]:
        raise ValueError("epochs must exceed warmup_epochs so that the backbone is fine-tuned")
    set_seed(config["seed"], deterministic=True)
    model, weights = build_model(config["model_name"], config["pretrained"],
                                 n_measurements=len(config.get("measurement_columns", [])))
    model.to(device)
    train_loader = make_loader(train_rows, root, config, training=True)
    val_loader = make_loader(val_rows, root, config)
    best, bad_epochs, history = (-np.inf, np.inf), 0, []
    for epoch in range(1, config["epochs"] + 1):
        head_only = epoch <= config["warmup_epochs"]
        if epoch in (1, config["warmup_epochs"] + 1):
            optimizer, bad_epochs = _optimizer(model, head_only, config), 0
        started = time.monotonic()
        train_loss, _ = run_epoch(model, train_loader, device, optimizer, head_only)
        val_loss, predictions = run_epoch(model, val_loader, device)
        auroc = float(roc_auc_score(predictions.label, predictions.probability))
        history.append({"epoch": epoch, "phase": "head" if head_only else "fine_tune",
                        "train_bce": train_loss, "validation_bce": val_loss, "validation_auroc": auroc,
                        "validation_average_precision":
                            float(average_precision_score(predictions.label, predictions.probability)),
                        "seconds": time.monotonic() - started})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(history[-1], flush=True)
        if head_only:
            continue
        if _improves(auroc, val_loss, best):
            best, bad_epochs = (auroc, val_loss), 0
            torch.save({"model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                        "config": dict(config), "weights": weights, "best_epoch": epoch,
                        "measurement_scaler": measurement_scaler,
                        "validation_auroc": auroc, "validation_bce": val_loss}, run_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= config["patience"]:
                print("Early stopping: no validation AUROC improvement", flush=True)
                break
    model, checkpoint = restore_model(run_dir / "best.pt", device)
    return model, checkpoint, pd.DataFrame(history)

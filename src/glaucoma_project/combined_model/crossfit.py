"""Cross-fitted image scores: every development image is scored by networks that never saw it.

The development images are split into folds that keep patient groups together. A network is trained
on all folds but one and scores the fold it did not see, so no image is ever scored by a network
fitted on it. A network that has memorised its training photographs gives scores that are too good
to learn from; these scores are honest.
"""
from pathlib import Path
from typing import Callable

import pandas as pd
import torch

from glaucoma_project.classification.baseline import grouped_folds
from glaucoma_project.image_model.model import restore_model
from glaucoma_project.image_model.predict import predict_rows
from glaucoma_project.image_model.train import train_for_epochs

Trainer = Callable[[pd.DataFrame, Path, dict, torch.device, int], torch.nn.Module]


def fold_assignments(development: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    """The fold (0 to n_folds - 1) of every development image, aligned with ``development``'s index.
    Images of one analysis group share a fold and both classes appear in every fold."""
    folds = pd.Series(-1, index=development.index)
    for fold, (_, held_out) in enumerate(grouped_folds(development, n_folds, seed)):
        folds.iloc[held_out] = fold
    return folds


def training_rows(development: pd.DataFrame, folds: pd.Series, held_out_fold: int) -> pd.DataFrame:
    return development[folds != held_out_fold]


def held_out_rows(development: pd.DataFrame, folds: pd.Series, held_out_fold: int) -> pd.DataFrame:
    return development[folds == held_out_fold]


def save_checkpoint(model: torch.nn.Module, config: dict, path: Path) -> None:
    """Write the weights in the format ``restore_model`` reads."""
    state = {name: tensor.cpu() for name, tensor in model.state_dict().items()}
    torch.save({"model_state": state, "config": dict(config)}, path)


def checkpoint_path(directory: Path, fold: int) -> Path:
    return directory / f"fold{fold}.pt"


def train_fold_models(development: pd.DataFrame, folds: pd.Series, config: dict, root: Path, directory: Path,
                      device: torch.device, epochs: int, trainer: Trainer = train_for_epochs) -> list[Path]:
    """Train one network per fold, on the images of the other folds, and save each checkpoint."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for fold in sorted(folds.unique()):
        model = trainer(training_rows(development, folds, fold), root, config, device, epochs)
        path = checkpoint_path(directory, fold)
        save_checkpoint(model, config, path)
        paths.append(path)
    return paths


def held_out_logits(development: pd.DataFrame, folds: pd.Series, checkpoints: list[Path], config: dict,
                    root: Path, device: torch.device) -> pd.DataFrame:
    """Logit of every development image from the network that did not train on it."""
    scored = []
    for fold, path in zip(sorted(folds.unique()), checkpoints):
        model, _ = restore_model(path, device)
        rows = held_out_rows(development, folds, fold)
        scored.append(predict_rows(model, rows, root, config, device)[["image_id", "label", "analysis_group", "logit"]])
    return pd.concat(scored, ignore_index=True)

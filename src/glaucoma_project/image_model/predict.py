"""Scoring: prediction tables for labelled rows, the operating threshold, and single photographs."""
import json
from pathlib import Path

import pandas as pd
import torch

from glaucoma_project.data.classification import letterbox, make_loader, make_transform
from glaucoma_project.evaluation.screening import select_threshold
from glaucoma_project.image_model.engine import run_epoch
from glaucoma_project.image_model.model import restore_model

METADATA = ["image_id", "binary_label", "dataset", "camera", "patient_id", "analysis_group",
            "classification_image"]


def attach_metadata(predictions: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Join each prediction to its row by image ID and check that every label still agrees."""
    if predictions.image_id.duplicated().any() or set(predictions.image_id) != set(rows.image_id):
        raise ValueError("Prediction identity mismatch or duplication")
    joined = predictions.merge(rows[METADATA], on="image_id", how="left", validate="one_to_one")
    if (joined.label != joined.binary_label.astype(int)).any():
        raise ValueError("Prediction label alignment failed")
    return joined


def predict_rows(model: torch.nn.Module, rows: pd.DataFrame, root: Path, config: dict,
                 device: torch.device) -> pd.DataFrame:
    """Probability for every row, with the row's metadata attached."""
    _, predictions = run_epoch(model, make_loader(rows, root, config), device)
    return attach_metadata(predictions, rows)


def operating_threshold(run_dir: Path) -> float:
    """The decision threshold of a run: chosen on its validation predictions to reach the
    configured sensitivity, with the highest specificity that allows."""
    # round_trip parsing returns exactly the probabilities that were saved
    predictions = pd.read_csv(run_dir / "validation_predictions.csv", float_precision="round_trip")
    target = json.loads((run_dir / "config.json").read_text())["target_sensitivity"]
    return select_threshold(predictions.label, predictions.probability, target)


def score_photo(run_dir: Path, photo: Path, device: torch.device) -> dict:
    """Suspicion score and screening flag for one new photograph of any size.

    Not for held-out images during development. The score is not a calibrated clinical probability.
    """
    model, _ = restore_model(run_dir / "best.pt", device)
    image = make_transform()(letterbox(photo)).unsqueeze(0).to(device)
    with torch.inference_mode():
        score = float(torch.sigmoid(model(image).reshape(-1))[0].cpu())
    threshold = operating_threshold(run_dir)
    return {"glaucoma_suspicion_score": score, "flag_at_threshold": score >= threshold,
            "threshold": threshold, "calibrated_clinical_probability": False}

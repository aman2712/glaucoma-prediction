"""Saved evaluation outputs for one partition of the image-only model."""
from pathlib import Path

import pandas as pd

from glaucoma_project.evaluation.screening import metric_values
from glaucoma_project.image_model.plots import plot_diagnostics, plot_errors

SMALL_CLASS = 10


def evaluation_tables(predictions: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metrics at the default 0.5 and at ``threshold``, and per-camera metrics at ``threshold``.
    A camera is flagged ``small_class_count`` when either class has fewer than 10 images."""
    overall = pd.DataFrame([
        {"operating_point": "default_0.5", **metric_values(predictions.label, predictions.probability, 0.5)},
        {"operating_point": "validation_selected",
         **metric_values(predictions.label, predictions.probability, threshold)}])
    cameras = []
    for camera, group in predictions.groupby("camera"):
        values = metric_values(group.label, group.probability, threshold)
        cameras.append({"camera": camera,
                        "small_class_count": min(values["positives"], values["negatives"]) < SMALL_CLASS,
                        **values})
    return overall, pd.DataFrame(cameras)


def save_report(name: str, predictions: pd.DataFrame, threshold: float, root: Path,
                directory: Path) -> pd.DataFrame:
    """Write predictions, metric tables and figures as ``<name>_*``. Returns the overall metrics."""
    overall, cameras = evaluation_tables(predictions, threshold)
    overall.insert(0, "partition", name)
    predictions.to_csv(directory / f"{name}_predictions.csv", index=False)
    overall.to_csv(directory / f"{name}_metrics.csv", index=False)
    cameras.to_csv(directory / f"{name}_camera_metrics.csv", index=False)
    plot_diagnostics(predictions, threshold, name, directory / f"{name}_curves.png")
    plot_errors(predictions, root, threshold, directory / f"{name}_errors.png")
    return overall

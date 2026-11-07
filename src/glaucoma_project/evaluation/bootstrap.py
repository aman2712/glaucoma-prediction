"""Bootstrap confidence intervals that resample whole patient groups."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from glaucoma_project.evaluation.screening import confusion_metrics, ranking_metrics, validated_scores

METRICS = ("auroc", "average_precision", "sensitivity", "specificity", "balanced_accuracy", "brier")


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    values = {**ranking_metrics(y, p), **confusion_metrics(y, p, threshold), "brier": brier_score_loss(y, p)}
    return {name: values[name] for name in METRICS}


def grouped_bootstrap(predictions: pd.DataFrame, threshold: float, seed: int, iterations: int) -> pd.DataFrame:
    """95% percentile intervals for ``METRICS`` at a fixed threshold.

    ``predictions`` has ``label``, ``probability`` and ``analysis_group``. Resampling whole groups
    keeps images of the same patient together. The intervals describe sampling noise for this
    trained model, not variation from retraining, and they cannot account for patient overlap
    that the dataset does not record. A metric that is undefined in a resample is skipped.
    """
    groups = list(predictions.reset_index(drop=True).groupby("analysis_group").indices.values())
    if len(groups) < 2:
        raise ValueError("At least two independent groups are needed for a bootstrap")
    y, p = validated_scores(predictions.label, predictions.probability)
    rng = np.random.default_rng(seed)
    samples = pd.DataFrame([
        _metrics(y[rows], p[rows], threshold)
        for rows in (np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                     for _ in range(iterations))])
    estimate = _metrics(y, p, threshold)
    records = []
    for name in METRICS:
        valid = samples[name].dropna()
        low, high = np.quantile(valid, [0.025, 0.975]) if len(valid) >= 2 else (np.nan, np.nan)
        records.append({"metric": name, "estimate": estimate[name], "lower_95": low, "upper_95": high,
                        "valid_resamples": len(valid), "requested_resamples": iterations,
                        "bootstrap_groups": len(groups)})
    return pd.DataFrame(records)

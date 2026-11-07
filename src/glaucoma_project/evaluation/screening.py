"""Metrics for a probability score used as a screening test at a fixed decision threshold.

Undefined values (sensitivity with no positives, AUROC with one class, ...) are NaN, never 0.
``ranking_metrics``, ``confusion_metrics`` and ``calibration_metrics`` expect arrays that
already passed ``validated_scores``.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, log_loss,
                             roc_auc_score, roc_curve)

ECE_BINS = 10


def validated_scores(y, p) -> tuple[np.ndarray, np.ndarray]:
    """Labels as ints and probabilities as floats; raises unless both are 1-D, equal length,
    labels are 0/1 and probabilities are finite and within [0, 1]."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or len(y) == 0:
        raise ValueError("Expected non-empty 1-D labels and probabilities of equal length")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected binary labels and finite probabilities in [0, 1]")
    return y.astype(int), p


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """AUROC and average precision (NaN when only one class is present)."""
    if len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "average_precision": float("nan")}
    return {"auroc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p))}


def confusion_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """Threshold-based rates and the confusion counts for ``p >= threshold``."""
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    sensitivity, specificity = _ratio(tp, tp + fn), _ratio(tn, tn + fp)
    return {"sensitivity": sensitivity, "specificity": specificity,
            "balanced_accuracy": (sensitivity + specificity) / 2, "accuracy": _ratio(tp + tn, len(y)),
            "precision": _ratio(tp, tp + fp), "npv": _ratio(tn, tn + fn),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Mean score, observed positive rate and count for each non-empty equal-width score bin."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    table = pd.DataFrame({"bin": bins, "score": p, "observed": y}).groupby("bin").agg(
        mean_score=("score", "mean"), observed_rate=("observed", "mean"), count=("score", "size"))
    return table.reset_index(drop=True)


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Brier score, log loss and expected calibration error (ECE) over ``ECE_BINS`` bins."""
    table = reliability_table(y, p, ECE_BINS)
    gaps = (table.observed_rate - table.mean_score).abs()
    return {"brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
            f"ece_{ECE_BINS}_bins": float((table["count"] * gaps).sum() / len(y))}


def metric_values(y, p, threshold: float = 0.5) -> dict:
    """Every screening metric at ``threshold``, plus the sample size and prevalence."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and within [0, 1]")
    y, p = validated_scores(y, p)
    return {"n": len(y), "positives": int(y.sum()), "negatives": int((1 - y).sum()),
            "prevalence": float(y.mean()), "threshold": threshold,
            **ranking_metrics(y, p), **calibration_metrics(y, p), **confusion_metrics(y, p, threshold)}


def select_threshold(y, p, target_sensitivity: float = 0.90) -> float:
    """The threshold with the highest specificity among those reaching ``target_sensitivity``
    on this data (ties go to the larger threshold). It does not promise the same sensitivity
    on new data."""
    y, p = validated_scores(y, p)
    if len(np.unique(y)) != 2 or not 0 < target_sensitivity <= 1:
        raise ValueError("Threshold selection needs both classes and a target sensitivity in (0, 1]")
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    reaching = np.flatnonzero(np.isfinite(thresholds) & (tpr >= target_sensitivity))
    best = max(reaching, key=lambda i: (1 - fpr[i], thresholds[i]))
    return float(thresholds[best])

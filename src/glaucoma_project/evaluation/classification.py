"""Classification metrics from docs/baseline_a.md: AUC, sensitivity, specificity,
sensitivity at 95% specificity, AUPRC and balanced accuracy, with bootstrap confidence intervals."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

METRICS = ["auc", "sensitivity", "specificity", "sensitivity_at_95_specificity", "auprc", "balanced_accuracy"]


def youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Score threshold maximising sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y, score)
    return float(thresholds[np.argmax(tpr - fpr)])


def sensitivity_at_specificity(y: np.ndarray, score: np.ndarray, specificity: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    return float(tpr[fpr <= 1 - specificity].max())


def classification_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    predicted = score >= threshold
    sensitivity = float(predicted[y == 1].mean())
    specificity = float((~predicted[y == 0]).mean())
    return {
        "auc": float(roc_auc_score(y, score)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "sensitivity_at_95_specificity": sensitivity_at_specificity(y, score),
        "auprc": float(average_precision_score(y, score)),
        "balanced_accuracy": (sensitivity + specificity) / 2,
    }


def evaluate(y, score, threshold: float, n_bootstrap: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Metrics with 95% bootstrap CIs (resampling images; the threshold stays fixed)."""
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    resampled = []
    for _ in range(n_bootstrap):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            resampled.append(classification_metrics(y[i], score[i], threshold))
    low, high = np.percentile(pd.DataFrame(resampled, columns=METRICS), [2.5, 97.5], axis=0)
    value = classification_metrics(y, score, threshold)
    return pd.DataFrame({"value": [value[m] for m in METRICS], "ci_low": low, "ci_high": high}, index=METRICS)

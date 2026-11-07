"""Paired comparison of two scores for the same images (used by the combined-model study)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def paired_auc_difference(y, score_a, score_b, groups, n_bootstrap: int = 2000, seed: int = 0) -> dict:
    """AUC(a) minus AUC(b) with a 95% percentile interval from a paired bootstrap.

    Each resample draws whole ``groups`` (patients, duplicates and possible repeat patients stay
    together) and scores both models on the same drawn images. Resamples with a single class are
    skipped and counted. The interval reflects the sampling of test images, not retraining.
    """
    y, a, b = np.asarray(y, dtype=int), np.asarray(score_a, float), np.asarray(score_b, float)
    members = list(pd.Series(np.arange(len(y))).groupby(np.asarray(groups)).apply(np.array))
    if len(members) < 2 or len(np.unique(y)) < 2:
        raise ValueError("Need at least two groups and both classes")
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_bootstrap):
        rows = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        if 0 < y[rows].sum() < len(rows):
            differences.append(roc_auc_score(y[rows], a[rows]) - roc_auc_score(y[rows], b[rows]))
    if len(differences) < 2:
        raise ValueError("Fewer than two valid bootstrap resamples")
    low, high = np.quantile(differences, [0.025, 0.975])
    return {"n_images": len(y), "n_positives": int(y.sum()), "n_groups": len(members),
            "auc_a": float(roc_auc_score(y, a)), "auc_b": float(roc_auc_score(y, b)),
            "difference": float(roc_auc_score(y, a) - roc_auc_score(y, b)), "ci_low": float(low),
            "ci_high": float(high), "valid_resamples": len(differences), "requested_resamples": n_bootstrap}

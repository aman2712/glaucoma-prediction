"""Paired comparison of two classifiers scored on the same images (docs/baseline_a.md)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def paired_auc_difference(y, score_a, score_b, n_bootstrap: int = 2000, seed: int = 0) -> dict:
    """AUC(a) - AUC(b) with a 95% bootstrap CI. Both models are scored on the same resampled images."""
    y, a, b = np.asarray(y, dtype=int), np.asarray(score_a, float), np.asarray(score_b, float)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_bootstrap):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            diffs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    low, high = np.percentile(diffs, [2.5, 97.5])
    p = min(1.0, 2 * min((np.array(diffs) <= 0).mean(), (np.array(diffs) >= 0).mean()))
    return {"auc_a": roc_auc_score(y, a), "auc_b": roc_auc_score(y, b),
            "difference": roc_auc_score(y, a) - roc_auc_score(y, b), "ci_low": low, "ci_high": high, "p_value": p}


def compare_on_shared_images(ours: pd.DataFrame, theirs: pd.DataFrame, **kwargs) -> dict:
    """ours: image_id, binary_label, score. theirs: image_id, probability. Joins on image_id."""
    merged = ours.merge(theirs[["image_id", "probability"]], on="image_id")
    result = paired_auc_difference(merged.binary_label, merged.score, merged.probability, **kwargs)
    return {"n_images": len(merged), "n_glaucoma": int(merged.binary_label.sum()), **result}

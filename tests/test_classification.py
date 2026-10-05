import numpy as np
import pandas as pd
import pytest

from glaucoma_project.classification.baseline import run_baseline
from glaucoma_project.evaluation.classification import (
    classification_metrics, evaluate, sensitivity_at_specificity, youden_threshold)


def test_metrics_for_perfect_separation():
    y = np.array([0, 0, 0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    result = classification_metrics(y, score, threshold=0.5)
    assert result["auc"] == 1 and result["sensitivity"] == 1 and result["specificity"] == 1
    assert result["balanced_accuracy"] == 1 and sensitivity_at_specificity(y, score) == 1
    assert 0.3 < youden_threshold(y, score) <= 0.8


def test_metrics_known_confusion_counts():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.6, 0.7, 0.3, 0.6, 0.8, 0.9])
    result = classification_metrics(y, score, threshold=0.5)
    assert result["sensitivity"] == 0.75 and result["specificity"] == 0.5


def test_bootstrap_interval_contains_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    score = y * 0.5 + rng.random(200)
    table = evaluate(y, score, threshold=0.7, n_bootstrap=200)
    assert ((table.ci_low <= table.value) & (table.value <= table.ci_high)).all()


def test_run_baseline_learns_a_separable_signal():
    rng = np.random.default_rng(0)
    rows = []
    for split, n in (("train", 200), ("validation", 100)):
        label = rng.integers(0, 2, n)
        rows.append(pd.DataFrame({"split": split, "source": "predicted", "binary_label": label,
                                  "group_id": [f"{split}{i}" for i in range(n)],
                                  "vcdr": 0.5 + 0.2 * label + 0.05 * rng.normal(size=n)}))
    config = {"features": ["vcdr"], "c_grid": [0.1, 1], "cv_folds": 3, "n_bootstrap": 50, "seed": 0}
    result = run_baseline(pd.concat(rows), "predicted", config)
    assert result.best_c in (0.1, 1) and result.validation.loc["auc", "value"] > 0.95

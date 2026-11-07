import numpy as np
import pandas as pd
import pytest

from glaucoma_project.evaluation.bootstrap import grouped_bootstrap
from glaucoma_project.evaluation.screening import metric_values, select_threshold, validated_scores


def test_known_values():
    y, p = [0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9]
    values = metric_values(y, p, 0.5)
    assert values["auroc"] == pytest.approx(0.75)
    assert values["brier"] == pytest.approx((0.01 + 0.64 + 0.09 + 0.01) / 4)
    assert (values["tn"], values["fp"], values["fn"], values["tp"]) == (1, 1, 0, 2)
    assert values["sensitivity"] == 1 and values["specificity"] == 0.5 and values["precision"] == pytest.approx(2 / 3)


def test_calibration_error_of_a_perfectly_calibrated_pair():
    y, p = [0, 0, 1, 1], [0.05, 0.05, 0.95, 0.95]
    assert metric_values(y, p)["ece_10_bins"] == pytest.approx(0.05)


def test_threshold_reaches_the_target_sensitivity_with_the_best_specificity():
    y, p = [0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9]
    threshold = select_threshold(y, p, 1.0)
    assert threshold == pytest.approx(0.7) and metric_values(y, p, threshold)["sensitivity"] == 1
    assert select_threshold([0, 1], [0.4, 0.4], 0.9) == 0.4


def test_undefined_metrics_are_nan_not_zero():
    values = metric_values([0, 0], [0.1, 0.2])
    assert all(np.isnan(values[k]) for k in ("auroc", "sensitivity", "precision"))
    with pytest.raises(ValueError):
        select_threshold([1, 1], [0.2, 0.3])


@pytest.mark.parametrize("p", [[np.nan, 0.2], [-0.1, 0.8], [0.2, 1.1]])
def test_invalid_probabilities_are_rejected(p):
    with pytest.raises(ValueError):
        validated_scores([0, 1], p)


def test_threshold_must_be_a_probability():
    with pytest.raises(ValueError):
        metric_values([0, 1], [0.2, 0.8], 1.5)


def test_bootstrap_resamples_whole_groups_and_is_reproducible():
    predictions = pd.DataFrame({"label": [0, 1, 0, 1, 0, 1], "probability": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
                                "analysis_group": ["a", "a", "b", "b", "c", "c"]})
    first, second = (grouped_bootstrap(predictions, 0.5, seed=0, iterations=30) for _ in range(2))
    pd.testing.assert_frame_equal(first, second)
    assert (first.bootstrap_groups == 3).all()
    assert first.set_index("metric").loc["auroc", "lower_95"] == 1.0

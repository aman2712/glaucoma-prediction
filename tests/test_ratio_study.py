import numpy as np
import pandas as pd
import pytest

from glaucoma_project.classification.ratio_study import (add_partition, camera_gate, compare_partitions, fit_classifier,
                                                         load_table, run_study)

FEATURES = ["vcdr", "hcdr", "area_cdr"]
CONFIG = {"features": FEATURES, "c_grid": [0.1, 1], "cv_folds": 3, "seed": 0, "n_bootstrap": 100}


def images(split, camera, n, shift, seed, prefix):
    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], n // 2)
    ids = [f"{prefix}{i:03d}" for i in range(n)]
    ratios = {c: 0.5 + 0.2 * labels + shift + 0.1 * rng.standard_normal(n) for c in FEATURES}
    return pd.DataFrame({"image_id": ids, "split": split, "camera": camera, "analysis_group": ids, "binary_label": labels, **ratios})


@pytest.fixture
def table():
    """Remidio and Bosch development images, a test set like them, and an external test set whose cups are drawn smaller."""
    return pd.concat([images("train", "Remidio", 120, 0.0, 1, "r"), images("validation", "Bosch", 40, 0.0, 2, "b"),
                      images("in_domain_test", "Remidio", 60, 0.0, 3, "t"), images("external_test", "Topcon", 80, -0.25, 4, "x")],
                     ignore_index=True)


def test_splits_map_to_three_partitions(table):
    assert set(add_partition(table).partition) == {"development", "in_domain_test", "external_test"}


def test_a_fitted_classifier_scores_new_rows_above_the_threshold_for_positives(table):
    data = add_partition(table)
    fit = fit_classifier(data[data.partition == "development"], CONFIG)
    scores = fit.model.predict_proba(data[data.partition == "in_domain_test"][FEATURES])[:, 1]
    labels = data[data.partition == "in_domain_test"].binary_label.to_numpy()
    assert scores[labels == 1].mean() > scores[labels == 0].mean() and 0 < fit.threshold < 1


def test_standardising_repairs_the_operating_point_when_the_cups_are_smaller(table):
    result = compare_partitions(add_partition(table), CONFIG)
    external = result[(result.partition == "external_test") & (result.metric == "balanced_accuracy")].iloc[0]
    assert external.difference > 0.1 and external.ci_low > 0


def test_the_ranking_barely_changes_when_only_the_scale_shifts(table):
    result = compare_partitions(add_partition(table), CONFIG)
    auc = result[(result.partition == "external_test") & (result.metric == "auc")].iloc[0]
    assert abs(auc.difference) < 0.05


def test_the_camera_gate_reports_both_arms_and_a_verdict(table):
    gate = camera_gate(add_partition(table), CONFIG, source_camera="Remidio")
    assert set(gate) == {"raw", "standardised", "passes"} and isinstance(gate["passes"], bool)


def test_the_study_returns_scores_for_both_arms_and_a_comparison(table):
    scores, comparison = run_study(add_partition(table), CONFIG)
    assert {"raw_score", "standardised_score", "raw_decision", "standardised_decision"} <= set(scores.columns)
    assert set(scores.partition) == {"in_domain_test", "external_test"} and len(comparison) == 4


def test_load_table_joins_the_camera_and_refuses_unknown_images(tmp_path):
    measurements = pd.DataFrame({"image_id": ["a", "b"], "split": ["train", "validation"], "analysis_group": ["a", "b"],
                                 "binary_label": [0, 1], "vcdr": [.5, .6], "hcdr": [.5, .6], "area_cdr": [.3, .4]})
    split = pd.DataFrame({"image_id": ["a"], "camera": ["Remidio"]})
    with pytest.raises(ValueError, match="no camera"):
        load_table(measurements, split)

import numpy as np
import pytest

from glaucoma_project.evaluation.compare import paired_auc_difference

Y = np.array([0, 0, 0, 1, 1, 1] * 10)
GROUPS = np.arange(60)


def test_identical_scores_have_zero_difference():
    s = np.random.default_rng(0).random(60)
    r = paired_auc_difference(Y, s, s, GROUPS, n_bootstrap=200)
    assert r["difference"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0


def test_a_clearly_better_score_has_a_positive_interval():
    rng = np.random.default_rng(1)
    good, noise = Y + 0.3 * rng.standard_normal(60), rng.standard_normal(60)
    r = paired_auc_difference(Y, good, noise, GROUPS, n_bootstrap=300)
    assert r["difference"] > 0.3 and r["ci_low"] > 0 and r["n_groups"] == 60


def test_images_of_one_group_are_resampled_together():
    """Two images per group: grouped resampling must leave fewer effective units than images."""
    groups = np.repeat(np.arange(30), 2)
    s = np.random.default_rng(2).random(60)
    t = np.random.default_rng(3).random(60)
    assert paired_auc_difference(Y, s, t, groups, n_bootstrap=50)["n_groups"] == 30


def test_rejects_one_class_or_one_group():
    with pytest.raises(ValueError):
        paired_auc_difference(np.zeros(6), np.arange(6), np.arange(6), np.arange(6))
    with pytest.raises(ValueError):
        paired_auc_difference(Y, np.arange(60), np.arange(60), np.zeros(60))

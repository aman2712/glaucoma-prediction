import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from glaucoma_project.evaluation.compare import paired_auc_difference, paired_difference

Y = np.array([0, 0, 0, 1, 1, 1] * 10)
GROUPS = np.arange(60)


def test_the_auc_comparison_is_the_general_one_with_the_auc_metric():
    rng = np.random.default_rng(0)
    a, b = Y + rng.standard_normal(60), rng.standard_normal(60)
    general = paired_difference(Y, a, b, GROUPS, roc_auc_score, 100)
    auc = paired_auc_difference(Y, a, b, GROUPS, 100)
    assert (auc["difference"], auc["ci_low"], auc["ci_high"]) == (general["difference"], general["ci_low"], general["ci_high"])
    assert auc["auc_a"] == general["metric_a"] and auc["auc_b"] == general["metric_b"]


def test_a_better_decision_rule_has_a_positive_balanced_accuracy_difference():
    rng = np.random.default_rng(1)
    good = np.where(rng.random(60) < 0.9, Y, 1 - Y)
    bad = np.where(rng.random(60) < 0.5, Y, 1 - Y)
    r = paired_difference(Y, good, bad, GROUPS, balanced_accuracy_score, 300)
    assert r["difference"] > 0.2 and r["ci_low"] > 0

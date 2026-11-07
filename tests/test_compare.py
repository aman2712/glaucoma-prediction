import numpy as np
import pandas as pd

from glaucoma_project.evaluation.compare import compare_on_shared_images, paired_auc_difference


def test_identical_scores_have_zero_difference():
    y = np.array([0, 0, 0, 1, 1, 1] * 10)
    s = np.random.default_rng(0).random(60)
    r = paired_auc_difference(y, s, s, n_bootstrap=200)
    assert r["difference"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0


def test_better_model_has_positive_difference_and_only_shared_images_used():
    y = [0, 0, 1, 1, 0, 1]
    ours = pd.DataFrame({"image_id": list("abcdef"), "binary_label": y, "score": [.1, .2, .8, .9, .3, .7]})
    theirs = pd.DataFrame({"image_id": list("abcdez"), "probability": [.5, .4, .6, .3, .5, .9]})
    r = compare_on_shared_images(ours, theirs, n_bootstrap=100)
    assert r["n_images"] == 5 and r["difference"] > 0

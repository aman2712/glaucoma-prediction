import numpy as np
import pandas as pd
import pytest

from glaucoma_project.combined_model.blend import (FEATURES, blend_logits, choose_regularisation, development_diagnostic,
                                                   development_table, fit_blend)

C_GRID = [0.01, 1, 100]


def synthetic_logits(n=80, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], n // 2)
    ids = [f"img{i:03d}" for i in range(n)]
    return ids, labels, rng


def logits_frame(ids, labels, values):
    return pd.DataFrame({"image_id": ids, "label": labels, "analysis_group": ids, "logit": values})


def measurements_for(ids, labels, rng, signal=0.3):
    noisy = lambda: 0.5 + signal * labels + 0.15 * rng.standard_normal(len(ids))
    return pd.DataFrame({"image_id": ids, "binary_label": labels, "vcdr": noisy(), "hcdr": noisy(), "area_cdr": noisy()})


@pytest.fixture
def table():
    ids, labels, rng = synthetic_logits()
    weak_image_scores = {seed: logits_frame(ids, labels, 0.1 * labels + rng.standard_normal(len(ids))) for seed in (1, 2, 3)}
    return development_table(weak_image_scores, measurements_for(ids, labels, rng))


def test_image_score_is_the_mean_logit_over_seeds_next_to_the_ratios():
    ids, labels, rng = synthetic_logits(n=10)
    by_seed = {1: logits_frame(ids, labels, np.zeros(10)), 2: logits_frame(ids, labels, np.full(10, 2.0))}
    result = development_table(by_seed, measurements_for(ids, labels, rng))
    assert result.image_logit.tolist() == [1.0] * 10
    assert {"binary_label", "analysis_group", *FEATURES} <= set(result.columns)


def test_seeds_that_scored_different_images_are_refused():
    ids, labels, rng = synthetic_logits(n=10)
    other = logits_frame(ids[::-1][:9] + ["extra"], np.append(labels[::-1][:9], 0), np.zeros(10))
    with pytest.raises(ValueError, match="different images"):
        development_table({1: logits_frame(ids, labels, np.zeros(10)), 2: other}, measurements_for(ids, labels, rng))


def test_regularisation_comes_from_the_grid(table):
    assert choose_regularisation(table, C_GRID, n_folds=4, seed=2440) in C_GRID


def test_blend_uses_the_ratios_when_the_image_score_is_weak(table):
    diagnostic = development_diagnostic(table, c=1, n_folds=4, seed=2440)
    assert diagnostic["blend_cv_auc"] > diagnostic["image_only_auc"]


def test_blend_scores_every_image_even_when_a_ratio_is_missing(table):
    model = fit_blend(table, c=1)
    missing = table.assign(vcdr=np.nan)
    scores = blend_logits(model, missing)
    assert len(scores) == len(table) and np.isfinite(scores).all()

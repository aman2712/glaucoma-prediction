import numpy as np
import pandas as pd
import pytest

from glaucoma_project.evaluation.metrics import agreement
from glaucoma_project.measurements import measure
from glaucoma_project.measurements.cdr import largest_component


def rectangle(top, bottom, left, right, size=100):
    mask = np.zeros((size, size), bool)
    mask[top:bottom, left:right] = True
    return mask


def test_measure_known_shapes():
    disc = rectangle(10, 50, 20, 50)   # 40 tall, 30 wide
    cup = rectangle(20, 40, 25, 40)    # 20 tall, 15 wide
    result = measure(disc, cup)
    assert result["vcdr"] == pytest.approx(0.5)
    assert result["hcdr"] == pytest.approx(0.5)
    assert result["area_cdr"] == pytest.approx(300 / 1200)
    assert (result["disc_area"], result["cup_area"]) == (1200, 300)


def test_stray_speck_does_not_change_measurement():
    disc, cup = rectangle(10, 50, 20, 50), rectangle(20, 40, 25, 40)
    noisy = cup.copy()
    noisy[90, 90] = True
    assert measure(disc, noisy) == measure(disc, cup)
    assert largest_component(noisy).sum() == cup.sum()


def test_empty_disc_gives_nan_ratios():
    result = measure(np.zeros((10, 10), bool), rectangle(2, 4, 2, 4, size=10))
    assert np.isnan(result["vcdr"]) and np.isnan(result["area_cdr"])


def test_agreement():
    expert = pd.Series([0.4, 0.5, 0.6, np.nan])
    result = agreement(expert + 0.1, expert)
    assert result["mae"] == pytest.approx(0.1) and result["pearson"] == pytest.approx(1.0) and result["n"] == 3

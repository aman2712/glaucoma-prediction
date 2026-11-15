import numpy as np
import pandas as pd
import pytest

from glaucoma_project.classification.standardise import standardise_within

COLUMNS = ["vcdr", "hcdr"]


def table():
    return pd.DataFrame({"group": ["a"] * 4 + ["b"] * 4,
                         "vcdr": [0.2, 0.4, 0.6, 0.8, 0.1, 0.2, 0.3, 0.4],
                         "hcdr": [0.5, 0.5, 0.7, 0.7, 0.3, 0.3, 0.3, 0.9],
                         "label": [0, 0, 1, 1] * 2})


def test_each_group_has_zero_mean_and_unit_spread():
    out = standardise_within(table(), COLUMNS, "group")
    for _, rows in out.groupby("group"):
        assert rows.vcdr.mean() == pytest.approx(0)
        assert rows.vcdr.std(ddof=0) == pytest.approx(1)


def test_other_columns_and_input_are_untouched():
    original = table()
    out = standardise_within(original, COLUMNS, "group")
    assert out.label.tolist() == original.label.tolist()
    assert original.vcdr.tolist() == table().vcdr.tolist()


def test_missing_values_stay_missing_and_do_not_enter_the_statistics():
    data = table()
    data.loc[0, "vcdr"] = np.nan
    out = standardise_within(data, COLUMNS, "group")
    assert np.isnan(out.loc[0, "vcdr"])
    assert out[out.group == "a"].vcdr.mean() == pytest.approx(0)


def test_a_constant_column_becomes_zero_not_nan():
    data = table().assign(hcdr=0.5)
    out = standardise_within(data, COLUMNS, "group")
    assert (out.hcdr == 0).all()

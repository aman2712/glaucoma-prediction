"""Per-group standardisation of the cup-to-disc ratios (experiment X1).

Experts of different datasets draw cups of different sizes, so a ratio of 0.5 means something
different in each. Each group's ratios are rescaled with that group's own mean and spread, which
uses the images but never the labels.
"""
import pandas as pd


def standardise_within(table: pd.DataFrame, columns: list[str], group: str) -> pd.DataFrame:
    """Copy of ``table`` whose ``columns`` have mean 0 and spread 1 inside each value of ``group``.
    Missing values stay missing and are left out of the statistics; a constant column becomes 0."""
    result = table.copy()
    for column in columns:
        within = result.groupby(group)[column]
        centred = result[column] - within.transform("mean")
        spread = within.transform(lambda values: values.std(ddof=0))
        result[column] = centred / spread.where(spread > 0, 1.0)  # a constant column is already 0 after centring
    return result

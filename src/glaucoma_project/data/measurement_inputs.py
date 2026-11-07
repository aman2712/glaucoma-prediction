"""Measurements as model inputs: joined to the image rows, then median-filled and standardised
with statistics from the training rows only."""
import numpy as np
import pandas as pd


def attach_measurements(rows: pd.DataFrame, measurements: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add ``columns`` to ``rows`` by image ID. Every row needs a measurement entry (its value may be
    NaN or empty where no disc was found), and the label must agree with the one in ``rows``."""
    if measurements.image_id.duplicated().any():
        raise ValueError("Duplicate image IDs in the measurements")
    joined = rows.merge(measurements[["image_id", "binary_label", *columns]], on="image_id", how="left",
                        suffixes=("", "_measured"), validate="one_to_one", indicator=True)
    if (joined["_merge"] != "both").any():
        raise ValueError("Some images have no measurement entry")
    if (joined.binary_label.astype(int) != joined.binary_label_measured.astype(int)).any():
        raise ValueError("Measurement labels disagree with the split labels")
    joined[columns] = joined[columns].replace("", np.nan).astype(float)  # tables read as text hold "" for NaN
    return joined.drop(columns=["_merge", "binary_label_measured"])


def fit_scaler(train_rows: pd.DataFrame, columns: list[str]) -> dict:
    """Median, mean and standard deviation of each column over the training rows (mean and
    standard deviation after median filling). Plain numbers, so it can be saved with a checkpoint."""
    values = train_rows[columns].astype(float)
    median = values.median()
    filled = values.fillna(median)
    return {"columns": list(columns), "median": median.tolist(), "mean": filled.mean().tolist(),
            "std": filled.std(ddof=0).tolist()}


def apply_scaler(rows: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    """Copy of ``rows`` with the scaler's columns median-filled and standardised."""
    columns = scaler["columns"]
    filled = rows[columns].astype(float).fillna(pd.Series(scaler["median"], index=columns))
    scaled = (filled - np.asarray(scaler["mean"])) / np.asarray(scaler["std"])
    return rows.assign(**{c: scaled[c] for c in columns})

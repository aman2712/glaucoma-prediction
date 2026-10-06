"""Baseline A classifier: logistic regression on disc/cup measurements."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from glaucoma_project.evaluation.classification import evaluate, youden_threshold

DEVELOPMENT_SPLITS = ("train", "validation")


@dataclass
class BaselineResult:
    model: Pipeline            # fitted on all development images
    threshold: float          # Youden's J on out-of-fold scores
    best_c: float
    cv_auc: pd.Series         # cross-validated AUC for each C
    development: pd.DataFrame  # out-of-fold metrics with CIs
    out_of_fold_score: np.ndarray
    labels: pd.Series


def make_model(c: float = 1.0) -> Pipeline:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=c, class_weight="balanced", max_iter=1000))


def labelled(table: pd.DataFrame) -> pd.DataFrame:
    """Rows with a binary glaucoma label (PAPILA 'suspect' eyes have none)."""
    return table.dropna(subset=["binary_label"])


def development_data(measurements: pd.DataFrame, source: str) -> pd.DataFrame:
    """Labelled train and validation images, measured from ``source`` ("predicted" or "expert") masks."""
    data = measurements[(measurements.source == source) & measurements.split.isin(DEVELOPMENT_SPLITS)]
    return labelled(data).reset_index(drop=True)


def grouped_folds(data: pd.DataFrame, folds: int, seed: int) -> list:
    """Stratified folds in which images of one duplicate group never straddle train and test."""
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(cv.split(data, data["binary_label"], data["group_id"]))


def cross_validated_auc(model, data: pd.DataFrame, features: list[str], folds: int, seed: int) -> float:
    return cross_val_score(model, data[features], data["binary_label"], cv=grouped_folds(data, folds, seed),
                           scoring="roc_auc").mean()


def run_baseline(measurements: pd.DataFrame, source: str, config: dict) -> BaselineResult:
    """Choose C by grouped cross-validation, the threshold from out-of-fold scores, then fit on everything.

    All development images (train and validation) are used, because the validation split alone has
    only 27 glaucoma cases. Test images are never touched here.
    """
    features, folds, seed = config["features"], config["cv_folds"], config["seed"]
    data = development_data(measurements, source)
    cv_auc = pd.Series({c: cross_validated_auc(make_model(c), data, features, folds, seed)
                        for c in config["c_grid"]})
    best_c = float(cv_auc.idxmax())
    score = cross_val_predict(make_model(best_c), data[features], data["binary_label"],
                              cv=grouped_folds(data, folds, seed), method="predict_proba")[:, 1]
    threshold = youden_threshold(data["binary_label"].to_numpy(), score)
    table = evaluate(data["binary_label"], score, threshold, config["n_bootstrap"], seed)
    model = make_model(best_c).fit(data[features], data["binary_label"])
    return BaselineResult(model, threshold, best_c, cv_auc, table, score, data["binary_label"])


def ablation(measurements: pd.DataFrame, source: str, config: dict, best_c: float) -> pd.Series:
    """Cross-validated AUC of the baseline against variants (extra features, a non-linear model)."""
    data = development_data(measurements, source)
    ratios, with_areas = config["features"], config["features"] + ["disc_area", "cup_area"]
    folds, seed = config["cv_folds"], config["seed"]
    boosting = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=100, random_state=seed)
    return pd.Series({
        "logistic regression (ratios)": cross_validated_auc(make_model(best_c), data, ratios, folds, seed),
        "logistic regression (ratios + areas)": cross_validated_auc(make_model(best_c), data, with_areas, folds, seed),
        "gradient boosting (ratios)": cross_validated_auc(boosting, data, ratios, folds, seed),
    })

"""Baseline A classifier: logistic regression on disc/cup measurements."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from glaucoma_project.evaluation.classification import evaluate, youden_threshold


@dataclass
class BaselineResult:
    model: Pipeline
    threshold: float
    best_c: float
    cv_auc: pd.Series       # mean cross-validated AUC on train, per C
    validation: pd.DataFrame  # metrics with CIs on the validation split
    validation_score: np.ndarray


def make_model(c: float = 1.0) -> Pipeline:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=c, class_weight="balanced", max_iter=1000))


def labelled(table: pd.DataFrame) -> pd.DataFrame:
    """Rows with a binary glaucoma label (PAPILA 'suspect' eyes have none)."""
    return table.dropna(subset=["binary_label"])


def cross_validated_auc(model, train: pd.DataFrame, features: list[str], folds: int, seed: int) -> float:
    """Mean AUC over stratified folds of the training split. Images of one group stay in one fold."""
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(cv.split(train, train["binary_label"], train["group_id"]))
    return cross_val_score(model, train[features], train["binary_label"], cv=splits, scoring="roc_auc").mean()


def select_c(train: pd.DataFrame, features: list[str], c_grid: list[float], folds: int, seed: int) -> pd.Series:
    """Cross-validated AUC on the training split for each value of C."""
    return pd.Series({c: cross_validated_auc(make_model(c), train, features, folds, seed) for c in c_grid})


def ablation(measurements: pd.DataFrame, source: str, config: dict, best_c: float) -> pd.Series:
    """Cross-validated train AUC of the baseline against variants (extra features, a non-linear model)."""
    train = labelled(measurements[(measurements.source == source) & (measurements.split == "train")])
    ratios, with_areas = config["features"], config["features"] + ["disc_area", "cup_area"]
    folds, seed = config["cv_folds"], config["seed"]
    boosting = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=100, random_state=seed)
    return pd.Series({
        "logistic regression (ratios)": cross_validated_auc(make_model(best_c), train, ratios, folds, seed),
        "logistic regression (ratios + areas)": cross_validated_auc(make_model(best_c), train, with_areas, folds, seed),
        "gradient boosting (ratios)": cross_validated_auc(boosting, train, ratios, folds, seed),
    })


def run_baseline(measurements: pd.DataFrame, source: str, config: dict) -> BaselineResult:
    """Fit on train, choose C by cross-validation on train, choose the threshold on validation.

    ``source`` is "predicted" or "expert": which masks the measurements came from.
    """
    features = config["features"]
    data = labelled(measurements[measurements.source == source])
    train, validation = data[data.split == "train"], data[data.split == "validation"]
    cv_auc = select_c(train, features, config["c_grid"], config["cv_folds"], config["seed"])
    best_c = float(cv_auc.idxmax())
    model = make_model(best_c).fit(train[features], train["binary_label"])
    score = model.predict_proba(validation[features])[:, 1]
    threshold = youden_threshold(validation["binary_label"].to_numpy(), score)
    table = evaluate(validation["binary_label"], score, threshold, config["n_bootstrap"], config["seed"])
    return BaselineResult(model, threshold, best_c, cv_auc, table, score)

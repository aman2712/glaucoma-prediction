"""The two-step blend: a simple classifier on a cross-fitted image score and the three CDR ratios."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from glaucoma_project.classification.baseline import cross_validated_auc, make_model
from glaucoma_project.combined_model.analysis import ensemble

RATIOS = ["vcdr", "hcdr", "area_cdr"]
FEATURES = ["image_logit", *RATIOS]


def development_table(logits_by_seed: dict[int, pd.DataFrame], measurements: pd.DataFrame) -> pd.DataFrame:
    """One row per development image: label, group, the mean held-out logit over seeds and the ratios.
    Raises if the seeds did not score the same images."""
    averaged = ensemble(list(logits_by_seed.values()))
    table = averaged.rename(columns={"logit": "image_logit", "label": "binary_label"})
    table = table[["image_id", "binary_label", "analysis_group", "image_logit"]]
    return table.merge(measurements[["image_id", *RATIOS]], on="image_id", how="left", validate="one_to_one")


def choose_regularisation(table: pd.DataFrame, c_grid: list[float], n_folds: int, seed: int) -> float:
    """The C of the grid with the best grouped cross-validated AUC (the Baseline A rule)."""
    auc_by_c = {c: cross_validated_auc(make_model(c), table, FEATURES, n_folds, seed) for c in c_grid}
    return float(max(auc_by_c, key=auc_by_c.get))


def fit_blend(table: pd.DataFrame, c: float) -> Pipeline:
    return make_model(c).fit(table[FEATURES], table["binary_label"])


def blend_logits(model: Pipeline, table: pd.DataFrame) -> np.ndarray:
    """Ranking score of the blend (the log-odds), defined for images with a missing ratio too."""
    return model.decision_function(table[FEATURES])


def development_diagnostic(table: pd.DataFrame, c: float, n_folds: int, seed: int) -> dict:
    """Development-only comparison, made before the test sets are opened: the held-out image score
    alone against the cross-validated blend."""
    return {"image_only_auc": float(roc_auc_score(table["binary_label"], table["image_logit"])),
            "blend_cv_auc": float(cross_validated_auc(make_model(c), table, FEATURES, n_folds, seed))}

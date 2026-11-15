"""Experiment X1: does standardising the cup-to-disc ratios within each dataset move the operating point?

Usage (design in docs/experiment_log.md, "X-series design"):
  python -m glaucoma_project.classification.ratio_study gate           development data only, no test image
  python -m glaucoma_project.classification.ratio_study freeze         records code and input hashes
  python -m glaucoma_project.classification.ratio_study run --out DIR  scores both test sets, once
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from glaucoma_project import COMMON_SPLIT_DIR, REPO_ROOT
from glaucoma_project.classification.baseline import cross_validated_auc, grouped_folds, make_model
from glaucoma_project.classification.standardise import standardise_within
from glaucoma_project.evaluation.classification import balanced_accuracy, classification_metrics, youden_threshold
from glaucoma_project.evaluation.compare import paired_difference
from glaucoma_project.files import read_table, write_json
from glaucoma_project.freeze import freeze, verify

STUDY_CONFIG = REPO_ROOT / "configs" / "x1.yaml"
MANIFEST = REPO_ROOT / "docs" / "frozen_x1.json"
CODE_PATHS = ["src/glaucoma_project/classification", "src/glaucoma_project/evaluation", "src/glaucoma_project/freeze.py",
              "src/glaucoma_project/files.py", "configs/x1.yaml"]
PARTITION_OF_SPLIT = {"train": "development", "validation": "development",
                      "in_domain_test": "in_domain_test", "external_test": "external_test"}
TEST_PARTITIONS = ("in_domain_test", "external_test")


@dataclass
class Fit:
    model: object
    threshold: float
    c: float


def load_table(measurements: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """The predicted ratios of every classification image, with its camera from the common split."""
    table = measurements.merge(split[["image_id", "camera"]], on="image_id", how="left")
    if table.camera.isna().any():
        raise ValueError("Some images have no camera in the common split")
    return add_partition(table)


def add_partition(table: pd.DataFrame) -> pd.DataFrame:
    return table.assign(partition=table.split.map(PARTITION_OF_SPLIT))


def fit_classifier(development: pd.DataFrame, config: dict) -> Fit:
    """The Baseline A recipe: C by grouped cross-validated AUC, threshold by Youden's J on out-of-fold scores."""
    features, folds, seed = config["features"], config["cv_folds"], config["seed"]
    cv_auc = {c: cross_validated_auc(make_model(c), development, features, folds, seed) for c in config["c_grid"]}
    best_c = max(cv_auc, key=cv_auc.get)
    out_of_fold = cross_val_predict(make_model(best_c), development[features], development.binary_label,
                                    cv=grouped_folds(development, folds, seed), method="predict_proba")[:, 1]
    threshold = youden_threshold(development.binary_label.to_numpy(), out_of_fold)
    return Fit(make_model(best_c).fit(development[features], development.binary_label), threshold, best_c)


def score(fit: Fit, rows: pd.DataFrame, features: list[str]) -> np.ndarray:
    return fit.model.predict_proba(rows[features])[:, 1]


def arm_scores(table: pd.DataFrame, config: dict, standardised: bool) -> pd.DataFrame:
    """One arm: ratios as measured, or standardised within each partition; fitted on development, scored on the tests."""
    features = config["features"]
    prepared = standardise_within(table, features, "partition") if standardised else table
    fit = fit_classifier(prepared[prepared.partition == "development"], config)
    tests = prepared[prepared.partition.isin(TEST_PARTITIONS)]
    values = score(fit, tests, features)
    return pd.DataFrame({"image_id": tests.image_id, "score": values, "decision": (values >= fit.threshold).astype(int)})


def run_study(table: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-image scores and decisions of both arms on the test sets, and the paired comparison."""
    arms = {name: arm_scores(table, config, standardised)
            for name, standardised in (("raw", False), ("standardised", True))}
    tests = table[table.partition.isin(TEST_PARTITIONS)][["image_id", "partition", "analysis_group", "binary_label"]]
    scores = tests.copy()
    for name, arm in arms.items():
        scores = scores.merge(arm.rename(columns={"score": f"{name}_score", "decision": f"{name}_decision"}), on="image_id")
    return scores, compare_scores(scores, config)


def compare_scores(scores: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Standardised minus raw, per test set: balanced accuracy at the arm's own threshold (primary) and AUC."""
    metrics = {"balanced_accuracy": ("decision", balanced_accuracy), "auc": ("score", roc_auc_score)}
    rows = []
    for partition, part in scores.groupby("partition"):
        for metric, (column, function) in metrics.items():
            result = paired_difference(part.binary_label, part[f"standardised_{column}"], part[f"raw_{column}"],
                                       part.analysis_group, function, config["n_bootstrap"], config["seed"])
            rows.append({"partition": partition, "metric": metric, "raw": result["metric_b"],
                         "standardised": result["metric_a"], "difference": result["difference"],
                         "ci_low": result["ci_low"], "ci_high": result["ci_high"],
                         "n_images": result["n_images"], "n_positives": result["n_positives"]})
    return pd.DataFrame(rows)


def compare_partitions(table: pd.DataFrame, config: dict) -> pd.DataFrame:
    return run_study(table, config)[1]


def camera_gate(table: pd.DataFrame, config: dict, source_camera: str = "Remidio") -> dict:
    """Development-only check: fit on one camera, score the others, with and without per-camera standardising."""
    features = config["features"]
    development = table[table.partition == "development"]
    outcome = {}
    for name, standardised in (("raw", False), ("standardised", True)):
        prepared = standardise_within(development, features, "camera") if standardised else development
        source, other = prepared[prepared.camera == source_camera], prepared[prepared.camera != source_camera]
        fit = fit_classifier(source, config)
        outcome[name] = classification_metrics(other.binary_label.to_numpy(), score(fit, other, features), fit.threshold)
        outcome[name]["n_images"], outcome[name]["n_positives"] = len(other), int(other.binary_label.sum())
    outcome["passes"] = bool(outcome["standardised"]["balanced_accuracy"] > outcome["raw"]["balanced_accuracy"])
    return outcome


def load_study(path: Path = STUDY_CONFIG) -> dict:
    return yaml.safe_load(path.read_text())


def study_table(study: dict) -> pd.DataFrame:
    split = read_table(COMMON_SPLIT_DIR / "all.csv")
    return load_table(pd.read_csv(REPO_ROOT / study["measurements"]), split)


def freeze_study(study: dict, manifest: Path = MANIFEST) -> dict:
    files = {"measurements": REPO_ROOT / study["measurements"], "common_split": COMMON_SPLIT_DIR / "all.csv"}
    return freeze(files, {"study": study}, manifest, CODE_PATHS)


def run_once(study: dict, out_dir: Path, manifest: Path = MANIFEST) -> pd.DataFrame:
    """Verify the freeze, refuse a second run, then score both test sets and write the tables."""
    verify(manifest)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists: the test sets are scored once")
    scores, comparison = run_study(study_table(study), study)
    out_dir.mkdir(parents=True)
    scores.to_csv(out_dir / "scores.csv", index=False)
    comparison.to_csv(out_dir / "comparison.csv", index=False)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["gate", "freeze", "run"])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    study = load_study()
    if args.action == "gate":
        gate = camera_gate(study_table(study), study)
        print(json.dumps(gate, indent=2))
        if args.out:
            write_json(args.out, gate)
    elif args.action == "freeze":
        print(freeze_study(study)["git_commit"])
    else:
        print(run_once(study, args.out).to_string(index=False))


if __name__ == "__main__":
    main()

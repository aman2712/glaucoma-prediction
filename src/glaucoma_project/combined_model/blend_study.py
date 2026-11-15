"""The P01b study: cross-fitted image scores blended with the three predicted CDR ratios.

Usage, in order (the design is in docs/experiment_log.md, "P01b design"):
  python -m glaucoma_project.combined_model.blend_study train --seed S   five networks, scores their held-out folds
  python -m glaucoma_project.combined_model.blend_study job N            seed of Slurm array task N
  python -m glaucoma_project.combined_model.blend_study fit              chooses C and fits the blend (development data only)
  python -m glaucoma_project.combined_model.blend_study freeze           records every checkpoint and input hash
  python -m glaucoma_project.combined_model.blend_study score --out DIR  scores both test sets, once
  python -m glaucoma_project.combined_model.blend_study analyse --scores DIR --out DIR
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from glaucoma_project import COMMON_SPLIT_DIR, PREPARED_ROOT, REPO_ROOT
from glaucoma_project.combined_model.blend import (RATIOS, blend_logits, choose_regularisation,
                                                   development_diagnostic, development_table, fit_blend)
from glaucoma_project.combined_model.crossfit import fold_assignments, held_out_logits, train_fold_models
from glaucoma_project.data.classification import check_images
from glaucoma_project.data.common_split import load_common_split
from glaucoma_project.evaluation.compare import paired_auc_difference
from glaucoma_project.files import write_json
from glaucoma_project.freeze import freeze, verify
from glaucoma_project.image_model.model import restore_model
from glaucoma_project.image_model.predict import predict_rows
from glaucoma_project.runtime import environment_record, pick_device

STUDY_CONFIG = REPO_ROOT / "configs" / "p01b.yaml"
MANIFEST = REPO_ROOT / "docs" / "frozen_p01b.json"
TEST_PARTITIONS = ("in_domain_test", "external_test")
CODE_PATHS = [f"src/glaucoma_project/{name}" for name in
              ("combined_model/blend_study.py", "combined_model/blend.py", "combined_model/crossfit.py",
               "combined_model/analysis.py", "image_model", "classification", "data", "evaluation",
               "runtime.py", "files.py", "freeze.py")] + ["configs/p01b.yaml"]


def load_study(path: Path = STUDY_CONFIG) -> dict:
    return yaml.safe_load(path.read_text())


def runs_directory(study: dict) -> Path:
    return REPO_ROOT / study["runs_dir"]


def seed_directory(study: dict, seed: int) -> Path:
    return runs_directory(study) / f"seed{seed}"


def checkpoint_name(seed: int, fold: int) -> str:
    return f"seed{seed}_fold{fold}"


def development_rows(splits_dir: Path) -> pd.DataFrame:
    parts = load_common_split(splits_dir)
    return pd.concat([parts["train"], parts["validation"]], ignore_index=True)


def read_measurements(study: dict) -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / study["measurements"])


# ---- training (development data only) ---------------------------------------------------------

def train_seed(study: dict, seed: int, root: Path = PREPARED_ROOT, device: torch.device | None = None,
               splits_dir: Path = COMMON_SPLIT_DIR) -> Path:
    """Train the fold networks of one seed and save them with their held-out scores.
    Refuses to reuse the seed's folder."""
    device = device or pick_device()
    directory = seed_directory(study, seed)
    directory.mkdir(parents=True)
    write_json(directory / "run.json", environment_record(device))
    development = development_rows(splits_dir)
    check_images(root, development)
    folds = fold_assignments(development, study["n_folds"], study["fold_seed"])
    config = {**study["recipe"], "seed": seed}
    checkpoints = train_fold_models(development, folds, config, root, directory, device, study["epochs"])
    held_out_logits(development, folds, checkpoints, config, root, device).to_csv(directory / "oof_logits.csv", index=False)
    return directory


def read_held_out_logits(study: dict) -> dict[int, pd.DataFrame]:
    return {seed: pd.read_csv(seed_directory(study, seed) / "oof_logits.csv", float_precision="round_trip")
            for seed in study["seeds"]}


def fit_study(study: dict) -> dict:
    """Choose C by grouped cross-validation and fit the blend on all development images."""
    blend_path = runs_directory(study) / "blend.joblib"
    if blend_path.exists():
        raise FileExistsError(f"{blend_path} already exists: the blend is already fitted")
    table = development_table(read_held_out_logits(study), read_measurements(study))
    c = choose_regularisation(table, study["c_grid"], study["n_folds"], study["fold_seed"])
    joblib.dump(fit_blend(table, c), blend_path)
    summary = {"c": c, "n_development_images": len(table),
               **development_diagnostic(table, c, study["n_folds"], study["fold_seed"])}
    write_json(runs_directory(study) / "development.json", summary)
    return summary


# ---- freezing and the one-shot test scoring ---------------------------------------------------

def freeze_study(study: dict, manifest: Path = MANIFEST) -> dict:
    """Record the code version and the hash of every checkpoint, the blend, the measurements and the split."""
    runs = runs_directory(study)
    files = {checkpoint_name(seed, fold): seed_directory(study, seed) / f"fold{fold}.pt"
             for seed in study["seeds"] for fold in range(study["n_folds"])}
    files |= {"blend": runs / "blend.joblib", "measurements": REPO_ROOT / study["measurements"],
              "common_split": COMMON_SPLIT_DIR / "all.csv"}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Not finished: {missing}")
    return freeze(files, {"study": study}, manifest, CODE_PATHS)


def frozen_checkpoints(study: dict, record: dict) -> list[Path]:
    return [REPO_ROOT / record["files"][checkpoint_name(seed, fold)]["path"]
            for seed in study["seeds"] for fold in range(study["n_folds"])]


def mean_image_logits(partitions: dict[str, pd.DataFrame], checkpoints: list[Path], root: Path,
                      device: torch.device) -> dict[str, np.ndarray]:
    """Per test set, the logit of every image averaged over all networks (one network in memory at a time)."""
    totals = {name: np.zeros(len(rows)) for name, rows in partitions.items()}
    for path in checkpoints:
        model, checkpoint = restore_model(path, device)
        for name, rows in partitions.items():
            logits = predict_rows(model, rows, root, checkpoint["config"], device).set_index("image_id").logit
            totals[name] += logits.reindex(rows.image_id).to_numpy()
    return {name: total / len(checkpoints) for name, total in totals.items()}


def load_test_partitions(root: Path, splits_dir: Path) -> dict[str, pd.DataFrame]:
    partitions = load_common_split(splits_dir)
    selected = {name: partitions[name] for name in TEST_PARTITIONS}
    for rows in selected.values():
        check_images(root, rows)
    return selected


def score_test_sets(study: dict, out_dir: Path, manifest: Path = MANIFEST, root: Path = PREPARED_ROOT,
                    device: torch.device | None = None, splits_dir: Path = COMMON_SPLIT_DIR) -> None:
    """Score both test sets with the frozen networks and blend, and save one table per test set.
    Refuses to run twice."""
    device = device or pick_device()
    record = verify(manifest)
    out_dir.mkdir(parents=True)  # an existing folder means the test sets were already used
    blend = joblib.load(REPO_ROOT / record["files"]["blend"]["path"])
    measurements = read_measurements(study)
    partitions = load_test_partitions(root, splits_dir)
    image_logits = mean_image_logits(partitions, frozen_checkpoints(study, record), root, device)
    for name, rows in partitions.items():
        table = rows[["image_id", "binary_label", "analysis_group"]].merge(
            measurements[["image_id", *RATIOS]], on="image_id", how="left", validate="one_to_one")
        table["image_logit"] = image_logits[name]
        table["blend_logit"] = blend_logits(blend, table)
        table.rename(columns={"binary_label": "label"}).drop(columns=RATIOS).to_csv(out_dir / f"{name}.csv", index=False)


# ---- analysis (opens no image) -----------------------------------------------------------------

def compare_scores(label: str, partition: str, table: pd.DataFrame, first: str, second: str, study: dict) -> dict:
    result = paired_auc_difference(table.label, table[first], table[second], table.analysis_group,
                                   study["n_bootstrap"], study["bootstrap_seed"])
    return {"partition": partition, "comparison": label, **result}


def analyse_study(study: dict, scores_dir: Path, out: Path, baseline_a: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Blend minus image score (primary) and blend minus measurements-only (secondary), per test set."""
    out.mkdir(parents=True, exist_ok=True)
    primary, secondary = [], []
    for partition in TEST_PARTITIONS:
        scores = pd.read_csv(scores_dir / f"{partition}.csv")
        primary.append(compare_scores("blend minus image score", partition, scores, "blend_logit", "image_logit", study))
        measured = baseline_a[(baseline_a.split == partition) & (baseline_a.source == "predicted")]
        shared = scores.merge(measured[["image_id", "score"]], on="image_id")
        secondary.append(compare_scores("blend minus measurements_only", partition, shared, "blend_logit", "score", study))
    tables = {"primary": pd.DataFrame(primary), "secondary": pd.DataFrame(secondary)}
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    return tables


# ---- command line ------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["train", "job", "fit", "freeze", "score", "analyse"])
    parser.add_argument("index", nargs="?", type=int, help="array task number (job only)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--scores", type=Path)
    args = parser.parse_args()
    study = load_study()
    if args.action == "job":
        print(study["seeds"][args.index])
    elif args.action == "train":
        print("Run folder:", train_seed(study, args.seed))
    elif args.action == "fit":
        print(json.dumps(fit_study(study), indent=2))
    elif args.action == "freeze":
        print(freeze_study(study)["git_commit"])
    elif args.action == "score":
        score_test_sets(study, args.out or runs_directory(study) / "final")
    else:
        baseline = pd.read_csv(REPO_ROOT / study["baseline_a_scores"])
        print(analyse_study(study, args.scores, args.out, baseline)["primary"].to_string(index=False))


if __name__ == "__main__":
    main()

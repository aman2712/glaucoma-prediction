"""Develop one image model: audit the data, train, and report on the validation split.

Usage: python -m glaucoma_project.image_model.develop [--config PATH] [--seed N] [--measurements FILE]
A config with ``measurement_columns`` also feeds those measurements (from ``--measurements``) to the
model. ``--seed`` overrides the config's seed and adds it to the run name, so each seed gets its own
run folder. Test images are never opened here; the final stage does that once, after freezing.
"""
import argparse
import importlib.metadata
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

from glaucoma_project import COMMON_SPLIT_DIR, PREPARED_ROOT, REPO_ROOT
from glaucoma_project.data.audit import audit_export
from glaucoma_project.data.classification import check_images
from glaucoma_project.data.common_split import check_common_split, load_common_split
from glaucoma_project.data.measurement_inputs import apply_scaler, attach_measurements, fit_scaler
from glaucoma_project.evaluation.screening import select_threshold
from glaucoma_project.files import read_table, write_json
from glaucoma_project.freeze import sha256_file
from glaucoma_project.image_model.plots import plot_learning_curves, plot_training_examples
from glaucoma_project.image_model.predict import predict_rows
from glaucoma_project.image_model.report import save_report
from glaucoma_project.image_model.train import fit
from glaucoma_project.runtime import git_commit, pick_device

RUNS_DIR = REPO_ROOT / "runs" / "image_model"
PACKAGES = ("torch", "torchvision", "numpy", "pandas", "scikit-learn", "Pillow", "matplotlib")


def develop(config: dict, root: Path = PREPARED_ROOT, runs_dir: Path = RUNS_DIR,
            splits_dir: Path = COMMON_SPLIT_DIR, measurements: Path | None = None) -> Path:
    """Run the whole development stage into ``runs_dir/<run_name>`` and return that folder.
    Refuses to reuse an existing run folder."""
    columns = config.get("measurement_columns", [])
    if bool(columns) != (measurements is not None):
        raise ValueError("measurement_columns and a measurements file go together")
    run_dir = runs_dir / config["run_name"]
    run_dir.mkdir(parents=True)
    write_json(run_dir / "config.json", config)
    device = pick_device()
    write_json(run_dir / "run.json", {
        "git_commit": git_commit(), "device": str(device), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "python": sys.version, "packages": {name: importlib.metadata.version(name) for name in PACKAGES}})

    _, summary = audit_export(root)
    check_common_split(root, splits_dir)
    write_json(run_dir / "input_audit.json", {
        "export_summary": summary, "common_split_sha256": sha256_file(splits_dir / "all.csv"),
        "measurements_sha256": sha256_file(measurements) if measurements else None,
        "patient_disjoint_claim": False})

    partitions = load_common_split(splits_dir)
    train_rows, val_rows = partitions["train"], partitions["validation"]
    scaler = None
    if columns:
        table = read_table(measurements)
        train_rows, val_rows = (attach_measurements(rows, table, columns) for rows in (train_rows, val_rows))
        scaler = fit_scaler(train_rows, columns)
        train_rows, val_rows = apply_scaler(train_rows, scaler), apply_scaler(val_rows, scaler)
    check_images(root, train_rows)
    check_images(root, val_rows)
    plot_training_examples(train_rows, root, run_dir / "training_examples.png", seed=config["seed"])
    model, _, history = fit(train_rows, val_rows, root, run_dir, config, device, scaler)
    plot_learning_curves(history, run_dir / "learning_curves.png")

    predictions = predict_rows(model, val_rows, root, config, device)
    threshold = select_threshold(predictions.label, predictions.probability, config["target_sensitivity"])
    predictions["predicted_at_validation_threshold"] = (predictions.probability >= threshold).astype(int)
    save_report("validation", predictions, threshold, root, run_dir)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "image_model.yaml")
    parser.add_argument("--seed", type=int, help="overrides the config's seed and is added to the run name")
    parser.add_argument("--measurements", type=Path, help="measurements file, for configs with measurement_columns")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    if args.seed is not None:
        settings.update(seed=args.seed, run_name=f"{settings['run_name']}_seed{args.seed}")
    print("Run folder:", develop(settings, runs_dir=args.runs_dir, measurements=args.measurements))

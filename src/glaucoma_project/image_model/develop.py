"""Develop the image-only model: audit the data, train, and report on the validation split.

Usage: python -m glaucoma_project.image_model.develop [--config PATH]
Test images are never opened here; ``final.py`` does that once, after freezing.
"""
import argparse
import importlib.metadata
import sys
from pathlib import Path

import torch
import yaml

from glaucoma_project import PREPARED_ROOT, REPO_ROOT
from glaucoma_project.data.audit import audit_export
from glaucoma_project.data.classification import check_images
from glaucoma_project.data.effective_splits import effective_split, save_effective_split
from glaucoma_project.evaluation.screening import select_threshold
from glaucoma_project.files import write_json
from glaucoma_project.freeze import sha256_file
from glaucoma_project.image_model.plots import plot_learning_curves, plot_training_examples
from glaucoma_project.image_model.predict import predict_rows
from glaucoma_project.image_model.report import save_report
from glaucoma_project.image_model.train import fit
from glaucoma_project.runtime import git_commit, pick_device

RUNS_DIR = REPO_ROOT / "runs" / "image_model"
PACKAGES = ("torch", "torchvision", "numpy", "pandas", "scikit-learn", "Pillow", "matplotlib")


def develop(config: dict, root: Path = PREPARED_ROOT, runs_dir: Path = RUNS_DIR) -> Path:
    """Run the whole development stage into ``runs_dir/<run_name>`` and return that folder.
    Refuses to reuse an existing run folder."""
    run_dir = runs_dir / config["run_name"]
    run_dir.mkdir(parents=True)
    write_json(run_dir / "config.json", config)
    device = pick_device()
    write_json(run_dir / "run.json", {
        "git_commit": git_commit(), "device": str(device), "cuda": torch.version.cuda,
        "python": sys.version, "packages": {name: importlib.metadata.version(name) for name in PACKAGES}})

    table, summary = audit_export(root)
    effective, partitions = effective_split(table)
    changes = save_effective_split(effective, partitions, run_dir / "effective_splits")
    write_json(run_dir / "input_audit.json", {
        "original_summary": summary, "original_root": root, "changed_rows": len(changes),
        "effective_manifest_sha256": sha256_file(run_dir / "effective_splits" / "all.csv"),
        "possible_filename_grouping": "same Bosch P-number, irrespective of source release; unverified precaution",
        "patient_disjoint_claim": False})

    train_rows, val_rows = partitions["train"], partitions["validation"]
    check_images(root, train_rows)
    check_images(root, val_rows)
    plot_training_examples(train_rows, root, run_dir / "training_examples.png", seed=config["seed"])
    model, _, history = fit(train_rows, val_rows, root, run_dir, config, device)
    plot_learning_curves(history, run_dir / "learning_curves.png")

    predictions = predict_rows(model, val_rows, root, config, device)
    threshold = select_threshold(predictions.label, predictions.probability, config["target_sensitivity"])
    predictions["predicted_at_validation_threshold"] = (predictions.probability >= threshold).astype(int)
    save_report("validation", predictions, threshold, root, run_dir)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "image_model.yaml")
    args = parser.parse_args()
    print("Run folder:", develop(yaml.safe_load(args.config.read_text())))

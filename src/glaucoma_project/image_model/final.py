"""Freeze the developed image-only model, then evaluate it once on the two test sets.

Usage: python -m glaucoma_project.image_model.final --run-dir RUN [--check-only]
The checkpoint, effective split and threshold are hashed first and checked again before any
test image is opened. The evaluation writes ``final_evaluation/`` once and refuses to overwrite it.
"""
import argparse
import json
import math
import warnings
from pathlib import Path

import pandas as pd
import torch

from glaucoma_project import PREPARED_ROOT
from glaucoma_project.data.classification import check_images
from glaucoma_project.data.effective_splits import partition_rows
from glaucoma_project.evaluation.bootstrap import grouped_bootstrap
from glaucoma_project.files import read_table, write_json
from glaucoma_project.freeze import sha256_file
from glaucoma_project.image_model.model import UNTRAINED, restore_model
from glaucoma_project.image_model.predict import operating_threshold, predict_rows
from glaucoma_project.image_model.report import save_report
from glaucoma_project.runtime import pick_device

TEST_PARTITIONS = ("in_domain_test", "external_test")
SPLIT_FILE = Path("effective_splits") / "all.csv"


def _load_checkpoint(run_dir: Path) -> dict:
    return torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)


def freeze_run(run_dir: Path) -> dict:
    """Record the checkpoint hash, effective-split hash and threshold in ``frozen_protocol.json``.
    Calling it again is fine only if nothing changed; a changed run must be a new run."""
    checkpoint = _load_checkpoint(run_dir)
    record = {
        "checkpoint_sha256": sha256_file(run_dir / "best.pt"),
        "effective_manifest_sha256": sha256_file(run_dir / SPLIT_FILE),
        "threshold": operating_threshold(run_dir), "threshold_source": "CHAKSU validation only",
        "target_validation_sensitivity": checkpoint["config"]["target_sensitivity"],
        "model_name": checkpoint["config"]["model_name"], "weights": checkpoint["weights"],
        "best_epoch": checkpoint["best_epoch"],
        "label_definition": "CHAKSU normal vs glaucoma suspect; PAPILA healthy vs clinical glaucoma",
        "probability_note": "Uncalibrated sigmoid score; not a validated clinical disease probability"}
    path = run_dir / "frozen_protocol.json"
    if not path.exists():
        write_json(path, record)
    elif json.loads(path.read_text()) != record:
        raise ValueError("The frozen run changed. Start a new development run; never tune on test data")
    return record


def verify_frozen(run_dir: Path) -> dict:
    """Return the frozen record, raising if the checkpoint or effective split changed since."""
    record = json.loads((run_dir / "frozen_protocol.json").read_text())
    if sha256_file(run_dir / "best.pt") != record["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after freezing")
    if sha256_file(run_dir / SPLIT_FILE) != record["effective_manifest_sha256"]:
        raise ValueError("Effective split changed after freezing")
    return record


def check_saved_run(run_dir: Path) -> None:
    """Consistency checks on a finished development run, before freezing. Touches no test data."""
    if (run_dir / "final_evaluation").exists():
        raise ValueError("final_evaluation already exists: inspect it, never overwrite or delete it")
    audit = json.loads((run_dir / "input_audit.json").read_text())
    if sha256_file(run_dir / SPLIT_FILE) != audit["effective_manifest_sha256"]:
        raise ValueError("The effective split differs from the one used for training")
    if json.loads((run_dir / "config.json").read_text()) != _load_checkpoint(run_dir)["config"]:
        raise ValueError("config.json differs from the checkpoint's configuration")
    saved = pd.read_csv(run_dir / "validation_metrics.csv", float_precision="round_trip")
    saved = saved.loc[saved.operating_point == "validation_selected", "threshold"]
    if len(saved) != 1 or not math.isclose(saved.iloc[0], operating_threshold(run_dir), abs_tol=1e-12):
        raise ValueError("The saved threshold does not match the saved validation predictions")


def final_evaluation(run_dir: Path, device: torch.device, root: Path = PREPARED_ROOT) -> pd.DataFrame:
    """Evaluate the frozen model once on the CHAKSU test split and PAPILA, using only the frozen
    checkpoint, threshold and effective split. Returns the overall metrics of both."""
    protocol = verify_frozen(run_dir)
    model, checkpoint = restore_model(run_dir / "best.pt", device)
    config = checkpoint["config"]
    if checkpoint["weights"] == UNTRAINED:
        warnings.warn("Software test only: this model has no pretrained weights")
    table = read_table(run_dir / SPLIT_FILE)
    out_dir = run_dir / "final_evaluation"
    out_dir.mkdir()  # a leftover folder means an interrupted run: inspect it, do not overwrite
    results = []
    for name in TEST_PARTITIONS:
        rows = partition_rows(table, name)
        check_images(root, rows)
        predictions = predict_rows(model, rows, root, config, device)
        results.append(save_report(name, predictions, protocol["threshold"], root, out_dir))
        grouped_bootstrap(predictions, protocol["threshold"], config["seed"], config["n_bootstrap"]).to_csv(
            out_dir / f"{name}_confidence_intervals.csv", index=False)
    comparison = pd.concat(results, ignore_index=True)
    comparison.to_csv(out_dir / "comparison.csv", index=False)
    write_json(out_dir / "completion.json", {"protocol": protocol, "complete": True})
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true",
                        help="check the saved run, without freezing or touching any test image")
    args = parser.parse_args()
    check_saved_run(args.run_dir)
    print("Saved run is consistent; threshold", operating_threshold(args.run_dir))
    if not args.check_only:
        freeze_run(args.run_dir)
        print(final_evaluation(args.run_dir, pick_device()).to_string(index=False))

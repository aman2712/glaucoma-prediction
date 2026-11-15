"""Experiment X5: a segmentation network trained with scale and colour augmentation and adapted to
each new set of photographs (batch-norm re-estimation), against the Baseline A network.

Usage (the design is in docs/experiment_log.md, "X-series design" and the X5 clarification):
  python -m glaucoma_project.segmentation.robust_study CONFIG gate-job N       "arm seed" of gate job N
  python -m glaucoma_project.segmentation.robust_study CONFIG gate-run ARM --seed S   one gate training
  python -m glaucoma_project.segmentation.robust_study CONFIG gate-verdict      development-only verdict

The gate trains on one camera of the development data and measures Dice on the other cameras, a stand-in
for a new camera. A variant that fails it is never scored on a test set.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from glaucoma_project import COMMON_SPLIT_DIR, PREPARED_ROOT
from glaucoma_project.combined_model.variant_study import GateRows, arm_config, gate_jobs, load_study, runs_directory
from glaucoma_project.data.segmentation import make_loader, segmentation_rows
from glaucoma_project.files import write_json
from glaucoma_project.runtime import environment_record, pick_device, set_seed
from glaucoma_project.segmentation.adapt import adapt_batchnorm
from glaucoma_project.segmentation.model import DiscCupUNet
from glaucoma_project.segmentation.predict import load_model
from glaucoma_project.segmentation.train import fit, validate

__all__ = ["gate_jobs", "gate_rows", "gate_verdict", "load_study", "run_gate", "runs_directory"]


def gate_rows(source_camera: str, splits_dir: Path = COMMON_SPLIT_DIR) -> GateRows:
    """Segmentation images of the development data only; no test image is ever listed."""
    train, validation = segmentation_rows("train", splits_dir), segmentation_rows("validation", splits_dir)
    on_source = lambda rows: rows[rows.camera == source_camera]
    development = pd.concat([train, validation], ignore_index=True)
    return GateRows(on_source(train), on_source(validation), development[development.camera != source_camera])


def dice_on(model: DiscCupUNet, rows: pd.DataFrame, root: Path, config: dict, device: torch.device,
            adapt: bool) -> dict:
    """Mean Dice of disc and cup on ``rows``; with ``adapt`` the batch-norm statistics are first
    re-estimated on the same rows' photographs (labels and masks are not used for that)."""
    loader = make_loader(None, config["batch_size"], root, config["num_workers"], training=False, rows=rows)
    if adapt:
        model = adapt_batchnorm(model, loader, device)
    scores = validate(model, loader, device)
    return {"disc": scores["val_dice_disc"], "cup": scores["val_dice_cup"]}


def run_gate(study: dict, arm: str, seed: int, gate: GateRows, root: Path, device: torch.device | None = None) -> Path:
    """Train one arm on the source camera and write its Dice scores. Never reuses a folder."""
    device = device or pick_device()
    config = arm_config(study, arm, seed)
    folder = runs_directory(study) / "gate" / f"{arm}_seed{seed}"
    folder.mkdir(parents=True)
    write_json(folder / "run.json", environment_record(device))
    set_seed(seed)
    profile, workers = config["augment_profile"], config["num_workers"]
    train_loader = make_loader(None, config["batch_size"], root, workers, training=True, profile=profile, rows=gate.train)
    validation_loader = make_loader(None, config["batch_size"], root, workers, training=False, rows=gate.validation)
    fit(DiscCupUNet(config["encoder"], config["pretrained"]).to(device), train_loader, validation_loader,
        folder, config, device)
    model = load_model(folder, config["encoder"], device)
    adapt = config["adapt"]
    scores = {"remidio_validation": dice_on(model, gate.validation, root, config, device, adapt),
              "shifted": dice_on(model, gate.shifted, root, config, device, adapt)}
    if adapt:  # descriptive only: what the augmentation alone achieves
        scores["shifted_without_adabn"] = dice_on(model, gate.shifted, root, config, device, adapt=False)
    write_json(folder / "scores.json", scores)
    return folder


def _mean_dice(study: dict, arm: str, key: str) -> float:
    """Mean over the gate seeds of the mean of disc and cup Dice."""
    values = []
    for seed in study["gate_seeds"]:
        scores = json.loads((runs_directory(study) / "gate" / f"{arm}_seed{seed}" / "scores.json").read_text())[key]
        values.append((scores["disc"] + scores["cup"]) / 2)
    return sum(values) / len(values)


def gate_verdict(study: dict) -> dict:
    """The variant passes if its Dice on the other cameras is higher than the baseline's and its Dice on
    the source camera's validation images is at most ``max_remidio_loss`` lower."""
    verdict = {arm: {"shifted_dice": _mean_dice(study, arm, "shifted"),
                     "remidio_validation_dice": _mean_dice(study, arm, "remidio_validation")}
               for arm in ("variant", "baseline")}
    first = runs_directory(study) / "gate" / f"variant_seed{study['gate_seeds'][0]}" / "scores.json"
    if "shifted_without_adabn" in json.loads(first.read_text()):
        verdict["variant"]["shifted_dice_without_adabn"] = _mean_dice(study, "variant", "shifted_without_adabn")
    verdict["passes"] = bool(
        verdict["variant"]["shifted_dice"] > verdict["baseline"]["shifted_dice"]
        and verdict["baseline"]["remidio_validation_dice"] - verdict["variant"]["remidio_validation_dice"]
        <= study["max_remidio_loss"])
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", type=Path)
    parser.add_argument("action", choices=["gate-job", "gate-run", "gate-verdict"])
    parser.add_argument("target", nargs="?", help="job number (gate-job) or arm (gate-run)")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    study = load_study(args.study)
    if args.action == "gate-job":
        print(*gate_jobs(study)[int(args.target)])
    elif args.action == "gate-run":
        print("Run folder:", run_gate(study, args.target, args.seed, gate_rows(study["source_camera"]), PREPARED_ROOT))
    else:
        verdict = gate_verdict(study)
        write_json(runs_directory(study) / "gate" / "verdict.json", verdict)
        print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()

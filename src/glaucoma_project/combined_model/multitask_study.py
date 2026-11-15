"""Experiment X6: one network that draws the disc and cup and classifies glaucoma, against the P01
image-only model.

Usage, in order (the design is in docs/experiment_log.md, "X-series design" and the X6 clarification):
  python -m glaucoma_project.combined_model.multitask_study CONFIG gate-run --seed S   one gate training
  (gate-job and gate-verdict are those of combined_model.variant_study; the baseline gate scores are copied)
  python -m glaucoma_project.combined_model.multitask_study CONFIG train --seed S      one final training
  python -m glaucoma_project.combined_model.multitask_study CONFIG freeze
  python -m glaucoma_project.combined_model.multitask_study CONFIG score --out DIR      both test sets, once
  (analyse is that of combined_model.variant_study)

Only images that have both a glaucoma label and expert masks are used for training. Validation and test
images need no masks.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from glaucoma_project import COMMON_SPLIT_DIR, PREPARED_ROOT, REPO_ROOT
from glaucoma_project.combined_model.variant_study import (TEST_PARTITIONS, GateRows, arm_config, checkpoint_name,
                                                           checkpoint_path, gate_rows, load_study, manifest_path,
                                                           runs_directory)
from glaucoma_project.data.common_split import load_common_split
from glaucoma_project.data.segmentation import PhotographDataset, SegmentationDataset
from glaucoma_project.files import write_json
from glaucoma_project.freeze import freeze, verify
from glaucoma_project.image_model.predict import attach_metadata
from glaucoma_project.runtime import environment_record, pick_device, set_seed
from glaucoma_project.segmentation.model import DiscCupUNet
from glaucoma_project.segmentation.multitask import fit_multitask, predict_logits

CODE_PATHS = [f"src/glaucoma_project/{name}" for name in
              ("combined_model/multitask_study.py", "combined_model/variant_study.py", "combined_model/analysis.py",
               "segmentation", "image_model/predict.py", "data", "evaluation", "runtime.py", "files.py", "freeze.py")]


def with_masks(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows that have expert disc and cup masks (the only ones the multi-task loss can use)."""
    return rows[pd.to_numeric(rows.segmentation_eligible) == 1]


def _loaders(train_rows: pd.DataFrame, validation_rows: pd.DataFrame, config: dict, root: Path) -> tuple[DataLoader, DataLoader]:
    train = DataLoader(SegmentationDataset(None, root, augment=True, rows=with_masks(train_rows)),
                       batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    validation = DataLoader(PhotographDataset(validation_rows, root), batch_size=config["batch_size"],
                            num_workers=config["num_workers"])
    return train, validation


def _train(config: dict, train_rows: pd.DataFrame, validation_rows: pd.DataFrame, folder: Path, root: Path,
           device: torch.device) -> DiscCupUNet:
    """Train one network into ``folder`` (new) and return it with the weights of its best epoch."""
    folder.mkdir(parents=True)
    write_json(folder / "run.json", environment_record(device))
    set_seed(config["seed"])
    model = DiscCupUNet(config["encoder"], config["pretrained"], classify=True).to(device)
    fit_multitask(model, *_loaders(train_rows, validation_rows, config, root), folder, config, device)
    return restore(folder / "best.pt", config, device)


def restore(path: Path, config: dict, device: torch.device) -> DiscCupUNet:
    model = DiscCupUNet(config["encoder"], pretrained=False, classify=True)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval()


def score_rows(model: DiscCupUNet, rows: pd.DataFrame, config: dict, root: Path, device: torch.device) -> pd.DataFrame:
    """Glaucoma logit and probability of every row, with the row's metadata, as the image models give them."""
    loader = DataLoader(PhotographDataset(rows, root), batch_size=config["batch_size"], num_workers=config["num_workers"])
    predictions = predict_logits(model, loader, device)
    return attach_metadata(predictions.assign(probability=torch.sigmoid(torch.tensor(predictions.logit.values)).numpy()), rows)


def run_gate(study: dict, seed: int, gate: GateRows, root: Path, device: torch.device | None = None) -> Path:
    """Train the variant on the source camera and write its scores for the other cameras. Never reuses a folder.
    The baseline arm of the gate is the P01 image-only recipe, whose gate scores already exist."""
    device = device or pick_device()
    config = arm_config(study, "variant", seed)
    folder = runs_directory(study) / "gate" / f"variant_seed{seed}"
    model = _train(config, gate.train, gate.validation, folder, root, device)
    score_rows(model, gate.shifted, config, root, device)[["image_id", "label", "logit"]].to_csv(folder / "scores.csv", index=False)
    return folder


def train_seed(study: dict, seed: int, root: Path = PREPARED_ROOT, device: torch.device | None = None,
               splits_dir: Path = COMMON_SPLIT_DIR) -> Path:
    """Train one final network on CHAKSU train, picking its epoch on CHAKSU validation."""
    partitions = load_common_split(splits_dir)
    config = arm_config(study, "variant", seed)
    folder = runs_directory(study) / config["run_name"]
    _train(config, partitions["train"], partitions["validation"], folder, root, device or pick_device())
    return folder


def freeze_study(study: dict) -> dict:
    """Record the code version and the hash of every checkpoint and of the common split."""
    files = {checkpoint_name(seed): checkpoint_path(study, seed) for seed in study["seeds"]}
    files["common_split"] = COMMON_SPLIT_DIR / "all.csv"
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Not finished: {missing}")
    configs = [f"configs/{study['name']}.yaml", study["variant_config"], study["baseline_config"]]
    return freeze(files, {"study": study}, manifest_path(study), CODE_PATHS + configs)


def score_test_sets(study: dict, out_dir: Path, root: Path = PREPARED_ROOT, device: torch.device | None = None,
                    splits_dir: Path = COMMON_SPLIT_DIR, manifest: Path | None = None) -> None:
    """Score both test sets with every frozen checkpoint, one ``<run>_<partition>.csv`` per pair. Refuses to run twice."""
    device = device or pick_device()
    record = verify(manifest or manifest_path(study))
    out_dir.mkdir(parents=True)  # an existing folder means the test sets were already used
    partitions = load_common_split(splits_dir)
    for seed in study["seeds"]:
        config = arm_config(study, "variant", seed)
        model = restore(REPO_ROOT / record["files"][checkpoint_name(seed)]["path"], config, device)
        for partition in TEST_PARTITIONS:
            score_rows(model, partitions[partition], config, root, device).assign(seed=seed).to_csv(
                out_dir / f"{config['run_name']}_{partition}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", type=Path)
    parser.add_argument("action", choices=["gate-run", "train", "freeze", "score"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    study = load_study(args.study)
    if args.action == "gate-run":
        print("Run folder:", run_gate(study, args.seed, gate_rows(load_common_split(), study["source_camera"]), PREPARED_ROOT))
    elif args.action == "train":
        print("Run folder:", train_seed(study, args.seed))
    elif args.action == "freeze":
        print(freeze_study(study)["git_commit"])
    else:
        score_test_sets(study, args.out or runs_directory(study) / "final")


if __name__ == "__main__":
    main()

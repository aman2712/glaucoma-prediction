"""Experiment X5, after the gate: train the robust segmentation network on CHAKSU train, fit the
Baseline A classifier on its measurements, freeze both, and score the test sets once.

Usage (the design is in docs/experiment_log.md, "X5 clarification"):
  python -m glaucoma_project.segmentation.robust_final CONFIG train             robust U-Net on CHAKSU train
  python -m glaucoma_project.segmentation.robust_final CONFIG fit-classifier    classifier on development measurements
  python -m glaucoma_project.segmentation.robust_final CONFIG freeze            records hashes and the code version
  python -m glaucoma_project.segmentation.robust_final CONFIG score --out DIR   both test sets, once
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from glaucoma_project import COMMON_SPLIT_DIR, PREPARED_ROOT, REPO_ROOT
from glaucoma_project.classification.baseline import labelled, run_baseline
from glaucoma_project.combined_model.variant_study import arm_config, load_study, runs_directory
from glaucoma_project.data.segmentation import make_loader, segmentation_rows
from glaucoma_project.evaluation.classification import evaluate
from glaucoma_project.evaluation.compare import paired_auc_difference
from glaucoma_project.evaluation.metrics import agreement, dice
from glaucoma_project.files import write_json
from glaucoma_project.freeze import freeze, verify
from glaucoma_project.measurements.cdr import measure
from glaucoma_project.measurements.extract import COLUMNS
from glaucoma_project.runtime import environment_record, pick_device, set_seed
from glaucoma_project.segmentation.adapt import adapt_batchnorm
from glaucoma_project.segmentation.model import DiscCupUNet
from glaucoma_project.segmentation.predict import load_model, predict_masks
from glaucoma_project.segmentation.train import fit

DEVELOPMENT_SPLITS = ("train", "validation")
TEST_SPLITS = ("in_domain_test", "external_test")
CODE_PATHS = [f"src/glaucoma_project/{name}" for name in
              ("segmentation", "data", "measurements", "classification", "evaluation", "runtime.py", "files.py",
               "freeze.py", "combined_model/variant_study.py")]


def model_directory(study: dict) -> Path:
    return runs_directory(study) / "final"


def _measure_pair(disc_and_cup: torch.Tensor) -> dict:
    return measure(disc_and_cup[0].numpy(), disc_and_cup[1].numpy())


def measure_rows(model: DiscCupUNet, rows: pd.DataFrame, root: Path, config: dict, device: torch.device,
                 adapt: bool) -> pd.DataFrame:
    """Per image: predicted ratios (vcdr, hcdr, area_cdr, areas), the same ratios from the expert masks
    (``expert_`` prefix) and the Dice of disc and cup. With ``adapt`` the batch-norm statistics are first
    re-estimated on these rows' photographs; no label or mask is used for that."""
    loader = make_loader(None, config["batch_size"], root, config["num_workers"], training=False, rows=rows)
    if adapt:
        model = adapt_batchnorm(model, loader, device)
    records = []
    for batch in loader:
        predicted, expert = predict_masks(model, batch["image"]), batch["mask"]
        overlap = dice(predicted, expert)
        for i, image_id in enumerate(batch["image_id"]):
            records.append({"image_id": image_id, **_measure_pair(predicted[i]),
                            **{f"expert_{k}": v for k, v in _measure_pair(expert[i]).items()},
                            "dice_disc": overlap[i, 0].item(), "dice_cup": overlap[i, 1].item()})
    return rows[[*COLUMNS, "split"]].merge(pd.DataFrame(records), on="image_id", validate="one_to_one")


def fit_classifier(model: DiscCupUNet, rows: pd.DataFrame, root: Path, config: dict, device: torch.device,
                   classifier_config: dict) -> dict:
    """The Baseline A recipe (grouped cross-validated C, Youden threshold) on the measurements this network
    makes of the development images, each split adapted on its own photographs."""
    measurements = pd.concat([measure_rows(model, rows[rows.split == split], root, config, device, adapt=True)
                              for split in DEVELOPMENT_SPLITS], ignore_index=True).assign(source="predicted")
    result = run_baseline(measurements, "predicted", classifier_config)
    return {"model": result.model, "threshold": result.threshold, "best_c": result.best_c}


def compare_with_baseline_a(measurements: pd.DataFrame, score: np.ndarray, baseline: pd.DataFrame,
                            n_bootstrap: int, seed: int) -> pd.DataFrame:
    """AUC of ``score`` minus the frozen Baseline A score of the same image, per split, paired bootstrap over
    analysis groups. ``measurements`` holds only labelled images and ``score`` is aligned with it."""
    table = measurements.assign(score=score)
    rows = []
    for split, part in table.groupby("split", sort=False):
        reference = baseline[(baseline.split == split) & (baseline.source == "predicted")]
        merged = part.merge(reference[["image_id", "score"]], on="image_id", suffixes=("", "_baseline"), how="left")
        if merged.score_baseline.isna().any():
            raise ValueError(f"A baseline score is missing for images of {split}")
        rows.append({"split": split, "n": len(merged), **paired_auc_difference(
            merged.binary_label, merged.score, merged.score_baseline, merged.analysis_group, n_bootstrap, seed)})
    return pd.DataFrame(rows)


def train_final(study: dict, root: Path = PREPARED_ROOT, device: torch.device | None = None,
                splits_dir: Path = COMMON_SPLIT_DIR) -> Path:
    """Train the robust network on CHAKSU train, picking the epoch on CHAKSU validation. Never reuses a folder."""
    device = device or pick_device()
    config = arm_config(study, "variant", study["final_seed"])
    folder = model_directory(study)
    folder.mkdir(parents=True)
    write_json(folder / "run.json", environment_record(device))
    set_seed(config["seed"])
    train_loader = make_loader("train", config["batch_size"], root, config["num_workers"], training=True,
                               profile=config["augment_profile"], splits_dir=splits_dir)
    validation_loader = make_loader("validation", config["batch_size"], root, config["num_workers"],
                                    splits_dir=splits_dir)
    fit(DiscCupUNet(config["encoder"], config["pretrained"]).to(device), train_loader, validation_loader,
        folder, config, device)
    return folder


def _development_rows(splits_dir: Path) -> pd.DataFrame:
    return pd.concat([segmentation_rows(split, splits_dir) for split in DEVELOPMENT_SPLITS], ignore_index=True)


def fit_final_classifier(study: dict, root: Path = PREPARED_ROOT, device: torch.device | None = None,
                         splits_dir: Path = COMMON_SPLIT_DIR) -> Path:
    device = device or pick_device()
    config = arm_config(study, "variant", study["final_seed"])
    path = model_directory(study) / "classifier.joblib"
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    classifier_config = yaml.safe_load((REPO_ROOT / study["classifier_config"]).read_text())
    model = load_model(model_directory(study), config["encoder"], device)
    joblib.dump(fit_classifier(model, _development_rows(splits_dir), root, config, device, classifier_config), path)
    return path


def freeze_final(study: dict, manifest: Path | None = None) -> dict:
    """Record the code version and the hash of the weights, the classifier, the common split and the
    Baseline A scores that the comparison uses."""
    folder = model_directory(study)
    files = {"segmentation_weights": folder / "best.pt", "classifier": folder / "classifier.joblib",
             "common_split": COMMON_SPLIT_DIR / "all.csv", "baseline_a_scores": REPO_ROOT / study["baseline_scores"]}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Not finished: {missing}")
    configs = [f"configs/{study['name']}.yaml", study["variant_config"], study["classifier_config"]]
    return freeze(files, {"study": study}, REPO_ROOT / "docs" / f"frozen_{study['name']}.json", CODE_PATHS + configs)


def score_test_sets(study: dict, out_dir: Path, root: Path = PREPARED_ROOT, device: torch.device | None = None,
                    splits_dir: Path = COMMON_SPLIT_DIR, manifest: Path | None = None) -> dict:
    """Score both test sets once with the frozen pipeline, each adapted on its own photographs. Refuses to run twice."""
    device = device or pick_device()
    record = verify(manifest or REPO_ROOT / "docs" / f"frozen_{study['name']}.json")
    out_dir.mkdir(parents=True)  # an existing folder means the test sets were already used
    config = arm_config(study, "variant", study["final_seed"])
    path = lambda name: REPO_ROOT / record["files"][name]["path"]
    model = load_model(path("segmentation_weights").parent, config["encoder"], device)
    classifier, baseline = joblib.load(path("classifier")), pd.read_csv(path("baseline_a_scores"))
    features = yaml.safe_load((REPO_ROOT / study["classifier_config"]).read_text())["features"]
    measurements = {adapt: pd.concat([measure_rows(model, segmentation_rows(split, splits_dir), root, config, device, adapt)
                                      for split in TEST_SPLITS], ignore_index=True) for adapt in (True, False)}
    main = labelled(measurements[True])
    score = classifier["model"].predict_proba(main[features])[:, 1]
    n, seed = study["n_bootstrap"], study["bootstrap_seed"]
    comparison = compare_with_baseline_a(main, score, baseline, n, seed)
    classification = pd.concat([evaluate(part.binary_label, score[main.split == split], classifier["threshold"], n, seed)
                                .assign(split=split).reset_index(names="metric")
                                for split, part in main.groupby("split", sort=False)], ignore_index=True)
    segmentation = pd.DataFrame([{"split": split, "adapted": adapt, "dice_disc": part.dice_disc.mean(),
                                  "dice_cup": part.dice_cup.mean(),
                                  **{f"vcdr_{k}": v for k, v in agreement(part.vcdr, part.expert_vcdr).items() if k != "n"}}
                                 for adapt, table in measurements.items() for split, part in table.groupby("split", sort=False)])
    for name, table in {"comparison": comparison, "classification": classification, "segmentation": segmentation,
                        "measurements_test": measurements[True]}.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    main.assign(score=score)[["image_id", "split", "binary_label", "score"]].to_csv(out_dir / "scores.csv", index=False)
    return {"comparison": comparison, "segmentation": segmentation}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", type=Path)
    parser.add_argument("action", choices=["train", "fit-classifier", "freeze", "score"])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    study = load_study(args.study)
    if args.action == "train":
        print("Run folder:", train_final(study))
    elif args.action == "fit-classifier":
        print("Classifier:", fit_final_classifier(study))
    elif args.action == "freeze":
        print(freeze_final(study)["git_commit"])
    else:
        results = score_test_sets(study, args.out or runs_directory(study) / "test")
        print(results["comparison"].to_string(index=False))


if __name__ == "__main__":
    main()

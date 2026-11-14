"""A study of one image-model variant against the P01 image-only model (experiments X2, X3, X4).

Usage, in order (the design is in docs/experiment_log.md, "X-series design"):
  python -m glaucoma_project.combined_model.variant_study CONFIG gate-job N    "arm seed" of gate job N
  python -m glaucoma_project.combined_model.variant_study CONFIG gate-run ARM --seed S   one gate training
  python -m glaucoma_project.combined_model.variant_study CONFIG gate-verdict  development-only verdict
  python -m glaucoma_project.combined_model.variant_study CONFIG job N         seed of training job N
  python -m glaucoma_project.combined_model.variant_study CONFIG train --seed S
  python -m glaucoma_project.combined_model.variant_study CONFIG freeze        records checkpoints and input hashes
  python -m glaucoma_project.combined_model.variant_study CONFIG score --out DIR   scores both test sets, once
  python -m glaucoma_project.combined_model.variant_study CONFIG analyse --scores DIR --out DIR

The gate trains on one camera of the development data and scores the others, a stand-in for a new
camera. A variant that fails it is never scored on a test set.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score

from glaucoma_project import COMMON_SPLIT_DIR, PREPARED_ROOT, REPO_ROOT
from glaucoma_project.combined_model.analysis import ensemble
from glaucoma_project.data.classification import check_images
from glaucoma_project.data.common_split import load_common_split
from glaucoma_project.evaluation.compare import paired_auc_difference
from glaucoma_project.files import write_json
from glaucoma_project.freeze import freeze, verify
from glaucoma_project.image_model.develop import develop
from glaucoma_project.image_model.model import restore_model
from glaucoma_project.image_model.predict import predict_rows
from glaucoma_project.image_model.train import fit
from glaucoma_project.runtime import environment_record, pick_device

TEST_PARTITIONS = ("in_domain_test", "external_test")
ARMS = ("variant", "baseline")
CODE_PATHS = [f"src/glaucoma_project/{name}" for name in
              ("combined_model/variant_study.py", "combined_model/analysis.py", "image_model", "data", "evaluation",
               "measurements/disc_boxes.py", "runtime.py", "files.py", "freeze.py")]


@dataclass
class GateRows:
    """Training and validation images of the source camera, and the development images of every other camera."""
    train: pd.DataFrame
    validation: pd.DataFrame
    shifted: pd.DataFrame


def load_study(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def runs_directory(study: dict) -> Path:
    return REPO_ROOT / study["runs_dir"]


def manifest_path(study: dict) -> Path:
    return REPO_ROOT / "docs" / f"frozen_{study['name']}.json"


def arm_config(study: dict, arm: str, seed: int) -> dict:
    """The recipe of an arm for one seed, with the run name carrying the seed."""
    config = yaml.safe_load((REPO_ROOT / study[f"{arm}_config"]).read_text())
    return {**config, "seed": seed, "run_name": f"{config['run_name']}_seed{seed}"}


def jobs(study: dict) -> list[int]:
    return list(study["seeds"])


def gate_jobs(study: dict) -> list[tuple[str, int]]:
    return [(arm, seed) for arm in ARMS for seed in study["gate_seeds"]]


def gate_rows(partitions: dict[str, pd.DataFrame], source_camera: str) -> GateRows:
    """Development images only; no test image is ever listed."""
    on_source = lambda rows: rows[rows.camera == source_camera]
    development = pd.concat([partitions["train"], partitions["validation"]], ignore_index=True)
    return GateRows(on_source(partitions["train"]), on_source(partitions["validation"]),
                    development[development.camera != source_camera])


def run_gate(study: dict, arm: str, seed: int, gate: GateRows, root: Path, device: torch.device | None = None) -> Path:
    """Train one arm on the source camera and write its scores for the other cameras. Never reuses a folder."""
    device = device or pick_device()
    config = arm_config(study, arm, seed)
    folder = runs_directory(study) / "gate" / f"{arm}_seed{seed}"
    folder.mkdir(parents=True)
    write_json(folder / "run.json", environment_record(device))
    for rows in (gate.train, gate.validation, gate.shifted):
        check_images(root, rows)
    model, _, _ = fit(gate.train, gate.validation, root, folder, config, device)
    predict_rows(model, gate.shifted, root, config, device)[["image_id", "label", "logit"]].to_csv(
        folder / "scores.csv", index=False)
    return folder


def gate_verdict(study: dict) -> dict:
    """Mean AUC over the gate seeds of each arm on the shifted cameras; passes if the variant is higher."""
    runs = runs_directory(study) / "gate"
    verdict = {}
    for arm in ARMS:
        aucs = []
        for seed in study["gate_seeds"]:
            scores = pd.read_csv(runs / f"{arm}_seed{seed}" / "scores.csv")
            aucs.append(float(roc_auc_score(scores.label, scores.logit)))
        verdict[arm] = {"aucs": aucs, "mean_auc": float(np.mean(aucs))}
    verdict["passes"] = bool(verdict["variant"]["mean_auc"] > verdict["baseline"]["mean_auc"])
    return verdict


def train_seed(study: dict, seed: int, root: Path = PREPARED_ROOT, splits_dir: Path = COMMON_SPLIT_DIR) -> Path:
    return develop(arm_config(study, "variant", seed), root=root, runs_dir=runs_directory(study), splits_dir=splits_dir)


def checkpoint_name(seed: int) -> str:
    return f"seed{seed}"


def checkpoint_path(study: dict, seed: int) -> Path:
    return runs_directory(study) / arm_config(study, "variant", seed)["run_name"] / "best.pt"


def freeze_study(study: dict, manifest: Path | None = None) -> dict:
    """Record the code version and the hash of every checkpoint, extra input and the common split."""
    files = {checkpoint_name(seed): checkpoint_path(study, seed) for seed in study["seeds"]}
    files |= {"common_split": COMMON_SPLIT_DIR / "all.csv"}
    files |= {Path(path).name: REPO_ROOT / path for path in study.get("inputs", [])}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Not finished: {missing}")
    configs = [f"configs/{study['name']}.yaml", study["variant_config"], study["baseline_config"]]
    return freeze(files, {"study": study}, manifest or manifest_path(study), CODE_PATHS + configs)


def score_test_sets(study: dict, out_dir: Path, manifest: Path | None = None, root: Path = PREPARED_ROOT,
                    device: torch.device | None = None, splits_dir: Path = COMMON_SPLIT_DIR) -> None:
    """Score both test sets with every frozen checkpoint; one ``<run>_<partition>.csv`` per pair.
    Refuses to run twice."""
    device = device or pick_device()
    record = verify(manifest or manifest_path(study))
    out_dir.mkdir(parents=True)  # an existing folder means the test sets were already used
    partitions = load_common_split(splits_dir)
    for seed in study["seeds"]:
        model, checkpoint = restore_model(REPO_ROOT / record["files"][checkpoint_name(seed)]["path"], device)
        for partition in TEST_PARTITIONS:
            rows = partitions[partition]
            check_images(root, rows)
            predictions = predict_rows(model, rows, root, checkpoint["config"], device)
            name = arm_config(study, "variant", seed)["run_name"]
            predictions.assign(seed=seed).to_csv(out_dir / f"{name}_{partition}.csv", index=False)


def _seed_scores(directory: Path, study: dict, arm: str, partition: str) -> dict[int, pd.DataFrame]:
    return {seed: pd.read_csv(directory / f"{arm_config(study, arm, seed)['run_name']}_{partition}.csv",
                              float_precision="round_trip") for seed in study["seeds"]}


def analyse_study(study: dict, scores: Path, out: Path) -> dict[str, pd.DataFrame]:
    """Variant minus the P01 image-only model, seed-averaged logits, paired bootstrap over patient groups,
    and the same difference for each seed. Opens no image and no checkpoint."""
    baseline_dir = REPO_ROOT / study["baseline_scores"]
    primary, per_seed = [], []
    for partition in TEST_PARTITIONS:
        variant = _seed_scores(scores, study, "variant", partition)
        baseline = _seed_scores(baseline_dir, study, "baseline", partition)
        a, b = ensemble(list(variant.values())), ensemble(list(baseline.values()))
        merged = a.merge(b[["image_id", "logit"]], on="image_id", suffixes=("", "_baseline"), validate="one_to_one")
        result = paired_auc_difference(merged.label, merged.logit, merged.logit_baseline, merged.analysis_group,
                                       study["n_bootstrap"], study["bootstrap_seed"])
        primary.append({"partition": partition, "comparison": f"{study['name']} minus P01 image-only", **result})
        per_seed += [{"partition": partition, "seed": seed,
                      "variant_auc": roc_auc_score(variant[seed].label, variant[seed].logit),
                      "baseline_auc": roc_auc_score(baseline[seed].label, baseline[seed].logit)} for seed in study["seeds"]]
    tables = {"primary": pd.DataFrame(primary), "per_seed": pd.DataFrame(per_seed)}
    tables["per_seed"]["difference"] = tables["per_seed"].variant_auc - tables["per_seed"].baseline_auc
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", type=Path)
    parser.add_argument("action", choices=["gate-job", "gate-run", "gate-verdict", "job", "train", "freeze", "score", "analyse"])
    parser.add_argument("target", nargs="?", help="job number (gate-job, job) or arm (gate-run)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--scores", type=Path)
    args = parser.parse_args()
    study = load_study(args.study)
    if args.action == "gate-job":
        print(*gate_jobs(study)[int(args.target)])
    elif args.action == "gate-run":
        gate = gate_rows(load_common_split(), study["source_camera"])
        print("Run folder:", run_gate(study, args.target, args.seed, gate, PREPARED_ROOT))
    elif args.action == "gate-verdict":
        verdict = gate_verdict(study)
        write_json(runs_directory(study) / "gate" / "verdict.json", verdict)
        print(json.dumps(verdict, indent=2))
    elif args.action == "job":
        print(jobs(study)[int(args.target)])
    elif args.action == "train":
        print("Run folder:", train_seed(study, args.seed))
    elif args.action == "freeze":
        print(freeze_study(study)["git_commit"])
    elif args.action == "score":
        score_test_sets(study, args.out or runs_directory(study) / "final")
    else:
        print(analyse_study(study, args.scores, args.out)["primary"].to_string(index=False))


if __name__ == "__main__":
    main()

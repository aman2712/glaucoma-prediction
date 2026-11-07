"""The P01 study: which runs belong to it, freezing them, and scoring the test sets once.

Usage (after every run has been trained with ``image_model.develop``):
  python -m glaucoma_project.combined_model.study job N       prints "config seed" of job N (for the Slurm array)
  python -m glaucoma_project.combined_model.study freeze      records every checkpoint and input hash
  python -m glaucoma_project.combined_model.study score       evaluates the frozen runs, once, on both test sets
Analysis of the saved scores is in ``combined_model/analysis.py`` and opens no image.
"""
import argparse
from pathlib import Path

import torch
import yaml

from glaucoma_project import COMMON_SPLIT_DIR, PREPARED_ROOT, REPO_ROOT
from glaucoma_project.data.classification import check_images
from glaucoma_project.data.common_split import load_common_split
from glaucoma_project.data.measurement_inputs import apply_scaler, attach_measurements
from glaucoma_project.files import read_table
from glaucoma_project.freeze import freeze, verify
from glaucoma_project.image_model.model import restore_model
from glaucoma_project.image_model.predict import predict_rows
from glaucoma_project.runtime import pick_device

STUDY_CONFIG = REPO_ROOT / "configs" / "p01.yaml"
MANIFEST = REPO_ROOT / "docs" / "frozen_p01.json"
TEST_PARTITIONS = ("in_domain_test", "external_test")
CODE_PATHS = [f"src/glaucoma_project/{name}" for name in
              ("combined_model", "image_model", "data", "evaluation", "runtime.py", "files.py", "freeze.py")] + [
              "configs/p01.yaml", "configs/p01_image_only.yaml", "configs/p01_combined.yaml"]


def load_study(path: Path = STUDY_CONFIG) -> dict:
    return yaml.safe_load(path.read_text())


def run_names(study: dict) -> dict[tuple[str, int], str]:
    """Run folder name of every (arm, seed) of the study."""
    names = {}
    for arm, config_path in study["arms"].items():
        name = yaml.safe_load((REPO_ROOT / config_path).read_text())["run_name"]
        names.update({(arm, seed): f"{name}_seed{seed}" for seed in study["seeds"]})
    return names


def jobs(study: dict) -> list[tuple[str, int]]:
    """Every training job of the study as (config path, seed): arms in order, then seeds."""
    return [(config, seed) for config in study["arms"].values() for seed in study["seeds"]]


def freeze_study(study: dict, manifest: Path = MANIFEST) -> dict:
    """Record the code version and the hash of every checkpoint, the measurements and the common split.
    Fails if a run is missing, so the study is frozen complete and only once."""
    runs_dir = REPO_ROOT / study["runs_dir"]
    files = {f"{arm}_seed{seed}": runs_dir / name / "best.pt" for (arm, seed), name in run_names(study).items()}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Runs not finished: {missing}")
    files |= {"measurements": REPO_ROOT / study["measurements"], "common_split": COMMON_SPLIT_DIR / "all.csv"}
    return freeze(files, {"study": study}, manifest, CODE_PATHS)


def score_test_sets(study: dict, out_dir: Path, manifest: Path = MANIFEST, root: Path = PREPARED_ROOT,
                    device: torch.device | None = None, splits_dir: Path = COMMON_SPLIT_DIR) -> None:
    """Score both test sets with every frozen run and save one ``<run>_<partition>.csv`` per pair
    (labels, groups, probability and logit). Refuses to run twice."""
    device = device or pick_device()
    record = verify(manifest)
    out_dir.mkdir(parents=True)  # an existing folder means the test sets were already used
    partitions, measurements = load_common_split(splits_dir), read_table(REPO_ROOT / study["measurements"])
    for (arm, seed), name in run_names(study).items():
        path = REPO_ROOT / record["files"][f"{arm}_seed{seed}"]["path"]
        model, checkpoint = restore_model(path, device)
        config, scaler = checkpoint["config"], checkpoint["measurement_scaler"]
        for partition in TEST_PARTITIONS:
            rows = partitions[partition]
            if scaler:
                rows = apply_scaler(attach_measurements(rows, measurements, scaler["columns"]), scaler)
            check_images(root, rows)
            predictions = predict_rows(model, rows, root, config, device)
            predictions.assign(arm=arm, seed=seed).to_csv(out_dir / f"{name}_{partition}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["job", "freeze", "score"])
    parser.add_argument("index", nargs="?", type=int, help="job number (job only)")
    parser.add_argument("--out", type=Path, help="folder for the test scores (score only)")
    args = parser.parse_args()
    settings = load_study()
    if args.action == "job":
        print(*jobs(settings)[args.index])
    elif args.action == "freeze":
        print(freeze_study(settings)["git_commit"])
    else:
        score_test_sets(settings, args.out or REPO_ROOT / settings["runs_dir"] / "final")

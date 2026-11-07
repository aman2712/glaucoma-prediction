"""P01 analysis of the saved test scores: arms, the primary paired comparison and the seed spread.

Usage: python -m glaucoma_project.combined_model.analysis --scores DIR --out DIR
Opens no image and no checkpoint. Each arm's score is the mean logit of its seeds.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from glaucoma_project import REPO_ROOT
from glaucoma_project.combined_model.study import TEST_PARTITIONS, load_study, run_names
from glaucoma_project.evaluation.bootstrap import grouped_bootstrap
from glaucoma_project.evaluation.compare import paired_auc_difference
from glaucoma_project.evaluation.screening import select_threshold

PRIMARY = ("combined", "image_only")


def ensemble(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """One row per image with the mean logit over the frames (one per seed) and its probability.
    Every frame must hold the same images and labels."""
    first = frames[0].sort_values("image_id").reset_index(drop=True)
    logits = []
    for frame in frames:
        frame = frame.sort_values("image_id").reset_index(drop=True)
        if not frame[["image_id", "label"]].equals(first[["image_id", "label"]]):
            raise ValueError("Runs scored different images or labels")
        logits.append(frame.logit.to_numpy())
    mean_logit = np.mean(logits, axis=0)
    return first.drop(columns=["logit", "probability", "seed"], errors="ignore").assign(
        logit=mean_logit, probability=1 / (1 + np.exp(-mean_logit)))


def _frames(directory: Path, study: dict, arm: str, partition: str) -> dict[int, pd.DataFrame]:
    """The saved scores of each seed of an arm."""
    names = run_names(study)
    return {seed: pd.read_csv(directory / f"{names[arm, seed]}_{partition}.csv", float_precision="round_trip")
            for seed in study["seeds"]}


def _validation_threshold(runs_dir: Path, study: dict, arm: str) -> float:
    """Threshold of the seed-averaged validation score: the Baseline B rule (highest specificity
    that keeps the target sensitivity), descriptive only."""
    names = run_names(study)
    validation = ensemble([pd.read_csv(runs_dir / names[arm, seed] / "validation_predictions.csv",
                                       float_precision="round_trip") for seed in study["seeds"]])
    return select_threshold(validation.label, validation.probability, study["target_sensitivity"])


def analyse(study: dict, scores: Path, runs_dir: Path, out: Path, baseline_a: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Write and return the P01 result tables. ``baseline_a`` holds the frozen Baseline A test scores
    (``image_id``, ``split``, ``source``, ``score``)."""
    out.mkdir(parents=True, exist_ok=True)
    n, seed = study["n_bootstrap"], study["bootstrap_seed"]
    seeds = {(arm, part): _frames(scores, study, arm, part) for arm in study["arms"] for part in TEST_PARTITIONS}
    arms = {key: ensemble(list(frames.values())) for key, frames in seeds.items()}
    thresholds = {arm: _validation_threshold(runs_dir, study, arm) for arm in study["arms"]}
    metrics, primary, secondary, per_seed = [], [], [], []
    for part in TEST_PARTITIONS:
        for arm in study["arms"]:
            table = grouped_bootstrap(arms[arm, part], thresholds[arm], seed, n)
            metrics.append(table.assign(arm=arm, partition=part, threshold=thresholds[arm]))
        a, b = (arms[arm, part] for arm in PRIMARY)
        primary.append({"partition": part, "comparison": "combined minus image_only",
                        **paired_auc_difference(a.label, a.logit, b.logit, a.analysis_group, n, seed)})
        measured = baseline_a[(baseline_a.split == part) & (baseline_a.source == "predicted")]
        for arm in study["arms"]:
            joined = arms[arm, part].merge(measured[["image_id", "score"]], on="image_id")
            secondary.append({"partition": part, "comparison": f"{arm} minus measurements_only",
                              **paired_auc_difference(joined.label, joined.logit, joined.score,
                                                      joined.analysis_group, n, seed)})
        for run_seed in study["seeds"]:
            aucs = {arm: roc_auc_score(seeds[arm, part][run_seed].label, seeds[arm, part][run_seed].logit)
                    for arm in PRIMARY}
            per_seed.append({"partition": part, "seed": run_seed, **aucs,
                             "difference": aucs["combined"] - aucs["image_only"]})
    tables = {"arm_metrics": pd.concat(metrics, ignore_index=True), "primary": pd.DataFrame(primary),
              "secondary": pd.DataFrame(secondary), "per_seed": pd.DataFrame(per_seed)}
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    settings = load_study()
    baseline = pd.read_csv(REPO_ROOT / settings["baseline_a_scores"])
    results = analyse(settings, args.scores, REPO_ROOT / settings["runs_dir"], args.out, baseline)
    print(results["primary"].to_string(index=False))

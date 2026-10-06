"""One-shot evaluation of the frozen Baseline A pipeline on the CHAKSU test set and PAPILA."""
import joblib
import numpy as np
import pandas as pd
import torch

from glaucoma_project import REPO_ROOT
from glaucoma_project.classification.baseline import labelled
from glaucoma_project.data import SegmentationDataset
from glaucoma_project.evaluation.classification import evaluate
from glaucoma_project.evaluation.metrics import agreement, dice
from glaucoma_project.freeze import verify
from glaucoma_project.measurements.extract import measure_split
from glaucoma_project.segmentation.predict import load_model, predict_dataset
from glaucoma_project.segmentation.train import pick_device

OUT_DIR = REPO_ROOT / "runs" / "final"
TEST_SPLITS = ("in_domain_test", "external_test")
SOURCES = ("predicted", "expert")


def _classification(measurements, classifiers, config):
    """Metrics at the frozen threshold, per test split and mask source, plus a vCDR-only reference
    and a PAPILA sensitivity analysis that counts 'suspect' eyes as glaucoma."""
    features, n, seed = config["features"], config["n_bootstrap"], config["seed"]
    tables, scores = [], []
    for split in TEST_SPLITS:
        for source in SOURCES:
            part = measurements[(measurements.split == split) & (measurements.source == source)]
            frozen = classifiers[source]
            variants = {"main": labelled(part)}
            if split == "external_test":
                variants["suspects counted as glaucoma"] = part.assign(binary_label=part.binary_label.fillna(1))
            for variant, data in variants.items():
                score = frozen["model"].predict_proba(data[features])[:, 1]
                table = evaluate(data.binary_label, score, frozen["threshold"], n, seed)
                tables.append(table.assign(split=split, source=source, analysis=variant, n=len(data),
                                           positives=int(data.binary_label.sum())).reset_index(names="metric"))
                if variant == "main":
                    scores.append(data[["image_id", "split", "source", "binary_label"]].assign(score=score))
        data = labelled(measurements[(measurements.split == split) & (measurements.source == "predicted")]).dropna(subset=["vcdr"])
        table = evaluate(data.binary_label, data.vcdr, 0.5, n, seed).loc[["auc"]]
        tables.append(table.assign(split=split, source="vcdr only (no model)", analysis="main", n=len(data),
                                   positives=int(data.binary_label.sum())).reset_index(names="metric"))
    return pd.concat(tables, ignore_index=True), pd.concat(scores, ignore_index=True)


def _segmentation(model, measurements):
    """Dice against the expert masks and agreement of the predicted CDRs with the expert CDRs."""
    rows = []
    for split in TEST_SPLITS:
        dataset = SegmentationDataset(split)
        expert = torch.stack([dataset[i]["mask"] for i in range(len(dataset))])
        scores = dice(predict_dataset(model, dataset), expert).mean(dim=0)
        part = measurements[measurements.split == split]
        predicted, truth = (part[part.source == s].set_index("image_id") for s in SOURCES)
        rows.append({"split": split, "dice_disc": scores[0].item(), "dice_cup": scores[1].item(),
                     **{f"{m}_{k}": v for m in ("vcdr", "hcdr", "area_cdr")
                        for k, v in agreement(predicted[m], truth[m]).items() if k != "n"}})
    return pd.DataFrame(rows)


def run_final_evaluation(config: dict) -> dict:
    """Verify the frozen pipeline, then evaluate it once. Refuses to run twice."""
    if (OUT_DIR / "classification.csv").exists():
        raise FileExistsError("The final evaluation has already been run")
    frozen = verify()
    path = lambda name: REPO_ROOT / frozen["files"][name]["path"]
    model = load_model(path("segmentation_weights").parent, device=pick_device())
    classifiers = {source: joblib.load(path(f"classifier_{source}")) for source in SOURCES}
    measurements = pd.concat([measure_split(split, source, model) for split in TEST_SPLITS for source in SOURCES])
    classification, scores = _classification(measurements, classifiers, config)
    segmentation = _segmentation(model, measurements)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    measurements.to_csv(OUT_DIR / "measurements_test.csv", index=False)
    scores.to_csv(OUT_DIR / "scores.csv", index=False)
    segmentation.to_csv(OUT_DIR / "segmentation.csv", index=False)
    classification.to_csv(OUT_DIR / "classification.csv", index=False)
    return {"classification": classification, "segmentation": segmentation}

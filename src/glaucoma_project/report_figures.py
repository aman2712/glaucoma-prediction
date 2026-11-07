"""Figures for the written reports. They show real patient images, so they are written to ``runs/``
(git-ignored) and never committed.

Usage: python -m glaucoma_project.report_figures --out runs/report/qualitative_examples.png
"""
import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from glaucoma_project import REPO_ROOT
from glaucoma_project.data.segmentation import SegmentationDataset
from glaucoma_project.freeze import verify_files
from glaucoma_project.runtime import pick_device
from glaucoma_project.segmentation.predict import load_model, predict_masks

BASELINE_MANIFEST = REPO_ROOT / "docs" / "frozen_baseline_a_common_split.json"
TEST_SPLIT = "in_domain_test"
DISC, CUP = "#00b4d8", "#f4a100"


def per_image_errors(measurements: pd.DataFrame, scores: pd.DataFrame, split: str = TEST_SPLIT) -> pd.DataFrame:
    """One row per image of ``split``: label, score, expert and predicted vertical CDR and their gap."""
    part = measurements[measurements.split == split]
    expert = part[part.source == "expert"].set_index("image_id").vcdr.rename("expert_vcdr")
    predicted = part[part.source == "predicted"].set_index("image_id").vcdr.rename("predicted_vcdr")
    table = pd.concat([expert, predicted], axis=1).dropna()
    table["error"] = (table.predicted_vcdr - table.expert_vcdr).abs()
    given = scores[(scores.split == split) & (scores.source == "predicted")].set_index("image_id")
    return table.join(given[["binary_label", "score"]], how="inner").reset_index()


def pick_examples(errors: pd.DataFrame, threshold: float) -> dict[str, str]:
    """Three images chosen by rule, not by eye: a correctly cleared healthy eye and a correctly flagged
    glaucoma eye, each with the median measurement error of its group, and the largest error overall."""
    def typical(group: pd.DataFrame) -> str:
        return group.iloc[(group.error - group.error.median()).abs().argmin()].image_id

    healthy = errors[(errors.binary_label == 0) & (errors.score < threshold)]
    glaucoma = errors[(errors.binary_label == 1) & (errors.score >= threshold)]
    if healthy.empty or glaucoma.empty:
        raise ValueError("Need at least one correctly classified image of each class")
    return {"healthy": typical(healthy), "glaucoma": typical(glaucoma), "largest error": errors.loc[errors.error.idxmax(), "image_id"]}


def crop_box(disc_mask: np.ndarray, margin: float = 1.0, size: int = 512) -> tuple[int, int, int, int]:
    """Square box (top, bottom, left, right) centred on the disc, as wide as the disc plus ``margin`` of its width each side."""
    rows, cols = np.flatnonzero(disc_mask.any(axis=1)), np.flatnonzero(disc_mask.any(axis=0))
    centre_r, centre_c = (rows[0] + rows[-1]) / 2, (cols[0] + cols[-1]) / 2
    half = max(rows[-1] - rows[0], cols[-1] - cols[0]) * (1 + margin) / 2
    half = min(half, size / 2)
    top, left = int(max(0, min(size - 2 * half, centre_r - half))), int(max(0, min(size - 2 * half, centre_c - half)))
    return top, top + int(2 * half), left, left + int(2 * half)


def qualitative_figure(out: Path, manifest: Path = BASELINE_MANIFEST) -> Path:
    """Expert outlines (solid) and the network's outlines (dashed) on three CHAKSU test images."""
    record = verify_files(manifest)
    weights = REPO_ROOT / record["files"]["segmentation_weights"]["path"]
    classifier = joblib.load(REPO_ROOT / record["files"]["classifier_predicted"]["path"])
    measurements = pd.read_csv(REPO_ROOT / "runs" / "final" / "common_split" / "measurements_test.csv")
    scores = pd.read_csv(REPO_ROOT / "runs" / "final" / "common_split" / "scores.csv")
    errors = per_image_errors(measurements, scores)
    chosen = pick_examples(errors, classifier["threshold"])
    dataset = SegmentationDataset(TEST_SPLIT)
    model = load_model(weights.parent, device=pick_device())
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, (name, image_id) in zip(axes, chosen.items()):
        item = dataset[int(np.flatnonzero(dataset.table.image_id == image_id)[0])]
        predicted = predict_masks(model, item["image"][None])[0].numpy().astype(float)
        expert = item["mask"].numpy()
        top, bottom, left, right = crop_box(expert[0] > 0)
        ax.imshow(item["image"].permute(1, 2, 0).numpy()[top:bottom, left:right])
        for channel, colour in enumerate((DISC, CUP)):
            for mask, style in ((expert, "solid"), (predicted, "dashed")):
                if mask[channel, top:bottom, left:right].any():
                    ax.contour(mask[channel, top:bottom, left:right], levels=[0.5], colors=colour,
                               linewidths=1.3, linestyles=style)
        row = errors[errors.image_id == image_id].iloc[0]
        ax.set_title(f"{name}\nvertical ratio: expert {row.expert_vcdr:.2f}, model {row.predicted_vcdr:.2f}", fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print({name: image_id for name, image_id in chosen.items()})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "report" / "qualitative_examples.png")
    print(qualitative_figure(parser.parse_args().out))

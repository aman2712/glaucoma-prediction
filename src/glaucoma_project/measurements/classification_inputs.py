"""Predicted cup-to-disc ratios for every classification image, for models that take them as input.

The frozen Baseline A segmentation network needs only the photograph, so unlike Baseline A's own
measurements these also cover images without expert masks. Expert masks and labels are never used.

Usage: python -m glaucoma_project.measurements.classification_inputs --out FILE
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from glaucoma_project import PREPARED_ROOT, REPO_ROOT
from glaucoma_project.data.common_split import load_common_split
from glaucoma_project.data.segmentation import read_image
from glaucoma_project.files import write_json
from glaucoma_project.freeze import sha256_file, verify_files
from glaucoma_project.measurements.cdr import measure
from glaucoma_project.runtime import pick_device
from glaucoma_project.segmentation.predict import load_model, predict_masks

BASELINE_MANIFEST = REPO_ROOT / "docs" / "frozen_baseline_a_common_split.json"
RATIOS = ["vcdr", "hcdr", "area_cdr"]


def predicted_measurements(rows: pd.DataFrame, model, root: Path = PREPARED_ROOT, batch_size: int = 8) -> pd.DataFrame:
    """One row per image: ``image_id`` and the three ratios (NaN where no disc is found)."""
    records = []
    for start in range(0, len(rows), batch_size):
        batch = rows.iloc[start:start + batch_size]
        images = torch.stack([read_image(root / path, "RGB") for path in batch.segmentation_image])
        records += [{"image_id": image_id, **{k: v for k, v in measure(m[0].numpy(), m[1].numpy()).items() if k in RATIOS}}
                    for image_id, m in zip(batch.image_id, predict_masks(model, images))]
    return pd.DataFrame(records, columns=["image_id", *RATIOS])


def write_measurements(out: Path, manifest: Path = BASELINE_MANIFEST, root: Path = PREPARED_ROOT) -> pd.DataFrame:
    """Measure every classification image of every split with the network frozen in ``manifest``.
    Writes ``out`` and ``out.with_suffix('.json')`` (weights hash, counts) and refuses to overwrite."""
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    record = verify_files(manifest)
    weights = REPO_ROOT / record["files"]["segmentation_weights"]["path"]
    model = load_model(weights.parent, device=pick_device())
    rows = pd.concat(load_common_split().values(), ignore_index=True)
    table = rows[["image_id", "dataset", "split", "analysis_group", "binary_label"]].merge(
        predicted_measurements(rows, model, root), on="image_id", validate="one_to_one")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    write_json(out.with_suffix(".json"), {
        "segmentation_weights_sha256": record["files"]["segmentation_weights"]["sha256"],
        "baseline_manifest": str(manifest.relative_to(REPO_ROOT)), "csv_sha256": sha256_file(out),
        "images": len(table), "no_disc_found": int(table.vcdr.isna().sum()),
        "images_per_split": table.split.value_counts().to_dict()})
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    print(write_measurements(parser.parse_args().out).groupby("split").size())

"""Predicted cup-to-disc ratios for every classification image, for models that take them as input.

The frozen Baseline A segmentation network needs only the photograph, so unlike Baseline A's own
measurements these also cover images without expert masks. Expert masks and labels are never used.

Usage: python -m glaucoma_project.measurements.classification_inputs --out FILE
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from glaucoma_project import PREPARED_ROOT, REPO_ROOT
from glaucoma_project.data.common_split import load_common_split
from glaucoma_project.data.segmentation import PhotographDataset, read_image
from glaucoma_project.files import write_json
from glaucoma_project.freeze import sha256_file, verify_files
from glaucoma_project.measurements.cdr import measure
from glaucoma_project.runtime import pick_device
from glaucoma_project.segmentation.adapt import adapt_batchnorm
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


def adapted_measurements(rows: pd.DataFrame, model, root: Path, device: torch.device, batch_size: int = 8) -> pd.DataFrame:
    """The three ratios of every row, after the model's batch-norm statistics are re-estimated on these rows'
    photographs (no label or mask is read). The model passed in is not modified."""
    adapted = adapt_batchnorm(model, DataLoader(PhotographDataset(rows, root), batch_size=batch_size), device)
    return predicted_measurements(rows, adapted, root, batch_size)


def write_measurements(out: Path, manifest: Path = BASELINE_MANIFEST, root: Path = PREPARED_ROOT,
                       adapt: bool = False) -> pd.DataFrame:
    """Measure every classification image of every split with the network frozen in ``manifest``; with ``adapt``
    each split is first used to re-estimate the batch-norm statistics. Writes ``out`` and
    ``out.with_suffix('.json')`` (weights hash, counts) and refuses to overwrite."""
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    record = verify_files(manifest)
    weights = REPO_ROOT / record["files"]["segmentation_weights"]["path"]
    device = pick_device()
    model = load_model(weights.parent, device=device)
    rows = pd.concat(load_common_split().values(), ignore_index=True)
    measure_split = (lambda part: adapted_measurements(part, model, root, device)) if adapt else \
        (lambda part: predicted_measurements(part, model, root))
    ratios = pd.concat([measure_split(part) for part in load_common_split().values()], ignore_index=True)
    table = rows[["image_id", "dataset", "split", "analysis_group", "binary_label"]].merge(
        ratios, on="image_id", validate="one_to_one")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    write_json(out.with_suffix(".json"), {
        "segmentation_weights_sha256": record["files"]["segmentation_weights"]["sha256"],
        "baseline_manifest": str(manifest.relative_to(REPO_ROOT)), "csv_sha256": sha256_file(out),
        "adapted_batch_norm": adapt, "images": len(table), "no_disc_found": int(table.vcdr.isna().sum()),
        "images_per_split": table.split.value_counts().to_dict()})
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=BASELINE_MANIFEST,
                        help="frozen manifest whose segmentation weights are used (default: Baseline A)")
    parser.add_argument("--adapt", action="store_true", help="re-estimate batch-norm statistics on each split")
    args = parser.parse_args()
    print(write_measurements(args.out, args.manifest, adapt=args.adapt).groupby("split").size())

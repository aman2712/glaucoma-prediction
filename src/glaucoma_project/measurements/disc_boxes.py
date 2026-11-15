"""Square crop boxes around the predicted optic disc, for experiment X2.

A camera's field of view changes how large the disc appears. Cutting a window whose side is a fixed
multiple of the disc's size gives every camera the same scale. Images with no predicted disc keep
the whole image. Expert masks and labels are never used.

Usage: python -m glaucoma_project.measurements.disc_boxes --out FILE
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from glaucoma_project import PREPARED_ROOT, REPO_ROOT
from glaucoma_project.data.common_split import load_common_split
from glaucoma_project.data.crops import BOX_COLUMNS
from glaucoma_project.data.segmentation import read_image
from glaucoma_project.freeze import sha256_file, verify_files
from glaucoma_project.measurements.cdr import largest_component
from glaucoma_project.measurements.classification_inputs import BASELINE_MANIFEST
from glaucoma_project.runtime import pick_device
from glaucoma_project.segmentation.predict import load_model, predict_masks

SIDE_FACTOR = 3.0
MIN_SIDE = 100   # px of the 512 px image; smaller boxes are spurious blobs (the smallest CHAKSU box is 147)


def crop_box(disc: np.ndarray, factor: float = SIDE_FACTOR) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of a square centred on the largest blob of ``disc``, with side
    ``factor`` times that blob's longer side. The box may reach past the image edge. An empty mask,
    or a box smaller than ``MIN_SIDE``, gives the whole image."""
    whole_image = (0, 0, disc.shape[1], disc.shape[0])
    blob = largest_component(disc > 0)
    if not blob.any():
        return whole_image
    rows, cols = np.flatnonzero(blob.any(axis=1)), np.flatnonzero(blob.any(axis=0))
    top, bottom, left, right = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    half = factor * max(bottom - top, right - left) / 2
    if 2 * half < MIN_SIDE:
        return whole_image
    centre_y, centre_x = (top + bottom) / 2, (left + right) / 2
    return (round(centre_x - half), round(centre_y - half), round(centre_x + half), round(centre_y + half))


def predicted_boxes(rows: pd.DataFrame, model, root: Path = PREPARED_ROOT, batch_size: int = 8) -> pd.DataFrame:
    records = []
    for start in range(0, len(rows), batch_size):
        batch = rows.iloc[start:start + batch_size]
        images = torch.stack([read_image(root / path, "RGB") for path in batch.segmentation_image])
        for image_id, masks in zip(batch.image_id, predict_masks(model, images)):
            records.append({"image_id": image_id, **dict(zip(BOX_COLUMNS, crop_box(masks[0].numpy())))})
    return pd.DataFrame(records, columns=["image_id", *BOX_COLUMNS])


def write_boxes(out: Path, manifest: Path = BASELINE_MANIFEST, root: Path = PREPARED_ROOT) -> pd.DataFrame:
    """Boxes for every classification image, from the segmentation network frozen in ``manifest``.
    Refuses to overwrite."""
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    record = verify_files(manifest)
    model = load_model((REPO_ROOT / record["files"]["segmentation_weights"]["path"]).parent, device=pick_device())
    rows = pd.concat(load_common_split().values(), ignore_index=True)
    table = predicted_boxes(rows, model, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print("sha256", sha256_file(out))
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    print(len(write_boxes(parser.parse_args().out)), "boxes")

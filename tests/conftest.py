"""A small synthetic prepared-data export, so the image-model code can be tested without patient data."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from glaucoma_project.data.segmentation import SPLITS
from glaucoma_project.files import write_json
from glaucoma_project.freeze import sha256_file

SIZES = {"train": 16, "validation": 12, "in_domain_test": 8, "external_test": 8}


def build_export(root: Path) -> pd.DataFrame:
    """Write a synthetic export under ``root`` and return its split table (alternating labels)."""
    rng = np.random.default_rng(2440)
    rows = []
    for split, count in SIZES.items():
        dataset = "papila" if split == "external_test" else "chaksu"
        release = "" if dataset == "papila" else ("test" if split == "in_domain_test" else "train")
        for i in range(count):
            image_id, label = f"{dataset}_{split}_{i:03d}", i % 2
            relative = f"prepared/{dataset}/{image_id}/image_224.png"
            pixels = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
            (root / relative).parent.mkdir(parents=True)
            Image.fromarray(pixels).save(root / relative)
            papila_patient = f"papila_{i // 2:03d}" if dataset == "papila" else ""
            rows.append({
                "image_id": image_id, "dataset": dataset, "chaksu_release": release,
                "camera": "Topcon" if dataset == "papila" else "Remidio", "patient_id": papila_patient,
                "patient_id_verified": "1" if dataset == "papila" else "0",
                "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                "group_id": papila_patient or image_id, "split": split, "classification_eligible": "1",
                "binary_label": str(label),
                "original_label": str(label) if dataset == "papila" else ("GLAUCOMA SUSPECT" if label else "NORMAL"),
                "classification_image": relative})
    table = pd.DataFrame(rows)
    write_export(root, table)
    return table


def write_export(root: Path, table: pd.DataFrame) -> None:
    """(Re)write the manifest, split tables and a matching summary for ``table``."""
    (root / "splits").mkdir(exist_ok=True)
    table.to_csv(root / "prepared_manifest.csv", index=False)
    table.to_csv(root / "splits" / "all.csv", index=False)
    eligible = table[table.classification_eligible == "1"]
    for name in SPLITS:
        eligible[eligible["split"] == name].to_csv(root / "splits" / f"{name}_classification.csv", index=False)
    write_json(root / "splits" / "split_summary.json", {
        "status": "PROVISIONAL", "near_duplicate_reviews_pending": 0,
        "all_splits_sha256": sha256_file(root / "splits" / "all.csv"),
        "manifest_sha256": sha256_file(root / "prepared_manifest.csv"),
        "classification_counts": {name: int((eligible["split"] == name).sum()) for name in SPLITS}})


@pytest.fixture
def export(tmp_path):
    """(root, table) of a fresh synthetic export."""
    return tmp_path, build_export(tmp_path)


@pytest.fixture
def rewrite_export():
    return write_export

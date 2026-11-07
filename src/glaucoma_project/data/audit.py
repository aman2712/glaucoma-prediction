"""Integrity audit of the prepared data export, run before the image-only model reads any image.

It reads split metadata only (never test images) and stops on the first inconsistency.
"""
import json
from pathlib import Path

import pandas as pd

from glaucoma_project import PREPARED_ROOT
from glaucoma_project.data.segmentation import SPLITS
from glaucoma_project.files import read_table
from glaucoma_project.freeze import sha256_file

REQUIRED_COLUMNS = {
    "image_id", "dataset", "chaksu_release", "camera", "patient_id", "patient_id_verified",
    "pixel_sha256", "group_id", "split", "classification_eligible", "binary_label",
    "original_label", "classification_image",
}
# The dataset and CHAKSU release each split must contain.
ROLES = {"train": ("chaksu", "train"), "validation": ("chaksu", "train"),
         "in_domain_test": ("chaksu", "test"), "external_test": ("papila", "")}
LABELS = {"chaksu": {"NORMAL": "0", "GLAUCOMA SUSPECT": "1"}, "papila": {"0": "0", "1": "1"}}
IDENTITY_COLUMNS = ("patient_id", "pixel_sha256", "group_id")


def audit_export(root: Path = PREPARED_ROOT) -> tuple[pd.DataFrame, dict]:
    """Return the split table and split summary of a prepared export, or raise ValueError
    (FileNotFoundError for missing files) if it is incomplete, altered or leaks between splits."""
    root = Path(root)
    _require_files(root)
    summary = json.loads((root / "splits" / "split_summary.json").read_text())
    _check_hashes(root, summary)
    table = read_table(root / "splits" / "all.csv")
    _check_table(table)
    _check_partition_files(root, table, summary)
    if summary.get("near_duplicate_reviews_pending", 0):
        raise ValueError("Resolve the pending near-duplicate review before training")
    active = table[table["split"].isin(SPLITS)]
    _check_roles(active)
    _check_labels(active[active.classification_eligible == "1"])
    _check_identities(active)
    return table, summary


def _require_files(root: Path) -> None:
    required = [root / "prepared_manifest.csv", root / "splits" / "all.csv",
                root / "splits" / "split_summary.json"]
    required += [root / "splits" / f"{name}_classification.csv" for name in SPLITS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The prepared data export is incomplete. Rebuild it with data_pipeline/prepare_data.py "
            "(never recreate a split by hand). Missing:\n" + "\n".join(missing))


def _check_hashes(root: Path, summary: dict) -> None:
    for key, path in (("all_splits_sha256", root / "splits" / "all.csv"),
                      ("manifest_sha256", root / "prepared_manifest.csv")):
        if not summary.get(key) or sha256_file(path) != summary[key]:
            raise ValueError(f"Data integrity check failed: {key}")


def _check_table(table: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(table)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if table.empty or table.image_id.duplicated().any() or (table.image_id == "").any():
        raise ValueError("Empty or repeated image identity in all.csv")
    if not table.classification_eligible.isin(["0", "1"]).all():
        raise ValueError("Invalid classification eligibility flag")
    if (table.pixel_sha256 == "").any() or (table.group_id == "").any():
        raise ValueError("Missing duplicate hash or group identity")


def _by_id(frame: pd.DataFrame, columns) -> pd.DataFrame:
    return frame.sort_values("image_id").reset_index(drop=True)[list(columns)]


def _check_partition_files(root: Path, table: pd.DataFrame, summary: dict) -> None:
    """Each per-split CSV must be exactly the eligible rows of all.csv, with the summary's count."""
    for name in SPLITS:
        part = read_table(root / "splits" / f"{name}_classification.csv")
        expected = table[(table["split"] == name) & (table.classification_eligible == "1")]
        if set(part.columns) != set(table.columns):
            raise ValueError(f"{name}: CSV columns differ from all.csv")
        if not _by_id(part, table.columns).equals(_by_id(expected, table.columns)):
            raise ValueError(f"{name}: rows differ from all.csv")
        if len(part) != summary.get("classification_counts", {}).get(name):
            raise ValueError(f"{name}: count differs from the split summary")


def _check_roles(active: pd.DataFrame) -> None:
    expected = active["split"].map(ROLES)
    crossed = [actual != role for actual, role in zip(zip(active.dataset, active.chaksu_release), expected)]
    if any(crossed):
        raise ValueError(f"Dataset or release crossed evaluation roles: {active.image_id[crossed].iloc[0]}")


def _normalise_label(label: str) -> str:
    """Upper-case, single-spaced, with the CHAKSU source's 'SUSUPECT' typo corrected."""
    return " ".join(label.upper().split()).replace("SUSUPECT", "SUSPECT")


def _check_labels(eligible: pd.DataFrame) -> None:
    if not eligible.binary_label.isin(["0", "1"]).all():
        raise ValueError("Non-binary classification target")
    mapped = [LABELS[dataset].get(_normalise_label(label))
              for dataset, label in zip(eligible.dataset, eligible.original_label)]
    wrong = [m != target for m, target in zip(mapped, eligible.binary_label)]
    if any(wrong):
        raise ValueError(f"Unexpected label mapping: {eligible.image_id[wrong].iloc[0]}")
    papila = eligible[eligible.dataset == "papila"]
    if ((papila.patient_id == "") | (papila.patient_id_verified != "1")).any():
        raise ValueError("PAPILA requires verified patient identities")


def _check_identities(active: pd.DataFrame) -> None:
    """A known patient, duplicate hash or group may not appear in two different splits."""
    for column in IDENTITY_COLUMNS:
        known = active[active[column] != ""]
        if (known.groupby(column)["split"].nunique() > 1).any():
            raise ValueError(f"Known {column} crosses splits; repair it upstream before training")

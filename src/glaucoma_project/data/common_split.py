"""The common split of record, shared by every model.

It is the pipeline's frozen split plus the precautionary grouping of ``effective_splits``,
written once to ``splits/`` and committed, so that no model recomputes or alters it. Each model
reads ``splits/all.csv`` and takes the rows of its own splits and task.

Build:  python -m glaucoma_project.data.common_split            (refuses to overwrite)
Check:  python -m glaucoma_project.data.common_split --check    (rebuild from the local export
        and compare split and group of every image with the committed file)
"""
import argparse
import tempfile
from pathlib import Path

import pandas as pd

from glaucoma_project import COMMON_SPLIT_DIR, PREPARED_ROOT
from glaucoma_project.data.audit import audit_export
from glaucoma_project.data.effective_splits import effective_split, partition_rows
from glaucoma_project.data.segmentation import SPLITS
from glaucoma_project.files import read_table, write_json
from glaucoma_project.freeze import sha256_file

MEMBERSHIP = ["image_id", "split", "analysis_group"]


def _counts(table: pd.DataFrame) -> dict:
    """Images per split that each task can use."""
    return {name: {"classification": int(((table["split"] == name) & (table.classification_eligible == "1")).sum()),
                   "segmentation": int(((table["split"] == name) & (table.segmentation_eligible == "1")).sum())}
            for name in SPLITS}


def build_common_split(root: Path = PREPARED_ROOT, directory: Path = COMMON_SPLIT_DIR) -> dict:
    """Audit the prepared export, apply the grouping rule and write ``all.csv``, ``changes.csv`` and
    ``common_split.json`` (hashes, counts, moved images) to ``directory``, which must not exist yet."""
    table, _ = audit_export(root)
    effective, _ = effective_split(table)
    directory.mkdir(parents=True)
    effective.to_csv(directory / "all.csv", index=False)
    moved = effective[effective["split"] != effective.original_split]
    moved_columns = ["image_id", "original_split", "split", "split_change_reason"]
    moved[moved_columns].to_csv(directory / "changes.csv", index=False)
    manifest = {"source_all_csv_sha256": sha256_file(root / "splits" / "all.csv"),
                "all_csv_sha256": sha256_file(directory / "all.csv"), "images": len(effective),
                "counts": _counts(effective), "moved_images": len(moved)}
    write_json(directory / "common_split.json", manifest)
    return manifest


def load_common_split(directory: Path = COMMON_SPLIT_DIR) -> dict[str, pd.DataFrame]:
    """The classification rows of each split, read from the committed ``all.csv`` (never recomputed)."""
    table = read_table(Path(directory) / "all.csv")
    return {name: partition_rows(table, name) for name in SPLITS}


def check_common_split(root: Path = PREPARED_ROOT, directory: Path = COMMON_SPLIT_DIR) -> None:
    """Raise unless rebuilding from ``root`` gives the same split and group for every image.
    Pixel hashes may differ between machines (library versions), so only membership is compared."""
    with tempfile.TemporaryDirectory() as scratch:
        build_common_split(root, Path(scratch) / "rebuilt")
        rebuilt = read_table(Path(scratch) / "rebuilt" / "all.csv")[MEMBERSHIP]
    committed = read_table(directory / "all.csv")[MEMBERSHIP]
    if not rebuilt.sort_values("image_id").reset_index(drop=True).equals(
            committed.sort_values("image_id").reset_index(drop=True)):
        raise ValueError("The committed common split differs from one rebuilt from the local export")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare with the committed split instead of building")
    if parser.parse_args().check:
        check_common_split()
        print("The committed common split matches the local export.")
    else:
        print(build_common_split())

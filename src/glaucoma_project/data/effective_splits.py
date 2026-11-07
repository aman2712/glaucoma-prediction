"""A stricter split derived from the pipeline's frozen one, for the image-only model.

CHAKSU Bosch images named ``P<number>_Image<n>`` may show the same patient in different
releases. The pipeline cannot confirm that, so every such image is grouped by its P-number
as a precaution. Groups are merged with the known groups (patient, pixel hash, duplicate
group), and then:

* a group spanning datasets or CHAKSU releases, or containing an already quarantined image,
  is quarantined, because it may leak between training and testing;
* a group spanning train and validation moves entirely to train.

The rule uses no image content and no model result. The input table is never changed.
"""
import re
from pathlib import Path

import pandas as pd

from glaucoma_project.data.segmentation import SPLITS

BOSCH_ID = re.compile(r"chaksu_Bosch_(?:train|test)_P(\d+)_Image[12]", re.IGNORECASE)
GROUP_COLUMNS = ("group_id", "patient_id", "pixel_sha256", "possible_filename_group")


class DisjointSets:
    """Union-find over image IDs. The root of every set is its smallest ID."""

    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _bosch_groups(table: pd.DataFrame) -> pd.Series:
    """'possible_bosch_P<n>' for Bosch images that carry a P-number, else ''."""
    numbers = table.image_id.str.extract(r"^" + BOSCH_ID.pattern + r"$", flags=re.IGNORECASE, expand=False)
    is_bosch = (table.dataset == "chaksu") & (table.camera == "Bosch")
    names = numbers.map(lambda n: f"possible_bosch_P{int(n)}" if isinstance(n, str) else "")
    return names.where(is_bosch, "")


def _analysis_groups(table: pd.DataFrame) -> pd.Series:
    """Connected components of images that share any known or possible identity."""
    sets = DisjointSets(table.image_id)
    for column in GROUP_COLUMNS:
        known = table[table[column] != ""]
        for _, ids in known.groupby(column).image_id:
            for other in ids.iloc[1:]:
                sets.union(ids.iloc[0], other)
    return table.image_id.map(sets.find)


def _destination(group: pd.DataFrame) -> tuple[str, str] | None:
    """New split and the reason for it, or None if the group can stay where it is."""
    splits = set(group["split"])
    releases = set(group.chaksu_release[group.dataset == "chaksu"])
    if any(name.startswith("quarantine") for name in splits):
        return "quarantine_inherited", "component includes previously quarantined image"
    if group.dataset.nunique() > 1:
        return "quarantine_cross_dataset", "possible shared identity across datasets"
    if len(releases) > 1:
        return "quarantine_possible_patient_overlap", "possible Bosch filename identity spans releases"
    if splits == {"train", "validation"}:
        return "train", "possible filename identity crosses development split; keep together in train"
    return None


def partition_rows(table: pd.DataFrame, name: str) -> pd.DataFrame:
    """Classification-eligible rows of one split, with an integer ``binary_label``."""
    rows = table[(table["split"] == name) & (table.classification_eligible == "1")]
    return rows.reset_index(drop=True).assign(binary_label=lambda r: r.binary_label.astype(int))


def effective_split(all_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Apply the grouping rule. Returns the full table (with ``original_split``,
    ``possible_filename_group``, ``analysis_group`` and ``split_change_reason`` added) and the
    classification rows of each split."""
    table = all_rows.copy()
    table["original_split"] = table["split"]
    table["possible_filename_group"] = _bosch_groups(table)
    table["analysis_group"] = _analysis_groups(table)
    table["split_change_reason"] = ""
    for _, group in table.groupby("analysis_group"):
        moved = _destination(group)
        if moved:
            table.loc[group.index, ["split", "split_change_reason"]] = moved
    active = table[table["split"].isin(SPLITS)]
    if (active.groupby("analysis_group")["split"].nunique() > 1).any():
        raise ValueError("Effective groups still cross evaluation splits")
    partitions = {name: partition_rows(table, name) for name in SPLITS}
    for name, rows in partitions.items():
        if set(rows.binary_label) != {0, 1}:
            raise ValueError(f"{name}: grouping leaves fewer than two classes; revise the protocol")
    return table, partitions


def split_counts(partitions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame([{"partition": name, "total": len(rows),
                          "negatives": int((rows.binary_label == 0).sum()),
                          "positives": int((rows.binary_label == 1).sum()),
                          "positive_fraction": rows.binary_label.mean()}
                         for name, rows in partitions.items()])


def save_effective_split(table: pd.DataFrame, partitions: dict[str, pd.DataFrame],
                         directory: Path) -> pd.DataFrame:
    """Write the effective split, its changes and its counts. Returns the rows that moved."""
    directory.mkdir(parents=True)
    table.to_csv(directory / "all.csv", index=False)
    for name, rows in partitions.items():
        rows.to_csv(directory / f"{name}_classification.csv", index=False)
    changes = table[table["split"] != table.original_split]
    changes.to_csv(directory / "changes.csv", index=False)
    split_counts(partitions).to_csv(directory / "counts.csv", index=False)
    return changes

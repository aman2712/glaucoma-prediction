import pandas as pd
import pytest

from glaucoma_project.data.common_split import build_common_split, check_common_split


def test_builds_the_split_of_record_with_counts_and_moved_images(export, tmp_path):
    root, _ = export
    manifest = build_common_split(root, tmp_path / "common")
    table = pd.read_csv(tmp_path / "common" / "all.csv")
    assert manifest["images"] == len(table) == 44 and manifest["moved_images"] == 0
    assert manifest["counts"]["train"] == {"classification": 16, "segmentation": 12}
    assert {"analysis_group", "original_split"} <= set(table.columns)
    assert (tmp_path / "common" / "changes.csv").exists() and (tmp_path / "common" / "common_split.json").exists()


def test_refuses_to_overwrite_an_existing_split(export, tmp_path):
    root, _ = export
    build_common_split(root, tmp_path / "common")
    with pytest.raises(FileExistsError):
        build_common_split(root, tmp_path / "common")


def test_check_passes_for_a_faithful_copy_and_fails_for_an_altered_one(export, tmp_path):
    root, _ = export
    build_common_split(root, tmp_path / "common")
    check_common_split(root, tmp_path / "common")
    path = tmp_path / "common" / "all.csv"
    altered = pd.read_csv(path, dtype=str, keep_default_na=False)
    altered.loc[0, "split"] = "validation"
    altered.to_csv(path, index=False)
    with pytest.raises(ValueError, match="differs"):
        check_common_split(root, tmp_path / "common")

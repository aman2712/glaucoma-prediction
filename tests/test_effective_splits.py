import pandas as pd

from glaucoma_project.data.effective_splits import DisjointSets, effective_split, partition_rows, save_effective_split


def test_unchanged_when_no_identity_overlaps(export):
    _, table = export
    effective, partitions = effective_split(table)
    assert (effective["split"] == effective.original_split).all()
    assert len(partitions["train"]) == 16 and partitions["train"].binary_label.dtype == int


def test_bosch_p_number_groups_and_quarantines_across_releases_or_keeps_development_together(export):
    _, table = export
    test_row = table[table["split"] == "in_domain_test"].index[0]
    val_row = table[table["split"] == "validation"].index[0]
    table.loc[0, ["image_id", "camera"]] = ["chaksu_Bosch_train_P10_Image1", "Bosch"]
    table.loc[test_row, ["image_id", "camera"]] = ["chaksu_Bosch_test_P10_Image2", "Bosch"]
    table.loc[1, "group_id"] = table.loc[0, "group_id"]  # a known duplicate of the P10 train image
    table.loc[2, ["image_id", "camera"]] = ["chaksu_Bosch_train_P1_Image1", "Bosch"]
    table.loc[val_row, ["image_id", "camera"]] = ["chaksu_Bosch_train_P1_Image2", "Bosch"]
    effective, _ = effective_split(table)
    assert effective.loc[[0, 1, test_row], "split"].str.startswith("quarantine").all()
    assert effective.loc[val_row, "split"] == "train"
    assert effective.loc[2, "analysis_group"] == effective.loc[val_row, "analysis_group"]
    assert table.loc[val_row, "split"] == "validation"  # the input is not modified


def test_result_does_not_depend_on_row_order(export):
    _, table = export
    columns = ["image_id", "split", "analysis_group"]
    a, _ = effective_split(table)
    b, _ = effective_split(table.sample(frac=1, random_state=7))
    pd.testing.assert_frame_equal(a[columns].sort_values("image_id").reset_index(drop=True),
                                  b[columns].sort_values("image_id").reset_index(drop=True))


def test_disjoint_sets_root_is_the_smallest_id():
    sets = DisjointSets(["c", "a", "b", "d"])
    sets.union("c", "b")
    sets.union("b", "a")
    assert {sets.find(i) for i in "abc"} == {"a"} and sets.find("d") == "d"


def test_saved_split_can_be_read_back_with_partition_rows(export, tmp_path):
    _, table = export
    effective, partitions = effective_split(table)
    changes = save_effective_split(effective, partitions, tmp_path / "out")
    saved = pd.read_csv(tmp_path / "out" / "all.csv", dtype=str, keep_default_na=False)
    assert changes.empty and partition_rows(saved, "validation").image_id.tolist() == partitions["validation"].image_id.tolist()

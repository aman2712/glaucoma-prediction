import pytest

from glaucoma_project.data.audit import audit_export
from glaucoma_project.files import read_table


def test_valid_export_passes(export):
    root, table = export
    audited, summary = audit_export(root)
    assert audited.equals(table) and summary["classification_counts"]["train"] == 16


def test_missing_export_names_the_problem(tmp_path):
    with pytest.raises(FileNotFoundError, match="incomplete"):
        audit_export(tmp_path / "missing")


def test_altered_split_file_is_detected(export):
    root, _ = export
    with (root / "splits" / "all.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="integrity"):
        audit_export(root)


def test_per_split_file_that_disagrees_with_all_csv_is_detected(export):
    root, _ = export
    path = root / "splits" / "train_classification.csv"
    damaged = read_table(path)
    damaged.loc[0, "binary_label"] = "1"
    damaged.to_csv(path, index=False)
    with pytest.raises(ValueError, match="differ"):
        audit_export(root)


def test_patient_shared_between_splits_is_rejected(export, rewrite_export):
    root, table = export
    table.loc[[0, 16], "patient_id"] = "shared_patient"
    rewrite_export(root, table)
    with pytest.raises(ValueError, match="crosses splits"):
        audit_export(root)


def test_papila_suspect_label_cannot_be_a_positive(export, rewrite_export):
    root, table = export
    table.loc[table[table["split"] == "external_test"].index[0], "original_label"] = "2"
    rewrite_export(root, table)
    with pytest.raises(ValueError, match="label mapping"):
        audit_export(root)


def test_split_holding_the_wrong_dataset_is_rejected(export, rewrite_export):
    root, table = export
    table.loc[0, ["dataset", "chaksu_release"]] = ["papila", ""]
    rewrite_export(root, table)
    with pytest.raises(ValueError, match="crossed evaluation roles"):
        audit_export(root)

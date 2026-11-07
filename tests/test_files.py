import json

import numpy as np

from glaucoma_project.files import read_table, write_json


def test_write_json_turns_undefined_numbers_into_null(tmp_path):
    write_json(tmp_path / "x.json", {"nan": float("nan"), "inf": np.float64("inf"), "n": np.int64(3), "p": tmp_path})
    assert json.loads((tmp_path / "x.json").read_text()) == {"nan": None, "inf": None, "n": 3, "p": str(tmp_path)}


def test_read_table_keeps_empty_cells_as_empty_text(tmp_path):
    (tmp_path / "t.csv").write_text("id,patient\na,\nb,p1\n")
    assert read_table(tmp_path / "t.csv").patient.tolist() == ["", "p1"]

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from glaucoma_project.combined_model import ratio_blend_study as rb
from glaucoma_project.data.common_split import load_common_split
from glaucoma_project.freeze import sha256_file
from glaucoma_project import REPO_ROOT
from glaucoma_project.measurements.classification_inputs import adapted_measurements, repo_relative

from test_robust_final import StubNet, rows_with_labels
from test_robust_segmentation import SIZE

CPU = torch.device("cpu")
RATIOS = ["vcdr", "hcdr", "area_cdr"]
TEST_PARTITIONS = ("in_domain_test", "external_test")


def test_adapted_measurements_give_the_three_ratios_of_every_row_and_leave_the_model_alone(tmp_path):
    _, table = rows_with_labels(tmp_path)
    model = StubNet()
    before = model.norm.running_mean.clone()
    measured = adapted_measurements(table, model, tmp_path, CPU)
    assert list(measured.columns) == ["image_id", *RATIOS] and list(measured.image_id) == list(table.image_id)
    assert measured[table.binary_label.values == 1].vcdr.mean() > measured[table.binary_label.values == 0].vcdr.mean()
    assert torch.equal(model.norm.running_mean, before)


@pytest.fixture
def study(export, common_split, tmp_path):
    """A development table with held-out image logits, test image logits and new ratios, all synthetic."""
    rng = np.random.default_rng(0)
    parts = load_common_split(common_split)
    development = pd.concat([parts["train"], parts["validation"]], ignore_index=True)
    label = development.binary_label.astype(int).to_numpy()
    for seed in (1, 2):
        folder = tmp_path / "p01b" / f"seed{seed}"
        folder.mkdir(parents=True)
        pd.DataFrame({"image_id": development.image_id, "label": label, "analysis_group": development.analysis_group,
                      "logit": label + rng.normal(0, 1, len(label))}).to_csv(folder / "oof_logits.csv", index=False)
    final = tmp_path / "p01b" / "final"
    final.mkdir()
    references = {}
    for name in TEST_PARTITIONS:
        rows = parts[name]
        y = rows.binary_label.astype(int).to_numpy()
        pd.DataFrame({"image_id": rows.image_id, "label": y, "analysis_group": rows.analysis_group,
                      "image_logit": y + rng.normal(0, 1, len(y)), "blend_logit": y + rng.normal(0, 1, len(y))}
                     ).to_csv(final / f"{name}.csv", index=False)
    everything = pd.concat([development] + [parts[n] for n in TEST_PARTITIONS], ignore_index=True)
    pd.DataFrame({"image_id": everything.image_id, **{c: rng.uniform(0.2, 0.8, len(everything)) for c in RATIOS}}
                 ).to_csv(tmp_path / "measurements.csv", index=False)
    tests = pd.concat([parts[n].assign(split=n) for n in TEST_PARTITIONS])
    pd.DataFrame({"image_id": tests.image_id, "split": tests.split, "binary_label": tests.binary_label.astype(float),
                  "score": rng.uniform(size=len(tests))}).to_csv(tmp_path / "x5_scores.csv", index=False)
    pd.DataFrame({"image_id": tests.image_id, "split": tests.split, "source": "predicted", "score": rng.uniform(size=len(tests))}
                 ).to_csv(tmp_path / "baseline_scores.csv", index=False)
    return {"name": "x9", "runs_dir": str(tmp_path / "runs"), "image_runs_dir": str(tmp_path / "p01b"),
            "image_scores": str(tmp_path / "p01b" / "final"), "measurements": str(tmp_path / "measurements.csv"),
            "references": {"x5 measurements only": str(tmp_path / "x5_scores.csv"),
                           "baseline a": str(tmp_path / "baseline_scores.csv")},
            "n_folds": 2, "fold_seed": 2440, "seeds": [1, 2], "c_grid": [0.1, 1], "n_bootstrap": 30, "bootstrap_seed": 0}


def test_fitting_uses_the_old_held_out_image_scores_with_the_new_ratios_and_writes_only_to_its_own_folder(study, tmp_path):
    summary = rb.fit_study(study)
    assert summary["c"] in study["c_grid"] and {"image_only_auc", "blend_cv_auc"} <= set(summary)
    assert (tmp_path / "runs" / "blend.joblib").is_file() and (tmp_path / "runs" / "development.json").is_file()
    assert not (tmp_path / "p01b" / "blend.joblib").exists()
    with pytest.raises(FileExistsError):
        rb.fit_study(study)


def test_a_frozen_file_that_changes_blocks_scoring_and_scoring_runs_once(study, tmp_path, monkeypatch):
    rb.fit_study(study)
    files = {"blend": tmp_path / "runs" / "blend.joblib", "measurements": tmp_path / "measurements.csv"}
    record = {"files": {n: {"path": str(p), "sha256": sha256_file(p)} for n, p in files.items()}}
    monkeypatch.setattr(rb, "verify", lambda manifest: record)
    out = tmp_path / "scores"
    rb.score_test_sets(study, out, manifest=tmp_path / "m.json")
    table = pd.read_csv(out / "external_test.csv")
    assert {"image_id", "label", "analysis_group", "image_logit", "p01b_blend_logit", "blend_logit"} <= set(table.columns)
    assert not table.blend_logit.isna().any()
    with pytest.raises(FileExistsError):
        rb.score_test_sets(study, out, manifest=tmp_path / "m.json")


def test_the_analysis_compares_the_blend_with_every_reference_on_the_shared_images(study, tmp_path, monkeypatch):
    rb.fit_study(study)
    monkeypatch.setattr(rb, "verify", lambda manifest: {"files": {}})
    rb.score_test_sets(study, tmp_path / "scores", manifest=tmp_path / "m.json")
    tables = rb.analyse_study(study, tmp_path / "scores", tmp_path / "analysis")
    primary, secondary = tables["primary"], tables["secondary"]
    assert set(primary.partition) == set(TEST_PARTITIONS) and set(primary.comparison) == {"x9 blend minus x5 measurements only"}
    assert set(secondary.comparison) == {"x9 blend minus image score", "x9 blend minus p01b blend",
                                         "x9 blend minus baseline a"}
    assert {"difference", "ci_low", "ci_high", "auc_a", "auc_b"} <= set(primary.columns) and (primary.n > 0).all()


def test_a_manifest_path_is_recorded_relative_to_the_repository_whether_given_relative_or_absolute(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    assert repo_relative(Path("docs/frozen_x5.json")) == "docs/frozen_x5.json"
    assert repo_relative(REPO_ROOT / "docs" / "frozen_x5.json") == "docs/frozen_x5.json"

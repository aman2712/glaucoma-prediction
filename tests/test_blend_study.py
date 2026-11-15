import json

import numpy as np
import pandas as pd
import pytest
import torch

from glaucoma_project.combined_model import blend_study as study_module
from glaucoma_project.combined_model.blend_study import (analyse_study, fit_study, score_test_sets, seed_directory,
                                                         train_seed)
from glaucoma_project.freeze import sha256_file
from glaucoma_project.runtime import git_commit

pytestmark = pytest.mark.filterwarnings("ignore:Software test only")
torch.set_num_threads(2)
CPU = torch.device("cpu")
RATIOS = ["vcdr", "hcdr", "area_cdr"]


@pytest.fixture
def study(export, tmp_path):
    root, rows = export
    measurements = tmp_path / "measurements.csv"
    rows[["image_id", "binary_label"]].assign(**{c: np.linspace(0.2, 0.8, len(rows)) for c in RATIOS}).to_csv(
        measurements, index=False)
    recipe = {"model_name": "resnet18", "pretrained": False, "batch_size": 8, "warmup_epochs": 1,
              "head_lr": 1e-3, "backbone_lr": 1e-4, "weight_decay": 1e-4, "augment": False}
    return {"recipe": recipe, "epochs": 2, "n_folds": 2, "fold_seed": 2440, "seeds": [1, 2], "c_grid": [0.1, 1],
            "runs_dir": str(tmp_path / "runs"), "measurements": str(measurements), "n_bootstrap": 30,
            "bootstrap_seed": 0}


@pytest.fixture
def trained(study, export, common_split):
    root, _ = export
    for seed in study["seeds"]:
        train_seed(study, seed, root, CPU, common_split)
    return study


def test_a_seed_writes_its_fold_models_held_out_scores_and_environment(trained):
    folder = seed_directory(trained, 1)
    assert {"fold0.pt", "fold1.pt", "oof_logits.csv", "run.json"} <= {p.name for p in folder.iterdir()}
    assert len(pd.read_csv(folder / "oof_logits.csv")) == 28  # every development image once


def test_a_seed_folder_is_never_reused(trained, export, common_split):
    with pytest.raises(FileExistsError):
        train_seed(trained, 1, export[0], CPU, common_split)


def test_fitting_saves_the_blend_and_the_development_diagnostic(trained, tmp_path):
    summary = fit_study(trained)
    folder = tmp_path / "runs"
    assert (folder / "blend.joblib").is_file() and summary["c"] in trained["c_grid"]
    assert {"image_only_auc", "blend_cv_auc"} <= set(json.loads((folder / "development.json").read_text()))


def manifest_for(study, tmp_path):
    runs = tmp_path / "runs"
    files = {f"seed{seed}_fold{fold}": runs / f"seed{seed}" / f"fold{fold}.pt"
             for seed in study["seeds"] for fold in range(study["n_folds"])}
    files["blend"] = runs / "blend.joblib"
    entries = {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in files.items()}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"git_commit": git_commit(), "code_paths": ["no_such_path"], "files": entries}))
    return path


def test_test_sets_are_scored_once_and_analysed(trained, export, common_split, tmp_path):
    root, _ = export
    fit_study(trained)
    manifest, scores = manifest_for(trained, tmp_path), tmp_path / "scores"
    score_test_sets(trained, scores, manifest, root, CPU, common_split)
    assert {p.name for p in scores.iterdir()} == {"in_domain_test.csv", "external_test.csv"}
    with pytest.raises(FileExistsError):
        score_test_sets(trained, scores, manifest, root, CPU, common_split)

    baseline = pd.concat([pd.read_csv(scores / f"{p}.csv")[["image_id"]].assign(split=p, source="predicted", score=0.5)
                          for p in ("in_domain_test", "external_test")])
    tables = analyse_study(trained, scores, tmp_path / "analysis", baseline)
    assert set(tables["primary"].partition) == {"in_domain_test", "external_test"}
    assert {"difference", "ci_low", "ci_high"} <= set(tables["primary"].columns) and len(tables["secondary"]) == 2


def test_a_changed_checkpoint_blocks_scoring(trained, export, common_split, tmp_path):
    fit_study(trained)
    manifest = manifest_for(trained, tmp_path)
    checkpoint = seed_directory(trained, 1) / "fold0.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"0")
    with pytest.raises(RuntimeError, match="Frozen file changed"):
        score_test_sets(trained, tmp_path / "scores", manifest, export[0], CPU, common_split)
    assert not (tmp_path / "scores").exists()

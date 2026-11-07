import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from glaucoma_project.combined_model import study as study_module
from glaucoma_project.combined_model.analysis import analyse, ensemble
from glaucoma_project.combined_model.study import run_names, score_test_sets
from glaucoma_project.freeze import sha256_file
from glaucoma_project.image_model.develop import develop
from glaucoma_project.runtime import git_commit

pytestmark = pytest.mark.filterwarnings("ignore:Software test only")
torch.set_num_threads(2)
RECIPE = {"model_name": "resnet18", "pretrained": False, "seed": 0, "batch_size": 8, "epochs": 2,
          "warmup_epochs": 1, "patience": 2, "head_lr": 1e-3, "backbone_lr": 1e-4, "weight_decay": 1e-4,
          "augment": False, "target_sensitivity": 0.9, "n_bootstrap": 20}
COLUMNS = ["vcdr", "hcdr", "area_cdr"]


@pytest.fixture
def trained(export, common_split, tmp_path):
    """Two arms x two seeds, trained on the synthetic export, and the study that describes them."""
    root, rows = export
    measurements = tmp_path / "measurements.csv"
    rows[["image_id", "binary_label"]].assign(**{c: np.linspace(0.2, 0.8, len(rows)) for c in COLUMNS}).to_csv(
        measurements, index=False)
    arms = {}
    for arm, extra in {"image_only": {}, "combined": {"measurement_columns": COLUMNS}}.items():
        arms[arm] = tmp_path / f"{arm}.yaml"
        arms[arm].write_text(yaml.safe_dump({**RECIPE, **extra, "run_name": f"p01_{arm}"}))
    study = {"arms": {k: str(v) for k, v in arms.items()}, "seeds": [1, 2], "runs_dir": str(tmp_path / "runs"),
             "measurements": str(measurements), "n_bootstrap": 30, "bootstrap_seed": 0, "target_sensitivity": 0.9}
    for (arm, seed), name in run_names(study).items():
        config = {**yaml.safe_load(arms[arm].read_text()), "seed": seed, "run_name": name}
        develop(config, root=root, runs_dir=tmp_path / "runs", splits_dir=common_split,
                measurements=measurements if arm == "combined" else None)
    return root, common_split, study, tmp_path


def manifest_for(study, tmp_path):
    files = {f"{arm}_seed{seed}": {"path": str(tmp_path / "runs" / name / "best.pt"),
                                   "sha256": sha256_file(tmp_path / "runs" / name / "best.pt")}
             for (arm, seed), name in run_names(study).items()}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"git_commit": git_commit(), "code_paths": ["no_such_path"], "files": files}))
    return path


def test_run_names_cover_every_arm_and_seed(trained):
    *_, study, _ = trained
    assert sorted(run_names(study).values()) == ["p01_combined_seed1", "p01_combined_seed2",
                                                 "p01_image_only_seed1", "p01_image_only_seed2"]


def test_freezing_needs_every_run(tmp_path, monkeypatch):
    monkeypatch.setattr(study_module, "REPO_ROOT", tmp_path)
    arm = tmp_path / "arm.yaml"
    arm.write_text("run_name: x\n")
    study = {"arms": {"image_only": str(arm)}, "seeds": [1], "runs_dir": "runs", "measurements": "m.csv"}
    with pytest.raises(FileNotFoundError, match="Runs not finished"):
        study_module.freeze_study(study, tmp_path / "m.json")


def test_test_sets_are_scored_once_and_analysed(trained):
    root, common, study, tmp_path = trained
    manifest = manifest_for(study, tmp_path)
    scores = tmp_path / "scores"
    score_test_sets(study, scores, manifest, root, torch.device("cpu"), common)
    assert len(list(scores.iterdir())) == 8  # 2 arms x 2 seeds x 2 test sets
    with pytest.raises(FileExistsError):
        score_test_sets(study, scores, manifest, root, torch.device("cpu"), common)

    baseline = pd.concat([pd.read_csv(scores / f"p01_combined_seed1_{p}.csv")[["image_id"]].assign(
        split=p, source="predicted", score=0.5) for p in ("in_domain_test", "external_test")])
    tables = analyse(study, scores, tmp_path / "runs", tmp_path / "analysis", baseline)
    assert set(tables["primary"].partition) == {"in_domain_test", "external_test"}
    assert {"difference", "ci_low", "ci_high"} <= set(tables["primary"].columns)
    assert len(tables["per_seed"]) == 4 and len(tables["secondary"]) == 4
    assert set(tables["arm_metrics"].arm) == {"image_only", "combined"}


def test_a_changed_checkpoint_blocks_scoring(trained):
    root, common, study, tmp_path = trained
    manifest = manifest_for(study, tmp_path)
    checkpoint = tmp_path / "runs" / "p01_combined_seed1" / "best.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"0")
    with pytest.raises(RuntimeError, match="Frozen file changed"):
        score_test_sets(study, tmp_path / "scores", manifest, root, torch.device("cpu"), common)
    assert not (tmp_path / "scores").exists()


def test_ensemble_averages_logits_and_rejects_different_images():
    frame = lambda logit, ids=("a", "b"): pd.DataFrame({"image_id": ids, "label": [0, 1], "logit": logit,
                                                        "probability": 0.5, "seed": 1})
    mean = ensemble([frame([0.0, 2.0]), frame([2.0, 4.0])])
    assert mean.logit.tolist() == [1.0, 3.0] and np.allclose(mean.probability, 1 / (1 + np.exp(-mean.logit)))
    with pytest.raises(ValueError, match="different images"):
        ensemble([frame([0.0, 1.0]), frame([0.0, 1.0], ids=("a", "c"))])

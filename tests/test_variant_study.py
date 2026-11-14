import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from glaucoma_project.combined_model import variant_study as vs
from glaucoma_project.data.common_split import load_common_split
from glaucoma_project.freeze import sha256_file
from glaucoma_project.image_model.develop import develop
from glaucoma_project.runtime import git_commit

pytestmark = pytest.mark.filterwarnings("ignore:Software test only")
torch.set_num_threads(2)
RECIPE = {"model_name": "resnet18", "pretrained": False, "seed": 0, "batch_size": 8, "epochs": 2,
          "warmup_epochs": 1, "patience": 2, "head_lr": 1e-3, "backbone_lr": 1e-4, "weight_decay": 1e-4,
          "augment": False, "target_sensitivity": 0.9, "n_bootstrap": 20}


def partitions_with_two_cameras(common_split):
    partitions = load_common_split(common_split)
    for name in ("train", "validation"):
        rows = partitions[name].copy()
        rows.loc[rows.index[::3], "camera"] = "Bosch"
        partitions[name] = rows
    return partitions


@pytest.fixture
def study(export, common_split, tmp_path):
    root, _ = export
    for name, run_name in (("variant", "x9_variant"), ("baseline", "p01_image_only")):
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump({**RECIPE, "run_name": run_name}))
    return {"name": "x9", "variant_config": str(tmp_path / "variant.yaml"), "baseline_config": str(tmp_path / "baseline.yaml"),
            "baseline_scores": str(tmp_path / "baseline_scores"), "seeds": [1, 2], "gate_seeds": [1, 2],
            "source_camera": "Bosch", "runs_dir": str(tmp_path / "runs"), "n_bootstrap": 30, "bootstrap_seed": 0,
            "target_sensitivity": 0.9}


def test_gate_jobs_list_the_variant_then_the_baseline_for_each_gate_seed(study):
    assert vs.gate_jobs(study) == [("variant", 1), ("variant", 2), ("baseline", 1), ("baseline", 2)]
    assert vs.jobs(study) == [1, 2]


def test_the_gate_trains_on_the_source_camera_and_scores_the_other_cameras(common_split):
    gate = vs.gate_rows(partitions_with_two_cameras(common_split), source_camera="Bosch")
    assert set(gate.train.camera) == {"Bosch"} and set(gate.validation.camera) == {"Bosch"}
    assert "Bosch" not in set(gate.shifted.camera) and len(gate.shifted) > 0
    assert not set(gate.shifted.image_id) & set(gate.train.image_id)


def test_gate_rows_never_include_a_test_image(common_split):
    gate = vs.gate_rows(partitions_with_two_cameras(common_split), source_camera="Bosch")
    tests = load_common_split(common_split)
    test_ids = set(tests["in_domain_test"].image_id) | set(tests["external_test"].image_id)
    assert not test_ids & (set(gate.train.image_id) | set(gate.validation.image_id) | set(gate.shifted.image_id))


def test_a_gate_run_writes_scores_for_the_shifted_images_and_is_never_reused(study, export, common_split):
    root, _ = export
    gate = vs.gate_rows(partitions_with_two_cameras(common_split), source_camera="Bosch")
    folder = vs.run_gate(study, "variant", 1, gate, root)
    scores = pd.read_csv(folder / "scores.csv")
    assert set(scores.image_id) == set(gate.shifted.image_id)
    with pytest.raises(FileExistsError):
        vs.run_gate(study, "variant", 1, gate, root)


def test_the_verdict_passes_only_when_the_variant_has_the_higher_mean_auc(study, tmp_path):
    runs = vs.runs_directory(study) / "gate"
    labels = np.tile([0, 1], 10)
    for arm, quality, seeds in (("variant", 2.0, [1, 2]), ("baseline", 0.5, [1, 2])):
        for seed in seeds:
            folder = runs / f"{arm}_seed{seed}"
            folder.mkdir(parents=True)
            noise = np.random.default_rng(seed).standard_normal(20)
            pd.DataFrame({"image_id": [f"i{k}" for k in range(20)], "label": labels,
                          "logit": quality * labels + noise}).to_csv(folder / "scores.csv", index=False)
    verdict = vs.gate_verdict(study)
    assert verdict["passes"] is True and verdict["variant"]["mean_auc"] > verdict["baseline"]["mean_auc"]


def test_the_verdict_is_refused_while_runs_are_missing(study):
    with pytest.raises(FileNotFoundError):
        vs.gate_verdict(study)


@pytest.fixture
def trained(study, export, common_split, tmp_path):
    root, _ = export
    for seed in study["seeds"]:
        for arm, key in (("variant", "variant_config"), ("baseline", "baseline_config")):
            config = {**yaml.safe_load(open(study[key]).read()), "seed": seed}
            config["run_name"] = f"{config['run_name']}_seed{seed}"
            runs = vs.runs_directory(study) if arm == "variant" else tmp_path / "baseline_runs"
            develop(config, root=root, runs_dir=runs, splits_dir=common_split)
    return root, common_split


def manifest_for(study, tmp_path):
    files = {f"seed{seed}": {"path": str(vs.runs_directory(study) / f"x9_variant_seed{seed}" / "best.pt"),
                             "sha256": sha256_file(vs.runs_directory(study) / f"x9_variant_seed{seed}" / "best.pt")}
             for seed in study["seeds"]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"git_commit": git_commit(), "code_paths": ["no_such_path"], "files": files}))
    return path


def test_scoring_writes_one_file_per_seed_and_test_set_and_refuses_a_second_run(study, trained, tmp_path):
    root, splits = trained
    manifest = manifest_for(study, tmp_path)
    out = tmp_path / "final"
    vs.score_test_sets(study, out, manifest, root, torch.device("cpu"), splits)
    assert sorted(p.name for p in out.glob("*.csv")) == [f"x9_variant_seed{s}_{p}.csv" for s in (1, 2)
                                                         for p in ("external_test", "in_domain_test")]
    with pytest.raises(FileExistsError):
        vs.score_test_sets(study, out, manifest, root, torch.device("cpu"), splits)


def test_the_analysis_compares_the_variant_with_the_baseline_on_each_test_set(study, trained, tmp_path):
    root, splits = trained
    manifest = manifest_for(study, tmp_path)
    out = tmp_path / "final"
    vs.score_test_sets(study, out, manifest, root, torch.device("cpu"), splits)
    base = tmp_path / "baseline_scores"
    base.mkdir()
    for seed in study["seeds"]:
        for partition in vs.TEST_PARTITIONS:
            frame = pd.read_csv(out / f"x9_variant_seed{seed}_{partition}.csv")
            frame.assign(logit=np.random.default_rng(seed).standard_normal(len(frame))).to_csv(
                base / f"p01_image_only_seed{seed}_{partition}.csv", index=False)
    tables = vs.analyse_study(study, out, tmp_path / "analysis")
    assert set(tables["primary"].partition) == set(vs.TEST_PARTITIONS)
    assert {"difference", "ci_low", "ci_high"} <= set(tables["primary"].columns)
    assert len(tables["per_seed"]) == 4

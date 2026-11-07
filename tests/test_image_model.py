import gc
import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from glaucoma_project.image_model import final
from glaucoma_project.image_model.develop import develop
from glaucoma_project.image_model.engine import set_training_mode
from glaucoma_project.image_model.model import ARCHITECTURES, build_model, head_parameters
from glaucoma_project.image_model.predict import attach_metadata, operating_threshold, score_photo
from glaucoma_project.image_model.report import evaluation_tables

pytestmark = pytest.mark.filterwarnings("ignore:Software test only")
torch.set_num_threads(2)
CPU = torch.device("cpu")
CONFIG = {"run_name": "test", "model_name": "resnet18", "pretrained": False, "seed": 2440, "batch_size": 8,
          "epochs": 4, "warmup_epochs": 1, "patience": 3, "head_lr": 1e-3, "backbone_lr": 1e-4,
          "weight_decay": 1e-4, "augment": True, "target_sensitivity": 0.9, "n_bootstrap": 20}


@pytest.mark.parametrize("name", sorted(ARCHITECTURES))
def test_every_architecture_returns_one_finite_logit(name):
    model, weights = build_model(name, pretrained=False)
    with torch.inference_mode():
        logits = model.eval()(torch.zeros(1, 3, 224, 224))
    assert tuple(logits.shape) == (1, 1) and torch.isfinite(logits).all() and weights == "NONE_TEST_ONLY"
    assert len(head_parameters(model)) >= 2
    del model
    gc.collect()


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model("vgg", pretrained=False)


def test_batchnorm_statistics_stay_fixed_while_weights_still_learn():
    model, _ = build_model("resnet18", pretrained=False)
    before = model.bn1.running_mean.clone()
    set_training_mode(model)
    model(torch.randn(2, 3, 224, 224)).sum().backward()
    torch.testing.assert_close(before, model.bn1.running_mean)
    assert model.conv1.weight.grad is not None


def test_predictions_are_joined_by_id_and_label_disagreement_is_caught(export):
    from glaucoma_project.data.effective_splits import effective_split
    rows = effective_split(export[1])[1]["validation"]
    shuffled = pd.DataFrame({"image_id": rows.image_id, "label": rows.binary_label, "probability": 0.5}).iloc[::-1]
    assert len(attach_metadata(shuffled, rows)) == len(rows)
    with pytest.raises(ValueError, match="alignment"):
        attach_metadata(shuffled.assign(label=1 - shuffled.label), rows)


def test_cameras_with_few_cases_are_flagged():
    predictions = pd.DataFrame({"label": [0, 1] * 6, "probability": np.linspace(0.1, 0.9, 12),
                                "camera": ["A"] * 12})
    _, cameras = evaluation_tables(predictions, 0.5)
    assert cameras.small_class_count.tolist() == [True]


def test_development_freeze_and_one_shot_final_evaluation(export, tmp_path):
    root, _ = export
    run = develop(CONFIG, root=root, runs_dir=tmp_path / "runs")
    expected = {"best.pt", "config.json", "run.json", "input_audit.json", "history.csv", "learning_curves.png",
                "validation_predictions.csv", "validation_metrics.csv", "validation_camera_metrics.csv"}
    assert expected <= {p.name for p in run.iterdir()}
    assert not (run / "final_evaluation").exists()  # development never touches the test splits
    final.check_saved_run(run)

    protocol = final.freeze_run(run)
    assert protocol["threshold"] == operating_threshold(run) and final.freeze_run(run) == protocol
    result = final.final_evaluation(run, CPU, root)
    assert result.partition.tolist() == ["in_domain_test", "in_domain_test", "external_test", "external_test"]
    assert (run / "final_evaluation" / "completion.json").exists()
    assert json.loads((run / "final_evaluation" / "completion.json").read_text())["complete"]
    with pytest.raises(FileExistsError):
        final.final_evaluation(run, CPU, root)
    with pytest.raises(ValueError, match="already exists"):
        final.check_saved_run(run)

    Image.new("RGB", (80, 40), (200, 100, 50)).save(tmp_path / "new.png")
    score = score_photo(run, tmp_path / "new.png", CPU)
    assert 0 <= score["glaucoma_suspicion_score"] <= 1 and score["threshold"] == protocol["threshold"]


def test_a_changed_checkpoint_is_refused_after_freezing(export, tmp_path):
    root, _ = export
    run = develop(CONFIG, root=root, runs_dir=tmp_path / "runs")
    final.freeze_run(run)
    (run / "best.pt").write_bytes((run / "best.pt").read_bytes() + b"0")
    with pytest.raises(ValueError, match="Checkpoint changed"):
        final.final_evaluation(run, CPU, root)
    assert not (run / "final_evaluation").exists()


def test_a_run_folder_is_never_reused(export, tmp_path):
    root, _ = export
    develop(CONFIG, root=root, runs_dir=tmp_path / "runs")
    with pytest.raises(FileExistsError):
        develop(CONFIG, root=root, runs_dir=tmp_path / "runs")

import numpy as np
import pandas as pd
import pytest
import torch

from glaucoma_project.data.classification import FundusDataset
from glaucoma_project.data.measurement_inputs import apply_scaler, attach_measurements, fit_scaler
from glaucoma_project.image_model.develop import develop
from glaucoma_project.image_model.model import build_model, head_parameters, restore_model

COLUMNS = ["vcdr", "hcdr", "area_cdr"]
CONFIG = {"run_name": "combined", "model_name": "resnet18", "pretrained": False, "seed": 2440, "batch_size": 8,
          "epochs": 3, "warmup_epochs": 1, "patience": 2, "head_lr": 1e-3, "backbone_lr": 1e-4,
          "weight_decay": 1e-4, "augment": False, "target_sensitivity": 0.9, "n_bootstrap": 20,
          "measurement_columns": COLUMNS}


def table(n=6):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"image_id": [f"i{k}" for k in range(n)], "binary_label": [k % 2 for k in range(n)],
                         **{c: rng.uniform(0.2, 0.8, n) for c in COLUMNS}})


def test_scaler_uses_training_statistics_and_fills_missing_with_the_median():
    train = table()
    train.loc[0, "vcdr"] = np.nan
    scaler = fit_scaler(train, COLUMNS)
    scaled = apply_scaler(train, scaler)
    assert scaled.vcdr.isna().sum() == 0
    assert np.allclose(scaled[COLUMNS].mean(), 0, atol=1e-9) and np.allclose(scaled[COLUMNS].std(ddof=0), 1)
    new = table(3).assign(vcdr=[np.nan, 0.5, 0.5])
    expected = (np.array([scaler["median"][0], 0.5]) - scaler["mean"][0]) / scaler["std"][0]
    assert np.allclose(apply_scaler(new, scaler).vcdr[:2], expected)  # not refitted on the new rows


def test_attach_checks_coverage_labels_and_duplicates():
    rows, measured = table(), table()
    assert len(attach_measurements(rows[["image_id", "binary_label"]], measured, COLUMNS)) == 6
    with pytest.raises(ValueError, match="no measurement entry"):
        attach_measurements(rows, measured.iloc[1:], COLUMNS)
    with pytest.raises(ValueError, match="labels disagree"):
        attach_measurements(rows, measured.assign(binary_label=1 - measured.binary_label), COLUMNS)
    with pytest.raises(ValueError, match="Duplicate"):
        attach_measurements(rows, pd.concat([measured, measured]), COLUMNS)


def test_model_with_measurements_uses_them_and_has_one_head():
    model, _ = build_model("resnet18", pretrained=False, n_measurements=3)
    images = torch.zeros(2, 3, 224, 224)
    model.eval()
    with torch.inference_mode():
        assert model(images, torch.zeros(2, 3)).shape == (2, 1)
        assert not torch.allclose(model(images, torch.zeros(2, 3)), model(images, torch.ones(2, 3)))
    assert model.head.in_features == 515 and len(head_parameters(model)) == 2
    with pytest.raises(ValueError, match="only be added to a ResNet"):
        build_model("efficientnet_b0", pretrained=False, n_measurements=3)


def test_dataset_yields_image_and_measurements_together(export):
    root, rows = export
    rows = rows[rows.split == "train"].assign(binary_label=lambda r: r.binary_label.astype(int))
    rows = rows.assign(**{c: 0.5 for c in COLUMNS})
    (image, measures), label, image_id = FundusDataset(rows, root, measurement_columns=COLUMNS)[0]
    assert image.shape == (3, 224, 224) and measures.tolist() == [0.5, 0.5, 0.5]


@pytest.mark.filterwarnings("ignore:Software test only")
def test_development_with_measurements_saves_the_scaler_and_reloads(export, common_split, tmp_path):
    root, rows = export
    measurements = rows[["image_id", "binary_label"]].assign(
        **{c: np.linspace(0.2, 0.8, len(rows)) for c in COLUMNS})
    path = tmp_path / "measurements.csv"
    measurements.to_csv(path, index=False)
    run = develop(CONFIG, root=root, runs_dir=tmp_path / "runs", splits_dir=common_split, measurements=path)
    model, checkpoint = restore_model(run / "best.pt", torch.device("cpu"))
    assert checkpoint["measurement_scaler"]["columns"] == COLUMNS and model.head.in_features == 515
    assert (pd.read_csv(run / "validation_predictions.csv").columns == "logit").any()
    with pytest.raises(ValueError, match="go together"):
        develop({**CONFIG, "run_name": "other"}, root=root, runs_dir=tmp_path / "runs", splits_dir=common_split)

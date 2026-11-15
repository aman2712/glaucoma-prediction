import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from glaucoma_project.segmentation import robust_final as rf

from test_robust_segmentation import SIZE, write_rows

CPU = torch.device("cpu")
CONFIG = {"batch_size": 2, "num_workers": 0}
BASELINE_A = {"features": ["vcdr", "hcdr", "area_cdr"], "c_grid": [1.0], "cv_folds": 2, "n_bootstrap": 20, "seed": 0}


class StubNet(nn.Module):
    """Draws a disc and a cup whose size follows the brightness of the photograph, so that bright
    photographs (label 1 in these tests) get a larger cup-to-disc ratio."""

    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm2d(3)

    def forward(self, image):
        self.norm(image)
        logits = torch.full((len(image), 2, SIZE, SIZE), -5.0)
        for i, one in enumerate(image):
            half = 10 if one.mean() > 0.6 else 4
            logits[i, 0, 16:48, 16:48] = 5.0
            logits[i, 1, 32 - half:32 + half, 32 - half:32 + half] = 5.0
        return logits


def rows_with_labels(root, n_train=12, n_validation=8):
    """Train and validation rows whose photograph brightness follows the label (even index: 0, odd: 1)."""
    names = [(f"t{i}", "Remidio", "train") for i in range(n_train)] + \
            [(f"v{i}", "Remidio", "validation") for i in range(n_validation)]
    splits = write_rows(root, names)
    table = pd.read_csv(splits / "all.csv")
    table["binary_label"] = [i % 2 for i in range(len(table))]
    table["analysis_group"] = table.image_id
    for row in table.itertuples():
        from PIL import Image
        value = 220 if row.binary_label else 90
        Image.fromarray(np.full((SIZE, SIZE, 3), value, np.uint8)).save(root / row.segmentation_image)
    table.to_csv(splits / "all.csv", index=False)
    return splits, table


def test_measuring_rows_gives_predicted_and_expert_ratios_and_dice(tmp_path):
    splits, table = rows_with_labels(tmp_path)
    measured = rf.measure_rows(StubNet(), table, tmp_path, CONFIG, CPU, adapt=False)
    assert list(measured.image_id) == list(table.image_id)
    assert {"vcdr", "hcdr", "area_cdr", "expert_vcdr", "dice_disc", "dice_cup", "binary_label", "analysis_group",
            "split"} <= set(measured.columns)
    assert measured.dice_disc.between(0, 1).all()
    assert measured[measured.binary_label == 1].vcdr.mean() > measured[measured.binary_label == 0].vcdr.mean()


def test_adaptation_is_applied_to_the_rows_being_measured_and_leaves_the_model_alone(tmp_path):
    _, table = rows_with_labels(tmp_path)
    model = StubNet()
    before = model.norm.running_mean.clone()
    rf.measure_rows(model, table, tmp_path, CONFIG, CPU, adapt=True)
    assert torch.equal(model.norm.running_mean, before)


def test_the_classifier_is_fitted_on_development_measurements_only(tmp_path):
    _, table = rows_with_labels(tmp_path)
    fitted = rf.fit_classifier(StubNet(), table, tmp_path, CONFIG, CPU, BASELINE_A)
    assert set(fitted) == {"model", "threshold", "best_c"} and 0 <= fitted["threshold"] <= 1
    assert len(fitted["model"].predict_proba(pd.DataFrame({"vcdr": [0.3], "hcdr": [0.3], "area_cdr": [0.1]}))) == 1


def test_comparison_is_paired_on_image_ids_and_reports_both_aucs():
    rng = np.random.default_rng(0)
    label = np.tile([0.0, 1.0], 40)
    good, poor = label + rng.normal(0, 0.3, 80), label + rng.normal(0, 1.5, 80)
    names = [f"i{k}" for k in range(80)]
    measurements = pd.DataFrame({"image_id": names, "split": "external_test", "binary_label": label,
                                 "analysis_group": names})
    baseline = pd.DataFrame({"image_id": names[::-1], "split": "external_test", "source": "predicted",
                             "binary_label": label[::-1], "score": poor[::-1]})
    table = rf.compare_with_baseline_a(measurements, good, baseline, n_bootstrap=50, seed=0)
    row = table.iloc[0]
    assert row.split == "external_test" and row.n == 80 and row.auc_a > row.auc_b
    assert row.ci_low < row.difference < row.ci_high


def test_a_missing_baseline_score_is_an_error():
    names = ["a", "b", "c", "d"]
    measurements = pd.DataFrame({"image_id": names, "split": "external_test", "binary_label": [0.0, 1.0, 0.0, 1.0],
                                 "analysis_group": names})
    baseline = pd.DataFrame({"image_id": names[:3], "split": "external_test", "source": "predicted",
                             "binary_label": [0.0, 1.0, 0.0], "score": [0.1, 0.9, 0.2]})
    with pytest.raises(ValueError, match="baseline"):
        rf.compare_with_baseline_a(measurements, np.array([0.1, 0.9, 0.2, 0.8]), baseline, 20, 0)


def test_scoring_runs_end_to_end_once_and_writes_every_table(tmp_path, monkeypatch):
    from PIL import Image
    names = [(f"{prefix}{i}", "Remidio", split) for prefix, split in
             (("t", "train"), ("v", "validation"), ("c", "in_domain_test"), ("p", "external_test")) for i in range(8)]
    splits = write_rows(tmp_path, names)
    table = pd.read_csv(splits / "all.csv")
    table["binary_label"] = [i % 2 for i in range(len(table))]
    table["analysis_group"] = table.image_id
    for row in table.itertuples():
        Image.fromarray(np.full((SIZE, SIZE, 3), 220 if row.binary_label else 90, np.uint8)).save(tmp_path / row.segmentation_image)
    table.to_csv(splits / "all.csv", index=False)
    baseline = table[table.split.isin(rf.TEST_SPLITS)][["image_id", "split", "binary_label"]].assign(
        source="predicted", score=lambda t: 0.3 + 0.4 * t.binary_label)
    baseline.to_csv(tmp_path / "baseline_scores.csv", index=False)
    classifier = rf.fit_classifier(StubNet(), table, tmp_path, CONFIG, CPU, BASELINE_A)
    joblib_path = tmp_path / "classifier.joblib"
    import joblib
    joblib.dump(classifier, joblib_path)
    import yaml
    (tmp_path / "variant.yaml").write_text(yaml.safe_dump({"run_name": "x9", "encoder": "resnet18", **CONFIG}))
    (tmp_path / "classifier.yaml").write_text(yaml.safe_dump(BASELINE_A))
    study = {"name": "x9", "final_seed": 1, "variant_config": str(tmp_path / "variant.yaml"),
             "classifier_config": str(tmp_path / "classifier.yaml"), "n_bootstrap": 20, "bootstrap_seed": 0,
             "runs_dir": str(tmp_path / "runs")}
    entry = lambda path: {"path": str(path)}
    record = {"files": {"segmentation_weights": entry(tmp_path / "best.pt"), "classifier": entry(joblib_path),
                        "baseline_a_scores": entry(tmp_path / "baseline_scores.csv")}}
    monkeypatch.setattr(rf, "verify", lambda manifest: record)
    monkeypatch.setattr(rf, "load_model", lambda *args, **kwargs: StubNet())

    result = rf.score_test_sets(study, tmp_path / "out", tmp_path, CPU, splits_dir=splits, manifest=tmp_path / "m.json")
    assert set(result["comparison"].split) == set(rf.TEST_SPLITS)
    produced = {p.name for p in (tmp_path / "out").iterdir()}
    assert produced == {"comparison.csv", "classification.csv", "segmentation.csv", "measurements_test.csv", "scores.csv"}
    assert set(result["segmentation"].adapted) == {True, False}
    with pytest.raises(FileExistsError):
        rf.score_test_sets(study, tmp_path / "out", tmp_path, CPU, splits_dir=splits, manifest=tmp_path / "m.json")

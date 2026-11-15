import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from glaucoma_project.combined_model import multitask_study as ms
from glaucoma_project.data.segmentation import PhotographDataset, SegmentationDataset
from glaucoma_project.segmentation import multitask as mt
from glaucoma_project.segmentation.model import DiscCupUNet

from test_robust_segmentation import SIZE, write_rows

CPU = torch.device("cpu")
torch.set_num_threads(2)
CONFIG = {"encoder": "resnet18", "pretrained": False, "epochs": 2, "batch_size": 4, "learning_rate": 1e-3,
          "weight_decay": 1e-4, "num_workers": 0, "augment_profile": "standard", "seed": 0}


def labelled_rows(root, names):
    """Rows of a tiny export whose photographs are bright for label 1 and dark for label 0."""
    splits = write_rows(root, names)
    table = pd.read_csv(splits / "all.csv")
    table["binary_label"] = [i % 2 for i in range(len(table))]
    table["analysis_group"] = table.image_id
    table["classification_eligible"] = 1
    table["patient_id"], table["classification_image"] = "", table.segmentation_image
    for row in table.itertuples():
        Image.fromarray(np.full((SIZE, SIZE, 3), 220 if row.binary_label else 60, np.uint8)).save(root / row.segmentation_image)
    table.to_csv(splits / "all.csv", index=False)
    return splits, table


def test_the_classifying_network_returns_masks_and_one_logit_per_image():
    model = DiscCupUNet("resnet18", pretrained=False, classify=True).eval()
    masks, logits = model(torch.rand(2, 3, SIZE, SIZE))
    assert masks.shape == (2, 2, SIZE, SIZE) and logits.shape == (2, 1)


def test_the_plain_network_is_unchanged():
    assert DiscCupUNet("resnet18", pretrained=False).eval()(torch.rand(1, 3, SIZE, SIZE)).shape == (1, 2, SIZE, SIZE)


def test_the_joint_loss_is_the_sum_of_the_segmentation_and_the_glaucoma_loss():
    masks, target = torch.randn(3, 2, 8, 8), (torch.rand(3, 2, 8, 8) > 0.5).float()
    logit, label = torch.randn(3, 1), torch.tensor([1.0, 0.0, 1.0])
    expected = mt.DiceBCELoss()(masks, target) + torch.nn.functional.binary_cross_entropy_with_logits(logit.reshape(-1), label)
    torch.testing.assert_close(mt.joint_loss(masks, logit, target, label), expected)


def test_the_dataset_items_carry_the_label_and_photographs_need_no_masks(tmp_path):
    splits, table = labelled_rows(tmp_path, [("a", "Remidio", "train"), ("b", "Remidio", "train")])
    items = SegmentationDataset("train", tmp_path, splits_dir=splits)
    assert [items[i]["label"] for i in range(2)] == [0.0, 1.0]
    for column in ("disc_mask", "cup_mask"):
        table[column] = "missing.png"
    photos = PhotographDataset(table, tmp_path)
    assert photos[1]["image"].shape == (3, SIZE, SIZE) and photos[1]["label"] == 1.0 and photos[1]["image_id"] == "b"


def test_training_writes_the_best_epoch_by_validation_auroc_and_the_history(tmp_path):
    names = [(f"t{i}", "Remidio", "train") for i in range(8)] + [(f"v{i}", "Remidio", "validation") for i in range(6)]
    splits, table = labelled_rows(tmp_path, names)
    train_rows, validation_rows = table[table.split == "train"], table[table.split == "validation"]
    run = tmp_path / "run"
    run.mkdir()
    torch.manual_seed(0)
    model = DiscCupUNet("resnet18", pretrained=False, classify=True).to(CPU)
    train_loader = DataLoader(SegmentationDataset(None, tmp_path, augment=True, rows=train_rows), batch_size=4, shuffle=True)
    validation_loader = DataLoader(PhotographDataset(validation_rows, tmp_path), batch_size=4)
    history = mt.fit_multitask(model, train_loader, validation_loader, run, CONFIG, CPU)
    assert list(history.epoch) == [1, 2] and history.validation_auroc.between(0, 1).all()
    assert (run / "best.pt").exists() and (run / "history.csv").exists()
    scores = mt.predict_logits(model, validation_loader, CPU)
    assert list(scores.columns) == ["image_id", "label", "logit"] and len(scores) == 6


def study_of(tmp_path):
    (tmp_path / "variant.yaml").write_text(yaml.safe_dump({**CONFIG, "run_name": "x9_multitask"}))
    (tmp_path / "baseline.yaml").write_text(yaml.safe_dump({"run_name": "p01_image_only"}))
    return {"name": "x9", "variant_config": str(tmp_path / "variant.yaml"), "baseline_config": str(tmp_path / "baseline.yaml"),
            "seeds": [1, 2], "gate_seeds": [1, 2], "source_camera": "Remidio", "runs_dir": str(tmp_path / "runs")}


def partitions_of(table):
    return {name: table[table.split == name] for name in ("train", "validation", "in_domain_test", "external_test")}


def test_training_images_must_have_masks_and_a_label_but_scored_images_need_neither_mask(tmp_path):
    names = [("a", "Remidio", "train"), ("b", "Remidio", "train"), ("c", "Bosch", "train"), ("v", "Remidio", "validation")]
    _, table = labelled_rows(tmp_path, names)
    table.loc[table.image_id == "b", "segmentation_eligible"] = 0
    gate = ms.gate_rows(partitions_of(table), "Remidio")
    assert list(ms.with_masks(gate.train).image_id) == ["a"] and list(gate.shifted.image_id) == ["c"]


def test_a_gate_run_writes_scores_for_the_shifted_cameras_and_a_folder_is_never_reused(tmp_path):
    names = ([(f"t{i}", "Remidio", "train") for i in range(8)] + [(f"v{i}", "Remidio", "validation") for i in range(6)]
             + [(f"b{i}", "Bosch", "train") for i in range(4)])
    _, table = labelled_rows(tmp_path, names)
    study = study_of(tmp_path)
    gate = ms.gate_rows(partitions_of(table), "Remidio")
    folder = ms.run_gate(study, 1, gate, tmp_path, CPU)
    scores = pd.read_csv(folder / "scores.csv")
    assert list(scores.columns) == ["image_id", "label", "logit"] and sorted(scores.image_id) == ["b0", "b1", "b2", "b3"]
    with pytest.raises(FileExistsError):
        ms.run_gate(study, 1, gate, tmp_path, CPU)

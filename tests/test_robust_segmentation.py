import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from glaucoma_project.data import SegmentationDataset
from glaucoma_project.data.segmentation import rescale, segmentation_rows
from glaucoma_project.segmentation import robust_study as rs
from glaucoma_project.segmentation.adapt import adapt_batchnorm

SIZE = 64
torch.set_num_threads(2)


def pair(size=SIZE):
    image = torch.rand(3, size, size)
    mask = torch.zeros(2, size, size)
    mask[:, 16:48, 16:48] = 1
    return image, mask


def test_a_scale_of_one_changes_nothing():
    image, mask = pair()
    new_image, new_mask = rescale(image, mask, 1.0, (0.3, 0.7))
    torch.testing.assert_close(new_image, image)
    assert torch.equal(new_mask, mask)


@pytest.mark.parametrize("factor", [0.6, 1.0, 1.6])
def test_rescaling_keeps_the_size_and_a_binary_mask(factor):
    image, mask = pair()
    new_image, new_mask = rescale(image, mask, factor, (0.5, 0.5))
    assert new_image.shape == image.shape and new_mask.shape == mask.shape
    assert set(new_mask.unique().tolist()) <= {0.0, 1.0}


def test_zooming_out_pads_with_black_and_shrinks_the_masks():
    image, mask = pair()
    new_image, new_mask = rescale(torch.ones_like(image), mask, 0.5, (0.0, 0.0))
    assert new_image[:, SIZE // 2:, SIZE // 2:].sum() == 0  # padding lies away from the corner the picture sits in
    assert 0 < new_mask.sum() < mask.sum()


def test_zooming_in_enlarges_the_masks_at_the_centre():
    image, mask = pair()
    _, new_mask = rescale(image, mask, 1.5, (0.5, 0.5))
    assert new_mask.sum() > mask.sum()


def write_rows(root, rows):
    """A tiny export with images, masks and a split table; ``rows`` are (image_id, camera, split)."""
    (root / "splits").mkdir(exist_ok=True)
    records = []
    for index, (image_id, camera, split) in enumerate(rows):
        for name, value in (("image", 100), ("disc", 255), ("cup", 255)):
            array = np.zeros((SIZE, SIZE, 3) if name == "image" else (SIZE, SIZE), np.uint8)
            if name == "image":
                array[...] = value + 5 * index
            else:
                array[16:48, 8:40] = value if name == "disc" else 0
                array[24:40, 16:32] = value
            Image.fromarray(array).save(root / f"{name}_{image_id}.png")
        records.append({"image_id": image_id, "dataset": "chaksu", "camera": camera, "split": split,
                        "segmentation_eligible": 1, "segmentation_image": f"image_{image_id}.png",
                        "disc_mask": f"disc_{image_id}.png", "cup_mask": f"cup_{image_id}.png"})
    pd.DataFrame(records).to_csv(root / "splits" / "all.csv", index=False)
    return root / "splits"


ROWS = [("r1", "Remidio", "train"), ("r2", "Remidio", "train"), ("b1", "Bosch", "train"),
        ("rv", "Remidio", "validation"), ("fv", "Forus", "validation"), ("t1", "Remidio", "in_domain_test")]


def test_the_robust_profile_gives_valid_items_and_an_unknown_profile_is_rejected(tmp_path):
    splits = write_rows(tmp_path, ROWS)
    dataset = SegmentationDataset("train", tmp_path, augment=True, splits_dir=splits, profile="robust")
    for _ in range(20):
        item = dataset[0]
        assert 0 <= item["image"].min() and item["image"].max() <= 1
        assert set(item["mask"].unique().tolist()) <= {0.0, 1.0}
    with pytest.raises(ValueError, match="profile"):
        SegmentationDataset("train", tmp_path, splits_dir=splits, profile="wild")


def test_a_dataset_can_be_built_from_chosen_rows(tmp_path):
    splits = write_rows(tmp_path, ROWS)
    rows = segmentation_rows("train", splits)
    dataset = SegmentationDataset(None, tmp_path, rows=rows[rows.camera == "Bosch"])
    assert [dataset[i]["image_id"] for i in range(len(dataset))] == ["b1"]


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv, self.norm = nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4)

    def forward(self, image):
        return self.norm(self.conv(image))


def loader_of(images):
    return DataLoader([{"image": image} for image in images], batch_size=2)


def test_adaptation_moves_the_statistics_to_the_target_and_leaves_the_weights_and_the_original_alone():
    torch.manual_seed(0)
    model = TinyNet().eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    target = loader_of([torch.rand(3, 8, 8) * 0.2 + 0.7 for _ in range(6)])
    adapted = adapt_batchnorm(model, target, torch.device("cpu"))
    after, original = adapted.state_dict(), model.state_dict()
    assert not adapted.training
    assert not torch.allclose(after["norm.running_mean"], before["norm.running_mean"])
    for key in ("conv.weight", "conv.bias", "norm.weight", "norm.bias"):
        assert torch.equal(after[key], before[key])
    assert all(torch.equal(original[k], before[k]) for k in before)
    expected = torch.cat([model.conv(batch["image"]) for batch in target]).mean(dim=(0, 2, 3))
    torch.testing.assert_close(after["norm.running_mean"], expected, atol=1e-5, rtol=1e-4)


def test_adaptation_refuses_an_empty_target():
    with pytest.raises(ValueError, match="no images"):
        adapt_batchnorm(TinyNet(), loader_of([]), torch.device("cpu"))


def test_the_gate_trains_on_the_source_camera_and_scores_the_other_cameras(tmp_path):
    gate = rs.gate_rows("Remidio", write_rows(tmp_path, ROWS))
    assert set(gate.train.camera) == {"Remidio"} and set(gate.validation.camera) == {"Remidio"}
    assert sorted(gate.shifted.image_id) == ["b1", "fv"]
    assert "t1" not in set(gate.train.image_id) | set(gate.validation.image_id) | set(gate.shifted.image_id)


def study_of(tmp_path):
    base = {"encoder": "resnet18", "pretrained": False, "epochs": 1, "batch_size": 2, "learning_rate": 1e-3,
            "weight_decay": 1e-4, "num_workers": 0, "seed": 0}
    (tmp_path / "variant.yaml").write_text(yaml.safe_dump({**base, "run_name": "x5_robust", "augment_profile": "robust",
                                                           "adapt": True}))
    (tmp_path / "baseline.yaml").write_text(yaml.safe_dump({**base, "run_name": "baseline", "augment_profile": "standard",
                                                            "adapt": False}))
    return {"name": "x5", "variant_config": str(tmp_path / "variant.yaml"),
            "baseline_config": str(tmp_path / "baseline.yaml"), "gate_seeds": [1, 2], "source_camera": "Remidio",
            "runs_dir": str(tmp_path / "runs"), "max_remidio_loss": 0.01}


def test_gate_jobs_list_the_variant_then_the_baseline_for_each_seed(tmp_path):
    assert rs.gate_jobs(study_of(tmp_path)) == [("variant", 1), ("variant", 2), ("baseline", 1), ("baseline", 2)]


def test_a_gate_run_writes_scores_and_a_folder_is_never_reused(tmp_path):
    study = study_of(tmp_path)
    gate = rs.gate_rows("Remidio", write_rows(tmp_path, ROWS))
    folder = rs.run_gate(study, "variant", 1, gate, tmp_path, torch.device("cpu"))
    scores = json.loads((folder / "scores.json").read_text())
    assert set(scores) == {"remidio_validation", "shifted", "shifted_without_adabn"}
    assert set(scores["shifted"]) == {"disc", "cup"}
    baseline = rs.run_gate(study, "baseline", 1, gate, tmp_path, torch.device("cpu"))
    assert set(json.loads((baseline / "scores.json").read_text())) == {"remidio_validation", "shifted"}
    with pytest.raises(FileExistsError):
        rs.run_gate(study, "variant", 1, gate, tmp_path, torch.device("cpu"))


def write_scores(study, arm, seed, validation, shifted):
    folder = rs.runs_directory(study) / "gate" / f"{arm}_seed{seed}"
    folder.mkdir(parents=True)
    (folder / "scores.json").write_text(json.dumps({
        "remidio_validation": {"disc": validation, "cup": validation}, "shifted": {"disc": shifted, "cup": shifted}}))


def test_the_verdict_needs_a_higher_shifted_dice_and_a_small_loss_on_the_source_camera(tmp_path):
    study = study_of(tmp_path)
    for seed in study["gate_seeds"]:
        write_scores(study, "baseline", seed, 0.90, 0.70)
        write_scores(study, "variant", seed, 0.895, 0.75)
    verdict = rs.gate_verdict(study)
    assert verdict["variant"]["shifted_dice"] == pytest.approx(0.75) and verdict["passes"]


def test_a_loss_of_more_than_the_allowance_on_the_source_camera_fails_the_gate(tmp_path):
    study = study_of(tmp_path)
    for seed in study["gate_seeds"]:
        write_scores(study, "baseline", seed, 0.90, 0.70)
        write_scores(study, "variant", seed, 0.88, 0.75)
    assert not rs.gate_verdict(study)["passes"]


def test_no_gain_on_the_shifted_cameras_fails_the_gate(tmp_path):
    study = study_of(tmp_path)
    for seed in study["gate_seeds"]:
        write_scores(study, "baseline", seed, 0.90, 0.70)
        write_scores(study, "variant", seed, 0.90, 0.70)
    assert not rs.gate_verdict(study)["passes"]

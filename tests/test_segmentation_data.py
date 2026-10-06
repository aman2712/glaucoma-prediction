import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from glaucoma_project.data import SegmentationDataset, make_loader
from glaucoma_project.visualization import overlay

SIZE = 32


@pytest.fixture
def root(tmp_path):
    """A tiny prepared-data folder with 3 images; the last is not segmentation-eligible."""
    (tmp_path / "splits").mkdir()
    rows = []
    for i in range(3):
        for name, value in (("image", 100), ("disc", 255), ("cup", 255)):
            array = np.zeros((SIZE, SIZE, 3) if name == "image" else (SIZE, SIZE), np.uint8)
            array[...] = value if name == "image" else 0
            if name != "image":
                array[8:24, 4:20] = value if name == "disc" else 0
                array[12:20, 8:16] = value
            Image.fromarray(array).save(tmp_path / f"{name}_{i}.png")
        rows.append({"image_id": f"img{i}", "dataset": "chaksu", "camera": "Remidio",
                     "segmentation_eligible": int(i < 2), "segmentation_image": f"image_{i}.png",
                     "disc_mask": f"disc_{i}.png", "cup_mask": f"cup_{i}.png"})
    pd.DataFrame(rows).to_csv(tmp_path / "splits" / "train_segmentation.csv", index=False)
    return tmp_path


def test_item_shapes_and_values(root):
    item = SegmentationDataset("train", root)[0]
    assert item["image"].shape == (3, SIZE, SIZE) and 0 <= item["image"].min() <= item["image"].max() <= 1
    assert item["mask"].shape == (2, SIZE, SIZE) and set(item["mask"].unique().tolist()) <= {0.0, 1.0}
    assert item["mask"][1].sum() < item["mask"][0].sum()  # cup is smaller than disc
    assert item["image_id"] == "img0"


def test_ineligible_images_are_dropped(root):
    assert len(SegmentationDataset("train", root)) == 2


def test_unknown_split_is_rejected(root):
    with pytest.raises(ValueError):
        SegmentationDataset("test", root)


def test_augmentation_keeps_mask_binary_and_image_in_range(root):
    dataset = SegmentationDataset("train", root, augment=True)
    for _ in range(20):
        item = dataset[0]
        assert set(item["mask"].unique().tolist()) <= {0.0, 1.0}
        assert 0 <= item["image"].min() and item["image"].max() <= 1


def test_loader_batches(root):
    batch = next(iter(make_loader("train", batch_size=2, root=root)))
    assert batch["image"].shape == (2, 3, SIZE, SIZE) and batch["mask"].shape == (2, 2, SIZE, SIZE)


def test_overlay_marks_only_boundaries(root):
    item = SegmentationDataset("train", root)[0]
    drawn = overlay(item["image"], item["mask"])
    assert drawn.shape == (SIZE, SIZE, 3)
    assert (drawn != item["image"].permute(1, 2, 0)).any()

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from glaucoma_project.data.classification import FundusDataset, IMAGE_SIZE, make_transform, segmentation_photo_path

SIZE = 512


@pytest.fixture
def root(tmp_path):
    """One 512 px photograph: dark, with a bright square at rows and columns 100 to 200."""
    pixels = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    pixels[100:200, 100:200] = 255
    folder = tmp_path / "prepared" / "x" / "a"
    folder.mkdir(parents=True)
    Image.fromarray(pixels).save(folder / "image_segmentation.png")
    Image.fromarray(pixels).resize((IMAGE_SIZE, IMAGE_SIZE)).save(folder / "image_224.png")
    return tmp_path


def rows():
    return pd.DataFrame({"image_id": ["a"], "binary_label": ["1"], "classification_image": ["prepared/x/a/image_224.png"],
                         "segmentation_image": ["prepared/x/a/image_segmentation.png"]})


def test_a_box_around_the_bright_square_fills_the_view_with_it(root):
    dataset = FundusDataset(rows(), root, boxes={"a": (100, 100, 200, 200)})
    tensor = dataset[0][0]
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE) and (tensor.amax(dim=(1, 2)) - tensor.amin(dim=(1, 2))).abs().max() < 1e-3


def test_a_box_beyond_the_edge_is_padded_with_black_not_refused(root):
    dataset = FundusDataset(rows(), root, boxes={"a": (-100, -100, 100, 100)})
    assert dataset[0][0].shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_without_boxes_the_whole_prepared_photograph_is_used(root):
    whole = FundusDataset(rows(), root)[0][0]
    assert whole.shape == (3, IMAGE_SIZE, IMAGE_SIZE) and whole.min() < whole.max()


def test_only_the_photograph_can_be_loaded_as_a_segmentation_image(root):
    with pytest.raises(ValueError):
        segmentation_photo_path(root, "prepared/x/a/disc_segmentation.png")


def test_per_image_normalisation_gives_each_channel_zero_mean_and_unit_spread(root):
    with Image.open(root / "prepared/x/a/image_224.png") as image:
        tensor = make_transform(per_image=True)(image.convert("RGB"))
    assert tensor.mean(dim=(1, 2)).abs().max() < 1e-4 and (tensor.std(dim=(1, 2), unbiased=False) - 1).abs().max() < 1e-3


def test_strong_augmentation_changes_the_pixels_and_keeps_the_shape(root):
    torch.manual_seed(0)
    with Image.open(root / "prepared/x/a/image_224.png") as image:
        photo = image.convert("RGB")
    standard, strong = make_transform(augment=True)(photo), make_transform(augment=True, profile="strong")(photo)
    assert strong.shape == standard.shape and not torch.allclose(strong, standard)


def test_an_unknown_augmentation_profile_is_refused():
    with pytest.raises(ValueError):
        make_transform(augment=True, profile="wild")

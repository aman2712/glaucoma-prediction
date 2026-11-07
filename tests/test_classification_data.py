import numpy as np
import pytest
from PIL import Image

from glaucoma_project.data.classification import (IMAGENET_MEAN, IMAGENET_STD, FundusDataset, check_images,
                                                  image_path, letterbox, make_loader)
from glaucoma_project.data.effective_splits import effective_split


def test_item_is_the_imagenet_normalised_photograph(export):
    root, table = export
    rows = effective_split(table)[1]["train"]
    check_images(root, rows)
    tensor, label, image_id = FundusDataset(rows, root)[0]
    with Image.open(root / rows.iloc[0].classification_image) as image:
        rgb = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255
    expected = (rgb - np.array(IMAGENET_MEAN)[:, None, None]) / np.array(IMAGENET_STD)[:, None, None]
    np.testing.assert_allclose(tensor.numpy(), expected, atol=1e-6)
    assert tensor.shape == (3, 224, 224) and float(label) == 0 and image_id == rows.iloc[0].image_id


@pytest.mark.parametrize("relative", ["../outside/image_224.png", "prepared/x/overlay_review.png", "/tmp/image_224.png"])
def test_path_guard_refuses_anything_but_a_prepared_photograph(export, relative):
    with pytest.raises(ValueError):
        image_path(export[0], relative)


def test_check_images_rejects_wrong_size(export):
    root, table = export
    rows = effective_split(table)[1]["train"]
    Image.new("RGB", (100, 100)).save(root / rows.iloc[0].classification_image)
    with pytest.raises(ValueError, match="224x224"):
        check_images(root, rows)


def test_training_loader_shuffles_reproducibly_and_validation_loader_does_not(export):
    root, table = export
    rows = effective_split(table)[1]["train"]
    config = {"seed": 1, "batch_size": 4, "augment": False}
    order = lambda: [i for _, _, ids in make_loader(rows, root, config, training=True) for i in ids]
    assert order() == order() != rows.image_id.tolist()
    assert [i for _, _, ids in make_loader(rows, root, config) for i in ids] == rows.image_id.tolist()


def test_letterbox_pads_with_black_and_centres_the_photo(tmp_path):
    Image.new("RGB", (80, 40), (200, 100, 50)).save(tmp_path / "photo.png")
    image = letterbox(tmp_path / "photo.png")
    assert image.size == (224, 224)
    assert image.getpixel((112, 0)) == (0, 0, 0) and image.getpixel((112, 112)) == (200, 100, 50)

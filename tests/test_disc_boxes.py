import numpy as np
import pandas as pd
import pytest

from glaucoma_project.data.crops import load_boxes
from glaucoma_project.measurements.disc_boxes import crop_box

SIZE = 512


def disc_at(top, left, height, width):
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[top:top + height, left:left + width] = 1
    return mask


def test_the_box_is_a_square_centred_on_the_disc_with_three_times_its_longer_side():
    assert crop_box(disc_at(200, 100, 40, 60), factor=3.0) == (100 + 30 - 90, 200 + 20 - 90, 100 + 30 + 90, 200 + 20 + 90)


def test_specks_away_from_the_disc_do_not_move_the_box():
    mask = disc_at(200, 100, 40, 60)
    mask[5, 5] = 1
    assert crop_box(mask, factor=3.0) == crop_box(disc_at(200, 100, 40, 60), factor=3.0)


def test_no_disc_gives_the_whole_image():
    assert crop_box(np.zeros((SIZE, SIZE), dtype=np.uint8), factor=3.0) == (0, 0, SIZE, SIZE)


def test_a_tiny_blob_is_taken_for_no_disc_and_gives_the_whole_image():
    assert crop_box(disc_at(200, 100, 8, 9), factor=3.0) == (0, 0, SIZE, SIZE)


def test_boxes_are_read_as_a_lookup_by_image_id(tmp_path):
    path = tmp_path / "boxes.csv"
    pd.DataFrame({"image_id": ["a"], "left": [1], "top": [2], "right": [30], "bottom": [40]}).to_csv(path, index=False)
    assert load_boxes(path) == {"a": (1, 2, 30, 40)}


def test_a_missing_boxes_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_boxes(tmp_path / "absent.csv")

import numpy as np
import pandas as pd
import pytest

from glaucoma_project.report_figures import crop_box, per_image_errors, pick_examples


def test_examples_are_picked_by_rule():
    errors = pd.DataFrame({"image_id": list("abcdefg"), "binary_label": [0, 0, 0, 1, 1, 1, 1],
                           "score": [.1, .2, .9, .9, .8, .2, .7], "error": [.01, .05, .5, .02, .06, .04, .20],
                           "expert_vcdr": .5, "predicted_vcdr": .5})
    picked = pick_examples(errors, threshold=0.5)
    assert picked["healthy"] in {"a", "b"} and picked["glaucoma"] in {"d", "e", "g"}
    assert picked["largest error"] == "c"  # the largest error counts even for a misclassified image


def test_needs_both_classes_correctly_classified():
    errors = pd.DataFrame({"image_id": ["a"], "binary_label": [0], "score": [.1], "error": [.1]})
    with pytest.raises(ValueError):
        pick_examples(errors, threshold=0.5)


def test_errors_join_expert_predicted_and_score_by_image():
    measurements = pd.DataFrame({"image_id": ["a", "a", "b", "b"], "split": "in_domain_test",
                                 "source": ["expert", "predicted"] * 2, "vcdr": [.5, .6, .4, np.nan]})
    scores = pd.DataFrame({"image_id": ["a", "b"], "split": "in_domain_test", "source": "predicted",
                           "binary_label": [1, 0], "score": [.9, .1]})
    table = per_image_errors(measurements, scores)
    assert table.image_id.tolist() == ["a"] and np.isclose(table.error[0], 0.1)  # b has no predicted disc


def test_crop_box_is_square_and_inside_the_image():
    mask = np.zeros((512, 512), bool)
    mask[250:290, 400:440] = True
    top, bottom, left, right = crop_box(mask)
    assert bottom - top == right - left and 0 <= top and bottom <= 512 and 0 <= left and right <= 512
    assert top <= 250 and bottom >= 290 and left <= 400 and right >= 440

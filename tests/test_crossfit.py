import pandas as pd
import pytest
import torch

from glaucoma_project.combined_model.crossfit import (fold_assignments, held_out_logits, train_fold_models,
                                                       training_rows)
from glaucoma_project.data.common_split import load_common_split
from glaucoma_project.image_model import train as train_module
from glaucoma_project.image_model.train import train_for_epochs

pytestmark = pytest.mark.filterwarnings("ignore:Software test only")
torch.set_num_threads(2)
CPU = torch.device("cpu")
RECIPE = {"model_name": "resnet18", "pretrained": False, "seed": 7, "batch_size": 8, "warmup_epochs": 1,
          "head_lr": 1e-3, "backbone_lr": 1e-4, "weight_decay": 1e-4, "augment": False}


@pytest.fixture
def development(common_split):
    parts = load_common_split(common_split)
    return pd.concat([parts["train"], parts["validation"]], ignore_index=True)


def test_every_image_gets_one_fold_and_groups_are_never_split(development):
    folds = fold_assignments(development, n_folds=4, seed=2440)
    assert sorted(folds.unique()) == [0, 1, 2, 3] and len(folds) == len(development)
    groups_per_fold = development.assign(fold=folds).groupby("analysis_group").fold.nunique()
    assert (groups_per_fold == 1).all()


def test_every_training_set_has_both_classes(development):
    folds = fold_assignments(development, n_folds=4, seed=2440)
    for fold in range(4):
        assert set(training_rows(development, folds, fold).binary_label) == {0, 1}


def test_fold_assignment_is_reproducible(development):
    first = fold_assignments(development, n_folds=4, seed=2440)
    assert first.equals(fold_assignments(development, n_folds=4, seed=2440))


def test_training_stops_after_exactly_the_requested_epochs(development, export, monkeypatch):
    root, _ = export
    epochs_run = []
    real_run_epoch = train_module.run_epoch
    monkeypatch.setattr(train_module, "run_epoch", lambda *a, **k: epochs_run.append(1) or real_run_epoch(*a, **k))
    train_for_epochs(development, root, RECIPE, CPU, epochs=3)
    assert len(epochs_run) == 3


def test_a_network_never_scores_images_it_was_trained_on(development, export, tmp_path):
    root, _ = export
    seen_by_network = []

    def recording_trainer(rows, root, config, device, epochs):
        seen_by_network.append(set(rows.image_id))
        return train_for_epochs(rows, root, config, device, epochs)

    folds = fold_assignments(development, n_folds=2, seed=2440)
    checkpoints = train_fold_models(development, folds, RECIPE, root, tmp_path, CPU, epochs=2, trainer=recording_trainer)
    scores = held_out_logits(development, folds, checkpoints, RECIPE, root, CPU)

    assert len(checkpoints) == 2 and all(path.is_file() for path in checkpoints)
    assert sorted(scores.image_id) == sorted(development.image_id) and scores.logit.notna().all()
    for fold, trained_on in enumerate(seen_by_network):
        held_out = set(development.image_id[folds == fold])
        assert trained_on.isdisjoint(held_out)


def test_training_for_too_few_epochs_is_refused(development, export):
    root, _ = export
    with pytest.raises(ValueError, match="must exceed warmup_epochs"):
        train_for_epochs(development, root, RECIPE, CPU, epochs=1)

"""Minimal handoff loader

Install torch/torchvision separately for model training. This NumPy loader does
not require them. It uses the explicit fixed ImageNet RGB normalisation below.

Four partitions exist, in increasing order of how "unseen" they must stay during
development:
  train/validation     - CHAKSU train-release only. Use freely during development.
  in_domain_test       - CHAKSU test-release, held out untouched. Load only after
                          the model and hyperparameters are frozen
                          (internal_evaluation=True).
  external_test        - PAPILA, a fully separate dataset/site. Load only after
                          the model and thresholds are frozen (external_evaluation=True).
"""
from pathlib import Path
import numpy as np
from PIL import Image
from prepare_data import read_csv

IMAGENET_MEAN = np.array([.485, .456, .406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([.229, .224, .225], dtype=np.float32)[:, None, None]

_EXPECTED = {
    'train': ('chaksu', 'train'),
    'validation': ('chaksu', 'train'),
    'in_domain_test': ('chaksu', 'test'),
    'external_test': ('papila', ''),
}


class ClassificationData:
    def __init__(self, artifacts, partition='train', internal_evaluation=False, external_evaluation=False):
        if partition not in _EXPECTED:
            raise ValueError('Unknown partition')
        if partition == 'in_domain_test' and not internal_evaluation:
            raise ValueError('Load the CHAKSU in-domain held-out test only after model and '
                              'hyperparameters are frozen.')
        if partition == 'external_test' and not external_evaluation:
            raise ValueError('Load external (PAPILA) data only after model and thresholds are frozen.')
        self.root = Path(artifacts)
        self.rows = read_csv(self.root / 'splits' / f'{partition}_classification.csv')
        expected_dataset, expected_release = _EXPECTED[partition]
        if any(row['dataset'] != expected_dataset for row in self.rows):
            raise ValueError('Dataset crossed its specified evaluation role')
        if expected_release and any(row.get('chaksu_release') != expected_release for row in self.rows):
            raise ValueError('CHAKSU release crossed its specified evaluation role')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['classification_image']) as image:
            rgb = np.asarray(image.convert('RGB'), dtype=np.float32).transpose(2, 0, 1) / 255
        x = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return x, int(row['binary_label']), row['image_id']

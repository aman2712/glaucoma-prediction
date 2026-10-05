"""Disc/cup segmentation data: reads the frozen split CSVs written by script/prepare_data.py."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from glaucoma_project import PREPARED_ROOT

SPLITS = ("train", "validation", "in_domain_test", "external_test")


def _read_image(path: Path, mode: str) -> torch.Tensor:
    """Load a PNG as float32 in [0, 1], shaped (C, H, W)."""
    array = np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0
    return torch.from_numpy(array if array.ndim == 3 else array[..., None]).permute(2, 0, 1)


class SegmentationDataset(Dataset):
    """Fundus images with expert optic disc and cup masks.

    Each item is a dict with
      image:    float (3, H, W) in [0, 1]
      mask:     float (2, H, W), channel 0 = disc, channel 1 = cup, values {0, 1}
      image_id: str
    Normalisation is left to the model. With ``augment`` the image and mask are
    flipped together and the image alone gets a brightness/contrast jitter.
    """

    def __init__(self, split: str, root: Path = PREPARED_ROOT, augment: bool = False):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        table = pd.read_csv(Path(root) / "splits" / f"{split}_segmentation.csv")
        self.table = table[table["segmentation_eligible"] == 1].reset_index(drop=True)
        self.root = Path(root)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict:
        row = self.table.iloc[index]
        image = _read_image(self.root / row["segmentation_image"], "RGB")
        disc = _read_image(self.root / row["disc_mask"], "L")
        cup = _read_image(self.root / row["cup_mask"], "L")
        mask = (torch.cat([disc, cup]) > 0.5).float()
        if self.augment:
            image, mask = self._augment(image, mask)
        return {"image": image, "mask": mask, "image_id": row["image_id"]}

    @staticmethod
    def _augment(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1) < 0.5:
            image, mask = image.flip(-1), mask.flip(-1)
        contrast, brightness = 1 + 0.4 * (torch.rand(1) - 0.5), 0.2 * (torch.rand(1) - 0.5)
        return (image * contrast + brightness).clamp(0, 1), mask


def make_loader(split: str, batch_size: int = 8, root: Path = PREPARED_ROOT,
                num_workers: int = 0, training: bool | None = None) -> DataLoader:
    """Shuffled and augmented when ``training`` (default: only for the train split)."""
    if training is None:
        training = split == "train"
    return DataLoader(SegmentationDataset(split, root, augment=training), batch_size=batch_size,
                      shuffle=training, num_workers=num_workers)

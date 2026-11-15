"""Disc/cup segmentation data. Which images belong to which split comes from the common split
of record (``splits/all.csv``); the images and masks come from the prepared data folder."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from glaucoma_project import COMMON_SPLIT_DIR, PREPARED_ROOT

SPLITS = ("train", "validation", "in_domain_test", "external_test")
PROFILES = ("standard", "robust")
SCALE_RANGE = (0.6, 1.6)


def read_image(path: Path, mode: str) -> torch.Tensor:
    """Load a PNG as float32 in [0, 1], shaped (C, H, W)."""
    array = np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0
    return torch.from_numpy(array if array.ndim == 3 else array[..., None]).permute(2, 0, 1)


def _label(row: pd.Series) -> float:
    return float(row["binary_label"]) if "binary_label" in row else float("nan")


def segmentation_rows(split: str, splits_dir: Path = COMMON_SPLIT_DIR) -> pd.DataFrame:
    """The segmentation-eligible rows of one split of the common split."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    table = pd.read_csv(Path(splits_dir) / "all.csv")
    return table[(table["split"] == split) & (table["segmentation_eligible"] == 1)].reset_index(drop=True)


def _window(old: int, new: int, place: float) -> tuple[slice, slice]:
    """Along one axis: which part of the zoomed picture goes to which part of the canvas."""
    offset = int(place * abs(new - old))
    if new >= old:
        return slice(offset, offset + old), slice(0, old)
    return slice(0, new), slice(offset, offset + new)


def rescale(image: torch.Tensor, mask: torch.Tensor, factor: float,
            position: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor]:
    """Zoom image and mask together by ``factor`` and return them at their original size.

    A factor above 1 cuts a window out of the enlarged picture, a factor below 1 places the shrunken
    picture on a black canvas. ``position`` (row, column), each in [0, 1], chooses where.
    """
    height, width = image.shape[-2:]
    size = (max(1, round(height * factor)), max(1, round(width * factor)))
    zoomed = F.interpolate(image[None], size=size, mode="bilinear", align_corners=False)[0]
    zoomed_mask = (F.interpolate(mask[None], size=size, mode="nearest")[0] > 0.5).float()
    (rows_from, rows_to), (cols_from, cols_to) = _window(height, size[0], position[0]), _window(width, size[1], position[1])
    canvas, canvas_mask = torch.zeros_like(image), torch.zeros_like(mask)
    canvas[..., rows_to, cols_to] = zoomed[..., rows_from, cols_from]
    canvas_mask[..., rows_to, cols_to] = zoomed_mask[..., rows_from, cols_from]
    return canvas, canvas_mask


class SegmentationDataset(Dataset):
    """Fundus images with expert optic disc and cup masks.

    Each item is a dict with
      image:    float (3, H, W) in [0, 1]
      mask:     float (2, H, W), channel 0 = disc, channel 1 = cup, values {0, 1}
      label:    float, the binary glaucoma label (NaN when the row has none)
      image_id: str
    Normalisation is left to the model. With ``augment`` the image and mask are
    flipped together and the image alone gets a brightness/contrast jitter. The "robust" profile also
    rescales image and mask together and jitters colour more strongly. ``rows`` replaces the split's rows.
    """

    def __init__(self, split: str | None, root: Path = PREPARED_ROOT, augment: bool = False,
                 splits_dir: Path = COMMON_SPLIT_DIR, profile: str = "standard", rows: pd.DataFrame | None = None):
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}")
        self.table = (segmentation_rows(split, splits_dir) if rows is None else rows).reset_index(drop=True)
        self.root = Path(root)
        self.augment = augment
        self.profile = profile

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict:
        row = self.table.iloc[index]
        image = read_image(self.root / row["segmentation_image"], "RGB")
        disc = read_image(self.root / row["disc_mask"], "L")
        cup = read_image(self.root / row["cup_mask"], "L")
        mask = (torch.cat([disc, cup]) > 0.5).float()
        if self.augment:
            image, mask = (self._augment_robust if self.profile == "robust" else self._augment)(image, mask)
        return {"image": image, "mask": mask, "label": _label(row), "image_id": row["image_id"]}

    @staticmethod
    def _augment(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1) < 0.5:
            image, mask = image.flip(-1), mask.flip(-1)
        contrast, brightness = 1 + 0.4 * (torch.rand(1) - 0.5), 0.2 * (torch.rand(1) - 0.5)
        return (image * contrast + brightness).clamp(0, 1), mask

    @staticmethod
    def _augment_robust(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Flip and rescale image and mask together, then jitter the colour of the image alone."""
        if torch.rand(1) < 0.5:
            image, mask = image.flip(-1), mask.flip(-1)
        factor = float(torch.empty(1).uniform_(*SCALE_RANGE))
        image, mask = rescale(image, mask, factor, tuple(torch.rand(2).tolist()))
        contrast, brightness = torch.empty(1).uniform_(0.6, 1.4), torch.empty(1).uniform_(-0.2, 0.2)
        gain = torch.empty(3, 1, 1).uniform_(0.85, 1.15)
        image = ((image * contrast + brightness) * gain).clamp(0, 1)
        return image ** float(torch.empty(1).uniform_(0.7, 1.4)), mask


def make_loader(split: str | None, batch_size: int = 8, root: Path = PREPARED_ROOT, num_workers: int = 0,
                training: bool | None = None, splits_dir: Path = COMMON_SPLIT_DIR, profile: str = "standard",
                rows: pd.DataFrame | None = None) -> DataLoader:
    """Shuffled and augmented when ``training`` (default: only for the train split)."""
    if training is None:
        training = split == "train"
    dataset = SegmentationDataset(split, root, augment=training, splits_dir=splits_dir, profile=profile, rows=rows)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training, num_workers=num_workers)


class PhotographDataset(Dataset):
    """Photographs of the given rows with their glaucoma label and no masks, for scoring and adaptation.
    Each item is a dict with image (3, H, W) in [0, 1], label (float) and image_id."""

    def __init__(self, rows: pd.DataFrame, root: Path = PREPARED_ROOT):
        self.table, self.root = rows.reset_index(drop=True), Path(root)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict:
        row = self.table.iloc[index]
        return {"image": read_image(self.root / row["segmentation_image"], "RGB"), "label": _label(row),
                "image_id": row["image_id"]}

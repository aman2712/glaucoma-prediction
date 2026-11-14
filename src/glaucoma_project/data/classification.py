"""Full-photograph data for the image-only model: dataset, loader and single-photo preparation."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from glaucoma_project import REPO_ROOT
from glaucoma_project.data.crops import load_boxes

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PHOTO_NAMES = {"image_224.png", "image_classification.png"}
SEGMENTATION_PHOTO_NAMES = {"image_segmentation.png"}
STRONG_COLOUR = {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}
GAMMA_RANGE = (0.7, 1.4)
GREYSCALE_PROBABILITY = 0.1
PROFILES = ("standard", "strong")


def _photo_path(root: Path, relative: str, allowed: set[str]) -> Path:
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError("Image paths must be relative to the prepared data folder")
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Image path escapes the prepared data folder")
    if path.name not in allowed:
        raise ValueError("Expected a prepared RGB photograph, not a mask or overlay")
    return path


def image_path(root: Path, relative: str) -> Path:
    """Resolve a ``classification_image`` entry, refusing anything that is not a prepared
    photograph inside ``root`` (so a mask or overlay can never be loaded by mistake)."""
    return _photo_path(root, relative, PHOTO_NAMES)


def segmentation_photo_path(root: Path, relative: str) -> Path:
    """Same checks for a ``segmentation_image`` entry (the 512 px photograph)."""
    return _photo_path(root, relative, SEGMENTATION_PHOTO_NAMES)


def check_images(root: Path, rows: pd.DataFrame) -> None:
    """Open every listed image and confirm it is a 224 x 224 RGB export."""
    for row in rows.itertuples():
        path = image_path(root, row.classification_image)
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared image: {path}")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(f"Expected a {IMAGE_SIZE}x{IMAGE_SIZE} RGB export: {path}")
            image.load()


class RandomGamma:
    """Brightens or darkens the mid-tones by a random power, drawn with the torch generator."""

    def __init__(self, low: float, high: float):
        self.low, self.high = low, high

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image.clamp(0, 1) ** float(torch.empty(1).uniform_(self.low, self.high))


class PerImageNormalise:
    """Each colour channel of one image to mean 0 and spread 1, so overall brightness and contrast
    of a camera cancel out."""

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        mean = image.mean(dim=(1, 2), keepdim=True)
        spread = image.std(dim=(1, 2), keepdim=True, unbiased=False)
        return (image - mean) / spread.clamp_min(1e-6)


def _augmentation(profile: str) -> list:
    if profile == "standard":
        return [transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05)]
    if profile == "strong":
        return [transforms.ColorJitter(**STRONG_COLOUR), transforms.RandomGrayscale(GREYSCALE_PROBABILITY)]
    raise ValueError(f"Unknown augmentation profile {profile!r}; choose from {PROFILES}")


def make_transform(augment: bool = False, profile: str = "standard", per_image: bool = False) -> transforms.Compose:
    """Normalisation (ImageNet's, or each image's own with ``per_image``), with optional colour
    augmentation of the chosen ``profile``. Whole-image geometry is untouched: no flips or rotations."""
    steps = _augmentation(profile) if augment else []
    to_tensor = [transforms.ToTensor()]
    if augment and profile == "strong":
        to_tensor.append(RandomGamma(*GAMMA_RANGE))
    normalise = PerImageNormalise() if per_image else transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    return transforms.Compose(steps + to_tensor + [normalise])


class FundusDataset(Dataset):
    """Items are (normalised image, label as float tensor, image_id). With ``measurement_columns``
    the first element is a pair (image, measurements), the measurements read from the rows as given."""

    def __init__(self, rows: pd.DataFrame, root: Path, augment: bool = False,
                 measurement_columns: list[str] | None = None, boxes: dict | None = None,
                 profile: str = "standard", per_image: bool = False):
        self.rows = rows.reset_index(drop=True)
        self.root = Path(root)
        self.boxes = boxes
        self.transform = make_transform(augment, profile, per_image)
        self.measurements = (torch.tensor(self.rows[measurement_columns].to_numpy(float), dtype=torch.float32)
                             if measurement_columns else None)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        tensor = self.transform(self._photograph(row))
        inputs = tensor if self.measurements is None else (tensor, self.measurements[index])
        return inputs, torch.tensor(float(row.binary_label)), row.image_id


    def _photograph(self, row) -> Image.Image:
        """The prepared photograph, or with ``boxes`` the window of the 512 px photograph around the
        disc, resized to the network's input size."""
        if self.boxes is None:
            with Image.open(image_path(self.root, row.classification_image)) as image:
                return image.convert("RGB")
        with Image.open(segmentation_photo_path(self.root, row.segmentation_image)) as image:
            window = image.convert("RGB").crop(self.boxes[row.image_id])  # outside the image reads as black
        return window.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)


def make_loader(rows: pd.DataFrame, root: Path, config: dict, training: bool = False) -> DataLoader:
    """Shuffled (with a seeded generator) and augmented only when ``training``."""
    boxes = load_boxes(REPO_ROOT / config["crop_boxes"]) if config.get("crop_boxes") else None
    dataset = FundusDataset(rows, root, augment=training and config["augment"],
                            measurement_columns=config.get("measurement_columns"), boxes=boxes,
                            profile=config.get("augment_profile", "standard"),
                            per_image=config.get("per_image_normalisation", False))
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=training, generator=generator)


def letterbox(path: Path) -> Image.Image:
    """Square-pad a new photograph with black and resize it exactly as data_pipeline/prepare_data.py
    does for the classification images."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), 0)
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)

"""Full-photograph data for the image-only model: dataset, loader and single-photo preparation."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PHOTO_NAMES = {"image_224.png", "image_classification.png"}


def image_path(root: Path, relative: str) -> Path:
    """Resolve a ``classification_image`` entry, refusing anything that is not a prepared
    photograph inside ``root`` (so a mask or overlay can never be loaded by mistake)."""
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError("classification_image must be relative to the prepared data folder")
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Image path escapes the prepared data folder")
    if path.name not in PHOTO_NAMES:
        raise ValueError("Expected the prepared RGB classification image, not a mask or overlay")
    return path


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


def make_transform(augment: bool = False) -> transforms.Compose:
    """ImageNet normalisation, with optional mild colour jitter. The whole photograph is kept:
    no crops, flips or rotations."""
    steps = [transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05)] if augment else []
    return transforms.Compose(steps + [transforms.ToTensor(),
                                       transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class FundusDataset(Dataset):
    """Items are (normalised image, label as float tensor, image_id)."""

    def __init__(self, rows: pd.DataFrame, root: Path, augment: bool = False):
        self.rows = rows.reset_index(drop=True)
        self.root = Path(root)
        self.transform = make_transform(augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        with Image.open(image_path(self.root, row.classification_image)) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.tensor(float(row.binary_label)), row.image_id


def make_loader(rows: pd.DataFrame, root: Path, config: dict, training: bool = False) -> DataLoader:
    """Shuffled (with a seeded generator) and augmented only when ``training``."""
    dataset = FundusDataset(rows, root, augment=training and config["augment"])
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

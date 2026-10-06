from pathlib import Path

import torch
from torch.utils.data import DataLoader

from glaucoma_project.data import SegmentationDataset
from glaucoma_project.segmentation.model import DiscCupUNet


def load_model(run_dir: Path, encoder: str = "resnet34", device: torch.device | None = None) -> DiscCupUNet:
    """Load the best checkpoint of a training run, ready for inference."""
    model = DiscCupUNet(encoder, pretrained=False)
    model.load_state_dict(torch.load(Path(run_dir) / "best.pt", map_location="cpu"))
    return model.to(device or "cpu").eval()


@torch.no_grad()
def predict_masks(model: DiscCupUNet, images: torch.Tensor) -> torch.Tensor:
    """Binary (N, 2, H, W) masks (disc, cup) on the CPU for a batch of images in [0, 1]."""
    device = next(model.parameters()).device
    return (torch.sigmoid(model(images.to(device))) > 0.5).cpu()


def predict_dataset(model: DiscCupUNet, dataset: SegmentationDataset, batch_size: int = 8) -> torch.Tensor:
    """Predicted masks for every item of a dataset, in dataset order."""
    loader = DataLoader(dataset, batch_size=batch_size)
    return torch.cat([predict_masks(model, batch["image"]) for batch in loader])

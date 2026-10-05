import segmentation_models_pytorch as smp
import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DiscCupUNet(nn.Module):
    """U-Net with a ResNet encoder. Takes RGB in [0, 1], returns (N, 2, H, W) logits (disc, cup)."""

    def __init__(self, encoder: str = "resnet34", pretrained: bool = True):
        super().__init__()
        self.net = smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None,
                            in_channels=3, classes=2)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net((image - self.mean) / self.std)

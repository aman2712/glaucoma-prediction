"""CNN architectures with a single-logit head, and checkpoint loading."""
from pathlib import Path

import torch
from torch import nn
from torchvision import models

ARCHITECTURES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
}
UNTRAINED = "NONE_TEST_ONLY"


def build_model(name: str, pretrained: bool = True) -> tuple[nn.Module, str]:
    """The architecture with its last layer replaced by one logit. Also returns the weights ID
    (``UNTRAINED`` without pretraining, which is for software tests only). A failed weight
    download raises: the model never silently falls back to random weights."""
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {name!r}; choose from {sorted(ARCHITECTURES)}")
    builder, weights = ARCHITECTURES[name]
    model = builder(weights=weights if pretrained else None)
    if isinstance(model, models.ResNet):
        model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    return model, str(weights) if pretrained else UNTRAINED


def head_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Parameters of the classification head; all others belong to the pretrained backbone."""
    head = model.fc if isinstance(model, models.ResNet) else model.classifier
    return list(head.parameters())


def restore_model(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    """Load a checkpoint written by ``train.fit`` into an evaluation-mode model."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model, _ = build_model(checkpoint["config"]["model_name"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), checkpoint

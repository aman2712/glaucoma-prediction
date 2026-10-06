import torch

from glaucoma_project.evaluation.metrics import dice, iou
from glaucoma_project.segmentation import DiscCupUNet
from glaucoma_project.segmentation.losses import DiceBCELoss


def test_model_output_shape():
    model = DiscCupUNet("resnet18", pretrained=False).eval()
    assert model(torch.rand(2, 3, 64, 64)).shape == (2, 2, 64, 64)


def test_dice_and_iou_known_values():
    target = torch.zeros(1, 2, 4, 4)
    target[:, :, :2] = 1  # 8 pixels per channel
    half = torch.zeros(1, 2, 4, 4)
    half[:, :, :1] = 1  # 4 pixels, all inside the target
    assert torch.allclose(dice(target, target), torch.ones(1, 2))
    assert torch.allclose(dice(half, target), torch.full((1, 2), 2 * 4 / (4 + 8)), atol=1e-4)
    assert torch.allclose(iou(half, target), torch.full((1, 2), 0.5), atol=1e-4)
    assert torch.allclose(dice(torch.zeros(1, 2, 4, 4), torch.zeros(1, 2, 4, 4)), torch.ones(1, 2))


def test_loss_is_low_for_perfect_and_high_for_wrong_logits():
    target = (torch.rand(2, 2, 8, 8) > 0.5).float()
    perfect = (target * 2 - 1) * 20
    assert DiceBCELoss()(perfect, target) < 0.01
    assert DiceBCELoss()(-perfect, target) > 1

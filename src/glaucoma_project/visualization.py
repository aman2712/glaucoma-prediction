"""Plotting helpers for checking images against their disc/cup masks."""
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

DISC_COLOUR = (0.0, 1.0, 1.0)
CUP_COLOUR = (1.0, 1.0, 0.0)


def _outline(mask: torch.Tensor, width: int = 5) -> torch.Tensor:
    """Boundary pixels of a binary (H, W) mask: the mask minus its erosion."""
    m = mask[None, None]
    eroded = -F.max_pool2d(-m, width, stride=1, padding=width // 2)
    return (m - eroded)[0, 0] > 0


def overlay(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Draw disc and cup outlines on an image. Returns (H, W, 3) floats in [0, 1]."""
    out = image.clone()
    for channel, colour in enumerate((DISC_COLOUR, CUP_COLOUR)):
        out[:, _outline(mask[channel])] = torch.tensor(colour)[:, None]
    return out.permute(1, 2, 0)


def show_overlays(dataset, indices, ncols: int = 4):
    """Grid of images with expert outlines (cyan = disc, yellow = cup)."""
    nrows = -(-len(indices) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, indices):
        item = dataset[i]
        row = dataset.table.iloc[i]
        ax.imshow(overlay(item["image"], item["mask"]).numpy())
        ax.set_title(f"{row['dataset']} · {row['camera']}", fontsize=9)
    fig.tight_layout()
    return fig


def plot_history(history):
    """Training loss and validation Dice (disc, cup) per epoch."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.4))
    left.plot(history["epoch"], history["train_loss"])
    left.set(title="Training loss", xlabel="epoch")
    right.plot(history["epoch"], history["val_dice_disc"], label="disc")
    right.plot(history["epoch"], history["val_dice_cup"], label="cup")
    right.set(title="Validation Dice", xlabel="epoch", ylim=(0.5, 1))
    right.legend()
    fig.tight_layout()
    return fig


def show_comparison(dataset, predicted, indices, scores):
    """Expert outlines (left) beside predicted outlines (right), one row per image.

    ``predicted`` holds masks for the whole dataset; ``scores`` is the per-image (disc, cup) Dice.
    """
    fig, axes = plt.subplots(len(indices), 2, figsize=(7, 3.5 * len(indices)), squeeze=False)
    for (expert_ax, predicted_ax), i in zip(axes, indices):
        item = dataset[i]
        expert_ax.imshow(overlay(item["image"], item["mask"]).numpy())
        predicted_ax.imshow(overlay(item["image"], predicted[i].float()).numpy())
        expert_ax.set_title(f"expert · {dataset.table.iloc[i]['camera']}", fontsize=9)
        predicted_ax.set_title(f"predicted · Dice disc {scores[i, 0]:.2f}, cup {scores[i, 1]:.2f}", fontsize=9)
        expert_ax.axis("off")
        predicted_ax.axis("off")
    fig.tight_layout()
    return fig


def plot_roc(curves: dict):
    """ROC curves from {name: (labels, scores)}."""
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for name, (y, score) in curves.items():
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="1 - specificity", ylabel="sensitivity", title="ROC (validation)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig

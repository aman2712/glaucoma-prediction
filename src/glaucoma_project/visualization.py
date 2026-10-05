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

"""Optic disc and cup measurements from binary masks. All lengths and areas are in pixels."""
import numpy as np
from scipy import ndimage


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest connected blob, removing stray predicted specks."""
    labels, count = ndimage.label(mask)
    if count <= 1:
        return mask.astype(bool)
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    return labels == (np.argmax(sizes) + 1)


def _extent(mask: np.ndarray) -> tuple[int, int]:
    """Vertical and horizontal extent (bounding-box height and width) of a mask."""
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any():
        return 0, 0
    top, bottom = np.flatnonzero(rows)[[0, -1]]
    left, right = np.flatnonzero(cols)[[0, -1]]
    return int(bottom - top + 1), int(right - left + 1)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else float("nan")


def measure(disc: np.ndarray, cup: np.ndarray) -> dict:
    """Cup-to-disc ratios and sizes from 2-D disc and cup masks.

    vcdr/hcdr: cup extent divided by disc extent (vertical/horizontal).
    area_cdr:  cup area divided by disc area.
    Ratios are NaN when the disc mask is empty.
    """
    disc, cup = largest_component(disc > 0), largest_component(cup > 0)
    disc_height, disc_width = _extent(disc)
    cup_height, cup_width = _extent(cup)
    return {
        "vcdr": _ratio(cup_height, disc_height),
        "hcdr": _ratio(cup_width, disc_width),
        "area_cdr": _ratio(cup.sum(), disc.sum()),
        "disc_area": int(disc.sum()),
        "cup_area": int(cup.sum()),
    }

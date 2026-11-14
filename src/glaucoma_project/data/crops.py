"""Crop boxes around the optic disc, stored one row per image (written by ``measurements.disc_boxes``)."""
from pathlib import Path

import pandas as pd

BOX_COLUMNS = ["left", "top", "right", "bottom"]


def load_boxes(path: Path) -> dict[str, tuple[int, int, int, int]]:
    """Crop boxes by image ID."""
    table = pd.read_csv(path)
    return {row.image_id: (row.left, row.top, row.right, row.bottom) for row in table.itertuples()}

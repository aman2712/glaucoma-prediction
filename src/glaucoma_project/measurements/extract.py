import pandas as pd
import torch

from glaucoma_project.data import make_loader
from glaucoma_project.measurements.cdr import measure
from glaucoma_project.segmentation.predict import predict_masks

COLUMNS = ["image_id", "dataset", "camera", "binary_label"]


def measure_split(split: str, source: str, model=None, batch_size: int = 8) -> pd.DataFrame:
    """Measurements for every image of a split, from expert masks or from a model's predictions.

    source: "expert" (ground-truth masks) or "predicted" (needs ``model``).
    Returns one row per image: identifiers and label, then vcdr, hcdr, area_cdr, disc_area, cup_area.
    """
    if source not in ("expert", "predicted"):
        raise ValueError(f"source must be 'expert' or 'predicted', got {source!r}")
    loader = make_loader(split, batch_size, training=False)
    rows = []
    for batch in loader:
        masks = batch["mask"] if source == "expert" else predict_masks(model, batch["image"])
        rows += [{"image_id": image_id, **measure(m[0].numpy(), m[1].numpy())}
                 for image_id, m in zip(batch["image_id"], masks)]
    table = loader.dataset.table[COLUMNS]
    return table.merge(pd.DataFrame(rows), on="image_id").assign(split=split, source=source)

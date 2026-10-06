import numpy as np
import pandas as pd

from glaucoma_project.classification.baseline import make_model
from glaucoma_project.final_evaluation import _classification

CONFIG = {"features": ["vcdr", "hcdr", "area_cdr"], "n_bootstrap": 20, "seed": 0}


def synthetic(split, source, n=120, suspects=0, seed=0):
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, n).astype(float)
    label[:suspects] = np.nan
    base = 0.5 + 0.2 * np.nan_to_num(label, nan=0.5) + 0.05 * rng.normal(size=n)
    return pd.DataFrame({"image_id": [f"{split}{source}{i}" for i in range(n)], "split": split, "source": source,
                         "binary_label": label, "vcdr": base, "hcdr": base, "area_cdr": base ** 2})


def test_classification_tables_cover_every_split_source_and_analysis():
    measurements = pd.concat([synthetic("in_domain_test", s) for s in ("predicted", "expert")]
                             + [synthetic("external_test", s, suspects=10) for s in ("predicted", "expert")])
    train = synthetic("train", "predicted", 200)
    frozen = {s: {"model": make_model(1).fit(train[CONFIG["features"]], train.binary_label), "threshold": 0.5}
              for s in ("predicted", "expert")}
    table, scores = _classification(measurements, frozen, CONFIG)
    keys = set(zip(table.split, table.source, table.analysis))
    assert ("external_test", "predicted", "suspects counted as glaucoma") in keys
    assert ("in_domain_test", "vcdr only (no model)", "main") in keys
    assert ("in_domain_test", "predicted", "suspects counted as glaucoma") not in keys
    main = table[(table.split == "external_test") & (table.source == "predicted") & (table.analysis == "main")]
    assert main.n.iloc[0] == 110 and set(main.metric) >= {"auc", "sensitivity", "specificity"}
    assert scores.binary_label.notna().all() and len(scores) == 120 * 2 + 110 * 2

"""Small file helpers shared across the project."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV with every column as text, so an empty patient ID stays empty instead of 'nan'."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _json_safe(value):
    """Convert to plain JSON types. NaN and infinity become null (strict JSON has no NaN)."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value) -> None:
    Path(path).write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n")

"""Time helpers robust to pandas datetime resolutions (s/ms/us/ns)."""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def epoch_seconds(values) -> np.ndarray:
    """Convert datetime-like values (any resolution, naive = UTC) to float epoch seconds."""
    s = pd.to_datetime(pd.Series(values), utc=True)
    return (s - _EPOCH).dt.total_seconds().to_numpy(dtype=float)

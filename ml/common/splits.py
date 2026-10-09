"""Temporal splitting and label availability.

Financial behaviour drifts over time and fraud labels arrive late, so every
model uses *time-based* splits:

    |------ train (60%) ------|-- validation (20%) --|-- test (20%) --|
                          train_end              val_end

The model is built at ``val_end`` ("now"). Training/validation labels are the
labels *reported before now*: a fraud reported later is "not (yet) fraud",
exactly what a production retraining job would see. Validation rows younger
than ``label_maturity_days`` are dropped from model selection/calibration
because their labels have not matured. Evaluation on the test period uses final
ground truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import Settings


def temporal_boundaries(event_time: pd.Series, settings: Settings | dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    tr = settings["training"]
    ts = pd.to_datetime(event_time, utc=True)
    start, end = ts.min(), ts.max()
    span = end - start
    train_end = start + span * float(tr["train_fraction"])
    val_end = start + span * (float(tr["train_fraction"]) + float(tr["validation_fraction"]))
    # align to midnight UTC for readability/reproducibility
    return train_end.floor("D"), val_end.floor("D")


def assign_split(event_time: pd.Series, train_end: pd.Timestamp, val_end: pd.Timestamp) -> np.ndarray:
    ts = pd.to_datetime(event_time, utc=True)
    return np.select([ts < train_end, ts < val_end], ["train", "validation"], default="test")


def known_labels(labels: pd.Series, reported_at: pd.Series, as_of: pd.Timestamp) -> np.ndarray:
    """Label as visible to a system at ``as_of`` (late/unreported fraud -> 0)."""
    rep = pd.to_datetime(reported_at, utc=True)
    return (labels.to_numpy(dtype=bool) & (rep < as_of).fillna(False).to_numpy(dtype=bool)).astype(int)

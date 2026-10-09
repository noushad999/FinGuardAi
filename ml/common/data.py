"""Loading the feature table joined with labels and temporal split assignment."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from common.config import Settings, get_settings
from ml.common.bundle import file_fingerprint
from ml.common.splits import assign_split, temporal_boundaries


@dataclass
class TrainingFrame:
    df: pd.DataFrame  # features + labels + split
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    dataset_version: str

    def part(self, split: str) -> pd.DataFrame:
        return self.df[self.df["split"] == split]


def load_training_frame(settings: Settings | None = None) -> TrainingFrame:
    settings = settings or get_settings()
    feat_path = settings.path("processed_dir") / "transaction_features.parquet"
    synth = settings.path("synthetic_dir")
    feats = pd.read_parquet(feat_path)
    labels = pd.read_parquet(synth / "transaction_labels.parquet")
    df = feats.merge(labels, on="transaction_id", how="left", validate="one_to_one")
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    train_end, val_end = temporal_boundaries(df["event_time"], settings)
    df["split"] = assign_split(df["event_time"], train_end, val_end)
    version = f"tx-{file_fingerprint(synth / 'transactions.parquet')}"
    return TrainingFrame(df=df, train_end=train_end, val_end=val_end, dataset_version=version)

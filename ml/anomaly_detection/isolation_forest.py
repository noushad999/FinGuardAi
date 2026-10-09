"""Model B: Isolation Forest - label-free transaction anomaly detection.

Trained on the training period *without labels*. It is useful for patterns no
one has labelled yet (new fraud typologies) and as a safety net next to the
supervised model. Scores are mapped to a percentile of the training-score
distribution so that the business policy can say "review the top 0.1%".
"""

from __future__ import annotations

import shutil
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from ml.common.bundle import BundleMetadata, load_metadata, save_metadata

N_QUANTILES = 1001


class PercentileMapper:
    """Maps a raw score to its percentile in a reference distribution."""

    def __init__(self, quantiles: list[float]) -> None:
        self.q = np.asarray(quantiles, dtype=float)
        self.levels = np.linspace(0, 100, len(self.q))

    @classmethod
    def fit(cls, scores: np.ndarray) -> PercentileMapper:
        return cls(np.quantile(scores, np.linspace(0, 1, N_QUANTILES)).tolist())

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        # right-continuous empirical CDF, resolved to the 0.1 percentile
        return np.interp(np.asarray(scores, dtype=float), self.q, self.levels)


def train_isolation_forest(X_train: np.ndarray, params: dict, seed: int) -> IsolationForest:
    model = IsolationForest(
        n_estimators=int(params.get("n_estimators", 300)),
        max_samples=min(int(params.get("max_samples", 4096)), len(X_train)),
        contamination=params.get("contamination", "auto"),
        random_state=seed,
        n_jobs=4,
    )
    return model.fit(X_train)


def anomaly_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Higher = more anomalous."""
    return -model.score_samples(X)


def save_bundle(bundle_dir: Path, model: IsolationForest, meta: BundleMetadata) -> None:
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    joblib.dump(model, bundle_dir / "model.joblib")
    save_metadata(bundle_dir, meta)


class IsolationForestModel:
    def __init__(self, bundle_dir: str | Path) -> None:
        self.dir = Path(bundle_dir)
        self.meta = load_metadata(self.dir)
        self.model: IsolationForest = joblib.load(self.dir / "model.joblib")
        self.model.n_jobs = 1  # single-row online scoring: thread fan-out costs more than it saves
        self.mapper = PercentileMapper(self.meta.extra["score_quantiles"])
        self.feature_names = self.meta.feature_names

    @property
    def version(self) -> str:
        return self.meta.version

    def score(self, X: np.ndarray) -> np.ndarray:
        return anomaly_scores(self.model, np.atleast_2d(X))

    def percentile(self, X: np.ndarray) -> np.ndarray:
        return self.mapper(self.score(X))

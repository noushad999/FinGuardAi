"""Model bundles: a self-describing directory per deployable model.

::

    models/fraud_xgb/
        metadata.json           # name, version, feature schema, thresholds, metrics, data lineage
        model.json | model.joblib
        calibrator.json         # optional isotonic calibration table
        feature_reference.json  # training feature distribution (drift monitoring)

The same directory is logged to MLflow and registered, so what the API loads
from ``models/`` and what the registry tracks are byte-identical.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class SchemaMismatchError(RuntimeError):
    """Raised when a model's feature schema does not match the running code."""


@dataclass
class BundleMetadata:
    name: str
    version: str
    model_type: str
    feature_names: list[str]
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default, allow_nan=False)


def sanitize(obj: Any) -> Any:
    """Replace NaN/inf with None recursively (strict JSON)."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    return obj


def save_metadata(bundle_dir: str | Path, meta: BundleMetadata) -> None:
    write_json(Path(bundle_dir) / "metadata.json", sanitize(asdict(meta)))


def load_metadata(bundle_dir: str | Path) -> BundleMetadata:
    with open(Path(bundle_dir) / "metadata.json", encoding="utf-8") as fh:
        return BundleMetadata(**json.load(fh))


def check_schema(meta: BundleMetadata, available: list[str]) -> None:
    missing = [f for f in meta.feature_names if f not in set(available)]
    if missing:
        raise SchemaMismatchError(f"{meta.name} {meta.version}: features missing from engine: {missing}")


def file_fingerprint(path: str | Path) -> str:
    """Short content hash used as a dataset version."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def feature_reference(df: pd.DataFrame, names: list[str], n_bins: int = 10) -> dict[str, Any]:
    """Quantile bins + proportions per feature (for PSI drift monitoring)."""
    ref: dict[str, Any] = {}
    for n in names:
        x = df[n].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1))) if len(x) else np.array([0.0, 1.0])
        inner = edges[1:-1]
        counts = np.bincount(np.searchsorted(inner, x, side="right"), minlength=len(inner) + 1)
        ref[n] = {
            "edges": inner.tolist(),
            "proportions": (counts / max(counts.sum(), 1)).tolist(),
            "mean": float(x.mean()) if len(x) else 0.0,
            "std": float(x.std()) if len(x) else 0.0,
        }
    return ref


class PlattCalibrator:
    """Logistic (Platt) calibration of a raw model margin: p = sigmoid(a * margin + b).

    Chosen over isotonic regression because validation sets contain only a few
    hundred positives: two parameters are stable, monotone, and preserve the
    model's ranking (so PR-AUC/ROC-AUC are unaffected by calibration).
    """

    def __init__(self, a: float = 1.0, b: float = 0.0) -> None:
        self.a = float(a)
        self.b = float(b)

    @classmethod
    def fit(cls, margins: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1e4, max_iter=1000)
        lr.fit(np.asarray(margins, dtype=float).reshape(-1, 1), np.asarray(labels).astype(int))
        return cls(float(lr.coef_[0, 0]), float(lr.intercept_[0]))

    def transform(self, margins: np.ndarray) -> np.ndarray:
        z = self.a * np.asarray(margins, dtype=float) + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

    def to_json(self, path: str | Path) -> None:
        write_json(path, {"method": "platt", "a": self.a, "b": self.b})

    @classmethod
    def from_json(cls, path: str | Path) -> PlattCalibrator:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return cls(d["a"], d["b"])

"""Shared XGBoost training / inference utilities (fraud, ATO and churn models)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from ml.common.bundle import BundleMetadata, PlattCalibrator, check_schema, load_metadata


def train_booster(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    params: dict[str, Any],
    feature_names: list[str],
    scale_pos_weight: float = 1.0,
    seed: int = 42,
) -> xgb.Booster:
    """Gradient-boosted trees with early stopping on validation PR-AUC."""
    dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
    dva = xgb.DMatrix(X_va, label=y_va, feature_names=feature_names)
    booster_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "max_depth": int(params.get("max_depth", 5)),
        "eta": float(params.get("learning_rate", 0.05)),
        "subsample": float(params.get("subsample", 0.8)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
        "min_child_weight": float(params.get("min_child_weight", 3)),
        "scale_pos_weight": float(scale_pos_weight),
        "seed": seed,
        "nthread": 4,
    }
    return xgb.train(
        booster_params,
        dtr,
        num_boost_round=int(params.get("n_estimators", 600)),
        evals=[(dva, "validation")],
        early_stopping_rounds=int(params.get("early_stopping_rounds", 50)),
        verbose_eval=False,
    )


def margins(booster: xgb.Booster, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    d = xgb.DMatrix(X, feature_names=feature_names)
    return booster.predict(d, output_margin=True, iteration_range=(0, booster.best_iteration + 1))


def contributions(booster: xgb.Booster, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Exact TreeSHAP values (last column = bias), in log-odds units."""
    d = xgb.DMatrix(X, feature_names=feature_names)
    return booster.predict(d, pred_contribs=True, iteration_range=(0, booster.best_iteration + 1))


class XGBModel:
    """A loaded XGBoost bundle: booster + calibration + schema."""

    def __init__(self, bundle_dir: str | Path, available_features: list[str] | None = None) -> None:
        self.dir = Path(bundle_dir)
        self.meta: BundleMetadata = load_metadata(self.dir)
        if available_features is not None:
            check_schema(self.meta, available_features)
        self.booster = xgb.Booster()
        self.booster.load_model(str(self.dir / "model.json"))
        # Online scoring is one row at a time: a single thread avoids OpenMP fan-out/contention.
        self.booster.set_param({"nthread": 1})
        best = self.meta.extra.get("best_iteration")
        self.best_iteration = int(best) if best is not None else None
        cal = self.dir / "calibrator.json"
        self.calibrator = PlattCalibrator.from_json(cal) if cal.exists() else None
        self.feature_names = self.meta.feature_names

    @property
    def version(self) -> str:
        return self.meta.version

    def _array(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].to_numpy(dtype=np.float32)
        return np.atleast_2d(np.asarray(X, dtype=np.float32))

    def _matrix(self, X: np.ndarray | pd.DataFrame) -> xgb.DMatrix:
        # nthread=1: building a DMatrix for a single row with a thread pool costs ~15 ms under contention
        return xgb.DMatrix(self._array(X), feature_names=self.feature_names, nthread=1)

    def _range(self) -> tuple[int, int]:
        return (0, self.best_iteration + 1) if self.best_iteration is not None else (0, 0)

    def predict_margin(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        # inplace_predict skips DMatrix construction entirely (fast path for online scoring)
        return np.asarray(
            self.booster.inplace_predict(self._array(X), iteration_range=self._range(), predict_type="margin")
        ).reshape(-1)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        m = self.predict_margin(X)
        if self.calibrator is not None:
            return self.calibrator.transform(m)
        return 1.0 / (1.0 + np.exp(-m))

    def contributions(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self._matrix(X), pred_contribs=True, iteration_range=self._range())

    def vector(self, features: dict[str, float]) -> np.ndarray:
        return np.array([[features[n] for n in self.feature_names]], dtype=np.float32)

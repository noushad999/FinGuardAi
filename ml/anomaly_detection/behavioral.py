"""Model C: behavioural anomaly detection against each customer's own baseline.

The feature engine already expresses the current transaction *relative to the
customer's history* (amount z-score vs. their own amounts, share of their
activity at this hour, never-seen device / district / recipient, velocity vs.
their own daily rate, share of balance moved...). This model turns those
deviations into one interpretable score without using any labels:

* each component ``x`` gets an upper-tail probability ``p = P(X >= x)`` under
  the population of *training* transactions,
* the score is Fisher's combination ``sum(-log p)`` - large when several
  independent-ish deviations happen at once,
* each component's ``-log p`` doubles as an explanation ("hour is in the top
  0.3% most unusual for this customer").

Customers with fewer than ``min_history`` prior transactions have no reliable
baseline; their score only uses components that do not need history.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.common.bundle import BundleMetadata, load_metadata, save_metadata, write_json

MIN_HISTORY = 3

# component name -> (description, function of feature frame -> value, needs_history)
COMPONENTS: dict[str, tuple[str, Any, bool]] = {
    "amount_deviation": (
        "Amount above the customer's usual range",
        lambda f: np.clip(f["amount_zscore"], 0, None),
        True,
    ),
    "hour_rarity": (
        "Unusual time of day for the customer",
        lambda f: -np.log(np.clip(f["hour_frequency"], 1e-4, 1)),
        True,
    ),
    "velocity": (
        "Burst of activity vs. the customer's normal rate",
        lambda f: np.log1p(f["txn_count_1h"]) + np.log1p(f["velocity_ratio_24h"]),
        False,
    ),
    "new_device": ("New device", lambda f: f["is_new_device"], True),
    "new_location": (
        "New district / implausible travel",
        lambda f: f["is_new_district"] + np.log1p(f["speed_from_prev_kmh"]) / 4,
        True,
    ),
    "new_recipient": ("New counterparty", lambda f: f["is_new_recipient"], True),
    "balance_drain": ("Large share of balance moved", lambda f: np.clip(f["amount_to_balance_ratio"], 0, 5), False),
    "auth_failures": (
        "Recent failed / wrong-PIN attempts",
        lambda f: f["wrong_pin_count_1h"] + f["failed_count_1h"] + (f["pin_attempts"] - 1),
        False,
    ),
    "pass_through": (
        "Money moved out shortly after it arrived",
        lambda f: np.where(
            f["log_minutes_since_last_inflow"] < np.log1p(120), np.clip(f["outflow_to_last_inflow_ratio"], 0, 3), 0.0
        ),
        False,
    ),
}
REQUIRED_FEATURES = sorted(
    {
        "amount_zscore",
        "hour_frequency",
        "txn_count_1h",
        "velocity_ratio_24h",
        "is_new_device",
        "is_new_district",
        "speed_from_prev_kmh",
        "is_new_recipient",
        "amount_to_balance_ratio",
        "wrong_pin_count_1h",
        "failed_count_1h",
        "pin_attempts",
        "log_minutes_since_last_inflow",
        "outflow_to_last_inflow_ratio",
        "n_prior_txns",
    }
)


Frame = pd.DataFrame | dict[str, np.ndarray]


def as_columns(f: pd.DataFrame | dict[str, float]) -> Frame:
    """A single feature dict becomes a dict of 1-element arrays (no pandas overhead online)."""
    if isinstance(f, dict):
        return {k: np.asarray([v], dtype=float) for k, v in f.items()}
    return f


def component_values(f: Frame) -> dict[str, np.ndarray]:
    return {name: np.asarray(fn(f), dtype=float) for name, (_, fn, _) in COMPONENTS.items()}


class BehavioralAnomalyModel:
    def __init__(
        self, reference: dict[str, dict[str, list[float]]], score_quantiles: list[float], version: str = "behavioral-v0"
    ) -> None:
        self.reference = reference
        self.score_q = np.asarray(score_quantiles, dtype=float)
        self.version = version

    # ---- fitting ----
    @classmethod
    def fit(cls, train_features: pd.DataFrame, version: str) -> BehavioralAnomalyModel:
        comps = component_values(train_features)
        reference: dict[str, dict[str, list[float]]] = {}
        for name in COMPONENTS:
            x = np.sort(comps[name])
            # store the sorted sample compressed to 2001 quantiles
            reference[name] = {"q": np.quantile(x, np.linspace(0, 1, 2001)).tolist()}
        model = cls(reference, [0.0, 1.0], version)
        scores = model.score_frame(train_features)
        model.score_q = np.quantile(scores, np.linspace(0, 1, 1001))
        return model

    # ---- scoring ----
    def _neglog_tail(self, name: str, x: np.ndarray) -> np.ndarray:
        q = np.asarray(self.reference[name]["q"])
        levels = np.linspace(0, 1, len(q))
        # P(X >= x) ~= 1 - F(x^-); ties at the minimum get p = 1
        cdf_below = np.interp(x, q, levels, left=0.0, right=1.0)
        at_min = x <= q[0]
        p = np.where(at_min, 1.0, np.clip(1.0 - cdf_below, 1.0 / (len(q) * 5), 1.0))
        return -np.log(p)

    def component_scores(self, f: pd.DataFrame | dict[str, float]) -> dict[str, np.ndarray]:
        f = as_columns(f)
        comps = component_values(f)
        has_hist = np.asarray(f["n_prior_txns"], dtype=float) >= MIN_HISTORY
        out = {}
        for name, (_, _, needs_hist) in COMPONENTS.items():
            s = self._neglog_tail(name, comps[name])
            out[name] = np.where(has_hist | (not needs_hist), s, 0.0)
        return out

    def score_frame(self, f: pd.DataFrame | dict[str, float]) -> np.ndarray:
        return np.sum(list(self.component_scores(f).values()), axis=0)

    def percentile(self, f: pd.DataFrame | dict[str, float]) -> np.ndarray:
        return np.interp(self.score_frame(f), self.score_q, np.linspace(0, 100, len(self.score_q)))

    def explain(self, features: dict[str, float], top: int = 3) -> list[dict[str, Any]]:
        cs = {k: float(v[0]) for k, v in self.component_scores(features).items()}
        ranked = sorted(cs.items(), key=lambda kv: kv[1], reverse=True)
        return [
            {"component": k, "reason": COMPONENTS[k][0], "tail_probability": float(np.exp(-v))}
            for k, v in ranked
            if v > 2.0
        ][:top]

    # ---- persistence ----
    def save(self, bundle_dir: Path, meta: BundleMetadata) -> None:
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        bundle_dir.mkdir(parents=True)
        write_json(
            bundle_dir / "reference.json", {"components": self.reference, "score_quantiles": self.score_q.tolist()}
        )
        save_metadata(bundle_dir, meta)

    @classmethod
    def load(cls, bundle_dir: str | Path) -> BehavioralAnomalyModel:
        bundle_dir = Path(bundle_dir)
        meta = load_metadata(bundle_dir)
        with open(bundle_dir / "reference.json", encoding="utf-8") as fh:
            ref = json.load(fh)
        return cls(ref["components"], ref["score_quantiles"], meta.version)

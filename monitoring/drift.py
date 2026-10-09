"""ML monitoring: feature drift, prediction drift and realised performance.

``python -m monitoring``

* **Feature drift** - Population Stability Index (PSI) of every model feature
  in the monitored window vs. the training distribution stored in the model
  bundle (``feature_reference.json``). Rule of thumb: < 0.1 stable,
  0.1-0.25 moderate shift, > 0.25 significant shift (investigate / retrain).
* **Prediction drift** - PSI of the fraud-probability distribution and the
  alert rate vs. the model's test period.
* **Performance when labels arrive** - labels are joined as they are reported;
  precision/recall of REVIEW+BLOCK decisions and PR-AUC per week, restricted to
  weeks whose labels have matured.

Source of monitored traffic: ``ops.risk_decisions`` (real-time scoring, with the
exact feature vectors used) when the database is reachable, otherwise the
offline test period (batch scores + feature table).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from common.config import Settings, get_settings, load_policy
from common.logging import get_logger
from ml.common.bundle import load_metadata, sanitize, write_json
from ml.common.metrics import classification_report

log = get_logger(__name__)
PSI_EPS = 1e-4

# Features that grow with the length of observed history (counters, ages). They
# drift by construction as the platform ages, so they are reported separately
# instead of raising alerts; a real deployment would normalise them by tenure.
CUMULATIVE_FEATURES = frozenset(
    {
        "n_prior_txns",
        "sender_out_degree",
        "merchant_txn_count",
        "merchant_unique_customers",
        "device_txn_count",
        "account_age_days",
        "receiver_account_age_days",
        "receiver_n_prior_txns",
        "customer_device_count",
        "merchant_age_days",
        "receiver_in_degree",
    }
)


def psi_from_reference(ref: dict[str, Any], values: np.ndarray) -> float:
    edges = np.asarray(ref["edges"], dtype=float)
    expected = np.asarray(ref["proportions"], dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    actual = counts / counts.sum()
    e = np.clip(expected, PSI_EPS, None)
    a = np.clip(actual, PSI_EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(v: float) -> str:
    if not np.isfinite(v):
        return "no_data"
    return "stable" if v < 0.1 else ("moderate" if v < 0.25 else "significant")


def feature_drift(reference: dict[str, Any], current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, ref in reference.items():
        if name not in current:
            continue
        x = current[name].to_numpy(dtype=float)
        v = psi_from_reference(ref, x)
        rows.append(
            {
                "feature": name,
                "psi": v,
                "status": psi_status(v),
                "cumulative": name in CUMULATIVE_FEATURES,
                "reference_mean": ref["mean"],
                "current_mean": float(np.nanmean(x)) if len(x) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def _reference_scores(settings: Settings) -> np.ndarray:
    s = pd.read_parquet(settings.path("processed_dir") / "fraud_scores.parquet", columns=["fraud_probability", "split"])
    return s.loc[s["split"] == "test", "fraud_probability"].to_numpy()


def _score_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))[1:-1]
    props = np.bincount(np.searchsorted(edges, reference, side="right"), minlength=len(edges) + 1) / len(reference)
    return psi_from_reference({"edges": edges.tolist(), "proportions": props.tolist()}, current)


def load_monitoring_window(settings: Settings) -> tuple[pd.DataFrame, str]:
    """Returns (frame with features + fraud_probability + decision + event_time + labels, source)."""
    try:
        from common.db import get_engine

        engine = get_engine()
        q = (
            "SELECT d.transaction_id, d.event_time, d.fraud_probability, d.decision, d.features, "
            "l.is_fraud, l.fraud_reported_at FROM ops.risk_decisions d "
            "LEFT JOIN raw.transaction_labels l USING (transaction_id) WHERE d.source = 'stream'"
        )
        df = pd.read_sql(q, engine)
        if len(df) >= 200:
            feats = pd.DataFrame([f if isinstance(f, dict) else json.loads(f or "{}") for f in df["features"]])
            df = pd.concat([df.drop(columns=["features"]).reset_index(drop=True), feats], axis=1)
            return df, "ops.risk_decisions"
    except Exception as exc:  # noqa: BLE001
        log.info("monitoring_db_unavailable", extra={"error": str(exc)[:120]})
    processed, synth = settings.path("processed_dir"), settings.path("synthetic_dir")
    feats = pd.read_parquet(processed / "transaction_features.parquet")
    scores = pd.read_parquet(processed / "fraud_scores.parquet")
    labels = pd.read_parquet(synth / "transaction_labels.parquet")
    df = feats.merge(scores, on="transaction_id").merge(labels, on="transaction_id")
    df = df[df["split"] == "test"]
    pol = load_policy()["thresholds"]
    risk = 100 * np.maximum(df["fraud_probability"], df.get("ato_probability", 0))
    df["decision"] = np.where(risk >= pol["block"], "BLOCK", np.where(risk >= pol["review"], "REVIEW", "APPROVE"))
    return df, "offline test period"


def performance_by_week(df: pd.DataFrame, as_of: pd.Timestamp, maturity_days: int) -> pd.DataFrame:
    d = df.copy()
    d["event_time"] = pd.to_datetime(d["event_time"], utc=True)
    rep = pd.to_datetime(d["fraud_reported_at"], utc=True)
    d["label_known"] = (d["is_fraud"].fillna(False).astype(bool) & (rep <= as_of)).astype(int)
    d["week"] = d["event_time"].dt.tz_convert(None).dt.to_period("W").dt.start_time
    rows = []
    for week, g in d.groupby("week"):
        flagged = g["decision"].isin(["REVIEW", "BLOCK"])
        y = g["label_known"].to_numpy()
        mature = (as_of - pd.Timestamp(week, tz="UTC")).days >= maturity_days + 7
        rep_ = classification_report(y, g["fraud_probability"].to_numpy(), 0.5) if y.any() else {}
        rows.append(
            {
                "week": pd.Timestamp(week).date().isoformat(),
                "transactions": int(len(g)),
                "flagged": int(flagged.sum()),
                "known_fraud": int(y.sum()),
                "precision_flagged": float(y[flagged.to_numpy()].mean()) if flagged.any() else None,
                "recall_flagged": float(flagged.to_numpy()[y == 1].mean()) if y.any() else None,
                "pr_auc": rep_.get("pr_auc"),
                "labels_mature": bool(mature),
            }
        )
    return pd.DataFrame(rows)


def run(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    bundle = settings.path("models_dir") / "fraud_xgb"
    meta = load_metadata(bundle)
    reference = json.loads((bundle / "feature_reference.json").read_text())
    df, source = load_monitoring_window(settings)
    drift = feature_drift(reference, df)
    ref_scores = _reference_scores(settings)
    pred_psi = _score_psi(ref_scores, df["fraud_probability"].to_numpy(dtype=float))
    as_of = pd.to_datetime(df["event_time"], utc=True).max() + pd.Timedelta(days=60)  # labels as known "later"
    as_of = min(as_of, pd.Timestamp.now(tz="UTC"))
    perf = performance_by_week(df, as_of, int(settings["training"].get("label_maturity_days", 21)))
    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_version": meta.version,
        "source": source,
        "rows": int(len(df)),
        "feature_drift": {
            "significant": drift.loc[(drift["status"] == "significant") & ~drift["cumulative"], "feature"].tolist(),
            "moderate": drift.loc[(drift["status"] == "moderate") & ~drift["cumulative"], "feature"].tolist(),
            "cumulative_features_shifted": drift.loc[
                (drift["status"] != "stable") & drift["cumulative"], "feature"
            ].tolist(),
            "top": drift.head(10).to_dict("records"),
        },
        "prediction_drift": {
            "fraud_probability_psi": pred_psi,
            "status": psi_status(pred_psi),
            "alert_rate": float(df["decision"].isin(["REVIEW", "BLOCK"]).mean()),
            "mean_fraud_probability": float(df["fraud_probability"].mean()),
        },
        "performance_by_week": perf.to_dict("records"),
    }
    write_json(settings.path("reports_dir") / "monitoring_report.json", sanitize(result))
    _persist(result, drift)
    log.info(
        "monitoring_report",
        extra={
            "source": source,
            "rows": len(df),
            "prediction_psi": round(pred_psi, 4),
            "significant_drift": len(result["feature_drift"]["significant"]),
        },
    )
    return result


def _persist(result: dict[str, Any], drift: pd.DataFrame) -> None:
    try:
        from sqlalchemy import text

        from common.db import get_engine

        engine = get_engine()
        run_at = datetime.now(UTC)
        rows = [
            {
                "r": run_at,
                "v": result["model_version"],
                "t": "feature_psi",
                "n": r.feature,
                "val": None if not np.isfinite(r.psi) else float(r.psi),
                "s": r.status,
                "d": None,
            }
            for r in drift.itertuples()
        ]
        pd_ = result["prediction_drift"]
        rows.append(
            {
                "r": run_at,
                "v": result["model_version"],
                "t": "prediction_psi",
                "n": "fraud_probability",
                "val": pd_["fraud_probability_psi"],
                "s": pd_["status"],
                "d": json.dumps(pd_),
            }
        )
        for w in result["performance_by_week"]:
            rows.append(
                {
                    "r": run_at,
                    "v": result["model_version"],
                    "t": "performance",
                    "n": w["week"],
                    "val": w["precision_flagged"],
                    "s": "mature" if w["labels_mature"] else "immature",
                    "d": json.dumps(sanitize(w)),
                }
            )
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO ops.model_monitoring (run_at, model_version, metric_type, name, value, "
                    "status, details) VALUES (:r, :v, :t, :n, :val, :s, CAST(:d AS JSONB))"
                ),
                rows,
            )
    except Exception as exc:  # noqa: BLE001 - reports are still written to disk
        log.info("monitoring_persist_skipped", extra={"error": str(exc)[:120]})


if __name__ == "__main__":  # pragma: no cover
    run()

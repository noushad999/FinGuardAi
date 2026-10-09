"""Churn prediction with snapshot-based labels and leakage-safe temporal splits.

Definition
----------
At a snapshot date ``T`` a customer is **eligible** if they made at least
``min_successful_txns_in_lookback`` successful transactions in
``[T - lookback_days, T)``. An eligible customer is **churned** if they make *no*
successful transaction in ``[T, T + inactivity_days)``.

Features only use ``[T - lookback_days, T)``; labels only use the future window.

Splits (no temporal overlap between any label window and a later snapshot)::

    train snapshots  T <= T_val - inactivity_days
    validation       T_val  = T_test - inactivity_days
    test             T_test = last snapshot whose label window is fully observed

The final model then scores every eligible customer at the end of the data,
predicting churn over the next ``inactivity_days``.
"""

from __future__ import annotations

import shutil
from typing import Any

import numpy as np
import pandas as pd

from common.config import Settings, get_settings
from common.logging import get_logger
from ml.common.bundle import BundleMetadata, PlattCalibrator, sanitize, save_metadata, write_json
from ml.common.metrics import calibration_report, classification_report, select_threshold
from ml.common.tracking import next_version, tracked_run
from ml.common.xgb import margins, train_booster

log = get_logger(__name__)
REGISTERED_NAME, PREFIX = "finguard-churn-xgb", "churn-xgb"

CHURN_FEATURES = [
    "n_success_lookback",
    "n_success_last15",
    "activity_trend",
    "active_days",
    "days_since_last_success",
    "log_avg_amount",
    "log_total_amount",
    "failed_ratio",
    "failed_last15",
    "network_failures_last15",
    "distinct_types",
    "p2p_ratio",
    "cash_out_ratio",
    "merchant_ratio",
    "recharge_ratio",
    "bill_ratio",
    "distinct_counterparties",
    "p2p_received",
    "mean_gap_days",
    "account_age_days",
    "n_devices",
]


def snapshot_features(
    tx: pd.DataFrame, customers: pd.DataFrame, T: pd.Timestamp, lookback: int, min_success: int
) -> pd.DataFrame:
    lo = T - pd.Timedelta(days=lookback)
    w = tx[(tx["event_time"] >= lo) & (tx["event_time"] < T)]
    ok = w[w["status"] == "SUCCESS"]
    g_ok = ok.groupby("customer_id")
    n_ok = g_ok.size()
    eligible = n_ok[n_ok >= min_success].index
    ok = ok[ok["customer_id"].isin(eligible)]
    w = w[w["customer_id"].isin(eligible)]
    g_ok, g_all = ok.groupby("customer_id"), w.groupby("customer_id")
    last15 = T - pd.Timedelta(days=15)
    f = pd.DataFrame(index=eligible)
    f["n_success_lookback"] = g_ok.size()
    f["n_success_last15"] = ok[ok["event_time"] >= last15].groupby("customer_id").size()
    f = f.fillna(0)
    rate_recent = f["n_success_last15"] / 15
    rate_before = (f["n_success_lookback"] - f["n_success_last15"]) / (lookback - 15)
    f["activity_trend"] = np.log((rate_recent + 0.02) / (rate_before + 0.02))
    f["active_days"] = g_ok["event_time"].agg(lambda s: s.dt.floor("D").nunique())
    f["days_since_last_success"] = (T - g_ok["event_time"].max()).dt.total_seconds() / 86400
    f["log_avg_amount"] = np.log1p(g_ok["amount"].mean())
    f["log_total_amount"] = np.log1p(g_ok["amount"].sum())
    failed = w[w["status"] == "FAILED"]
    f["failed_ratio"] = (failed.groupby("customer_id").size() / g_all.size()).reindex(f.index).fillna(0)
    f["failed_last15"] = failed[failed["event_time"] >= last15].groupby("customer_id").size()
    f["network_failures_last15"] = (
        failed[(failed["event_time"] >= last15) & (failed["failure_reason"] == "NETWORK_TIMEOUT")]
        .groupby("customer_id")
        .size()
    )
    types = pd.crosstab(ok["customer_id"], ok["transaction_type"])
    f["distinct_types"] = (types > 0).sum(axis=1)
    for t, col in [
        ("P2P", "p2p_ratio"),
        ("CASH_OUT", "cash_out_ratio"),
        ("MERCHANT_PAYMENT", "merchant_ratio"),
        ("MOBILE_RECHARGE", "recharge_ratio"),
        ("BILL_PAYMENT", "bill_ratio"),
    ]:
        f[col] = types.get(t, 0) / f["n_success_lookback"]
    f["distinct_counterparties"] = g_ok["receiver_id"].nunique()
    f["p2p_received"] = ok[ok["transaction_type"] == "P2P"].groupby("receiver_id").size().reindex(f.index)
    gaps = (
        ok.sort_values("event_time")
        .groupby("customer_id")["event_time"]
        .agg(lambda s: s.diff().dt.total_seconds().mean() / 86400 if len(s) > 1 else lookback)
    )
    f["mean_gap_days"] = gaps
    reg = pd.to_datetime(customers.set_index("customer_id")["registration_date"], utc=True).reindex(f.index)
    f["account_age_days"] = (T - reg).dt.days
    f["n_devices"] = w[w["device_id"].notna()].groupby("customer_id")["device_id"].nunique()
    f = f.fillna(0).astype(float)
    f.index.name = "customer_id"
    return f.reset_index().assign(snapshot=T)


def churn_labels(tx: pd.DataFrame, snap: pd.DataFrame, T: pd.Timestamp, horizon: int) -> np.ndarray:
    fut = tx[
        (tx["event_time"] >= T) & (tx["event_time"] < T + pd.Timedelta(days=horizon)) & (tx["status"] == "SUCCESS")
    ]
    active = set(fut["customer_id"])
    return (~snap["customer_id"].isin(active)).astype(int).to_numpy()


def plan_snapshots(
    start: pd.Timestamp, end: pd.Timestamp, lookback: int, horizon: int, every: int
) -> dict[str, list[pd.Timestamp]]:
    snaps = []
    T = start + pd.Timedelta(days=lookback)
    while T + pd.Timedelta(days=horizon) <= end:
        snaps.append(T)
        T += pd.Timedelta(days=every)
    if len(snaps) < 3:
        raise ValueError("Not enough history for churn train/validation/test snapshots; increase SIM_DAYS")
    t_test = snaps[-1]
    t_val = max(s for s in snaps if s <= t_test - pd.Timedelta(days=horizon))
    train = [s for s in snaps if s <= t_val - pd.Timedelta(days=horizon)]
    if not train:
        raise ValueError("No training snapshots available; increase SIM_DAYS")
    return {"train": train, "validation": [t_val], "test": [t_test]}


def run(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    cfg = settings["churn"]
    lookback, horizon = int(cfg["lookback_days"]), int(cfg["inactivity_days"])
    min_s, every = int(cfg["min_successful_txns_in_lookback"]), int(cfg["snapshot_every_days"])
    synth = settings.path("synthetic_dir")
    tx = pd.read_parquet(synth / "transactions.parquet")
    tx["event_time"] = pd.to_datetime(tx["event_time"], utc=True)
    customers = pd.read_parquet(synth / "customers.parquet")
    start, end = tx["event_time"].min().floor("D"), tx["event_time"].max()
    plan = plan_snapshots(start, end, lookback, horizon, every)

    frames = {}
    for split, snaps in plan.items():
        parts = []
        for T in snaps:
            s = snapshot_features(tx, customers, T, lookback, min_s)
            s["churned"] = churn_labels(tx, s, T, horizon)
            parts.append(s)
        frames[split] = pd.concat(parts, ignore_index=True)
    X = {k: v[CHURN_FEATURES].to_numpy(dtype=np.float32) for k, v in frames.items()}
    y = {k: v["churned"].to_numpy() for k, v in frames.items()}

    params = dict(settings["training"]["xgb"])
    params.update({"max_depth": 4, "n_estimators": 400})
    seed = int(settings["training"]["random_state"])
    booster = train_booster(X["train"], y["train"], X["validation"], y["validation"], params, CHURN_FEATURES, 1.0, seed)
    cal = PlattCalibrator.fit(margins(booster, X["validation"], CHURN_FEATURES), y["validation"])
    p_val = cal.transform(margins(booster, X["validation"], CHURN_FEATURES))
    thr = select_threshold(y["validation"], p_val)
    p_test = cal.transform(margins(booster, X["test"], CHURN_FEATURES))
    t = thr["max_f1_threshold"]
    test_report = classification_report(y["test"], p_test, t, ks=(100, 250))
    calib = calibration_report(y["test"], p_test)
    # naive baseline: recency only
    base = classification_report(y["test"], frames["test"]["days_since_last_success"].to_numpy(), 30.0, ks=(100,))
    evaluation = {
        "definition": {"lookback_days": lookback, "inactivity_days": horizon, "min_successful_txns_in_lookback": min_s},
        "snapshots": {k: [s.isoformat() for s in v] for k, v in plan.items()},
        "rows": {k: int(len(v)) for k, v in frames.items()},
        "churn_rate": {k: float(v["churned"].mean()) for k, v in frames.items()},
        "test": {
            "roc_auc": test_report["roc_auc"],
            "pr_auc": test_report["pr_auc"],
            "threshold": t,
            "precision": test_report["precision"],
            "recall": test_report["recall"],
            "f1": test_report["f1"],
            "brier": calib["brier"],
            "ece": calib["ece"],
            "reliability": calib["reliability"],
        },
        "baseline_recency_only": {"roc_auc": base["roc_auc"], "pr_auc": base["pr_auc"]},
    }

    bundle = settings.path("models_dir") / "churn_xgb"
    version = next_version(REGISTERED_NAME, bundle, PREFIX)
    with tracked_run("churn-xgb", {"model_family": "churn"}) as run:
        if bundle.exists():
            shutil.rmtree(bundle)
        bundle.mkdir(parents=True)
        booster.save_model(str(bundle / "model.json"))
        cal.to_json(bundle / "calibrator.json")
        meta = BundleMetadata(
            name="churn_xgb",
            version=version,
            model_type="xgboost_binary_classifier",
            feature_names=CHURN_FEATURES,
            params=params,
            metrics=sanitize(evaluation),
            thresholds=thr,
            data={"snapshots": evaluation["snapshots"]},
            extra={"best_iteration": int(booster.best_iteration), "mlflow_run_id": run.run_id},
        )
        save_metadata(bundle, meta)
        run.log_params({"lookback_days": lookback, "inactivity_days": horizon, "version": version})
        run.log_metrics({k: v for k, v in evaluation["test"].items() if isinstance(v, float)}, prefix="test.")
        run.register_bundle(bundle, REGISTERED_NAME, "Customer churn probability (next 30 days)")

    # ---- score all eligible customers "now" ----
    now = end.ceil("D")
    current = snapshot_features(tx, customers, now, lookback, min_s)
    current["churn_probability"] = cal.transform(
        margins(booster, current[CHURN_FEATURES].to_numpy(np.float32), CHURN_FEATURES)
    )
    current["churn_model_version"] = version
    current["as_of"] = now
    current[
        [
            "customer_id",
            "churn_probability",
            "churn_model_version",
            "as_of",
            "days_since_last_success",
            "activity_trend",
            "n_success_lookback",
        ]
    ].to_parquet(settings.path("processed_dir") / "churn_scores.parquet", index=False)
    write_json(settings.path("reports_dir") / "churn.json", sanitize(evaluation))
    log.info(
        "churn_trained",
        extra={
            "version": version,
            "roc_auc": evaluation["test"]["roc_auc"],
            "pr_auc": evaluation["test"]["pr_auc"],
            "brier": evaluation["test"]["brier"],
        },
    )
    return evaluation


if __name__ == "__main__":  # pragma: no cover
    run()

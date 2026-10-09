"""Unified customer intelligence profile.

Combines, per customer:
* transaction-level model outputs over the last 30 days (fraud / ATO probability),
* churn probability (next 30 days), customer segment,
* graph signals (mule score percentile, membership of a suspicious cluster),
* behavioural descriptors (stability of weekly activity, recent activity vs. baseline).

The overall ``risk_level`` uses the *same* business policy thresholds as the
real-time engine (configs/policy.yaml) so a customer page and a transaction
decision never disagree about what "high risk" means.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from common.config import Settings, get_settings, load_policy
from common.logging import get_logger

log = get_logger(__name__)


def _weekly_stats(tx: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    ok = tx[(tx["status"] == "SUCCESS") & (tx["event_time"] >= as_of - pd.Timedelta(weeks=12))]
    week = ((as_of - ok["event_time"]).dt.days // 7).clip(upper=11)
    counts = pd.crosstab(ok["customer_id"], week).reindex(columns=range(12), fill_value=0)
    recent = counts[0]
    baseline = counts[list(range(1, 9))].mean(axis=1)
    cv = counts.std(axis=1) / counts.mean(axis=1).replace(0, np.nan)
    return pd.DataFrame({"txn_last_7d": recent, "weekly_baseline": baseline, "weekly_cv": cv})


def build_profiles(
    tx: pd.DataFrame,
    customers: pd.DataFrame,
    scores: pd.DataFrame | None,
    churn: pd.DataFrame | None,
    segments: pd.DataFrame | None,
    graph: pd.DataFrame | None,
    policy: dict[str, Any],
) -> pd.DataFrame:
    tx = tx.copy()
    tx["event_time"] = pd.to_datetime(tx["event_time"], utc=True)
    as_of = tx["event_time"].max().ceil("D")
    review, block = policy["thresholds"]["review"] / 100, policy["thresholds"]["block"] / 100
    prof = (
        customers[["customer_id", "district", "customer_segment", "account_status"]]
        .rename(columns={"customer_segment": "product_tier"})
        .set_index("customer_id")
    )

    if scores is not None and len(scores):
        recent = tx[tx["event_time"] >= as_of - pd.Timedelta(days=30)][["transaction_id", "customer_id"]]
        s = recent.merge(scores, on="transaction_id", how="inner")
        g = s.groupby("customer_id")
        prof["fraud_probability_30d_max"] = g["fraud_probability"].max()
        prof["fraud_probability_30d_mean"] = g["fraud_probability"].mean()
        if "ato_probability" in s:
            prof["ato_probability_30d_max"] = g["ato_probability"].max()
        prof["flagged_txn_30d"] = s[(s["fraud_probability"] >= review)].groupby("customer_id").size()
    for col in (
        "fraud_probability_30d_max",
        "fraud_probability_30d_mean",
        "ato_probability_30d_max",
        "flagged_txn_30d",
    ):
        if col not in prof:
            prof[col] = np.nan
    prof["flagged_txn_30d"] = prof["flagged_txn_30d"].fillna(0).astype(int)

    if churn is not None and len(churn):
        prof["churn_probability"] = churn.set_index("customer_id")["churn_probability"]
    else:
        prof["churn_probability"] = np.nan
    if segments is not None and len(segments):
        prof["segment_name"] = segments.set_index("customer_id")["segment_name"]
    prof["segment_name"] = prof.get("segment_name", pd.Series(dtype=object)).fillna("Unsegmented (low activity)")
    if graph is not None and len(graph):
        gi = graph.set_index("customer_id")
        prof["mule_score_percentile"] = gi["mule_score_percentile"]
        prof["in_suspicious_cluster"] = gi["in_suspicious_cluster"].fillna(False).astype(bool)
        prof["community_id"] = gi["community_id"]
    else:
        prof["mule_score_percentile"] = np.nan
        prof["in_suspicious_cluster"] = False
        prof["community_id"] = np.nan

    wk = _weekly_stats(tx, as_of)
    prof = prof.join(wk)
    prof[["txn_last_7d", "weekly_baseline"]] = prof[["txn_last_7d", "weekly_baseline"]].fillna(0)
    ratio = prof["txn_last_7d"] / prof["weekly_baseline"].replace(0, np.nan)
    prof["recent_activity"] = np.select(
        [
            prof["txn_last_7d"].eq(0) & prof["weekly_baseline"].eq(0),
            prof["txn_last_7d"].eq(0),
            ratio >= 2.5,
            ratio <= 0.3,
        ],
        ["Inactive", "No activity this week", "Elevated", "Low"],
        default="Normal",
    )
    churn_p = prof["churn_probability"]
    prof["behavior"] = np.select(
        [prof["weekly_baseline"].eq(0), churn_p.ge(0.5), prof["weekly_cv"].lt(0.8), prof["weekly_cv"].lt(1.5)],
        ["Dormant", "Declining", "Stable", "Variable"],
        default="Erratic",
    )

    fraud_max = prof["fraud_probability_30d_max"].fillna(0)
    ato_max = prof["ato_probability_30d_max"].fillna(0)
    mule_pct = prof["mule_score_percentile"].fillna(0)
    graph_risk = np.where((mule_pct >= 99) & prof["in_suspicious_cluster"], 0.9, np.where(mule_pct >= 97, 0.6, 0.0))
    score = np.maximum.reduce([fraud_max.to_numpy(), ato_max.to_numpy(), graph_risk])
    prof["risk_score"] = np.round(score * 100).astype(int)
    prof["risk_level"] = np.select([score >= block, score >= review], ["High", "Medium"], default="Low")

    reasons = []
    for _, r in prof.iterrows():
        rs = []
        if r["fraud_probability_30d_max"] >= review:
            rs.append(f"{int(r['flagged_txn_30d'])} transaction(s) above the review threshold in the last 30 days")
        if r["ato_probability_30d_max"] >= review:
            rs.append("Recent activity consistent with account takeover")
        if r["mule_score_percentile"] >= 97:
            rs.append(f"Money-mule indicators (graph score percentile {r['mule_score_percentile']:.0f})")
        if r["in_suspicious_cluster"]:
            rs.append("Member of a suspicious transaction cluster")
        if r["recent_activity"] == "Elevated":
            rs.append("Activity this week well above the customer's baseline")
        reasons.append(rs)
    prof["risk_reasons"] = reasons
    prof["as_of"] = as_of
    return prof.reset_index()


def run(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    synth, processed = settings.path("synthetic_dir"), settings.path("processed_dir")

    def opt(name: str) -> pd.DataFrame | None:
        p = processed / name
        return pd.read_parquet(p) if p.exists() else None

    prof = build_profiles(
        pd.read_parquet(synth / "transactions.parquet"),
        pd.read_parquet(synth / "customers.parquet"),
        opt("fraud_scores.parquet"),
        opt("churn_scores.parquet"),
        opt("customer_segments.parquet"),
        opt("graph_customer_metrics.parquet"),
        load_policy(),
    )
    prof.to_parquet(processed / "customer_profiles.parquet", index=False)
    log.info("customer_profiles_built", extra={"customers": len(prof), **prof["risk_level"].value_counts().to_dict()})
    return prof


if __name__ == "__main__":  # pragma: no cover
    run()

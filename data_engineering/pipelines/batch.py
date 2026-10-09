"""Batch data pipelines (the offline analytics path).

    synthetic extracts ──ingest_lake──> lake raw zone (Parquet, partitioned by event_date)
    lake raw zone ──load_raw──> PostgreSQL raw.*  ──dbt──> analytics.* (star schema)
    ML outputs ──load_ml──> PostgreSQL ml.*

Every step is idempotent: lake partitions are overwritten, warehouse tables are
fully refreshed inside a transaction (TRUNCATE + COPY), so an Airflow retry or
a manual re-run never duplicates data.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pyarrow as pa

from common.config import Settings, get_settings
from common.db import copy_dataframe, get_engine, init_schema, record_run
from common.logging import get_logger
from common.storage import Lake

log = get_logger(__name__)

ENTITIES = ("customers", "merchants", "devices", "transaction_labels")


def ingest_lake(settings: Settings | None = None) -> dict[str, Any]:
    """Copy source extracts into the lake raw zone."""
    settings = settings or get_settings()
    synth = settings.path("synthetic_dir")
    lake = Lake(settings)
    lake.ensure_bucket()
    out: dict[str, Any] = {"lake": lake.uri}
    tx = pd.read_parquet(synth / "transactions.parquet")
    tx["event_date"] = tx["event_time"].dt.strftime("%Y-%m-%d")
    lake.write_table(
        pa.Table.from_pandas(tx, preserve_index=False), "raw", "transactions", partition_cols=["event_date"]
    )
    out["transactions"] = len(tx)
    for name in ENTITIES:
        df = pd.read_parquet(synth / f"{name}.parquet")
        lake.write_table(pa.Table.from_pandas(df, preserve_index=False), "raw", name, "data.parquet")
        out[name] = len(df)
    log.info("lake_ingested", extra=out)
    return out


def _read_lake(lake: Lake, name: str) -> pd.DataFrame:
    if name == "transactions":
        df = lake.read_table("raw", "transactions").to_pandas()
        return df.drop(columns=["event_date"], errors="ignore").sort_values("event_time").reset_index(drop=True)
    return lake.read_table("raw", name, "data.parquet").to_pandas()


def load_raw(settings: Settings | None = None) -> dict[str, int]:
    """Lake raw zone -> PostgreSQL raw schema (full refresh)."""
    settings = settings or get_settings()
    engine = get_engine()
    init_schema(engine)
    lake = Lake(settings)
    started = datetime.now(UTC)
    counts: dict[str, int] = {}
    order = ["customers", "merchants", "devices", "transactions", "transaction_labels"]
    for name in order:
        df = _read_lake(lake, name)
        json_cols = ["metadata"] if name == "transactions" else []
        counts[name] = copy_dataframe(engine, df, f"raw.{name}", json_cols=json_cols)
        log.info("raw_table_loaded", extra={"table": f"raw.{name}", "rows": counts[name]})
    record_run(engine, "load_raw", started, "success", sum(counts.values()), counts)
    return counts


ML_TABLES = {
    "fraud_scores.parquet": ("ml.transaction_scores", []),
    "customer_segments.parquet": ("ml.customer_segments", []),
    "churn_scores.parquet": ("ml.churn_scores", []),
    "graph_customer_metrics.parquet": ("ml.graph_customer_metrics", []),
    "graph_communities.parquet": ("ml.graph_communities", []),
    "graph_suspicious_edges.parquet": ("ml.graph_suspicious_edges", []),
    "customer_profiles.parquet": ("ml.customer_profiles", ["risk_reasons"]),
}
ML_COLUMNS = {
    "ml.transaction_scores": [
        "transaction_id",
        "fraud_probability",
        "ato_probability",
        "anomaly_percentile",
        "behavioral_percentile",
        "split",
        "fraud_model_version",
        "ato_model_version",
    ],
    "ml.customer_segments": [
        "customer_id",
        "segment_id",
        "segment_name",
        "hdbscan_cluster",
        "n_txn",
        "monthly_spend",
        "segmentation_version",
    ],
}


def load_ml(settings: Settings | None = None) -> dict[str, int]:
    """Batch ML outputs -> PostgreSQL ml schema (full refresh)."""
    settings = settings or get_settings()
    engine = get_engine()
    init_schema(engine)
    processed = settings.path("processed_dir")
    started = datetime.now(UTC)
    counts: dict[str, int] = {}
    with engine.connect() as conn:
        table_cols = {
            t: [
                r[0]
                for r in conn.exec_driver_sql(
                    "SELECT column_name FROM information_schema.columns WHERE table_schema=%s AND table_name=%s "
                    "ORDER BY ordinal_position",
                    tuple(t.split(".")),
                )
            ]
            for t, _ in ML_TABLES.values()
        }
    for fname, (table, json_cols) in ML_TABLES.items():
        path = processed / fname
        if not path.exists():
            log.warning("ml_output_missing", extra={"file": fname})
            continue
        df = pd.read_parquet(path)
        cols = [c for c in table_cols[table] if c in df.columns]
        for c in (
            "device_share_max_accounts",
            "device_component_size",
            "community_size",
            "segment_id",
            "hdbscan_cluster",
            "n_txn",
            "community_id",
            "flagged_txn_30d",
            "risk_score",
            "size",
            "max_device_sharing",
        ):
            if c in cols and table != "ml.customer_profiles" and c in df:
                df[c] = df[c].round().astype("Int64")
        if table == "ml.customer_profiles":
            for c in ("flagged_txn_30d", "risk_score"):
                df[c] = df[c].round().astype("Int64")
        counts[table] = copy_dataframe(engine, df[cols], table, json_cols=json_cols)
        log.info("ml_table_loaded", extra={"table": table, "rows": counts[table]})
    record_run(engine, "load_ml", started, "success", sum(counts.values()), counts)
    return counts

"""Run data-quality checks.

python -m data_engineering.validation                 # synthetic extracts (pre-load gate)
python -m data_engineering.validation --source bronze # stream output written by Spark to the lake
python -m data_engineering.validation --fail-on-error # non-zero exit if an error check fails (Airflow)
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from common.config import get_settings
from common.logging import get_logger
from data_engineering.validation.quality import build_report, write_report

log = get_logger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["synthetic", "lake-raw", "bronze"], default="synthetic")
    p.add_argument("--fail-on-error", action="store_true")
    a = p.parse_args()
    settings = get_settings()
    synth = settings.path("synthetic_dir")
    refs = {
        "customers": pd.read_parquet(synth / "customers.parquet"),
        "devices": pd.read_parquet(synth / "devices.parquet"),
        "merchants": pd.read_parquet(synth / "merchants.parquet"),
    }
    if a.source == "synthetic":
        tx = pd.read_parquet(synth / "transactions.parquet")
        refs["labels"] = pd.read_parquet(synth / "transaction_labels.parquet")
    else:
        from common.storage import Lake

        lake = Lake(settings)
        zone = ("raw", "transactions") if a.source == "lake-raw" else ("bronze", "transactions")
        tx = lake.read_table(*zone).to_pandas()
        tx = tx.drop(
            columns=[
                c
                for c in ("event_date", "ingested_at", "kafka_partition", "kafka_offset", "schema_version")
                if c in tx.columns
            ]
        )
        if "metadata" in tx.columns and not pd.api.types.is_string_dtype(tx["metadata"]):
            tx["metadata"] = tx["metadata"].map(lambda v: v if isinstance(v, str) else json.dumps(v))
    report = build_report(a.source, tx, **refs)
    j, md = write_report(report, settings.path("reports_dir"))
    _persist(report)
    summary = {k: v for k, v in report.to_dict().items() if k != "checks"}
    log.info("data_quality_done", extra={**summary, "report": str(md)})
    print(json.dumps(summary, indent=2))
    if a.fail_on_error and not report.passed:
        sys.exit(1)


def _persist(report) -> None:  # noqa: ANN001
    try:
        from sqlalchemy import text

        from common.db import get_engine

        rows = [
            {
                "r": report.run_at,
                "d": report.dataset,
                "c": c.name,
                "s": c.severity,
                "p": c.passed,
                "f": c.failing_rows,
                "x": json.dumps({"description": c.description, "sample": c.sample}),
            }
            for c in report.checks
        ]
        with get_engine().begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO ops.data_quality_results (run_at, dataset, check_name, severity, passed, "
                    "failing_rows, details) VALUES (:r, :d, :c, :s, :p, :f, CAST(:x AS JSONB))"
                ),
                rows,
            )
    except Exception as exc:  # noqa: BLE001 - report files are still written
        log.info("dq_persist_skipped", extra={"error": str(exc)[:120]})


if __name__ == "__main__":
    main()

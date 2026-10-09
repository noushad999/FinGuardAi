"""End-to-end tests against the running Docker stack (`make up pipeline apps`).

Covers: Kafka -> risk engine -> PostgreSQL, Kafka -> Spark -> lake/PostgreSQL,
API -> model/Redis state, and the dbt warehouse.
"""

import json
import time
import uuid
from datetime import UTC, datetime

import httpx
import pandas as pd
import pytest
from sqlalchemy import text

from tests.integration.conftest import API, KAFKA, S3

pytestmark = pytest.mark.integration


def _wait(fn, timeout=120, every=2.0):
    end = time.time() + timeout
    while time.time() < end:
        out = fn()
        if out:
            return out
        time.sleep(every)
    return fn()


def _events(n: int, tag: str):
    """Events for a dedicated test customer (keeps demo customers' online state clean)."""
    now = datetime.now(UTC).replace(microsecond=0)
    return [
        {
            "transaction_id": f"ITEST_{tag}_{i:04d}",
            "event_time": now.isoformat(),
            "customer_id": "CUS_9990042",
            "sender_id": "CUS_9990042",
            "receiver_id": "CUS_9990043",
            "amount": 250.0 + i,
            "transaction_type": "P2P",
            "channel": "APP",
            "device_id": "DEV_ITEST_0001",
            "district": "Dhaka",
            "latitude": 23.8,
            "longitude": 90.4,
            "status": "SUCCESS",
            "balance_before": 10_000.0,
            "balance_after": 9_750.0 - i,
            "metadata": {"pin_attempts": 1},
        }
        for i in range(n)
    ]


def _produce(events, extra_raw=()):
    from confluent_kafka import Producer

    p = Producer({"bootstrap.servers": KAFKA})
    for e in events:
        p.produce("fg.transactions", key=e["customer_id"].encode(), value=json.dumps(e).encode())
    for raw in extra_raw:
        p.produce("fg.transactions", key=b"CUS_9990042", value=raw)
    assert p.flush(30) == 0


def test_api_health_uses_redis_and_postgres():
    h = httpx.get(f"{API}/health", timeout=10).json()
    assert h["status"] == "ok"
    assert h["state_store"] == "RedisStateStore" and h["repository"] == "postgres"


def test_kafka_to_risk_engine_to_postgres_with_dedup_and_dlq(engine):
    tag = uuid.uuid4().hex[:8]
    events = _events(20, tag)
    # 20 valid events + 5 duplicates + 2 invalid messages
    _produce(events + events[:5], extra_raw=[b"{broken json", json.dumps({**events[0], "amount": -1}).encode()])

    def done():
        with engine.connect() as c:
            n = c.execute(
                text("SELECT count(*) FROM ops.risk_decisions WHERE transaction_id LIKE :p"), {"p": f"ITEST_{tag}_%"}
            ).scalar()
        return n if n == 20 else None

    assert _wait(done, timeout=180) == 20, "every valid event scored exactly once"
    with engine.connect() as c:
        row = c.execute(
            text("SELECT decision, model_version, features FROM ops.risk_decisions WHERE transaction_id = :t"),
            {"t": f"ITEST_{tag}_0000"},
        ).one()
    assert row.decision in ("APPROVE", "REVIEW", "BLOCK") and row.model_version.startswith("fraud-xgb")
    assert "txn_count_1m" in row.features
    metrics = httpx.get("http://localhost:9102/metrics", timeout=5).text
    assert "finguard_dlq_messages_total" in metrics and "finguard_duplicate_events_total" in metrics


def test_kafka_to_spark_to_lake_and_postgres(engine):
    from urllib.parse import urlparse

    import pyarrow.dataset as ds
    import pyarrow.fs as pafs

    tag = uuid.uuid4().hex[:8]
    _produce(_events(10, tag))
    fs = pafs.S3FileSystem(
        access_key="finguard", secret_key="finguard-secret", endpoint_override=urlparse(S3).netloc, scheme="http"
    )

    def in_bronze():
        try:
            t = (
                ds.dataset("finguard-lake/bronze/transactions", filesystem=fs, format="parquet", partitioning="hive")
                .to_table(columns=["transaction_id"])
                .to_pandas()
            )
        except Exception:
            return None
        return (t["transaction_id"].str.startswith(f"ITEST_{tag}")).sum() == 10 or None

    assert _wait(in_bronze, timeout=240, every=10), "Spark wrote the events to the bronze zone"
    with engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM ops.stream_metrics_minute")).scalar() > 0
        assert c.execute(text("SELECT count(*) FROM ops.stream_customer_windows")).scalar() > 0


def test_api_scores_with_live_state_and_persists(engine):
    body = _events(1, uuid.uuid4().hex[:8])[0]
    body.update(amount=45_000, balance_before=46_000, device_id="DEV_BRAND_NEW", metadata={"pin_attempts": 4})
    r = httpx.post(f"{API}/risk/score", json=body, timeout=10)
    assert r.status_code == 200
    res = r.json()
    assert res["decision"] in ("REVIEW", "BLOCK"), res
    assert res["risk_reasons"]
    with engine.connect() as c:
        assert (
            c.execute(
                text("SELECT source FROM ops.risk_decisions WHERE transaction_id = :t"), {"t": body["transaction_id"]}
            ).scalar()
            == "api"
        )
    prof = httpx.get(f"{API}/customer/CUS_0000042", timeout=10).json()
    assert prof["live_state"]["prior_transactions"] > 0 and prof["customer_segment"]


def test_warehouse_marts_are_queryable_and_consistent(engine):
    with engine.connect() as c:
        tx = c.execute(text("SELECT count(*) FROM raw.transactions")).scalar()
        fact = c.execute(text("SELECT count(*) FROM analytics.fact_transactions")).scalar()
        kpis = pd.read_sql(text("SELECT sum(transactions) AS n FROM analytics.mart_daily_kpis"), c)
    assert tx == fact == int(kpis["n"].iloc[0])

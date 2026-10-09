"""Spark Structured Streaming logic tested with Spark in batch mode (same transformation functions)."""

import importlib.util
import json
import shutil

import pytest

pytestmark = pytest.mark.spark
if importlib.util.find_spec("pyspark") is None or shutil.which("java") is None:
    pytest.skip("pyspark/java not available", allow_module_level=True)


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    s = (
        SparkSession.builder.master("local[1]")
        .appName("fg-test")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    yield s
    s.stop()


def kafka_frame(spark, payloads):
    from pyspark.sql import Row

    rows = [Row(value=bytearray(p.encode()), partition=0, offset=i, timestamp=None) for i, p in enumerate(payloads)]
    return spark.createDataFrame(rows, "value binary, partition int, offset long, timestamp timestamp")


GOOD = {
    "transaction_id": "T1",
    "event_time": "2026-05-01T10:00:10Z",
    "customer_id": "CUS_0000001",
    "receiver_id": "CUS_0000002",
    "amount": 100.0,
    "transaction_type": "P2P",
    "status": "SUCCESS",
    "device_id": "D1",
    "metadata": {"pin_attempts": "1"},
}


def test_parse_validates_and_quarantines(spark):
    from streaming.spark_jobs.transactions_stream import parse, valid_records

    payloads = [
        json.dumps(GOOD),
        "{broken",
        json.dumps({**GOOD, "transaction_id": "T2", "amount": -1}),
        json.dumps({**GOOD, "transaction_id": "T3", "customer_id": "X"}),
        json.dumps({**GOOD, "transaction_id": "T4", "transaction_type": "TELEPORT"}),
    ]
    parsed = parse(kafka_frame(spark, payloads))
    reasons = {r["invalid_reason"] for r in parsed.collect()}
    assert reasons == {None, "malformed_json", "invalid_amount", "invalid_customer_id", "invalid_transaction_type"}
    valid = valid_records(parsed).collect()
    assert len(valid) == 1 and valid[0]["transaction_id"] == "T1" and str(valid[0]["event_date"]) == "2026-05-01"


def test_window_aggregates(spark):
    from streaming.spark_jobs.transactions_stream import customer_windows, minute_metrics, parse, valid_records

    evs = [
        json.dumps(
            {
                **GOOD,
                "transaction_id": f"T{i}",
                "event_time": f"2026-05-01T10:0{i}:00Z",
                "receiver_id": f"CUS_000000{i % 2 + 2}",
                "status": "FAILED" if i == 1 else "SUCCESS",
            }
        )
        for i in range(4)
    ]
    df = valid_records(parse(kafka_frame(spark, evs))).dropDuplicates(["transaction_id"])
    w = customer_windows(df, "5 minutes", "1 minute").filter("window_start = '2026-05-01 09:59:00'").collect()
    assert len(w) == 1 and w[0]["txn_count"] == 4 and w[0]["failed_count"] == 1 and w[0]["unique_receivers"] == 2
    m = {str(r["window_start"]): r["txn_count"] for r in minute_metrics(df).collect()}
    assert sum(m.values()) == 4 and len(m) == 4

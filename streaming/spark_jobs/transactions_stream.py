"""Spark Structured Streaming job: Kafka -> validation -> lake (bronze) + event-time window aggregates.

    spark-submit streaming/spark_jobs/transactions_stream.py

Pipeline
--------
1. **Source**: Kafka topic ``fg.transactions`` (JSON values, key = customer_id).
2. **Parse & validate**: ``from_json`` with an explicit schema; records that fail
   parsing or basic validation (missing ids, non-positive amount, unknown type,
   bad customer id) go to the lake **quarantine** zone with a reason.
3. **Bronze** (append-only, at-least-once): every valid record + Kafka
   partition/offset + ingestion time, Parquet partitioned by ``event_date``.
   Bronze deliberately keeps late and duplicate records - it is the replayable
   history; the warehouse layer de-duplicates on ``transaction_id``.
4. **Aggregates** (event time): ``withWatermark("event_time", WATERMARK)``
   bounds state and tolerates out-of-order data; duplicates are dropped with
   ``dropDuplicatesWithinWatermark(transaction_id)``; then
   * per-customer sliding windows (5 min / 1 min slide, 1 h / 15 min slide):
     count, amount sum/avg, distinct receivers, distinct devices, failures;
   * platform-wide 1-minute throughput.
   Windows are emitted in *append* mode (once final, i.e. when the watermark
   passes the window end) and upserted into PostgreSQL, so re-processing after
   a restart is idempotent. Events later than the watermark are dropped from
   the aggregates (Spark reports them as ``numRowsDroppedByWatermark``) but are
   still present in bronze.
5. **Checkpoints** in the lake make the query restartable with exactly-once
   state semantics for the Spark-managed parts.

This file is intentionally self-contained (configuration via environment
variables, Python 3.8 compatible) because it runs inside the Spark image rather
than the application image.
"""

from __future__ import annotations

import json
import os
import time

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

KAFKA = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9094")
TOPIC = os.environ.get("KAFKA_TRANSACTIONS_TOPIC", "fg.transactions")
LAKE = os.environ.get("LAKE_URI", "data/lake").rstrip("/")
if LAKE.startswith("s3://"):
    LAKE = "s3a://" + LAKE[len("s3://"):]
CHECKPOINTS = os.environ.get("CHECKPOINT_URI", LAKE + "/_checkpoints/transactions_stream")
WATERMARK = os.environ.get("STREAM_WATERMARK", "10 minutes")
TRIGGER = os.environ.get("STREAM_TRIGGER", "15 seconds")
MAX_OFFSETS = os.environ.get("STREAM_MAX_OFFSETS_PER_TRIGGER", "20000")
STARTING_OFFSETS = os.environ.get("STREAM_STARTING_OFFSETS", "earliest")
PG = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": int(os.environ.get("POSTGRES_PORT", "5432")),
    "dbname": os.environ.get("POSTGRES_DB", "finguard"),
    "user": os.environ.get("POSTGRES_USER", "finguard"),
    "password": os.environ.get("POSTGRES_PASSWORD", "finguard"),
}
TRANSACTION_TYPES = ["P2P", "MERCHANT_PAYMENT", "CASH_IN", "CASH_OUT", "BILL_PAYMENT", "MOBILE_RECHARGE",
                     "BANK_TRANSFER"]

SCHEMA = T.StructType([
    T.StructField("schema_version", T.StringType()),
    T.StructField("transaction_id", T.StringType()),
    T.StructField("event_time", T.TimestampType()),
    T.StructField("sender_id", T.StringType()),
    T.StructField("receiver_id", T.StringType()),
    T.StructField("customer_id", T.StringType()),
    T.StructField("merchant_id", T.StringType()),
    T.StructField("amount", T.DoubleType()),
    T.StructField("fee", T.DoubleType()),
    T.StructField("currency", T.StringType()),
    T.StructField("transaction_type", T.StringType()),
    T.StructField("channel", T.StringType()),
    T.StructField("device_id", T.StringType()),
    T.StructField("district", T.StringType()),
    T.StructField("latitude", T.DoubleType()),
    T.StructField("longitude", T.DoubleType()),
    T.StructField("status", T.StringType()),
    T.StructField("failure_reason", T.StringType()),
    T.StructField("balance_before", T.DoubleType()),
    T.StructField("balance_after", T.DoubleType()),
    T.StructField("metadata", T.MapType(T.StringType(), T.StringType())),
])


def build_spark() -> SparkSession:
    builder = (SparkSession.builder.appName("finguard-transactions-stream")
               .config("spark.sql.session.timeZone", "UTC")
               .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "4"))
               .config("spark.sql.streaming.stateStore.providerClass",
                       "org.apache.spark.sql.execution.streaming.state.HDFSBackedStateStoreProvider"))
    if LAKE.startswith("s3a://"):
        builder = (builder
                   .config("spark.hadoop.fs.s3a.endpoint", os.environ.get("S3_ENDPOINT", "http://localhost:8333"))
                   .config("spark.hadoop.fs.s3a.access.key", os.environ.get("S3_ACCESS_KEY", "finguard"))
                   .config("spark.hadoop.fs.s3a.secret.key", os.environ.get("S3_SECRET_KEY", "finguard-secret"))
                   .config("spark.hadoop.fs.s3a.path.style.access", "true")
                   .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
                   .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"))
    packages = os.environ.get("SPARK_KAFKA_PACKAGES")
    if packages:
        builder = builder.config("spark.jars.packages", packages)
    return builder.getOrCreate()


def parse(raw: DataFrame) -> DataFrame:
    """Decode Kafka records and attach a validation reason (null = valid)."""
    df = raw.select(
        F.col("value").cast("string").alias("json"),
        F.col("partition").alias("kafka_partition"),
        F.col("offset").alias("kafka_offset"),
        F.col("timestamp").alias("kafka_timestamp"),
    ).withColumn("data", F.from_json("json", SCHEMA))
    # from_json is PERMISSIVE (broken JSON -> all-null struct), so detect malformed payloads explicitly
    reason = (F.when(F.col("data").isNull() | F.get_json_object("json", "$").isNull(), F.lit("malformed_json"))
              .when(F.col("data.transaction_id").isNull(), F.lit("missing_transaction_id"))
              .when(F.col("data.event_time").isNull(), F.lit("invalid_event_time"))
              .when(F.col("data.customer_id").isNull() | ~F.col("data.customer_id").rlike(r"^CUS_\d+$"),
                    F.lit("invalid_customer_id"))
              .when(F.col("data.amount").isNull() | (F.col("data.amount") <= 0), F.lit("invalid_amount"))
              .when(~F.col("data.transaction_type").isin(TRANSACTION_TYPES), F.lit("invalid_transaction_type"))
              .otherwise(F.lit(None)))
    return df.withColumn("invalid_reason", reason)


def valid_records(parsed: DataFrame) -> DataFrame:
    return (parsed.filter(F.col("invalid_reason").isNull())
            .select("data.*", "kafka_partition", "kafka_offset")
            .withColumn("metadata", F.to_json("metadata"))
            .withColumn("ingested_at", F.current_timestamp())
            .withColumn("event_date", F.to_date("event_time")))


def customer_windows(deduped: DataFrame, size: str, slide: str) -> DataFrame:
    return (deduped.groupBy(F.window("event_time", size, slide), "customer_id")
            .agg(F.count("*").alias("txn_count"),
                 F.sum("amount").alias("amount_sum"),
                 F.avg("amount").alias("amount_avg"),
                 F.approx_count_distinct("receiver_id").alias("unique_receivers"),
                 F.approx_count_distinct("device_id").alias("unique_devices"),
                 F.sum(F.when(F.col("status") == "FAILED", 1).otherwise(0)).alias("failed_count"))
            .select("customer_id", F.col("window.start").alias("window_start"),
                    F.col("window.end").alias("window_end"), F.lit(size).alias("window_size"), "txn_count",
                    "amount_sum", "amount_avg", "unique_receivers", "unique_devices", "failed_count"))


def minute_metrics(deduped: DataFrame) -> DataFrame:
    return (deduped.groupBy(F.window("event_time", "1 minute"))
            .agg(F.count("*").alias("txn_count"), F.sum("amount").alias("amount_sum"),
                 F.sum(F.when(F.col("status") == "FAILED", 1).otherwise(0)).alias("failed_count"),
                 F.approx_count_distinct("customer_id").alias("unique_customers"))
            .select(F.col("window.start").alias("window_start"), "txn_count", "amount_sum", "failed_count",
                    "unique_customers"))


def _upsert(table: str, key: list[str], columns: list[str]):
    """foreachBatch sink: idempotent upsert of a (small) aggregate micro-batch into PostgreSQL."""
    def write(batch: DataFrame, batch_id: int) -> None:
        rows = [tuple(r[c] for c in columns) for r in batch.select(*columns).collect()]
        if not rows:
            return
        import psycopg2
        from psycopg2.extras import execute_values

        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns if c not in key)
        sql = (f"INSERT INTO {table} ({', '.join(columns)}) VALUES %s "
               f"ON CONFLICT ({', '.join(key)}) DO UPDATE SET {updates}, updated_at = now()")
        with psycopg2.connect(**PG) as conn, conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=1000)
        print(json.dumps({"event": "upsert", "table": table, "batch_id": batch_id, "rows": len(rows)}), flush=True)
    return write


def main() -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel(os.environ.get("SPARK_LOG_LEVEL", "WARN"))
    raw = (spark.readStream.format("kafka")
           .option("kafka.bootstrap.servers", KAFKA)
           .option("subscribe", TOPIC)
           .option("startingOffsets", STARTING_OFFSETS)
           .option("maxOffsetsPerTrigger", MAX_OFFSETS)
           .option("failOnDataLoss", "false")
           .load())
    parsed = parse(raw)
    valid = valid_records(parsed)

    queries = []
    queries.append(
        parsed.filter(F.col("invalid_reason").isNotNull())
        .select("json", "invalid_reason", "kafka_partition", "kafka_offset", "kafka_timestamp")
        .writeStream.format("parquet")
        .option("path", LAKE + "/quarantine/transactions")
        .option("checkpointLocation", CHECKPOINTS + "/quarantine")
        .trigger(processingTime=TRIGGER).queryName("quarantine").start())
    queries.append(
        valid.writeStream.format("parquet")
        .option("path", LAKE + "/bronze/transactions")
        .option("checkpointLocation", CHECKPOINTS + "/bronze")
        .partitionBy("event_date")
        .trigger(processingTime=TRIGGER).queryName("bronze").start())

    deduped = (valid.withWatermark("event_time", WATERMARK)
               .dropDuplicatesWithinWatermark(["transaction_id"]))
    window_cols = ["customer_id", "window_start", "window_end", "window_size", "txn_count", "amount_sum",
                   "amount_avg", "unique_receivers", "unique_devices", "failed_count"]
    for size, slide, name in (("5 minutes", "1 minute", "windows_5m"), ("1 hour", "15 minutes", "windows_1h")):
        queries.append(
            customer_windows(deduped, size, slide).writeStream.outputMode("append")
            .foreachBatch(_upsert("ops.stream_customer_windows", ["customer_id", "window_size", "window_start"],
                                  window_cols))
            .option("checkpointLocation", CHECKPOINTS + "/" + name)
            .trigger(processingTime=TRIGGER).queryName(name).start())
    queries.append(
        minute_metrics(deduped).writeStream.outputMode("append")
        .foreachBatch(_upsert("ops.stream_metrics_minute", ["window_start"],
                              ["window_start", "txn_count", "amount_sum", "failed_count", "unique_customers"]))
        .option("checkpointLocation", CHECKPOINTS + "/minute_metrics")
        .trigger(processingTime=TRIGGER).queryName("minute_metrics").start())

    # Lightweight progress logging (throughput, watermark, late rows dropped).
    run_seconds = float(os.environ.get("STREAM_RUN_SECONDS", "0"))
    started = time.time()
    while any(q.isActive for q in queries):
        time.sleep(30)
        for q in queries:
            p = q.lastProgress
            if p:
                state = (p.get("stateOperators") or [{}])
                print(json.dumps({
                    "event": "progress", "query": q.name, "batch": p.get("batchId"),
                    "input_rows": p.get("numInputRows"), "rows_per_sec": p.get("processedRowsPerSecond"),
                    "watermark": (p.get("eventTime") or {}).get("watermark"),
                    "late_rows_dropped": sum(s.get("numRowsDroppedByWatermark", 0) for s in state),
                    "state_rows": sum(s.get("numRowsTotal", 0) for s in state)}), flush=True)
        if run_seconds and time.time() - started > run_seconds:
            for q in queries:
                q.stop()
    spark.stop()


if __name__ == "__main__":
    main()

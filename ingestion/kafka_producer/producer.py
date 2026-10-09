"""Transaction replay producer.

Replays the synthetic transaction log into Kafka in event-time order, at a
configurable rate, simulating a live MFS platform:

    python -m ingestion.kafka_producer --from-split test --rate 200
    python -m ingestion.kafka_producer --limit 5000 --duplicate-rate 0.01 --invalid-rate 0.005 --late-rate 0.01

* Messages are keyed by ``customer_id`` so each customer's events stay ordered
  within one partition (the feature engine relies on per-customer order).
* The producer is idempotent (``enable.idempotence``, ``acks=all``) - broker
  retries cannot create duplicates.
* "Chaos" options inject duplicates, invalid records and late events to exercise
  de-duplication, the dead-letter topic and Spark watermarking.
* Every message is validated against :class:`~ingestion.schemas.TransactionEvent`
  before sending (except deliberately invalid chaos records).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from typing import Any

import pandas as pd
from confluent_kafka import KafkaException, Producer

from common.config import Settings, get_settings
from common.logging import get_logger
from ingestion.schemas import TransactionEvent, event_from_row

log = get_logger(__name__)


class TransactionProducer:
    def __init__(self, settings: Settings | None = None, producer: Any | None = None) -> None:
        settings = settings or get_settings()
        self.topic = settings["kafka"]["topics"]["transactions"]
        self.producer = producer or Producer(
            {
                "bootstrap.servers": settings["kafka"]["bootstrap_servers"],
                "enable.idempotence": True,
                "acks": "all",
                "compression.type": "lz4",
                "linger.ms": 20,
                "retries": 10,
                "retry.backoff.ms": 200,
                "delivery.timeout.ms": 120_000,
            }
        )
        self.delivered = 0
        self.failed = 0

    def _on_delivery(self, err, msg) -> None:  # noqa: ANN001 - librdkafka callback signature
        if err is not None:
            self.failed += 1
            log.error("delivery_failed", extra={"error": str(err), "key": msg.key()})
        else:
            self.delivered += 1

    def send(self, key: str, value: str) -> None:
        while True:
            try:
                self.producer.produce(self.topic, key=key.encode(), value=value.encode(), on_delivery=self._on_delivery)
                break
            except BufferError:  # local queue full: let librdkafka drain, then retry
                self.producer.poll(0.5)
            except KafkaException as exc:
                log.error("produce_failed", extra={"error": str(exc)})
                raise
        self.producer.poll(0)

    def flush(self, timeout: float = 60) -> int:
        return self.producer.flush(timeout)


def _load(settings: Settings, from_split: str | None, start: str | None, limit: int | None) -> pd.DataFrame:
    tx = pd.read_parquet(settings.path("synthetic_dir") / "transactions.parquet")
    tx = tx.sort_values(["event_time", "transaction_id"])
    if from_split:
        from ml.common.splits import temporal_boundaries

        train_end, val_end = temporal_boundaries(tx["event_time"], settings)
        tx = tx[tx["event_time"] >= {"validation": train_end, "test": val_end}[from_split]]
    if start:
        tx = tx[tx["event_time"] >= pd.Timestamp(start, tz="UTC")]
    if limit:
        tx = tx.head(limit)
    return tx


def replay(
    settings: Settings | None = None,
    *,
    from_split: str | None = "test",
    start: str | None = None,
    limit: int | None = None,
    rate: float | None = None,
    duplicate_rate: float = 0.0,
    invalid_rate: float = 0.0,
    late_rate: float = 0.0,
    seed: int = 7,
    producer: TransactionProducer | None = None,
) -> dict[str, int]:
    settings = settings or get_settings()
    rate = rate if rate is not None else float(settings["kafka"]["producer_rate_per_sec"])
    tx = _load(settings, from_split, start, limit)
    prod = producer or TransactionProducer(settings)
    rnd = random.Random(seed)
    stats = {"sent": 0, "duplicates": 0, "invalid": 0, "late": 0, "rejected_by_producer": 0}
    late_buffer: list[tuple[int, str, str]] = []
    t0 = time.time()
    for i, row in enumerate(tx.to_dict("records")):
        event = event_from_row(row)
        try:
            payload = TransactionEvent(**event).to_json()
        except Exception as exc:  # noqa: BLE001 - source data problem: never publish unvalidated
            stats["rejected_by_producer"] += 1
            log.warning(
                "producer_validation_failed",
                extra={"transaction_id": event.get("transaction_id"), "error": str(exc)[:200]},
            )
            continue
        key = event["customer_id"]
        if late_rate and rnd.random() < late_rate:
            # deliver this event ~200-2000 messages later (arrives late / out of order)
            late_buffer.append((i + rnd.randint(200, 2000), key, payload))
            stats["late"] += 1
        else:
            prod.send(key, payload)
            stats["sent"] += 1
        if duplicate_rate and rnd.random() < duplicate_rate:
            prod.send(key, payload)  # at-least-once upstream: same transaction_id twice
            stats["duplicates"] += 1
        if invalid_rate and rnd.random() < invalid_rate:
            bad = json.loads(payload)
            bad.update(
                rnd.choice(
                    [
                        {"amount": -abs(bad["amount"])},
                        {"customer_id": "??"},
                        {"event_time": "not-a-date"},
                        {"transaction_type": "TELEPORT"},
                    ]
                )
            )
            prod.send(key, json.dumps(bad))
            stats["invalid"] += 1
        due = [x for x in late_buffer if x[0] <= i]
        for item in due:
            prod.send(item[1], item[2])
            late_buffer.remove(item)
        if rate and rate > 0:
            expected = stats["sent"] / rate
            elapsed = time.time() - t0
            if expected > elapsed:
                time.sleep(expected - elapsed)
        if i and i % 5000 == 0:
            log.info("replay_progress", extra={**stats, "rows": i, "eps": round(i / (time.time() - t0), 1)})
    for _, key, payload in late_buffer:
        prod.send(key, payload)
    prod.flush()
    stats["delivered"] = prod.delivered
    stats["delivery_failed"] = prod.failed
    log.info("replay_done", extra={**stats, "seconds": round(time.time() - t0, 1)})
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Replay synthetic transactions into Kafka")
    p.add_argument(
        "--from-split",
        choices=["validation", "test", "all"],
        default="test",
        help="start of the replay window (default: test period = 'after the model was trained')",
    )
    p.add_argument("--start", help="ISO timestamp to start from (overrides --from-split)")
    p.add_argument("--limit", type=int)
    p.add_argument("--rate", type=float, help="events/second (0 = as fast as possible)")
    p.add_argument("--duplicate-rate", type=float, default=0.0)
    p.add_argument("--invalid-rate", type=float, default=0.0)
    p.add_argument("--late-rate", type=float, default=0.0)
    a = p.parse_args()
    from ingestion.kafka_admin import ensure_topics

    ensure_topics()
    print(
        json.dumps(
            replay(
                from_split=None if a.from_split == "all" else a.from_split,
                start=a.start,
                limit=a.limit,
                rate=a.rate,
                duplicate_rate=a.duplicate_rate,
                invalid_rate=a.invalid_rate,
                late_rate=a.late_rate,
            ),
            indent=2,
        )
    )

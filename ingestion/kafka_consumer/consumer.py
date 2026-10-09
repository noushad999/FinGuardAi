"""Validating Kafka consumer with retries and a dead-letter topic (DLQ).

Delivery semantics: **at-least-once**. Offsets are committed only after the
handler finished a batch, so a crash re-delivers the batch. Downstream writes
are idempotent (``ON CONFLICT DO NOTHING`` on ``transaction_id``) and the risk
engine de-duplicates events before touching online state.

Failure handling:
* malformed JSON / schema violations -> DLQ immediately (retrying cannot fix data),
* handler exceptions -> retried with exponential backoff; if a batch still
  fails, each record is retried individually and only the poison records go to
  the DLQ, so one bad record never blocks a partition.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from common.config import Settings, get_settings
from common.logging import get_logger
from ingestion.schemas import TransactionEvent
from monitoring import metrics as m

log = get_logger(__name__)

Handler = Callable[[list[TransactionEvent]], None]


class ValidatingConsumer:
    def __init__(
        self,
        handler: Handler,
        settings: Settings | None = None,
        group_id: str | None = None,
        batch_size: int = 250,
        max_retries: int = 3,
        consumer: Any | None = None,
        dlq_producer: Any | None = None,
    ) -> None:
        settings = settings or get_settings()
        kafka = settings["kafka"]
        self.topic = kafka["topics"]["transactions"]
        self.dlq_topic = kafka["topics"]["dead_letter"]
        self.handler = handler
        self.batch_size = batch_size
        self.max_retries = max_retries
        if consumer is None:
            from confluent_kafka import Consumer

            consumer = Consumer(
                {
                    "bootstrap.servers": kafka["bootstrap_servers"],
                    "group.id": group_id or kafka["consumer_group"],
                    "enable.auto.commit": False,
                    "auto.offset.reset": "earliest",
                    "max.poll.interval.ms": 300_000,
                    "session.timeout.ms": 30_000,
                }
            )
        if dlq_producer is None:
            from confluent_kafka import Producer

            dlq_producer = Producer(
                {"bootstrap.servers": kafka["bootstrap_servers"], "acks": "all", "enable.idempotence": True}
            )
        self.consumer = consumer
        self.dlq = dlq_producer
        self.stats = {"consumed": 0, "valid": 0, "dlq": 0, "batches": 0}

    # ------------------------------------------------------------------ DLQ
    def _to_dlq(self, msg: Any, reason: str, error: str) -> None:
        headers = [
            ("error_reason", reason.encode()),
            ("error", error[:1000].encode()),
            ("source_topic", str(msg.topic()).encode()),
            ("source_partition", str(msg.partition()).encode()),
            ("source_offset", str(msg.offset()).encode()),
            ("failed_at", datetime.now(UTC).isoformat().encode()),
        ]
        self.dlq.produce(self.dlq_topic, key=msg.key(), value=msg.value(), headers=headers)
        self.dlq.poll(0)
        self.stats["dlq"] += 1
        m.DLQ_MESSAGES.labels(reason).inc()

    # ------------------------------------------------------------- decoding
    def _decode(self, msg: Any) -> TransactionEvent | None:
        try:
            payload = json.loads(msg.value())
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
            self._to_dlq(msg, "malformed_json", str(exc))
            m.EVENTS_CONSUMED.labels("malformed").inc()
            return None
        try:
            event = TransactionEvent(**payload)
        except (ValidationError, ValueError, TypeError) as exc:
            self._to_dlq(msg, "schema_validation", str(exc))
            m.EVENTS_CONSUMED.labels("invalid").inc()
            return None
        m.EVENTS_CONSUMED.labels("valid").inc()
        return event

    # ------------------------------------------------------------ handling
    def _with_retry(self, events: list[TransactionEvent]) -> bool:
        delay = 0.2
        for attempt in range(1, self.max_retries + 1):
            try:
                self.handler(events)
                return True
            except Exception as exc:  # noqa: BLE001 - any handler failure is retried
                m.PROCESSING_ERRORS.inc()
                log.warning("handler_failed", extra={"attempt": attempt, "error": str(exc)[:300], "batch": len(events)})
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
        return False

    def process_messages(self, msgs: list[Any]) -> None:
        pairs: list[tuple[Any, TransactionEvent]] = []
        for msg in msgs:
            if msg.error():
                log.warning("kafka_message_error", extra={"error": str(msg.error())})
                continue
            self.stats["consumed"] += 1
            ev = self._decode(msg)
            if ev is not None:
                pairs.append((msg, ev))
        if not pairs:
            return
        self.stats["valid"] += len(pairs)
        if not self._with_retry([ev for _, ev in pairs]):
            # isolate poison records
            for msg, ev in pairs:
                if not self._with_retry([ev]):
                    self._to_dlq(msg, "processing_failed", "handler failed after retries")

    def update_lag(self) -> int:
        total = 0
        try:
            assignment = self.consumer.assignment()
            if not assignment:
                return 0
            positions = self.consumer.position(assignment)
            for tp in positions:
                _, high = self.consumer.get_watermark_offsets(tp, timeout=2, cached=False)
                lag = max(0, high - tp.offset) if tp.offset >= 0 else high
                m.CONSUMER_LAG.labels(str(tp.partition)).set(lag)
                total += lag
            m.CONSUMER_LAG_TOTAL.set(total)
        except Exception as exc:  # noqa: BLE001 - metrics must not kill the consumer
            log.debug("lag_update_failed", extra={"error": str(exc)})
        return total

    def run(
        self,
        max_messages: int | None = None,
        idle_timeout_s: float | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, int]:
        self.consumer.subscribe([self.topic])
        log.info("consumer_started", extra={"topic": self.topic})
        last_msg = time.time()
        last_lag = 0.0
        try:
            while True:
                if should_stop and should_stop():
                    break
                msgs = self.consumer.consume(num_messages=self.batch_size, timeout=1.0)
                now = time.time()
                if now - last_lag > 5:
                    self.update_lag()
                    last_lag = now
                if not msgs:
                    if idle_timeout_s is not None and now - last_msg > idle_timeout_s:
                        break
                    continue
                last_msg = now
                t0 = time.perf_counter()
                self.process_messages(msgs)
                self.dlq.flush(10)
                self.consumer.commit(asynchronous=False)
                self.stats["batches"] += 1
                m.BATCH_SECONDS.observe(time.perf_counter() - t0)
                if max_messages is not None and self.stats["consumed"] >= max_messages:
                    break
        finally:
            self.update_lag()
            self.consumer.close()
            log.info("consumer_stopped", extra=self.stats)
        return self.stats

"""Streaming risk engine: Kafka ``fg.transactions`` -> score -> decisions/alerts.

    python -m services.risk_engine.worker

For each valid event (see :mod:`ingestion.kafka_consumer.consumer`):
1. skip if the transaction_id was already processed (duplicate),
2. compute features from the online store, score with all models, apply policy,
3. commit the event into online state (Redis),
4. persist the decision (ops.risk_decisions) and REVIEW/BLOCK alerts
   (ops.fraud_alerts + Kafka ``fg.alerts``).

Prometheus metrics are served on ``METRICS_PORT`` (default 9102).
"""

from __future__ import annotations

import os
import signal
import time
from typing import Any

from common.config import Settings, get_settings
from common.logging import get_logger
from features.store import RedisStateStore, StateStore, build_store
from ingestion.schemas import TransactionEvent
from monitoring import metrics as m
from services.risk_engine.dedup import InMemoryDeduplicator, RedisDeduplicator
from services.risk_engine.scorer import RiskScorer
from services.risk_engine.sink import DecisionSink, NullSink, PostgresSink

log = get_logger(__name__)


class RiskEngineWorker:
    def __init__(self, scorer: RiskScorer, sink: DecisionSink, dedup: Any, store_features: bool = True) -> None:
        self.scorer = scorer
        self.sink = sink
        self.dedup = dedup
        self.store_features = store_features
        self.processed = 0

    def handle(self, events: list[TransactionEvent]) -> None:
        """Batch handler used by the ValidatingConsumer."""
        decisions = []
        for ev in events:
            if not self.dedup.claim(ev.transaction_id):
                m.DUPLICATES.inc()
                continue
            t0 = time.perf_counter()
            res = self.scorer.score(ev.to_engine_dict(), commit=True, include_features=self.store_features)
            m.SCORING_LATENCY.labels("stream").observe(time.perf_counter() - t0)
            m.DECISIONS.labels("stream", res["decision"]).inc()
            m.RISK_SCORE.labels("stream").observe(res["risk_score"])
            m.FRAUD_PROBABILITY.labels("stream").observe(res["fraud_probability"])
            m.LAST_EVENT_TS.set(ev.event_time.timestamp())
            res.update(
                {
                    "event_time": ev.event_time.isoformat(),
                    "source": "stream",
                    "transaction_type": ev.transaction_type,
                    "amount": ev.amount,
                }
            )
            decisions.append(res)
        self.sink.write(decisions)
        self.processed += len(decisions)


def build_worker(
    settings: Settings | None = None, store: StateStore | None = None, sink: DecisionSink | None = None
) -> RiskEngineWorker:
    settings = settings or get_settings()
    store = store if store is not None else build_store(settings["redis"]["url"] or None)
    scorer = RiskScorer.from_settings(settings, store=store)
    if isinstance(store, RedisStateStore):
        dedup: Any = RedisDeduplicator(store.client)
    else:
        dedup = InMemoryDeduplicator()
    if sink is None:
        try:
            from confluent_kafka import Producer

            from common.db import get_engine

            engine = get_engine()
            with engine.connect() as conn:
                conn.exec_driver_sql("SELECT 1")
            alerts = Producer({"bootstrap.servers": settings["kafka"]["bootstrap_servers"], "acks": "all"})
            sink = PostgresSink(engine, alerts, settings["kafka"]["topics"]["alerts"])
        except Exception as exc:  # noqa: BLE001 - run without persistence rather than not at all
            log.warning("database_unavailable_using_null_sink", extra={"error": str(exc)[:200]})
            sink = NullSink()
    return RiskEngineWorker(scorer, sink, dedup)


def main() -> None:  # pragma: no cover - exercised by the docker integration test
    from prometheus_client import start_http_server

    from ingestion.kafka_admin import ensure_topics
    from ingestion.kafka_consumer.consumer import ValidatingConsumer

    settings = get_settings()
    start_http_server(int(os.environ.get("METRICS_PORT", "9102")))
    for attempt in range(30):
        try:
            ensure_topics(settings)
            break
        except Exception as exc:  # noqa: BLE001 - kafka still starting
            log.warning("kafka_not_ready", extra={"attempt": attempt, "error": str(exc)[:200]})
            time.sleep(5)
    worker = build_worker(settings)
    stopping = {"flag": False}

    def _stop(*_: Any) -> None:
        stopping["flag"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    consumer = ValidatingConsumer(worker.handle, settings)
    consumer.run(should_stop=lambda: stopping["flag"])


if __name__ == "__main__":  # pragma: no cover
    main()

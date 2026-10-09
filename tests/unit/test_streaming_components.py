"""Kafka producer/consumer, DLQ, retries, de-duplication and the risk-engine worker (no broker needed)."""

import json

import pandas as pd
import pytest

from ingestion.kafka_consumer.consumer import ValidatingConsumer
from ingestion.schemas import TransactionEvent, event_from_row
from services.risk_engine.dedup import InMemoryDeduplicator


class Msg:
    def __init__(self, value, key=b"k", offset=0):
        self._v, self._k, self._o = value, key, offset

    def value(self):
        return self._v

    def key(self):
        return self._k

    def error(self):
        return None

    def topic(self):
        return "fg.transactions"

    def partition(self):
        return 0

    def offset(self):
        return self._o


class FakeProducer:
    def __init__(self):
        self.sent = []

    def produce(self, topic, key=None, value=None, headers=None, on_delivery=None):
        self.sent.append({"topic": topic, "key": key, "value": value, "headers": dict(headers or [])})
        if on_delivery:
            on_delivery(None, type("M", (), {"key": lambda self: key})())

    def poll(self, t=0):
        return 0

    def flush(self, t=0):
        return 0


SETTINGS = {
    "kafka": {
        "topics": {"transactions": "fg.transactions", "dead_letter": "fg.dlq"},
        "bootstrap_servers": "x",
        "consumer_group": "g",
    }
}
GOOD = {
    "transaction_id": "TXN0000001",
    "event_time": "2026-05-01T10:00:00Z",
    "customer_id": "CUS_0000001",
    "receiver_id": "CUS_0000002",
    "amount": 100,
    "transaction_type": "P2P",
}


def test_invalid_messages_go_to_dlq_valid_to_handler():
    seen, dlq = [], FakeProducer()
    c = ValidatingConsumer(lambda evs: seen.extend(evs), SETTINGS, consumer=object(), dlq_producer=dlq)
    c.process_messages(
        [Msg(json.dumps(GOOD).encode()), Msg(b"{not json"), Msg(json.dumps({**GOOD, "amount": -3}).encode())]
    )
    assert [e.transaction_id for e in seen] == ["TXN0000001"]
    reasons = sorted(m["headers"]["error_reason"].decode() for m in dlq.sent)
    assert reasons == ["malformed_json", "schema_validation"]
    assert all(m["topic"] == "fg.dlq" for m in dlq.sent)


def test_handler_retries_then_isolates_poison_record(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(evs):
        calls["n"] += 1
        if any(e.transaction_id == "TXN_POISON" for e in evs):
            raise RuntimeError("boom")

    dlq = FakeProducer()
    c = ValidatingConsumer(handler, SETTINGS, consumer=object(), dlq_producer=dlq, max_retries=2)
    c.process_messages(
        [Msg(json.dumps(GOOD).encode()), Msg(json.dumps({**GOOD, "transaction_id": "TXN_POISON"}).encode())]
    )
    assert len(dlq.sent) == 1 and json.loads(dlq.sent[0]["value"])["transaction_id"] == "TXN_POISON"
    assert dlq.sent[0]["headers"]["error_reason"] == b"processing_failed"
    assert calls["n"] == 2 + 1 + 2  # batch retries, then good record once, poison record retried


def test_consumer_commits_after_processing():
    class FakeConsumer:
        def __init__(self):
            self.batches = [[Msg(json.dumps(GOOD).encode())]]
            self.commits = 0

        def subscribe(self, topics):
            pass

        def consume(self, num_messages, timeout):
            return self.batches.pop() if self.batches else []

        def commit(self, asynchronous=False):
            self.commits += 1

        def assignment(self):
            return []

        def close(self):
            pass

    fc = FakeConsumer()
    c = ValidatingConsumer(lambda evs: None, SETTINGS, consumer=fc, dlq_producer=FakeProducer())
    stats = c.run(max_messages=1)
    assert stats["consumed"] == 1 and fc.commits == 1


def test_dedup():
    d = InMemoryDeduplicator(capacity=2)
    assert d.claim("a") and not d.claim("a") and d.claim("b") and d.claim("c")
    assert d.claim("a")  # evicted after capacity


def test_producer_replay_with_chaos(small_data):
    from ingestion.kafka_producer.producer import TransactionProducer, replay

    settings, _ = small_data
    fake = FakeProducer()
    prod = TransactionProducer(settings, producer=fake)
    stats = replay(
        settings,
        from_split=None,
        limit=500,
        rate=0,
        duplicate_rate=0.05,
        invalid_rate=0.02,
        late_rate=0.02,
        producer=prod,
    )
    assert stats["sent"] + stats["late"] == 500
    assert len(fake.sent) == 500 + stats["duplicates"] + stats["invalid"]
    keys = {m["key"] for m in fake.sent}
    assert all(k.startswith(b"CUS_") for k in keys)  # partitioned by customer
    ok = 0
    for m in fake.sent:
        try:
            TransactionEvent(**json.loads(m["value"]))
            ok += 1
        except Exception:
            pass
    assert ok == 500 + stats["duplicates"]


@pytest.mark.ml
def test_worker_scores_dedups_and_persists(small_env):
    from features.store import InMemoryStateStore
    from services.risk_engine.scorer import RiskScorer
    from services.risk_engine.sink import NullSink
    from services.risk_engine.worker import RiskEngineWorker

    settings, ds, _ = small_env
    sink = NullSink()
    w = RiskEngineWorker(RiskScorer.from_settings(settings, store=InMemoryStateStore()), sink, InMemoryDeduplicator())
    rows = ds.transactions.head(50).to_dict("records")
    events = [TransactionEvent(**event_from_row(r)) for r in rows]
    w.handle(events)
    w.handle(events[:10])  # redelivery
    assert w.processed == 50 and len(sink.decisions) == 50
    assert {d["source"] for d in sink.decisions} == {"stream"}
    assert pd.Series([d["transaction_id"] for d in sink.decisions]).is_unique

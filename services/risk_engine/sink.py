"""Persistence of risk decisions and alerts (PostgreSQL), plus the alert topic."""

from __future__ import annotations

import json
from typing import Any, Protocol

from common.logging import get_logger

log = get_logger(__name__)

DECISION_COLUMNS = [
    "transaction_id",
    "customer_id",
    "event_time",
    "scored_at",
    "source",
    "transaction_type",
    "amount",
    "fraud_probability",
    "account_takeover_probability",
    "anomaly_percentile",
    "behavioral_anomaly_percentile",
    "risk_score",
    "risk_level",
    "decision",
    "risk_reasons",
    "model_version",
    "model_versions",
    "policy_version",
    "latency_ms",
    "features",
]
ALERT_COLUMNS = [
    "transaction_id",
    "customer_id",
    "event_time",
    "risk_score",
    "risk_level",
    "decision",
    "reasons",
    "model_version",
]
JSON_FIELDS = {"risk_reasons", "model_versions", "features", "reasons"}


class DecisionSink(Protocol):
    def write(self, decisions: list[dict[str, Any]]) -> None: ...


class NullSink:
    """Keeps the last N decisions in memory (tests / no-database demo)."""

    def __init__(self, keep: int = 5000) -> None:
        self.keep = keep
        self.decisions: list[dict[str, Any]] = []

    def write(self, decisions: list[dict[str, Any]]) -> None:
        self.decisions.extend(decisions)
        self.decisions = self.decisions[-self.keep :]


def _row(d: dict[str, Any], cols: list[str]) -> tuple:
    return tuple(json.dumps(d.get(c)) if c in JSON_FIELDS and d.get(c) is not None else d.get(c) for c in cols)


class PostgresSink:
    """Idempotent bulk writes (``ON CONFLICT DO NOTHING``) to ops.risk_decisions / ops.fraud_alerts."""

    def __init__(self, engine: Any, alert_producer: Any | None = None, alert_topic: str | None = None) -> None:
        self.engine = engine
        self.alert_producer = alert_producer
        self.alert_topic = alert_topic

    def write(self, decisions: list[dict[str, Any]]) -> None:
        if not decisions:
            return
        from psycopg2.extras import execute_values

        alerts = [dict(d, reasons=d.get("risk_reasons")) for d in decisions if d["decision"] != "APPROVE"]
        raw = self.engine.raw_connection()
        try:
            cur = raw.cursor()
            execute_values(
                cur,
                f"INSERT INTO ops.risk_decisions ({', '.join(DECISION_COLUMNS)}) VALUES %s "
                "ON CONFLICT (transaction_id) DO NOTHING",
                [_row(d, DECISION_COLUMNS) for d in decisions],
                page_size=500,
            )
            if alerts:
                execute_values(
                    cur,
                    f"INSERT INTO ops.fraud_alerts ({', '.join(ALERT_COLUMNS)}) VALUES %s "
                    "ON CONFLICT (transaction_id) DO NOTHING",
                    [_row(a, ALERT_COLUMNS) for a in alerts],
                )
            raw.commit()
        except Exception:
            raw.rollback()
            raise
        finally:
            raw.close()
        if self.alert_producer is not None and alerts:
            for a in alerts:
                payload = {k: a.get(k) for k in ALERT_COLUMNS}
                self.alert_producer.produce(
                    self.alert_topic, key=a["customer_id"].encode(), value=json.dumps(payload, default=str).encode()
                )
            self.alert_producer.poll(0)

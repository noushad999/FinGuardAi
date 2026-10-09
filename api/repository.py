"""Read models for the API.

``SqlRepository`` reads the warehouse/operational tables (production mode).
``FileRepository`` reads the local Parquet outputs so the API is fully usable
without PostgreSQL (development, tests, quick demos).
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sqlalchemy import text


def _clean(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and np.isnan(v):
            v = None
        elif isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, np.bool_):
            v = bool(v)
        elif isinstance(v, pd.Timestamp):
            v = v.isoformat()
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        elif v is pd.NaT:
            v = None
        out[k] = v
    return out


class Repository(Protocol):
    name: str

    def ok(self) -> bool: ...
    def customer_profile(self, customer_id: str) -> dict[str, Any] | None: ...
    def customer_exists(self, customer_id: str) -> bool: ...
    def transaction(self, transaction_id: str) -> dict[str, Any] | None: ...
    def predictions(self, transaction_id: str) -> list[dict[str, Any]]: ...
    def recent_decisions(self, customer_id: str, limit: int = 10) -> list[dict[str, Any]]: ...
    def alerts(self, limit: int, decision: str | None, customer_id: str | None) -> list[dict[str, Any]]: ...
    def save_decision(self, decision: dict[str, Any]) -> None: ...


class SqlRepository:
    name = "postgres"

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def _rows(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        with self.engine.connect() as conn:
            return [_clean(dict(r._mapping)) for r in conn.execute(text(sql), params)]

    def ok(self) -> bool:
        try:
            self._rows("SELECT 1 AS x")
            return True
        except Exception:  # noqa: BLE001
            return False

    def customer_profile(self, customer_id: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM ml.customer_profiles WHERE customer_id = :c", c=customer_id)
        return rows[0] if rows else None

    def customer_exists(self, customer_id: str) -> bool:
        return bool(self._rows("SELECT 1 FROM raw.customers WHERE customer_id = :c", c=customer_id))

    def transaction(self, transaction_id: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM raw.transactions WHERE transaction_id = :t", t=transaction_id)
        return rows[0] if rows else None

    def predictions(self, transaction_id: str) -> list[dict[str, Any]]:
        rt = self._rows(
            "SELECT source, model_version, fraud_probability, account_takeover_probability, risk_score, risk_level, "
            "decision, risk_reasons, scored_at FROM ops.risk_decisions WHERE transaction_id = :t",
            t=transaction_id,
        )
        batch = self._rows(
            "SELECT 'batch' AS source, fraud_model_version AS model_version, fraud_probability, "
            "ato_probability AS account_takeover_probability, split AS evaluation_split "
            "FROM ml.transaction_scores WHERE transaction_id = :t",
            t=transaction_id,
        )
        return rt + batch

    def recent_decisions(self, customer_id: str, limit: int = 10) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT transaction_id, event_time, transaction_type, amount, risk_score, decision, risk_reasons "
            "FROM ops.risk_decisions WHERE customer_id = :c ORDER BY event_time DESC NULLS LAST LIMIT :n",
            c=customer_id,
            n=limit,
        )

    def alerts(self, limit: int, decision: str | None, customer_id: str | None) -> list[dict[str, Any]]:
        where, params = ["1=1"], {"n": limit}
        if decision:
            where.append("decision = :d")
            params["d"] = decision
        if customer_id:
            where.append("customer_id = :c")
            params["c"] = customer_id
        return self._rows(
            f"SELECT alert_id, transaction_id, customer_id, created_at, event_time, risk_score, risk_level, decision, "
            f"reasons, model_version, status FROM ops.fraud_alerts WHERE {' AND '.join(where)} "
            f"ORDER BY created_at DESC, alert_id DESC LIMIT :n",
            **params,
        )

    def save_decision(self, decision: dict[str, Any]) -> None:
        from services.risk_engine.sink import PostgresSink

        PostgresSink(self.engine).write([decision])


class FileRepository:
    """Parquet-backed read model (no database needed)."""

    name = "files"

    def __init__(self, synthetic_dir: Path, processed_dir: Path, keep_decisions: int = 5000) -> None:
        self.synthetic_dir = Path(synthetic_dir)
        self.processed_dir = Path(processed_dir)
        self._tx: pd.DataFrame | None = None
        self._scores: pd.DataFrame | None = None
        self._profiles: pd.DataFrame | None = None
        self._customers: set[str] | None = None
        self.decisions: deque[dict[str, Any]] = deque(maxlen=keep_decisions)

    def _load(self) -> None:
        if self._tx is not None:
            return
        tx_path = self.synthetic_dir / "transactions.parquet"
        self._tx = pd.read_parquet(tx_path).set_index("transaction_id") if tx_path.exists() else pd.DataFrame()
        sp = self.processed_dir / "fraud_scores.parquet"
        self._scores = pd.read_parquet(sp).set_index("transaction_id") if sp.exists() else pd.DataFrame()
        pp = self.processed_dir / "customer_profiles.parquet"
        self._profiles = pd.read_parquet(pp).set_index("customer_id") if pp.exists() else pd.DataFrame()
        cp = self.synthetic_dir / "customers.parquet"
        self._customers = set(pd.read_parquet(cp, columns=["customer_id"])["customer_id"]) if cp.exists() else set()

    def ok(self) -> bool:
        return (self.synthetic_dir / "transactions.parquet").exists()

    def customer_profile(self, customer_id: str) -> dict[str, Any] | None:
        self._load()
        assert self._profiles is not None
        if customer_id not in self._profiles.index:
            return None
        return _clean({"customer_id": customer_id, **self._profiles.loc[customer_id].to_dict()})

    def customer_exists(self, customer_id: str) -> bool:
        self._load()
        return customer_id in (self._customers or set())

    def transaction(self, transaction_id: str) -> dict[str, Any] | None:
        self._load()
        assert self._tx is not None
        if transaction_id not in self._tx.index:
            return None
        d = _clean({"transaction_id": transaction_id, **self._tx.loc[transaction_id].to_dict()})
        if isinstance(d.get("metadata"), str):
            d["metadata"] = json.loads(d["metadata"])
        return d

    def predictions(self, transaction_id: str) -> list[dict[str, Any]]:
        self._load()
        out = [d for d in self.decisions if d["transaction_id"] == transaction_id]
        assert self._scores is not None
        if transaction_id in self._scores.index:
            s = self._scores.loc[transaction_id]
            out.append(
                _clean(
                    {
                        "source": "batch",
                        "model_version": s.get("fraud_model_version"),
                        "fraud_probability": s.get("fraud_probability"),
                        "account_takeover_probability": s.get("ato_probability"),
                        "evaluation_split": s.get("split"),
                    }
                )
            )
        return out

    def recent_decisions(self, customer_id: str, limit: int = 10) -> list[dict[str, Any]]:
        rows = [d for d in reversed(self.decisions) if d["customer_id"] == customer_id]
        return rows[:limit]

    def alerts(self, limit: int, decision: str | None, customer_id: str | None) -> list[dict[str, Any]]:
        rows = [
            d
            for d in reversed(self.decisions)
            if d["decision"] != "APPROVE"
            and (decision is None or d["decision"] == decision)
            and (customer_id is None or d["customer_id"] == customer_id)
        ]
        return [
            {
                k: d.get(k)
                for k in (
                    "transaction_id",
                    "customer_id",
                    "scored_at",
                    "risk_score",
                    "risk_level",
                    "decision",
                    "risk_reasons",
                    "model_version",
                )
            }
            for d in rows[:limit]
        ]

    def save_decision(self, decision: dict[str, Any]) -> None:
        keep = (
            "transaction_id",
            "customer_id",
            "event_time",
            "scored_at",
            "transaction_type",
            "amount",
            "fraud_probability",
            "account_takeover_probability",
            "risk_score",
            "risk_level",
            "decision",
            "risk_reasons",
            "model_version",
            "source",
        )
        self.decisions.append({k: decision.get(k) for k in keep})

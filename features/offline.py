"""Offline (batch) feature generation by replaying history through the online engine.

``build_transaction_features`` walks the transaction log in event-time order and
calls :meth:`FeatureEngine.process` for each event - the same function the
real-time risk engine calls. The resulting table is point-in-time correct:
row *i* only uses information from events strictly before it.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from common.logging import get_logger
from common.timeutils import epoch_seconds
from features.definitions import FEATURE_NAMES
from features.engine import EngineConfig, FeatureEngine
from features.reference import ReferenceData
from features.store import InMemoryStateStore, StateStore

log = get_logger(__name__)

ENGINE_INPUT_COLUMNS = [
    "transaction_id",
    "event_time",
    "sender_id",
    "receiver_id",
    "customer_id",
    "merchant_id",
    "amount",
    "transaction_type",
    "channel",
    "device_id",
    "district",
    "latitude",
    "longitude",
    "status",
    "failure_reason",
    "balance_before",
    "metadata",
]


def _records(df: pd.DataFrame) -> list[dict]:
    recs = df[ENGINE_INPUT_COLUMNS].astype(object).where(df[ENGINE_INPUT_COLUMNS].notna(), None)
    recs["event_time"] = epoch_seconds(df["event_time"])
    return recs.to_dict("records")


def replay(
    transactions: pd.DataFrame, engine: FeatureEngine, collect: bool = True, chunk_size: int = 200_000
) -> pd.DataFrame | None:
    """Replay transactions (any order; sorted here) through ``engine``."""
    tx = transactions.sort_values(["event_time", "transaction_id"], kind="stable")
    rows: list[list[float]] = []
    t0 = time.time()
    for start in range(0, len(tx), chunk_size):
        chunk = tx.iloc[start : start + chunk_size]
        for rec in _records(chunk):
            feats = engine.process(rec, commit=True)
            if collect:
                rows.append([feats[n] for n in FEATURE_NAMES])
        log.info(
            "replay_progress",
            extra={
                "processed": min(start + chunk_size, len(tx)),
                "total": len(tx),
                "seconds": round(time.time() - t0, 1),
            },
        )
    if not collect:
        return None
    out = pd.DataFrame(np.asarray(rows, dtype=np.float32), columns=FEATURE_NAMES)
    out.insert(0, "transaction_id", tx["transaction_id"].to_numpy())
    out.insert(1, "event_time", tx["event_time"].to_numpy())
    out.insert(2, "customer_id", tx["customer_id"].to_numpy())
    return out


def build_transaction_features(
    transactions: pd.DataFrame,
    reference: ReferenceData,
    config: EngineConfig | None = None,
    store: StateStore | None = None,
) -> pd.DataFrame:
    engine = FeatureEngine(store or InMemoryStateStore(), reference, config)
    out = replay(transactions, engine, collect=True)
    assert out is not None
    return out


def build_from_dir(synthetic_dir: str | Path, out_path: str | Path, config: EngineConfig) -> Path:
    synthetic_dir = Path(synthetic_dir)
    tx = pd.read_parquet(synthetic_dir / "transactions.parquet")
    reference = ReferenceData.from_parquet(synthetic_dir)
    feats = build_transaction_features(tx, reference, config)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path, index=False)
    log.info("features_written", extra={"path": str(out_path), "rows": len(feats), "n_features": len(FEATURE_NAMES)})
    return out_path


def hydrate_store(
    transactions: pd.DataFrame,
    reference: ReferenceData,
    target: StateStore,
    until: pd.Timestamp | None,
    config: EngineConfig | None = None,
    batch_size: int = 5_000,
) -> int:
    """Warm an online store with state from history strictly before ``until``.

    This is how a new online store (e.g. empty Redis) is bootstrapped so that
    real-time features match what the model saw during training.
    """
    tx = transactions
    if until is not None:
        tx = tx[pd.to_datetime(tx["event_time"], utc=True) < until]
    mem = InMemoryStateStore()
    replay(tx, FeatureEngine(mem, reference, config), collect=False)
    items = list(mem.items())
    for i in range(0, len(items), batch_size):
        target.put_many(dict(items[i : i + batch_size]))
    log.info(
        "store_hydrated",
        extra={"entities": len(items), "events": len(tx), "until": str(until) if until is not None else None},
    )
    return len(items)

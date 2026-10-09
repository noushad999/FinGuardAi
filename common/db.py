"""PostgreSQL helpers: engine creation, DDL bootstrap and fast bulk loading (COPY)."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from common.config import PROJECT_ROOT, get_settings
from common.logging import get_logger

log = get_logger(__name__)
DDL_DIR = PROJECT_ROOT / "warehouse" / "schema"


def get_engine(url: str | None = None) -> Engine:
    return create_engine(url or get_settings()["database"]["url"], pool_pre_ping=True, future=True)


def init_schema(engine: Engine, ddl_dir: Path = DDL_DIR) -> list[str]:
    applied = []
    with engine.begin() as conn:
        for f in sorted(ddl_dir.glob("*.sql")):
            conn.exec_driver_sql(f.read_text(encoding="utf-8"))
            applied.append(f.name)
    log.info("schema_initialised", extra={"files": applied})
    return applied


def _prepare(df: pd.DataFrame, json_cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in json_cols:
        if c in out:
            out[c] = out[c].map(
                lambda v: (
                    None
                    if v is None or (isinstance(v, float) and np.isnan(v))
                    else (v if isinstance(v, str) else json.dumps(_plain(v)))
                )
            )
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].map(lambda t: None if pd.isna(t) else t.isoformat())
        elif pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].map(lambda b: None if pd.isna(b) else ("t" if b else "f"))
    return out


def _plain(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, np.generic):
        return v.item()
    return v


def copy_dataframe(
    engine: Engine,
    df: pd.DataFrame,
    table: str,
    *,
    truncate: bool = True,
    json_cols: Iterable[str] = (),
    chunk_rows: int = 250_000,
) -> int:
    """Bulk-load ``df`` into ``table`` with COPY. ``truncate`` makes the load idempotent."""
    df = _prepare(df, json_cols)
    cols = ",".join(f'"{c}"' for c in df.columns)
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        if truncate:
            cur.execute(f"TRUNCATE TABLE {table}")
        for start in range(0, len(df), chunk_rows):
            buf = io.StringIO()
            df.iloc[start : start + chunk_rows].to_csv(
                buf, index=False, header=False, na_rep="\\N", quoting=csv.QUOTE_MINIMAL
            )
            buf.seek(0)
            cur.copy_expert(f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv, NULL '\\N')", buf)
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()
    return len(df)


def record_run(
    engine: Engine, pipeline: str, started: datetime, status: str, rows: int, details: dict[str, Any]
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO ops.pipeline_runs (pipeline, started_at, finished_at, status, rows_written, details) "
                "VALUES (:p, :s, :f, :st, :r, CAST(:d AS JSONB))"
            ),
            {
                "p": pipeline,
                "s": started,
                "f": datetime.now(UTC),
                "st": status,
                "r": rows,
                "d": json.dumps(details, default=str),
            },
        )

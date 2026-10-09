"""Data access for the dashboard: warehouse marts (PostgreSQL), the API and Prometheus."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg2://finguard:finguard@localhost:5432/finguard")
API_URL = os.environ.get("API_URL", "http://localhost:8000")
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")

_engine = None


def engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _engine


def query(sql: str, **params: Any) -> pd.DataFrame:
    with engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def table_exists(qualified: str) -> bool:
    schema, name = qualified.split(".")
    df = query("SELECT 1 FROM information_schema.tables WHERE table_schema = :s AND table_name = :n", s=schema, n=name)
    return not df.empty


def api_get(path: str) -> dict[str, Any] | None:
    try:
        r = httpx.get(f"{API_URL}{path}", timeout=5)
        return r.json() if r.status_code == 200 else {"_status": r.status_code, **_safe_json(r)}
    except httpx.HTTPError:
        return None


def _safe_json(r: httpx.Response) -> dict[str, Any]:
    try:
        return r.json()
    except json.JSONDecodeError:
        return {}


def prom(expr: str) -> float | None:
    """Instant Prometheus query returning a single scalar (None if unavailable)."""
    try:
        r = httpx.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": expr}, timeout=4).json()
        res = r.get("data", {}).get("result", [])
        return float(res[0]["value"][1]) if res else None
    except (httpx.HTTPError, ValueError, KeyError, IndexError):
        return None


def prom_range(expr: str, minutes: int = 30, step: str = "30s") -> pd.DataFrame:
    import time

    end = time.time()
    try:
        r = httpx.get(
            f"{PROMETHEUS_URL}/api/v1/query_range",
            params={"query": expr, "start": end - minutes * 60, "end": end, "step": step},
            timeout=5,
        ).json()
        rows = []
        for series in r.get("data", {}).get("result", []):
            label = ",".join(f"{k}={v}" for k, v in series["metric"].items() if k != "__name__") or "value"
            rows += [
                {"time": pd.to_datetime(float(t), unit="s"), "series": label, "value": float(v)}
                for t, v in series["values"]
            ]
        return pd.DataFrame(rows)
    except (httpx.HTTPError, ValueError, KeyError):
        return pd.DataFrame()

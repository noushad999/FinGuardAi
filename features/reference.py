"""Static reference data (slowly changing dimensions) needed at scoring time."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from common.timeutils import epoch_seconds

RISK_LEVEL = {"low": 0, "medium": 1, "high": 2}


def _to_epoch(series: pd.Series) -> pd.Series:
    return pd.Series(epoch_seconds(series), index=series.index)


@dataclass
class ReferenceData:
    """Customer registration dates and merchant attributes, keyed by id."""

    customer_reg_ts: dict[str, float] = field(default_factory=dict)
    merchant_reg_ts: dict[str, float] = field(default_factory=dict)
    merchant_risk: dict[str, int] = field(default_factory=dict)
    merchant_category: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_frames(cls, customers: pd.DataFrame, merchants: pd.DataFrame) -> ReferenceData:
        c_ts = _to_epoch(customers["registration_date"])
        m_ts = _to_epoch(merchants["registration_date"])
        return cls(
            customer_reg_ts=dict(zip(customers["customer_id"], c_ts.astype(float))),
            merchant_reg_ts=dict(zip(merchants["merchant_id"], m_ts.astype(float))),
            merchant_risk={
                m: RISK_LEVEL.get(r, 1) for m, r in zip(merchants["merchant_id"], merchants["risk_category"])
            },
            merchant_category=dict(zip(merchants["merchant_id"], merchants["merchant_category"])),
        )

    @classmethod
    def from_parquet(cls, directory: str | Path) -> ReferenceData:
        d = Path(directory)
        return cls.from_frames(
            pd.read_parquet(d / "customers.parquet", columns=["customer_id", "registration_date"]),
            pd.read_parquet(
                d / "merchants.parquet",
                columns=["merchant_id", "registration_date", "risk_category", "merchant_category"],
            ),
        )

    @classmethod
    def from_database(cls, engine) -> ReferenceData:  # pragma: no cover - exercised in integration tests
        customers = pd.read_sql("SELECT customer_id, registration_date FROM raw.customers", engine)
        merchants = pd.read_sql(
            "SELECT merchant_id, registration_date, risk_category, merchant_category FROM raw.merchants", engine
        )
        return cls.from_frames(customers, merchants)

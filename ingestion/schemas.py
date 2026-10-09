"""Transaction event contract (JSON on Kafka ``fg.transactions``).

JSON (not Avro/Protobuf) is used deliberately: one producer, a handful of
consumers, human-debuggable messages, and the contract is enforced in code by
this Pydantic model on both the producing and consuming side. A schema registry
with Avro becomes worthwhile once many independent teams publish/consume the
topic (listed as future work). ``schema_version`` travels in every message so
consumers can reject or route unknown versions (schema drift).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from simulator.constants import CHANNELS, TRANSACTION_TYPES

SCHEMA_VERSION = "1.0"
TransactionType = Literal[
    "P2P", "MERCHANT_PAYMENT", "CASH_IN", "CASH_OUT", "BILL_PAYMENT", "MOBILE_RECHARGE", "BANK_TRANSFER"
]
Channel = Literal["APP", "USSD", "AGENT"]
assert set(TransactionType.__args__) == set(TRANSACTION_TYPES)  # type: ignore[attr-defined]
assert set(Channel.__args__) == set(CHANNELS)  # type: ignore[attr-defined]


class TransactionEvent(BaseModel):
    """A single MFS transaction as published to Kafka / accepted by the API."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    schema_version: str = SCHEMA_VERSION
    transaction_id: str = Field(min_length=6, max_length=64)
    event_time: datetime
    sender_id: str | None = None
    receiver_id: str | None = None
    customer_id: str = Field(pattern=r"^CUS_\d+$")
    merchant_id: str | None = None
    amount: float = Field(gt=0, le=10_000_000)
    fee: float = Field(default=0.0, ge=0)
    currency: Literal["BDT"] = "BDT"
    transaction_type: TransactionType
    channel: Channel = "APP"
    device_id: str | None = None
    district: str | None = None
    latitude: float | None = Field(default=None, ge=20.0, le=27.5)
    longitude: float | None = Field(default=None, ge=87.5, le=93.5)
    status: Literal["SUCCESS", "FAILED"] | None = None
    failure_reason: str | None = None
    balance_before: float | None = Field(default=None, ge=0)
    balance_after: float | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_time")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            v = v.replace(tzinfo=UTC)
        if v.year < 2000 or v > datetime.now(UTC).replace(year=datetime.now(UTC).year + 5):
            raise ValueError("event_time out of plausible range")
        return v

    @field_validator("metadata", mode="before")
    @classmethod
    def _parse_metadata(cls, v: Any) -> Any:
        if v is None:
            return {}
        if isinstance(v, str):
            return json.loads(v) if v.strip() else {}
        return v

    @model_validator(mode="after")
    def _consistency(self) -> TransactionEvent:
        if self.schema_version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
            raise ValueError(f"unsupported schema_version {self.schema_version}")
        if self.transaction_type == "MERCHANT_PAYMENT" and not self.merchant_id:
            raise ValueError("merchant payments require merchant_id")
        if self.transaction_type == "P2P" and not self.receiver_id:
            raise ValueError("P2P transfers require receiver_id")
        if self.status == "FAILED" and not self.failure_reason:
            raise ValueError("failed transactions require failure_reason")
        return self

    def to_engine_dict(self) -> dict[str, Any]:
        d = self.model_dump()
        d["event_time"] = self.event_time.timestamp()
        return d

    def to_json(self) -> str:
        return self.model_dump_json()


def event_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a transactions.parquet row into a JSON-ready event dict."""
    out = {k: (None if _is_nan(v) else v) for k, v in row.items()}
    et = out["event_time"]
    out["event_time"] = et.isoformat() if hasattr(et, "isoformat") else et
    out["schema_version"] = SCHEMA_VERSION
    return out


def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and v != v

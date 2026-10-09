"""Pydantic request/response models for the FinGuard API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ingestion.schemas import TransactionEvent


class TransactionScoreRequest(TransactionEvent):
    """A transaction to score. ``transaction_id`` may be omitted (one is generated)."""

    transaction_id: str = Field(default="", max_length=64)

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "transaction_id": "TXN_DEMO_0001",
                "event_time": "2026-06-20T20:15:00Z",
                "customer_id": "CUS_0000042",
                "sender_id": "CUS_0000042",
                "receiver_id": "CUS_0001999",
                "amount": 18500,
                "transaction_type": "P2P",
                "channel": "APP",
                "device_id": "DEV_NEW_000001",
                "district": "Sylhet",
                "latitude": 24.89,
                "longitude": 91.87,
                "balance_before": 21000,
                "metadata": {"pin_attempts": 3},
            }
        },
    )


class ReasonDetail(BaseModel):
    feature: str
    value: float
    contribution: float
    reason: str


class RiskScoreResponse(BaseModel):
    transaction_id: str
    customer_id: str
    fraud_probability: float = Field(ge=0, le=1)
    account_takeover_probability: float = Field(ge=0, le=1)
    account_takeover_risk_score: int = Field(ge=0, le=100)
    account_takeover_reasons: list[str]
    anomaly_percentile: float | None
    behavioral_anomaly_percentile: float | None
    risk_score: int = Field(ge=0, le=100)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    decision: Literal["APPROVE", "REVIEW", "BLOCK"]
    risk_reasons: list[str]
    reason_details: list[ReasonDetail]
    score_driver: str
    explanation_note: str
    model_version: str
    model_versions: dict[str, str | None]
    policy_version: str
    scored_at: datetime
    latency_ms: float
    state_committed: bool = False


class LiveState(BaseModel):
    prior_transactions: int
    last_transaction_at: datetime | None
    known_devices: int
    known_counterparties: int
    known_districts: int
    transactions_last_24h: int


class CustomerProfileResponse(BaseModel):
    customer_id: str
    risk_level: str | None = None
    risk_score: int | None = None
    fraud_probability: float | None = Field(None, description="Max fraud probability over the last 30 days")
    account_takeover_probability: float | None = None
    churn_probability: float | None = None
    customer_segment: str | None = None
    behavior: str | None = None
    recent_activity: str | None = None
    mule_score_percentile: float | None = None
    in_suspicious_cluster: bool | None = None
    flagged_transactions_30d: int | None = None
    risk_reasons: list[str] = []
    district: str | None = None
    product_tier: str | None = None
    profile_as_of: datetime | None = None
    live_state: LiveState | None = None
    recent_decisions: list[dict[str, Any]] = []


class TransactionResponse(BaseModel):
    transaction: dict[str, Any]
    predictions: list[dict[str, Any]]


class AlertsResponse(BaseModel):
    count: int
    alerts: list[dict[str, Any]]


class ModelInfoResponse(BaseModel):
    models: dict[str, dict[str, Any]]
    policy: dict[str, Any]
    feature_count: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    models_loaded: bool
    model_versions: dict[str, str | None]
    state_store: str
    state_store_ok: bool
    repository: str
    repository_ok: bool

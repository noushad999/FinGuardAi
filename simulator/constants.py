"""Enumerations and schema constants shared by the simulator and downstream code."""

from __future__ import annotations

TRANSACTION_TYPES: tuple[str, ...] = (
    "P2P",
    "MERCHANT_PAYMENT",
    "CASH_IN",
    "CASH_OUT",
    "BILL_PAYMENT",
    "MOBILE_RECHARGE",
    "BANK_TRANSFER",
)
CHANNELS: tuple[str, ...] = ("APP", "USSD", "AGENT")
STATUSES: tuple[str, ...] = ("SUCCESS", "FAILED")
FAILURE_REASONS: tuple[str, ...] = (
    "INSUFFICIENT_BALANCE",
    "WRONG_PIN",
    "NETWORK_TIMEOUT",
    "LIMIT_EXCEEDED",
)

# Business counterparty types; the prefix of the id encodes the type.
MERCHANT_TYPES: dict[str, str] = {
    "MER": "merchant",
    "AGT": "cash_agent",
    "BIL": "biller",
    "TEL": "telecom_operator",
    "BNK": "bank",
}

# Retail merchant categories with (relative share, base risk category).
MERCHANT_CATEGORIES: dict[str, tuple[float, str]] = {
    "grocery": (0.24, "low"),
    "restaurant": (0.14, "low"),
    "pharmacy": (0.08, "low"),
    "fashion": (0.10, "low"),
    "electronics": (0.08, "medium"),
    "e_commerce": (0.14, "medium"),
    "transport": (0.07, "low"),
    "education": (0.04, "low"),
    "online_gaming": (0.05, "high"),
    "forex_crypto_services": (0.02, "high"),
    "travel_agency": (0.04, "medium"),
}

# Per-transaction limits (BDT); exceeding them fails with LIMIT_EXCEEDED.
TXN_LIMITS: dict[str, float] = {
    "P2P": 50_000,
    "MERCHANT_PAYMENT": 100_000,
    "CASH_IN": 50_000,
    "CASH_OUT": 50_000,
    "BILL_PAYMENT": 100_000,
    "MOBILE_RECHARGE": 2_000,
    "BANK_TRANSFER": 100_000,
}

CASH_OUT_FEE_RATE = 0.0185  # typical MFS cash-out charge

OCCUPATIONS: tuple[str, ...] = (
    "student",
    "salaried_private",
    "government_service",
    "business_owner",
    "farmer",
    "day_laborer",
    "garment_worker",
    "homemaker",
    "freelancer",
    "transport_worker",
    "retired",
)

FRAUD_SCENARIOS: tuple[str, ...] = (
    "velocity_attack",
    "sudden_high_value",
    "new_device_high_value",
    "impossible_travel",
    "device_sharing_ring",
    "suspicious_merchant",
    "rapid_fund_movement",
    "structuring",
    "mule_collection",
    "failed_auth_then_success",
    "unusual_recipient_scam",
    "abnormal_nighttime",
)

# Scenarios in which someone other than the legitimate owner controls the account.
ATO_SCENARIOS: frozenset[str] = frozenset(
    {
        "velocity_attack",
        "new_device_high_value",
        "impossible_travel",
        "failed_auth_then_success",
        "abnormal_nighttime",
    }
)

# Public event schema (what goes to Kafka / the lake). Labels are NOT part of it.
TRANSACTION_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "event_time",
    "sender_id",
    "receiver_id",
    "customer_id",
    "merchant_id",
    "amount",
    "fee",
    "currency",
    "transaction_type",
    "channel",
    "device_id",
    "district",
    "latitude",
    "longitude",
    "status",
    "failure_reason",
    "balance_before",
    "balance_after",
    "metadata",
)

LABEL_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "is_fraud",
    "is_ato",
    "fraud_scenario",
    "fraud_case_id",
    "fraud_reported_at",
)

"""Feature catalogue: the single source of truth for feature names and meaning.

The model bundle stores this list; loading a model whose schema differs from
the code's feature list fails fast (see ``ml.common.bundle``).
"""

from __future__ import annotations

from simulator.constants import TRANSACTION_TYPES

FEATURE_GROUPS: dict[str, dict[str, str]] = {
    "transaction": {
        "amount": "Transaction amount (BDT)",
        "log_amount": "log1p(amount)",
        **{f"type_{t.lower()}": f"One-hot: transaction_type == {t}" for t in TRANSACTION_TYPES},
        "channel_app": "One-hot: channel == APP",
        "channel_ussd": "One-hot: channel == USSD",
        "channel_agent": "One-hot: channel == AGENT",
        "merchant_risk_level": "Merchant risk category (0 low, 1 medium, 2 high, -1 no merchant)",
        "merchant_age_days": "Days since merchant registration (-1 no merchant)",
        "merchant_txn_count": "Prior transactions seen at this merchant",
        "merchant_unique_customers": "Distinct customers previously seen at this merchant (capped)",
        "is_near_threshold": "Amount within 15% below the structuring threshold",
        "amount_to_balance_ratio": "Debit amount / wallet balance before the transaction",
        "pin_attempts": "PIN attempts in the authentication session",
    },
    "behavioral": {
        "n_prior_txns": "Customer's prior transaction attempts",
        "account_age_days": "Days since customer registration",
        "txn_count_1m": "Customer attempts in the previous 60 s",
        "txn_count_5m": "Customer attempts in the previous 5 min",
        "txn_count_1h": "Customer attempts in the previous hour",
        "txn_count_24h": "Customer attempts in the previous 24 h",
        "amount_sum_1h": "Sum of customer amounts in the previous hour",
        "amount_sum_24h": "Sum of customer amounts in the previous 24 h",
        "avg_amount_hist": "Geometric mean of the customer's historical debit amounts",
        "amount_std_hist": "Std of log debit amounts (customer history)",
        "amount_zscore": "z-score of log(amount) vs customer history",
        "amount_to_avg_ratio": "amount / historical geometric mean",
        "unique_recipients_24h": "Distinct counterparties in the previous 24 h",
        "unique_devices_24h": "Distinct devices used in the previous 24 h",
        "failed_ratio_hist": "Share of failed historical attempts",
        "failed_count_1h": "Failed attempts in the previous hour",
        "wrong_pin_count_1h": "WRONG_PIN failures in the previous hour",
        "velocity_ratio_24h": "24 h count / customer's average daily count",
    },
    "temporal": {
        "hour_local": "Local hour of day (Asia/Dhaka)",
        "day_of_week": "0=Monday .. 6=Sunday",
        "is_weekend": "Friday or Saturday (Bangladesh weekend)",
        "is_night": "Local time between 00:00 and 05:00",
        "log_seconds_since_prev": "log1p(seconds since the customer's previous attempt)",
        "hour_frequency": "Smoothed share of the customer's history within +-1h of this hour",
    },
    "device": {
        "device_missing": "No customer device (agent-initiated)",
        "is_new_device": "Device never seen for this customer",
        "device_txn_count": "Prior transactions on this device (all accounts)",
        "accounts_per_device": "Distinct accounts seen on this device (incl. current)",
        "customer_device_count": "Distinct devices the customer used before",
    },
    "geographic": {
        "is_new_district": "District never seen for this customer",
        "district_frequency": "Share of the customer's history in this district",
        "distance_from_prev_km": "Distance from the previous transaction location",
        "speed_from_prev_kmh": "Implied travel speed since the previous transaction (capped)",
    },
    "network": {
        "sender_out_degree": "Distinct counterparties the customer paid before",
        "is_new_recipient": "Counterparty never paid before by this customer (P2P / merchant)",
        "receiver_in_degree": "Distinct senders that paid the receiving customer before",
        "receiver_inflow_count_24h": "Inflows to the receiving customer in the previous 24 h",
        "receiver_account_age_days": "Receiving customer's account age (-1 if not a customer)",
        "receiver_n_prior_txns": "Receiving customer's own prior transactions",
        "log_minutes_since_last_inflow": "log1p(minutes since the customer's last inflow)",
        "outflow_to_last_inflow_ratio": "Debit amount / last inflow amount (24 h)",
        "inflow_sum_24h": "Money received by the customer in the previous 24 h",
    },
}

FEATURE_NAMES: list[str] = [name for group in FEATURE_GROUPS.values() for name in group]
FEATURE_DESCRIPTIONS: dict[str, str] = {k: v for g in FEATURE_GROUPS.values() for k, v in g.items()}

# Features used by the account-takeover model: signals that the *controller* of
# the account changed rather than the owner behaving badly.
ATO_FEATURES: list[str] = [
    "is_new_device",
    "device_missing",
    "accounts_per_device",
    "device_txn_count",
    "customer_device_count",
    "is_new_district",
    "district_frequency",
    "distance_from_prev_km",
    "speed_from_prev_kmh",
    "hour_frequency",
    "is_night",
    "hour_local",
    "amount_zscore",
    "amount_to_avg_ratio",
    "amount_to_balance_ratio",
    "is_new_recipient",
    "txn_count_5m",
    "txn_count_1h",
    "log_seconds_since_prev",
    "wrong_pin_count_1h",
    "failed_count_1h",
    "pin_attempts",
    "n_prior_txns",
    "unique_devices_24h",
    "type_p2p",
    "type_cash_out",
    "log_amount",
]

assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)), "duplicate feature names"
assert set(ATO_FEATURES) <= set(FEATURE_NAMES)

"""Human-readable explanations from SHAP contributions.

XGBoost computes exact TreeSHAP values natively (``pred_contribs=True``), so we
don't need the separate ``shap`` package at serving time. Positive
contributions (in log-odds) push the prediction towards fraud; we turn the
largest ones into short sentences.

Wording matters: SHAP describes the *model*, not the world. Reasons therefore
say a factor "contributed to the model's risk score" - never that it "caused
fraud".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from features.definitions import FEATURE_DESCRIPTIONS

EXPLANATION_NOTE = (
    "Reasons list the factors that contributed most strongly to the model's prediction "
    "(SHAP attributions). They describe the model's reasoning, not proof or cause of fraud."
)

Template = Callable[[float, dict[str, float]], str | None]


def _fmt_amount(v: float) -> str:
    return f"BDT {v:,.0f}"


TEMPLATES: dict[str, Template] = {
    "amount_to_avg_ratio": lambda v, f: (
        f"Transaction amount {v:.1f}x the customer's typical amount" if v >= 1.5 else None
    ),
    "amount_zscore": lambda v, f: "Amount unusually high for this customer" if v >= 1.5 else None,
    "amount": lambda v, f: f"High transaction amount ({_fmt_amount(v)})" if v >= 10_000 else None,
    "log_amount": lambda v, f: f"High transaction amount ({_fmt_amount(np.expm1(v))})" if v >= 9.2 else None,
    "amount_to_balance_ratio": lambda v, f: (
        f"Transaction uses {min(v, 1):.0%} of the wallet balance" if v >= 0.6 else None
    ),
    "is_new_device": lambda v, f: "New device for this customer" if v >= 1 else None,
    "accounts_per_device": lambda v, f: f"Device linked to {v:.0f} different accounts" if v >= 2 else None,
    "device_txn_count": lambda v, f: "Device has little or no transaction history" if v <= 2 else None,
    "is_new_district": lambda v, f: "Unusual location (district never used by this customer)" if v >= 1 else None,
    "district_frequency": lambda v, f: "Location rarely used by this customer" if v < 0.1 else None,
    "distance_from_prev_km": lambda v, f: f"{v:.0f} km from the previous transaction" if v >= 50 else None,
    "speed_from_prev_kmh": lambda v, f: (
        f"Implied travel speed of {v:.0f} km/h since the previous transaction" if v >= 150 else None
    ),
    "is_new_recipient": lambda v, f: "New recipient never paid before" if v >= 1 else None,
    "receiver_in_degree": lambda v, f: (
        f"Recipient has received money from {v:.0f} different senders" if v >= 4 else None
    ),
    "receiver_inflow_count_24h": lambda v, f: (
        f"Recipient received {v:.0f} transfers in the last 24h" if v >= 3 else None
    ),
    "receiver_account_age_days": lambda v, f: f"Recipient account is only {v:.0f} days old" if 0 <= v <= 60 else None,
    "txn_count_1m": lambda v, f: f"{v:.0f} transactions in the previous minute" if v >= 2 else None,
    "txn_count_5m": lambda v, f: f"{v:.0f} transactions in the previous 5 minutes" if v >= 2 else None,
    "txn_count_1h": lambda v, f: f"{v:.0f} transactions in the previous hour" if v >= 3 else None,
    "velocity_ratio_24h": lambda v, f: f"Activity {v:.1f}x the customer's normal daily rate" if v >= 3 else None,
    "log_seconds_since_prev": lambda v, f: (
        f"Only {np.expm1(v):.0f} seconds after the previous transaction" if v <= np.log1p(120) else None
    ),
    "wrong_pin_count_1h": lambda v, f: f"{v:.0f} wrong-PIN attempts in the last hour" if v >= 1 else None,
    "failed_count_1h": lambda v, f: f"{v:.0f} failed attempts in the last hour" if v >= 1 else None,
    "pin_attempts": lambda v, f: f"{v:.0f} PIN attempts in this session" if v >= 2 else None,
    "is_night": lambda v, f: "Night-time transaction" if v >= 1 and f.get("hour_frequency", 1) < 0.1 else None,
    "hour_frequency": lambda v, f: "Unusual time of day for this customer" if v < 0.05 else None,
    "hour_local": lambda v, f: None,
    "merchant_risk_level": lambda v, f: "High-risk merchant category" if v >= 2 else None,
    "merchant_age_days": lambda v, f: f"Merchant registered only {v:.0f} days ago" if 0 <= v <= 90 else None,
    "merchant_unique_customers": lambda v, f: "Merchant has very few prior customers" if 0 <= v <= 5 else None,
    "merchant_txn_count": lambda v, f: None,
    "is_near_threshold": lambda v, f: (
        "Amount just below the reporting threshold (possible structuring)" if v >= 1 else None
    ),
    "outflow_to_last_inflow_ratio": lambda v, f: (
        "Funds moved out shortly after being received"
        if v >= 0.8 and f.get("log_minutes_since_last_inflow", 99) <= np.log1p(120)
        else None
    ),
    "log_minutes_since_last_inflow": lambda v, f: (
        f"Money received {np.expm1(v):.0f} minutes earlier is being moved out" if v <= np.log1p(60) else None
    ),
    "inflow_sum_24h": lambda v, f: f"Received {_fmt_amount(v)} in the last 24h" if v >= 20_000 else None,
    "unique_recipients_24h": lambda v, f: f"Paid {v:.0f} different counterparties in the last 24h" if v >= 4 else None,
    "account_age_days": lambda v, f: f"Account is only {v:.0f} days old" if v <= 30 else None,
    "n_prior_txns": lambda v, f: "Very little transaction history" if v <= 2 else None,
    "type_cash_out": lambda v, f: "Cash-out transaction" if v >= 1 else None,
    "type_p2p": lambda v, f: None,
    "unique_devices_24h": lambda v, f: f"{v:.0f} different devices used in the last 24h" if v >= 2 else None,
    "customer_device_count": lambda v, f: f"Customer has used {v:.0f} different devices" if v >= 3 else None,
    "amount_sum_1h": lambda v, f: f"{_fmt_amount(v)} moved in the previous hour" if v >= 20_000 else None,
    "amount_sum_24h": lambda v, f: f"{_fmt_amount(v)} moved in the previous 24h" if v >= 50_000 else None,
    "sender_out_degree": lambda v, f: "Customer has paid very few counterparties before" if v <= 1 else None,
    "failed_ratio_hist": lambda v, f: f"{v:.0%} of the customer's past attempts failed" if v >= 0.2 else None,
    "device_missing": lambda v, f: None,
}

GENERIC_REASON = "Combination of several weaker risk signals (see reason_details)"


def reasons_from_contributions(
    features: dict[str, float], contributions: dict[str, float], max_reasons: int = 4, min_contribution: float = 0.05
) -> list[dict[str, Any]]:
    """Turn per-feature SHAP values into ordered human-readable reasons."""
    ranked = sorted(
        ((k, v) for k, v in contributions.items() if v > min_contribution), key=lambda kv: kv[1], reverse=True
    )
    reasons: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, contrib in ranked:
        value = float(features.get(name, 0.0))
        template = TEMPLATES.get(name)
        text = template(value, features) if template else None
        if not text or text in seen:
            continue
        seen.add(text)
        reasons.append(
            {"feature": name, "value": round(value, 4), "contribution": round(float(contrib), 4), "reason": text}
        )
        if len(reasons) >= max_reasons:
            break
    if not reasons and ranked:
        name, contrib = ranked[0]
        reasons.append(
            {
                "feature": name,
                "value": round(float(features.get(name, 0.0)), 4),
                "contribution": round(float(contrib), 4),
                "reason": f"{GENERIC_REASON}; strongest: {FEATURE_DESCRIPTIONS.get(name, name)}",
            }
        )
    return reasons


def global_importance(contribs: np.ndarray, names: list[str]) -> list[dict[str, float]]:
    """Mean |SHAP| per feature (bias column excluded)."""
    imp = np.abs(contribs[:, : len(names)]).mean(axis=0)
    order = np.argsort(-imp)
    return [{"feature": names[i], "mean_abs_shap": float(imp[i])} for i in order]

"""Latent behavioural personas for synthetic customers.

A persona is *hidden* ground truth: it drives how a customer transacts but is
never exposed as a feature. Downstream segmentation has to rediscover it from
behaviour, and we can measure how well it does (adjusted Rand index).

All numbers are illustrative assumptions chosen to produce plausible,
heavy-tailed MFS behaviour; they are not calibrated to any real provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from simulator.constants import TRANSACTION_TYPES

# Median amount (BDT) per transaction type before persona / customer scaling.
BASE_AMOUNT_MEDIAN: dict[str, float] = {
    "P2P": 1500.0,
    "MERCHANT_PAYMENT": 650.0,
    "CASH_IN": 3000.0,
    "CASH_OUT": 2500.0,
    "BILL_PAYMENT": 1300.0,
    "MOBILE_RECHARGE": 90.0,
    "BANK_TRANSFER": 9000.0,
}
AMOUNT_SIGMA: dict[str, float] = {
    "P2P": 0.95,
    "MERCHANT_PAYMENT": 0.9,
    "CASH_IN": 0.8,
    "CASH_OUT": 0.8,
    "BILL_PAYMENT": 0.6,
    "MOBILE_RECHARGE": 0.7,
    "BANK_TRANSFER": 0.7,
}


@dataclass(frozen=True)
class Persona:
    name: str
    weight: float  # share of customer base
    daily_rate: float  # median transactions per active day
    rate_sigma: float  # log-normal spread of rate across customers
    type_mix: dict[str, float]
    amount_scale: float
    peak_hour: float  # local time
    hour_spread: float
    night_prob: float  # probability a txn happens 00:00-05:00 local
    smartphone_prob: float
    travel_trips_per_month: float
    churn_prob: float  # probability of churning during the simulated period
    topup_prob: float  # prob. of topping up (cash-in) when balance is short
    occupations: dict[str, float] = field(default_factory=dict)
    age_mean: float = 32.0
    age_sd: float = 9.0

    def mix_vector(self) -> list[float]:
        total = sum(self.type_mix.get(t, 0.0) for t in TRANSACTION_TYPES)
        return [self.type_mix.get(t, 0.0) / total for t in TRANSACTION_TYPES]


PERSONAS: tuple[Persona, ...] = (
    Persona(
        name="digital_native",
        weight=0.20,
        daily_rate=0.75,
        rate_sigma=0.55,
        type_mix={
            "P2P": 0.24,
            "MERCHANT_PAYMENT": 0.36,
            "CASH_IN": 0.10,
            "CASH_OUT": 0.04,
            "BILL_PAYMENT": 0.07,
            "MOBILE_RECHARGE": 0.16,
            "BANK_TRANSFER": 0.03,
        },
        amount_scale=1.1,
        peak_hour=19.0,
        hour_spread=3.5,
        night_prob=0.03,
        smartphone_prob=0.98,
        travel_trips_per_month=0.5,
        churn_prob=0.06,
        topup_prob=0.9,
        occupations={"salaried_private": 0.45, "freelancer": 0.25, "business_owner": 0.1, "student": 0.2},
        age_mean=27,
        age_sd=5,
    ),
    Persona(
        name="salary_earner",
        weight=0.20,
        daily_rate=0.35,
        rate_sigma=0.5,
        type_mix={
            "P2P": 0.30,
            "MERCHANT_PAYMENT": 0.12,
            "CASH_IN": 0.12,
            "CASH_OUT": 0.12,
            "BILL_PAYMENT": 0.18,
            "MOBILE_RECHARGE": 0.10,
            "BANK_TRANSFER": 0.06,
        },
        amount_scale=1.5,
        peak_hour=13.0,
        hour_spread=4.0,
        night_prob=0.01,
        smartphone_prob=0.85,
        travel_trips_per_month=0.3,
        churn_prob=0.07,
        topup_prob=0.9,
        occupations={"salaried_private": 0.5, "government_service": 0.35, "garment_worker": 0.15},
        age_mean=36,
        age_sd=8,
    ),
    Persona(
        name="student",
        weight=0.14,
        daily_rate=0.30,
        rate_sigma=0.6,
        type_mix={
            "P2P": 0.30,
            "MERCHANT_PAYMENT": 0.17,
            "CASH_IN": 0.10,
            "CASH_OUT": 0.12,
            "BILL_PAYMENT": 0.02,
            "MOBILE_RECHARGE": 0.29,
            "BANK_TRANSFER": 0.0,
        },
        amount_scale=0.45,
        peak_hour=16.0,
        hour_spread=4.5,
        night_prob=0.06,
        smartphone_prob=0.9,
        travel_trips_per_month=0.4,
        churn_prob=0.18,
        topup_prob=0.7,
        occupations={"student": 1.0},
        age_mean=21,
        age_sd=2.5,
    ),
    Persona(
        name="remittance_receiver",
        weight=0.20,
        daily_rate=0.10,
        rate_sigma=0.5,
        type_mix={
            "P2P": 0.06,
            "MERCHANT_PAYMENT": 0.03,
            "CASH_IN": 0.03,
            "CASH_OUT": 0.68,
            "BILL_PAYMENT": 0.05,
            "MOBILE_RECHARGE": 0.15,
            "BANK_TRANSFER": 0.0,
        },
        amount_scale=1.2,
        peak_hour=11.0,
        hour_spread=3.0,
        night_prob=0.005,
        smartphone_prob=0.45,
        travel_trips_per_month=0.1,
        churn_prob=0.08,
        topup_prob=0.25,
        occupations={"homemaker": 0.45, "farmer": 0.3, "day_laborer": 0.15, "retired": 0.1},
        age_mean=42,
        age_sd=12,
    ),
    Persona(
        name="small_business",
        weight=0.10,
        daily_rate=1.3,
        rate_sigma=0.5,
        type_mix={
            "P2P": 0.34,
            "MERCHANT_PAYMENT": 0.06,
            "CASH_IN": 0.26,
            "CASH_OUT": 0.22,
            "BILL_PAYMENT": 0.04,
            "MOBILE_RECHARGE": 0.05,
            "BANK_TRANSFER": 0.03,
        },
        amount_scale=2.6,
        peak_hour=14.0,
        hour_spread=4.0,
        night_prob=0.02,
        smartphone_prob=0.8,
        travel_trips_per_month=0.6,
        churn_prob=0.05,
        topup_prob=0.95,
        occupations={"business_owner": 0.8, "transport_worker": 0.2},
        age_mean=38,
        age_sd=9,
    ),
    Persona(
        name="occasional_user",
        weight=0.16,
        daily_rate=0.05,
        rate_sigma=0.6,
        type_mix={
            "P2P": 0.25,
            "MERCHANT_PAYMENT": 0.10,
            "CASH_IN": 0.15,
            "CASH_OUT": 0.20,
            "BILL_PAYMENT": 0.08,
            "MOBILE_RECHARGE": 0.22,
            "BANK_TRANSFER": 0.0,
        },
        amount_scale=0.9,
        peak_hour=15.0,
        hour_spread=4.5,
        night_prob=0.02,
        smartphone_prob=0.6,
        travel_trips_per_month=0.1,
        churn_prob=0.30,
        topup_prob=0.6,
        occupations={"day_laborer": 0.3, "farmer": 0.2, "transport_worker": 0.2, "homemaker": 0.2, "retired": 0.1},
        age_mean=40,
        age_sd=13,
    ),
)

PERSONA_BY_NAME: dict[str, Persona] = {p.name: p for p in PERSONAS}

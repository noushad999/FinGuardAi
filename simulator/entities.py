"""Generation of static entities: customers, merchants/agents/billers, devices, mule rings.

Everything is vectorised with numpy so the LARGE preset (250k customers) stays fast.
The result is a :class:`World` object holding both the public entity tables and
hidden ground truth (persona, mule ring, churn date) used only for labels/evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common.geo import DISTRICTS
from simulator.constants import MERCHANT_CATEGORIES, OCCUPATIONS
from simulator.personas import PERSONAS

DAY = 86_400


@dataclass
class World:
    """All static entities plus per-customer behavioural parameters (numpy arrays)."""

    start_ts: int  # epoch seconds (UTC) of the simulation start (local midnight)
    end_ts: int
    local_offset_s: int
    rng: np.random.Generator

    customers: pd.DataFrame
    merchants: pd.DataFrame
    # per-customer arrays aligned with ``customers`` row order
    persona_idx: np.ndarray
    home_district_idx: np.ndarray
    home_lat: np.ndarray
    home_lon: np.ndarray
    active_from: np.ndarray  # epoch seconds
    active_to: np.ndarray  # epoch seconds (churn date or end)
    churn_ts: np.ndarray  # epoch seconds or -1
    rate: np.ndarray  # transactions/day (after global scaling)
    amount_mult: np.ndarray
    peak_hour: np.ndarray
    hour_spread: np.ndarray
    night_prob: np.ndarray
    type_probs: np.ndarray  # (n_customers, n_types)
    smartphone: np.ndarray  # bool
    initial_balance: np.ndarray
    topup_prob: np.ndarray
    is_mule: np.ndarray
    mule_ring: np.ndarray  # ring id or -1
    # counterparties (padded with -1)
    contacts: np.ndarray
    fav_merchants: np.ndarray
    home_agents: np.ndarray
    billers: np.ndarray
    telco: np.ndarray
    bank: np.ndarray
    # devices
    device_registry: dict[str, dict] = field(default_factory=dict)
    primary_device: np.ndarray = field(default_factory=lambda: np.array([]))
    secondary_device: np.ndarray = field(default_factory=lambda: np.array([]))
    device_change_ts: np.ndarray = field(default_factory=lambda: np.array([]))
    # merchant index helpers
    merchant_ids: np.ndarray = field(default_factory=lambda: np.array([]))
    retail_idx: np.ndarray = field(default_factory=lambda: np.array([]))
    agent_idx: np.ndarray = field(default_factory=lambda: np.array([]))
    biller_idx: np.ndarray = field(default_factory=lambda: np.array([]))
    telco_idx: np.ndarray = field(default_factory=lambda: np.array([]))
    bank_idx: np.ndarray = field(default_factory=lambda: np.array([]))
    fraud_merchant_idx: np.ndarray = field(default_factory=lambda: np.array([]))
    merchant_district_idx: np.ndarray = field(default_factory=lambda: np.array([]))
    merchant_reg_ts: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def n_customers(self) -> int:
        return len(self.customers)

    @property
    def customer_ids(self) -> np.ndarray:
        return self.customers["customer_id"].to_numpy()

    def new_device(self, owner_idx: int | None, ts: int, device_type: str | None = None) -> str:
        """Register a brand-new device (e.g. an attacker handset) and return its id."""
        dev_id = f"DEV_{len(self.device_registry) + 1:08d}"
        rng = self.rng
        dtype = device_type or ("smartphone" if rng.random() < 0.95 else "feature_phone")
        os_name = _pick_os(rng, dtype)
        self.device_registry[dev_id] = {
            "device_type": dtype,
            "os": os_name,
            "device_age_days": int(rng.integers(0, 60)),
            "owner_idx": owner_idx,
            "created_ts": ts,
        }
        return dev_id


def _pick_os(rng: np.random.Generator, device_type: str) -> str:
    if device_type == "feature_phone":
        return "KaiOS" if rng.random() < 0.3 else "Proprietary"
    return "Android" if rng.random() < 0.93 else "iOS"


def _weighted_choice(rng: np.random.Generator, n: int, weights: np.ndarray) -> np.ndarray:
    p = np.asarray(weights, dtype=float)
    return rng.choice(len(p), size=n, p=p / p.sum())


def _pad_choices(
    rng: np.random.Generator, pool: np.ndarray, counts: np.ndarray, width: int, pool_weights: np.ndarray | None = None
) -> np.ndarray:
    """For each row pick ``counts[i]`` items from ``pool`` (with weights), padded with -1."""
    out = np.full((len(counts), width), -1, dtype=np.int64)
    if len(pool) == 0:
        return out
    p = None if pool_weights is None else pool_weights / pool_weights.sum()
    draws = rng.choice(pool, size=(len(counts), width), p=p)
    mask = np.arange(width)[None, :] < counts[:, None]
    out[mask] = draws[mask]
    return out


def build_world(
    n_customers: int,
    start_date: str,
    days: int,
    local_offset_hours: int,
    mule_fraction: float,
    seed: int,
) -> World:
    """Create customers, merchants, counterparties and devices."""
    rng = np.random.default_rng(seed)
    local_offset_s = local_offset_hours * 3600
    # Simulation starts at local midnight of start_date.
    start_ts = int(pd.Timestamp(start_date, tz="UTC").timestamp()) - local_offset_s
    end_ts = start_ts + days * DAY

    # ---------------- customers ----------------
    persona_w = np.array([p.weight for p in PERSONAS])
    persona_idx = _weighted_choice(rng, n_customers, persona_w)
    district_w = np.array([d.weight for d in DISTRICTS])
    home_district_idx = _weighted_choice(rng, n_customers, district_w)
    d_lat = np.array([d.lat for d in DISTRICTS])
    d_lon = np.array([d.lon for d in DISTRICTS])
    home_lat = d_lat[home_district_idx] + rng.normal(0, 0.04, n_customers)
    home_lon = d_lon[home_district_idx] + rng.normal(0, 0.04, n_customers)
    upazila = np.array(
        [f"{DISTRICTS[di].name}-U{rng.integers(1, DISTRICTS[di].n_upazilas + 1):02d}" for di in home_district_idx]
    )

    ages = np.empty(n_customers)
    occupations = np.empty(n_customers, dtype=object)
    rate = np.empty(n_customers)
    amount_mult = np.empty(n_customers)
    peak_hour = np.empty(n_customers)
    hour_spread = np.empty(n_customers)
    night_prob = np.empty(n_customers)
    smartphone = np.empty(n_customers, dtype=bool)
    type_probs = np.empty((n_customers, 7))
    churn_flag = np.empty(n_customers, dtype=bool)
    topup_prob = np.empty(n_customers)
    trips_pm = np.empty(n_customers)
    for pi, p in enumerate(PERSONAS):
        m = persona_idx == pi
        k = int(m.sum())
        if k == 0:
            continue
        ages[m] = np.clip(rng.normal(p.age_mean, p.age_sd, k), 18, 80)
        occ_names = list(p.occupations)
        occ_w = np.array([p.occupations[o] for o in occ_names])
        occupations[m] = np.array(occ_names, dtype=object)[_weighted_choice(rng, k, occ_w)]
        rate[m] = p.daily_rate * rng.lognormal(0, p.rate_sigma, k)
        amount_mult[m] = p.amount_scale * rng.lognormal(0, 0.45, k)
        peak_hour[m] = p.peak_hour + rng.normal(0, 1.5, k)
        hour_spread[m] = p.hour_spread * rng.uniform(0.7, 1.3, k)
        night_prob[m] = p.night_prob * rng.lognormal(0, 0.6, k)
        smartphone[m] = rng.random(k) < p.smartphone_prob
        type_probs[m] = rng.dirichlet(np.array(p.mix_vector()) * 40 + 1e-3, k)
        churn_flag[m] = rng.random(k) < p.churn_prob
        topup_prob[m] = p.topup_prob
        trips_pm[m] = p.travel_trips_per_month * rng.lognormal(0, 0.5, k)
    # Night-shift workers: legitimate night activity (overlaps with fraud signal).
    night_shift = np.isin(occupations, ["transport_worker", "garment_worker"]) & (rng.random(n_customers) < 0.35)
    peak_hour[night_shift] = rng.normal(1.5, 1.0, int(night_shift.sum()))
    night_prob[night_shift] = rng.uniform(0.15, 0.35, int(night_shift.sum()))
    # feature-phone users cannot use the app channel; they use USSD
    peak_hour = np.mod(peak_hour, 24)

    # Registration: 82% before the window (up to 5 years), 18% during it.
    registered_during = rng.random(n_customers) < 0.18
    reg_ts = np.where(
        registered_during,
        start_ts + rng.uniform(0, 0.8, n_customers) * (end_ts - start_ts),
        start_ts - rng.uniform(30, 5 * 365, n_customers) * DAY,
    ).astype(np.int64)

    # Churn date: after at least 45 days of observation.
    churn_ts = np.full(n_customers, -1, dtype=np.int64)
    lo = np.maximum(start_ts + 45 * DAY, reg_ts + 40 * DAY)
    valid = churn_flag & (lo < end_ts - 5 * DAY)
    churn_ts[valid] = (lo[valid] + rng.uniform(0, 1, int(valid.sum())) * (end_ts - lo[valid])).astype(np.int64)
    active_from = np.maximum(reg_ts, start_ts)
    active_to = np.where(churn_ts > 0, churn_ts, end_ts)

    # ---------------- mules ----------------
    n_mules = max(8, int(round(mule_fraction * n_customers)))
    # Mules look like recently-registered occasional users / students.
    candidates = np.where(np.isin(persona_idx, [2, 5]))[0]
    mule_idx = rng.choice(candidates, size=min(n_mules, len(candidates)), replace=False)
    is_mule = np.zeros(n_customers, dtype=bool)
    is_mule[mule_idx] = True
    reg_ts[mule_idx] = (start_ts + rng.uniform(-60, 0.5 * days, len(mule_idx)) * DAY).astype(np.int64)
    active_from = np.maximum(reg_ts, start_ts)
    churn_ts[mule_idx] = -1
    active_to[mule_idx] = end_ts
    mule_ring = np.full(n_customers, -1, dtype=np.int64)
    ring_id = 0
    order = rng.permutation(mule_idx)
    pos = 0
    while pos < len(order):
        size = int(rng.integers(3, 9))
        mule_ring[order[pos : pos + size]] = ring_id
        pos += size
        ring_id += 1

    segment = np.select(
        [amount_mult > 2.5, amount_mult > 1.2],
        ["priority", "mass_affluent"],
        default="mass",
    )
    customers = pd.DataFrame(
        {
            "customer_id": [f"CUS_{i + 1:07d}" for i in range(n_customers)],
            "age": ages.round().astype(int),
            "registration_date": pd.to_datetime(reg_ts, unit="s", utc=True).floor("D"),
            "occupation_category": occupations.astype(str),
            "district": [DISTRICTS[i].name for i in home_district_idx],
            "upazila": upazila,
            "customer_segment": segment,
            "kyc_level": np.where(rng.random(n_customers) < 0.85, "full", "basic"),
        }
    )

    # ---------------- merchants & other counterparties ----------------
    n_retail = max(120, n_customers // 10)
    n_agents = max(60, n_customers // 25)
    cat_names = list(MERCHANT_CATEGORIES)
    cat_w = np.array([MERCHANT_CATEGORIES[c][0] for c in cat_names])
    retail_cat = np.array(cat_names)[_weighted_choice(rng, n_retail, cat_w)]
    retail_risk = np.array([MERCHANT_CATEGORIES[c][1] for c in retail_cat], dtype=object)
    # A few merchants are fraudulent storefronts (ground truth); they look like
    # recently-registered medium/high risk e-commerce/electronics shops.
    # Fraudulent storefronts are short-lived: they open throughout the period,
    # scam for a few weeks and disappear. ~10% of legitimate merchants are also
    # newly registered during the window, so "new merchant" alone is not proof.
    n_fraud_merchants = max(6, n_retail // 40)
    fraud_m = rng.choice(
        np.where(np.isin(retail_cat, ["e_commerce", "electronics", "online_gaming"]))[0],
        size=n_fraud_merchants,
        replace=False,
    )
    retail_reg = start_ts - rng.uniform(30, 8 * 365, n_retail) * DAY
    new_legit = rng.random(n_retail) < 0.10
    retail_reg[new_legit] = start_ts + rng.uniform(0, 0.9 * days, int(new_legit.sum())) * DAY
    retail_reg[fraud_m] = start_ts + rng.uniform(0, 0.93 * days, n_fraud_merchants) * DAY
    retail_risk[fraud_m] = np.where(rng.random(n_fraud_merchants) < 0.6, "high", "medium")
    retail_district = _weighted_choice(rng, n_retail, district_w)
    agent_district = _weighted_choice(rng, n_agents, district_w)

    rows = []
    for i in range(n_retail):
        rows.append(("MER", f"MER_{i + 1:06d}", retail_cat[i], retail_district[i], retail_risk[i], retail_reg[i]))
    for i in range(n_agents):
        rows.append(
            (
                "AGT",
                f"AGT_{i + 1:06d}",
                "cash_agent",
                agent_district[i],
                "medium" if rng.random() < 0.1 else "low",
                start_ts - rng.uniform(60, 6 * 365) * DAY,
            )
        )
    billers = [
        "electricity",
        "gas",
        "water",
        "internet",
        "tv",
        "insurance",
        "education_fees",
        "electricity",
        "water",
        "internet",
        "credit_card",
        "loan_emi",
    ]
    for i, b in enumerate(billers):
        rows.append(("BIL", f"BIL_{i + 1:04d}", f"utility_{b}", 0, "low", start_ts - 3000 * DAY))
    for i in range(4):
        rows.append(("TEL", f"TEL_{i + 1:04d}", "telecom_operator", 0, "low", start_ts - 4000 * DAY))
    for i in range(10):
        rows.append(("BNK", f"BNK_{i + 1:04d}", "bank", 0, "low", start_ts - 5000 * DAY))
    merchants = pd.DataFrame(
        rows, columns=["prefix", "merchant_id", "merchant_category", "district_idx", "risk_category", "reg_ts"]
    )
    merchants["merchant_type"] = merchants["prefix"].map(
        {"MER": "merchant", "AGT": "cash_agent", "BIL": "biller", "TEL": "telecom_operator", "BNK": "bank"}
    )
    merchants["district"] = [DISTRICTS[int(i)].name for i in merchants["district_idx"]]
    merchants.loc[merchants["prefix"].isin(["BIL", "TEL", "BNK"]), "district"] = "Dhaka"
    merchants["registration_date"] = pd.to_datetime(merchants["reg_ts"].astype(np.int64), unit="s", utc=True).dt.floor(
        "D"
    )
    merchant_ids = merchants["merchant_id"].to_numpy()
    retail_idx = np.where(merchants["prefix"] == "MER")[0]
    agent_idx = np.where(merchants["prefix"] == "AGT")[0]
    biller_idx = np.where(merchants["prefix"] == "BIL")[0]
    telco_idx = np.where(merchants["prefix"] == "TEL")[0]
    bank_idx = np.where(merchants["prefix"] == "BNK")[0]
    fraud_merchant_idx = retail_idx[fraud_m]
    merchant_district_idx = merchants["district_idx"].to_numpy().astype(int)
    merchants["is_fraudulent"] = False
    merchants.loc[fraud_merchant_idx, "is_fraudulent"] = True

    # ---------------- counterparties per customer ----------------
    # P2P contacts: biased to the same district and towards remittance receivers
    # (family members in villages receive money from earners in cities).
    n_contacts = rng.integers(2, 13, n_customers)
    receiver_w = np.where(persona_idx == 3, 3.0, 1.0) * np.where(is_mule, 0.0, 1.0)
    contacts = _pad_choices(rng, np.arange(n_customers), n_contacts, 12, receiver_w)
    # replace ~50% with same-district contacts
    by_district = {di: np.where((home_district_idx == di) & ~is_mule)[0] for di in range(len(DISTRICTS))}
    same = rng.random(contacts.shape) < 0.5
    for di, members in by_district.items():
        if len(members) == 0:
            continue
        rows_m = np.where(home_district_idx == di)[0]
        sub = same[rows_m] & (contacts[rows_m] >= 0)
        picks = rng.choice(members, size=sub.shape)
        block = contacts[rows_m]
        block[sub] = picks[sub]
        contacts[rows_m] = block
    self_ref = contacts == np.arange(n_customers)[:, None]
    contacts[self_ref] = (contacts[self_ref] + 1) % n_customers

    fav_merchants = np.full((n_customers, 8), -1, dtype=np.int64)
    home_agents = np.full((n_customers, 3), -1, dtype=np.int64)
    retail_by_d = {di: retail_idx[merchant_district_idx[retail_idx] == di] for di in range(len(DISTRICTS))}
    agents_by_d = {di: agent_idx[merchant_district_idx[agent_idx] == di] for di in range(len(DISTRICTS))}
    legit_retail = np.setdiff1d(retail_idx, fraud_merchant_idx)
    n_fav = rng.integers(2, 9, n_customers)
    n_ag = rng.integers(1, 4, n_customers)
    for di in range(len(DISTRICTS)):
        rows_d = np.where(home_district_idx == di)[0]
        if len(rows_d) == 0:
            continue
        local_retail = np.setdiff1d(retail_by_d[di], fraud_merchant_idx)
        pool_r = local_retail if len(local_retail) >= 3 else legit_retail
        fav_merchants[rows_d] = _pad_choices(rng, pool_r, n_fav[rows_d], 8)
        # a couple of national/e-commerce merchants too
        ecommerce = rng.random(len(rows_d)) < 0.5
        fav_merchants[rows_d[ecommerce], 0] = rng.choice(legit_retail, int(ecommerce.sum()))
        pool_a = agents_by_d[di] if len(agents_by_d[di]) else agent_idx
        home_agents[rows_d] = _pad_choices(rng, pool_a, n_ag[rows_d], 3)
    billers_arr = _pad_choices(rng, biller_idx, rng.integers(1, 4, n_customers), 3)
    telco = rng.choice(telco_idx, n_customers, p=[0.45, 0.3, 0.2, 0.05])
    bank = rng.choice(bank_idx, n_customers)

    initial_balance = np.round(rng.lognormal(np.log(1500), 1.0, n_customers) * amount_mult, -1)

    world = World(
        start_ts=start_ts,
        end_ts=end_ts,
        local_offset_s=local_offset_s,
        rng=rng,
        customers=customers,
        merchants=merchants.drop(columns=["prefix", "district_idx", "reg_ts"]),
        persona_idx=persona_idx,
        home_district_idx=home_district_idx,
        home_lat=home_lat,
        home_lon=home_lon,
        active_from=active_from,
        active_to=active_to,
        churn_ts=churn_ts,
        rate=rate,
        amount_mult=amount_mult,
        peak_hour=peak_hour,
        hour_spread=hour_spread,
        night_prob=np.clip(night_prob, 0, 0.5),
        type_probs=type_probs,
        smartphone=smartphone,
        initial_balance=initial_balance,
        topup_prob=topup_prob,
        is_mule=is_mule,
        mule_ring=mule_ring,
        contacts=contacts,
        fav_merchants=fav_merchants,
        home_agents=home_agents,
        billers=billers_arr,
        telco=telco,
        bank=bank,
        merchant_ids=merchant_ids,
        retail_idx=retail_idx,
        agent_idx=agent_idx,
        biller_idx=biller_idx,
        telco_idx=telco_idx,
        bank_idx=bank_idx,
        fraud_merchant_idx=fraud_merchant_idx,
        merchant_district_idx=merchant_district_idx,
        merchant_reg_ts=merchants["reg_ts"].to_numpy().astype(np.int64),
    )
    world.customers["_night_shift"] = night_shift
    world.customers["_trips_per_month"] = trips_pm
    _assign_devices(world, reg_ts)
    return world


def _assign_devices(world: World, reg_ts: np.ndarray) -> None:
    """Primary device per customer, optional secondary device and phone upgrades.

    ~3% of customers share their primary device with a family member, which is
    *legitimate* device sharing and overlaps with the device-farm fraud signal.
    """
    rng = world.rng
    n = world.n_customers
    primary = np.empty(n, dtype=object)
    secondary = np.full(n, None, dtype=object)
    change_ts = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        dtype = "smartphone" if world.smartphone[i] else "feature_phone"
        primary[i] = world.new_device(i, int(reg_ts[i]), dtype)
    shared = rng.random(n) < 0.03
    for i in np.where(shared)[0]:
        j = world.contacts[i, 0]
        if j >= 0 and j != i:
            primary[i] = primary[j]
    has_second = rng.random(n) < 0.10
    for i in np.where(has_second)[0]:
        secondary[i] = world.new_device(i, int(reg_ts[i]))
    # legitimate phone change during the window -> a genuinely "new device"
    changers = (rng.random(n) < 0.12) & ~has_second
    for i in np.where(changers)[0]:
        lo, hi = world.active_from[i], world.active_to[i]
        if hi - lo < 10 * DAY:
            continue
        t = int(lo + rng.uniform(0.1, 0.9) * (hi - lo))
        secondary[i] = world.new_device(i, t, "smartphone")
        change_ts[i] = t
    world.primary_device = primary
    world.secondary_device = secondary
    world.device_change_ts = change_ts

    # keep occupations list import used (documented choice list)
    assert set(world.customers["occupation_category"]).issubset(set(OCCUPATIONS))

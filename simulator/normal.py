"""Vectorised generation of *legitimate* customer behaviour.

The generator works in two steps:

1. Sample *when* each customer transacts (non-homogeneous Poisson process via
   thinning: weekly and month-start seasonality, festival spikes, and a gradual
   activity decline before a customer churns).
2. "Materialise" each (customer, timestamp) pair into a transaction: type,
   amount, counterparty, device, channel and location (home or travelling).

Legitimate behaviour intentionally contains things that also appear in fraud:
new recipients, occasional large payments, phone upgrades (new device), travel,
night-shift workers, bursts of transactions within a few minutes, and amounts
just under the reporting threshold for small businesses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.geo import DISTRICTS
from simulator.constants import TRANSACTION_TYPES
from simulator.entities import DAY, World
from simulator.personas import AMOUNT_SIGMA, BASE_AMOUNT_MEDIAN

T_IDX = {t: i for i, t in enumerate(TRANSACTION_TYPES)}
CH_APP, CH_USSD, CH_AGENT = 0, 1, 2

# Approximate festival dates (local) that drive remittance spikes.
FESTIVALS = ("2026-03-20", "2026-05-27", "2025-03-31", "2025-06-07", "2027-03-10", "2027-05-17")

D_LAT = np.array([d.lat for d in DISTRICTS])
D_LON = np.array([d.lon for d in DISTRICTS])
D_W = np.array([d.weight for d in DISTRICTS])


def _time_weight(world: World, ts: np.ndarray, cust: np.ndarray) -> np.ndarray:
    local = ts + world.local_offset_s
    days = local // DAY
    dow = (days + 3) % 7  # 1970-01-01 was a Thursday -> Monday=0
    w = np.ones(len(ts))
    w[dow == 4] *= 0.8  # Friday
    w[dow == 5] *= 0.9  # Saturday
    dom = pd.to_datetime(local, unit="s").day.to_numpy()
    w[dom <= 7] *= 1.15
    for f in FESTIVALS:
        fday = int(pd.Timestamp(f).timestamp()) // DAY
        w[(days >= fday - 6) & (days <= fday)] *= 1.6
    churn = world.churn_ts[cust]
    decay_zone = (churn > 0) & (ts > churn - 30 * DAY)
    frac = np.clip((churn - ts) / (30 * DAY), 0, 1)
    w[decay_zone] *= 0.15 + 0.85 * frac[decay_zone]
    return w


def sample_event_times(world: World, n_primary: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample (customer_index, ts) pairs for roughly ``n_primary`` events."""
    rng = world.rng
    active_days = np.clip((world.active_to - world.active_from) / DAY, 0, None)
    base = world.rate * active_days
    w_max = 1.15 * 1.6
    probe = rng.uniform(world.start_ts, world.end_ts, 20_000).astype(np.int64)
    accept = _time_weight(world, probe, np.zeros(len(probe), dtype=np.int64)).mean() / w_max
    scale = n_primary / max(base.sum(), 1e-9)
    world.rate = world.rate * scale  # store calibrated rate (used by fraud baselines)
    counts = rng.poisson(base * scale / accept * 1.02)
    cust = np.repeat(np.arange(world.n_customers), counts)
    u = rng.random(len(cust))
    lo = world.active_from[cust]
    hi = world.active_to[cust]
    day_ts = lo + u * (hi - lo)
    keep = rng.random(len(cust)) < _time_weight(world, day_ts.astype(np.int64), cust) / w_max
    cust, day_ts = cust[keep], day_ts[keep]

    # time of day (local)
    local_day_start = ((day_ts.astype(np.int64) + world.local_offset_s) // DAY) * DAY - world.local_offset_s
    night = rng.random(len(cust)) < world.night_prob[cust]
    hour = np.where(
        night,
        rng.uniform(0, 5, len(cust)),
        np.mod(rng.normal(world.peak_hour[cust], world.hour_spread[cust]), 24),
    )
    ts = (local_day_start + hour * 3600).astype(np.int64)
    ok = (ts >= world.active_from[cust]) & (ts < world.active_to[cust])
    cust, ts = cust[ok], ts[ok]
    if len(ts) > n_primary:
        sel = np.sort(rng.choice(len(ts), n_primary, replace=False))
        cust, ts = cust[sel], ts[sel]
    return cust, ts


def _pick_from_padded(rng: np.random.Generator, table: np.ndarray, rows: np.ndarray, skew: float = 1.8) -> np.ndarray:
    """Pick one valid entry per row of a -1 padded table, favouring early columns."""
    sub = table[rows]
    k = (sub >= 0).sum(axis=1)
    k = np.maximum(k, 1)
    pos = np.floor(k * rng.random(len(rows)) ** skew).astype(np.int64)
    return sub[np.arange(len(rows)), np.minimum(pos, k - 1)]


def _round_amounts(amount: np.ndarray, ttype: np.ndarray) -> np.ndarray:
    out = np.where(amount < 1000, np.round(amount, -1), np.round(amount / 50) * 50)
    rech = ttype == T_IDX["MOBILE_RECHARGE"]
    out[rech] = np.clip(np.round(amount[rech] / 10) * 10, 20, 1500)
    bank = ttype == T_IDX["BANK_TRANSFER"]
    out[bank] = np.maximum(out[bank], 500)
    return np.maximum(out, 10.0)


def _trip_table(world: World) -> pd.DataFrame:
    rng = world.rng
    active_days = np.clip((world.active_to - world.active_from) / DAY, 0, None)
    trips_pm = world.customers["_trips_per_month"].to_numpy()
    n_trips = rng.poisson(trips_pm * active_days / 30.0)
    cust = np.repeat(np.arange(world.n_customers), n_trips)
    start = world.active_from[cust] + rng.random(len(cust)) * (world.active_to[cust] - world.active_from[cust])
    dur = rng.uniform(1, 4, len(cust)) * DAY
    dest = rng.choice(len(DISTRICTS), len(cust), p=D_W / D_W.sum())
    same = dest == world.home_district_idx[cust]
    dest[same] = (dest[same] + rng.integers(1, len(DISTRICTS), int(same.sum()))) % len(DISTRICTS)
    return pd.DataFrame(
        {"cust": cust, "trip_start": start.astype(np.int64), "trip_end": (start + dur).astype(np.int64), "dest": dest}
    ).sort_values("trip_start")


def materialize(
    world: World, cust: np.ndarray, ts: np.ndarray, trips: pd.DataFrame, type_override: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Turn (customer, timestamp) pairs into fully specified legitimate events."""
    rng = world.rng
    n = len(cust)
    # ---- type ----
    if type_override is None:
        cum = np.cumsum(world.type_probs[cust], axis=1)
        ttype = (rng.random(n)[:, None] > cum).sum(axis=1).clip(0, len(TRANSACTION_TYPES) - 1)
    else:
        ttype = type_override
    ttype = ttype.astype(np.int8)

    # ---- amount ----
    med = np.array([BASE_AMOUNT_MEDIAN[t] for t in TRANSACTION_TYPES])[ttype]
    sig = np.array([AMOUNT_SIGMA[t] for t in TRANSACTION_TYPES])[ttype]
    amount = med * world.amount_mult[cust] * rng.lognormal(0, sig)
    big = rng.random(n) < 0.004  # legitimate large purchases / family support
    amount[big] *= rng.uniform(4, 12, int(big.sum()))
    # small businesses sometimes move amounts just under the reporting threshold
    sb = (world.persona_idx[cust] == 4) & (rng.random(n) < 0.01)
    amount[sb] = rng.uniform(0.85, 0.999, int(sb.sum())) * 25_000
    amount = _round_amounts(amount, ttype)

    # ---- location: home or travelling ----
    ev = pd.DataFrame({"i": np.arange(n), "cust": cust, "ts": ts}).sort_values("ts")
    if len(trips):
        m = pd.merge_asof(ev, trips, left_on="ts", right_on="trip_start", by="cust", direction="backward")
        travelling = (m["ts"] <= m["trip_end"]).fillna(False).to_numpy(dtype=bool)
        dest = m["dest"].fillna(-1).to_numpy().astype(np.int64)
        order = m["i"].to_numpy()
        trav = np.zeros(n, dtype=bool)
        dest_arr = np.full(n, -1, dtype=np.int64)
        trav[order] = travelling
        dest_arr[order] = dest
    else:
        trav = np.zeros(n, dtype=bool)
        dest_arr = np.full(n, -1, dtype=np.int64)
    district = np.where(trav, dest_arr, world.home_district_idx[cust])
    lat = np.where(
        trav, D_LAT[np.clip(dest_arr, 0, None)] + rng.normal(0, 0.03, n), world.home_lat[cust] + rng.normal(0, 0.01, n)
    )
    lon = np.where(
        trav, D_LON[np.clip(dest_arr, 0, None)] + rng.normal(0, 0.03, n), world.home_lon[cust] + rng.normal(0, 0.01, n)
    )

    # ---- counterparties ----
    cp_cust = np.full(n, -1, dtype=np.int64)
    merch = np.full(n, -1, dtype=np.int64)
    p2p = ttype == T_IDX["P2P"]
    rows = np.where(p2p)[0]
    cp_cust[rows] = _pick_from_padded(rng, world.contacts, cust[rows])
    new_rcpt = rows[rng.random(len(rows)) < 0.04]
    cp_cust[new_rcpt] = rng.integers(0, world.n_customers, len(new_rcpt))
    bad = (cp_cust == cust) & p2p
    cp_cust[bad] = (cp_cust[bad] + 1) % world.n_customers

    mp = np.where(ttype == T_IDX["MERCHANT_PAYMENT"])[0]
    merch[mp] = _pick_from_padded(rng, world.fav_merchants, cust[mp])
    legit_retail = np.setdiff1d(world.retail_idx, world.fraud_merchant_idx)
    explore = mp[rng.random(len(mp)) < 0.08]
    merch[explore] = rng.choice(legit_retail, len(explore))
    # while travelling, shop locally
    trav_mp = mp[trav[mp] & (rng.random(len(mp)) < 0.6)]
    for di in np.unique(district[trav_mp]):
        local = legit_retail[world.merchant_district_idx[legit_retail] == di]
        sel = trav_mp[district[trav_mp] == di]
        if len(local):
            merch[sel] = rng.choice(local, len(sel))

    # a merchant cannot be paid before it exists: fall back to an established one
    early = mp[ts[mp] < world.merchant_reg_ts[merch[mp]]]
    if len(early):
        established = legit_retail[world.merchant_reg_ts[legit_retail] < world.start_ts]
        merch[early] = rng.choice(established, len(early))

    cash = np.where(np.isin(ttype, [T_IDX["CASH_IN"], T_IDX["CASH_OUT"]]))[0]
    merch[cash] = _pick_from_padded(rng, world.home_agents, cust[cash], skew=1.0)
    trav_cash = cash[trav[cash]]
    for di in np.unique(district[trav_cash]):
        local = world.agent_idx[world.merchant_district_idx[world.agent_idx] == di]
        sel = trav_cash[district[trav_cash] == di]
        if len(local):
            merch[sel] = rng.choice(local, len(sel))
    bill = np.where(ttype == T_IDX["BILL_PAYMENT"])[0]
    merch[bill] = _pick_from_padded(rng, world.billers, cust[bill], skew=1.0)
    rech = np.where(ttype == T_IDX["MOBILE_RECHARGE"])[0]
    merch[rech] = world.telco[cust[rech]]
    bank = np.where(ttype == T_IDX["BANK_TRANSFER"])[0]
    merch[bank] = world.bank[cust[bank]]

    # ---- device & channel ----
    device = world.primary_device[cust].copy()
    sec = world.secondary_device[cust]
    chg = world.device_change_ts[cust]
    upgraded = (chg > 0) & (ts >= chg)
    device[upgraded] = sec[upgraded]
    use_second = (chg < 0) & np.array([s is not None for s in sec]) & (rng.random(n) < 0.2)
    device[use_second] = sec[use_second]
    dev_type = np.array([world.device_registry[d]["device_type"] for d in device])
    channel = np.where(dev_type == "smartphone", CH_APP, CH_USSD).astype(np.int8)
    ussd_fallback = (channel == CH_APP) & (rng.random(n) < 0.04)
    channel[ussd_fallback] = CH_USSD
    cin = ttype == T_IDX["CASH_IN"]
    channel[cin] = CH_AGENT
    device = device.astype(object)
    device[cin] = None

    return {
        "ts": ts.astype(np.int64),
        "cust": cust.astype(np.int64),
        "ttype": ttype,
        "cp_cust": cp_cust,
        "merch": merch,
        "amount": amount.astype(float),
        "drain": np.full(n, np.nan),
        "device": device,
        "channel": channel,
        "district": district.astype(np.int64),
        "lat": lat,
        "lon": lon,
    }


def generate_normal_events(world: World, n_target: int) -> pd.DataFrame:
    """Generate ~``n_target`` legitimate events (including short follow-up bursts)."""
    rng = world.rng
    followup_rate = 0.10
    n_primary = int(n_target / (1 + followup_rate))
    cust, ts = sample_event_times(world, n_primary)
    trips = _trip_table(world)
    ev = materialize(world, cust, ts, trips)

    # Follow-up bursts: a second action a few minutes later (pay, recharge, send).
    eligible = np.where(ev["ttype"] != T_IDX["CASH_IN"])[0]
    k = min(len(eligible), n_target - len(ts))
    k = max(k, 0)
    base_i = rng.choice(eligible, k, replace=False) if k else np.array([], dtype=np.int64)
    f_cust = ev["cust"][base_i]
    f_ts = ev["ts"][base_i] + rng.integers(30, 600, k)
    f_type = rng.choice([T_IDX["MERCHANT_PAYMENT"], T_IDX["MOBILE_RECHARGE"], T_IDX["P2P"]], k, p=[0.4, 0.35, 0.25])
    fu = materialize(world, f_cust, f_ts, trips, type_override=f_type)
    # keep device/location consistent with the originating event
    for col in ("device", "channel", "district", "lat", "lon"):
        fu[col] = ev[col][base_i]
    fu["channel"] = np.where(fu["channel"] == CH_AGENT, CH_APP, fu["channel"]).astype(np.int8)
    df = pd.concat([pd.DataFrame(ev), pd.DataFrame(fu)], ignore_index=True)
    df = df[df["ts"] < world.end_ts]
    df["source"] = "normal"
    return df

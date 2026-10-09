"""Fraud scenario engine.

Each scenario creates one or more *cases*; a case is a coherent story (who is
the victim, who is the perpetrator, which device/location, what sequence of
transactions) with a ground-truth label on every transaction it produces.

Design principles
-----------------
* Every fraud transaction has an explainable reason (``fraud_scenario``).
* Fraud is *not* perfectly separable: several scenarios (unusual-recipient scams,
  suspicious merchants, mule collection) are executed by the genuine customer on
  their own device and location, with amounts close to their normal range. At the
  same time normal behaviour contains new devices, travel, night activity and
  large payments (see ``simulator.normal``).
* ATO drains are expressed as a *fraction of the balance at that moment*
  (``drain``) and resolved by the ledger, so fraudsters can only steal what is
  there and failed attempts on empty wallets occur naturally.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from common.geo import DISTRICTS, haversine_km
from simulator.constants import ATO_SCENARIOS, FRAUD_SCENARIOS
from simulator.entities import DAY, World
from simulator.normal import CH_AGENT, CH_APP, CH_USSD, D_LAT, D_LON, T_IDX

# Rough expected number of transactions per case, used to budget case counts.
AVG_TXNS_PER_CASE: dict[str, float] = {
    "velocity_attack": 7.5,
    "sudden_high_value": 1.0,
    "new_device_high_value": 1.5,
    "impossible_travel": 2.0,
    "device_sharing_ring": 6.0,
    "suspicious_merchant": 6.0,
    "rapid_fund_movement": 2.0,
    "structuring": 7.0,
    "mule_collection": 11.0,
    "failed_auth_then_success": 4.5,
    "unusual_recipient_scam": 1.0,
    "abnormal_nighttime": 3.5,
}


class FraudInjector:
    """Generates labelled fraud events on top of the legitimate event table."""

    def __init__(self, world: World, normal: pd.DataFrame, structuring_threshold: float):
        self.w = world
        self.rng = world.rng
        self.threshold = structuring_threshold
        self.events: list[dict] = []
        self.case_seq = 0
        # Per-customer baselines computed from legitimate behaviour.
        out = normal[normal["ttype"] != T_IDX["CASH_IN"]]
        med = out.groupby("cust")["amount"].median()
        self.median_amount = np.full(world.n_customers, 1000.0)
        self.median_amount[med.index.to_numpy()] = med.to_numpy()
        hours = ((normal["ts"] + world.local_offset_s) % DAY) / 3600
        night_share = (hours < 5).groupby(normal["cust"]).mean()
        self.night_share = np.zeros(world.n_customers)
        self.night_share[night_share.index.to_numpy()] = night_share.to_numpy()
        counts = normal.groupby("cust").size()
        self.n_events = np.zeros(world.n_customers, dtype=np.int64)
        self.n_events[counts.index.to_numpy()] = counts.to_numpy()
        # sorted per-customer timeline for "last legit event before t" lookups
        tl = normal[["cust", "ts", "district", "device"]].sort_values(["cust", "ts"])
        self.tl_cust = tl["cust"].to_numpy()
        self.tl_ts = tl["ts"].to_numpy()
        self.tl_district = tl["district"].to_numpy()
        self.tl_device = tl["device"].to_numpy()
        self.tl_start = np.searchsorted(self.tl_cust, np.arange(world.n_customers), side="left")
        self.tl_end = np.searchsorted(self.tl_cust, np.arange(world.n_customers), side="right")

        self.mules = np.where(world.is_mule)[0]
        self.rings: dict[int, np.ndarray] = {
            int(r): np.where(world.mule_ring == r)[0] for r in np.unique(world.mule_ring[world.mule_ring >= 0])
        }
        # Each ring operates 1-2 shared "device farm" handsets.
        self.ring_devices: dict[int, list[str]] = {
            r: [world.new_device(None, world.start_ts) for _ in range(int(self.rng.integers(1, 3)))] for r in self.rings
        }
        self.non_mules = np.where(~world.is_mule)[0]

    # ------------------------------------------------------------------ helpers
    def _new_case(self) -> int:
        self.case_seq += 1
        return self.case_seq

    def _ev(
        self,
        *,
        ts: float,
        cust: int,
        ttype: str,
        scenario: str,
        case: int,
        cp_cust: int = -1,
        merch: int = -1,
        amount: float = np.nan,
        drain: float = np.nan,
        device: str | None = None,
        channel: int = CH_APP,
        district: int | None = None,
        lat: float | None = None,
        lon: float | None = None,
        forced_fail: str | None = None,
        pin_attempts: int = 1,
        allow_topup: bool = False,
    ) -> None:
        w = self.w
        if district is None:
            district = int(w.home_district_idx[cust])
            lat = w.home_lat[cust] + self.rng.normal(0, 0.01)
            lon = w.home_lon[cust] + self.rng.normal(0, 0.01)
        elif lat is None:
            lat = D_LAT[district] + self.rng.normal(0, 0.03)
            lon = D_LON[district] + self.rng.normal(0, 0.03)
        if not np.isnan(amount):
            amount = float(np.round(amount, -1)) if amount < 1000 else float(np.round(amount / 50) * 50)
        self.events.append(
            dict(
                ts=int(ts),
                cust=int(cust),
                ttype=T_IDX[ttype],
                cp_cust=int(cp_cust),
                merch=int(merch),
                amount=amount,
                drain=drain,
                device=device,
                channel=channel,
                district=int(district),
                lat=float(lat),
                lon=float(lon),
                source="fraud",
                is_fraud=True,
                is_ato=scenario in ATO_SCENARIOS,
                fraud_scenario=scenario,
                fraud_case_id=case,
                forced_fail=forced_fail,
                pin_attempts=pin_attempts,
                allow_topup=allow_topup,
            )
        )

    def _victim(self, t0: int, min_history: int = 3, exclude_night: bool = False) -> int | None:
        w = self.w
        for _ in range(50):
            v = int(self.rng.choice(self.non_mules))
            if w.active_from[v] > t0 - 7 * DAY or t0 >= w.end_ts:
                continue
            if self.n_events[v] < min_history and self.rng.random() < 0.9:
                continue
            if exclude_night and self.night_share[v] > 0.02:
                continue
            return v
        return None

    def _own_device(self, v: int, t: int) -> str:
        w = self.w
        if w.device_change_ts[v] > 0 and t >= w.device_change_ts[v]:
            return w.secondary_device[v]
        return w.primary_device[v]

    def _mule(self, t: int | None = None) -> int:
        """A mule account that already exists at time ``t``."""
        pool = self.mules
        if t is not None:
            active = pool[self.w.active_from[pool] <= t - DAY]
            pool = active if len(active) else pool
        return int(self.rng.choice(pool))

    def _ring_of(self, mule: int) -> int:
        return int(self.w.mule_ring[mule])

    def _agent_in(self, district: int) -> int:
        w = self.w
        local = w.agent_idx[w.merchant_district_idx[w.agent_idx] == district]
        return int(self.rng.choice(local if len(local) else w.agent_idx))

    def _attacker_device(self) -> str:
        # half of attacks use a ring device farm handset, half a fresh handset
        if self.rng.random() < 0.5 and self.ring_devices:
            r = int(self.rng.choice(list(self.ring_devices)))
            return str(self.rng.choice(self.ring_devices[r]))
        return self.w.new_device(None, self.w.start_ts)

    def _t0(self) -> int:
        w = self.w
        return int(self.rng.uniform(w.start_ts + 8 * DAY, w.end_ts - 2 * DAY))

    def _last_legit(self, v: int, t: int) -> tuple[int, int, str | None] | None:
        s, e = self.tl_start[v], self.tl_end[v]
        if e <= s:
            return None
        pos = s + np.searchsorted(self.tl_ts[s:e], t, side="right") - 1
        if pos < s:
            return None
        return int(self.tl_ts[pos]), int(self.tl_district[pos]), self.tl_device[pos]

    def _drain_targets(self, v_district: int, t: int) -> tuple[str, int, int]:
        """Where does stolen money go: P2P to a mule or cash-out at an agent."""
        if self.rng.random() < 0.65:
            return "P2P", self._mule(t), -1
        return "CASH_OUT", -1, self._agent_in(v_district)

    # ---------------------------------------------------------------- scenarios
    def velocity_attack(self) -> None:
        t = self._t0()
        v = self._victim(t)
        if v is None:
            return
        case = self._new_case()
        dev = self._attacker_device()
        district = (
            int(self.w.home_district_idx[v]) if self.rng.random() < 0.4 else int(self.rng.integers(0, len(DISTRICTS)))
        )
        for _ in range(int(self.rng.integers(4, 12))):
            t += int(self.rng.exponential(45)) + 5
            ttype, cp, merch = self._drain_targets(district, t)
            self._ev(
                ts=t,
                cust=v,
                ttype=ttype,
                cp_cust=cp,
                merch=merch,
                drain=self.rng.uniform(0.25, 0.6),
                device=dev,
                district=district,
                scenario="velocity_attack",
                case=case,
            )

    def sudden_high_value(self) -> None:
        t = self._t0()
        v = self._victim(t)
        if v is None:
            return
        case = self._new_case()
        amt = self.median_amount[v] * self.rng.uniform(4, 18)
        self._ev(
            ts=t,
            cust=v,
            ttype="P2P",
            cp_cust=self._mule(t),
            amount=min(amt, 49_000),
            device=self._own_device(v, t),
            scenario="sudden_high_value",
            case=case,
            allow_topup=True,
        )

    def new_device_high_value(self) -> None:
        t = self._t0()
        v = self._victim(t)
        if v is None:
            return
        case = self._new_case()
        dev = self.w.new_device(None, t)
        district = int(self.w.home_district_idx[v])
        for _ in range(int(self.rng.integers(1, 3))):
            t += int(self.rng.integers(60, 1800))
            ttype, cp, merch = self._drain_targets(district, t)
            self._ev(
                ts=t,
                cust=v,
                ttype=ttype,
                cp_cust=cp,
                merch=merch,
                drain=self.rng.uniform(0.6, 0.95),
                device=dev,
                scenario="new_device_high_value",
                case=case,
            )

    def impossible_travel(self) -> None:
        t = self._t0()
        v = self._victim(t, min_history=5)
        if v is None:
            return
        last = self._last_legit(v, t)
        if last is None:
            return
        t_last, d_last, _ = last
        far = [
            i
            for i, d in enumerate(DISTRICTS)
            if haversine_km(d.lat, d.lon, DISTRICTS[d_last].lat, DISTRICTS[d_last].lon) > 200
        ]
        if not far:
            return
        case = self._new_case()
        d_far = int(self.rng.choice(far))
        t = t_last + int(self.rng.uniform(8, 50) * 60)
        dev = self._own_device(v, t) if self.rng.random() < 0.5 else self._attacker_device()
        for _ in range(int(self.rng.integers(1, 4))):
            ttype, cp, merch = self._drain_targets(d_far, t)
            self._ev(
                ts=t,
                cust=v,
                ttype=ttype,
                cp_cust=cp,
                merch=merch,
                drain=self.rng.uniform(0.4, 0.9),
                device=dev,
                district=d_far,
                scenario="impossible_travel",
                case=case,
            )
            t += int(self.rng.integers(60, 900))

    def device_sharing_ring(self) -> None:
        if not self.rings:
            return
        r = int(self.rng.choice(list(self.rings)))
        members = list(self.rings[r])
        extra = list(self.rng.choice(self.mules, int(self.rng.integers(1, 4))))  # accounts borrowed from other rings
        accounts = list(dict.fromkeys(members + extra))
        t = self._t0()
        accounts = [a for a in accounts if self.w.active_from[a] <= t - DAY]
        if len(accounts) < 2:
            return
        head = accounts[0]
        dev = str(self.rng.choice(self.ring_devices[r]))
        case = self._new_case()
        district = int(self.w.home_district_idx[head])
        for acc in accounts:
            t += int(self.rng.integers(120, 5400))
            if acc == head:
                self._ev(
                    ts=t,
                    cust=acc,
                    ttype="CASH_OUT",
                    merch=self._agent_in(district),
                    drain=self.rng.uniform(0.8, 0.98),
                    device=dev,
                    district=district,
                    scenario="device_sharing_ring",
                    case=case,
                )
            else:
                self._ev(
                    ts=t,
                    cust=acc,
                    ttype="P2P",
                    cp_cust=head,
                    drain=self.rng.uniform(0.8, 0.98),
                    device=dev,
                    district=district,
                    scenario="device_sharing_ring",
                    case=case,
                )

    def suspicious_merchant(self) -> None:
        w = self.w
        m = int(self.rng.choice(w.fraud_merchant_idx))
        reg = int(pd.Timestamp(w.merchants.loc[m, "registration_date"]).timestamp())
        t = reg + int(self.rng.uniform(1, 25) * DAY)  # scam shops operate soon after opening
        if t >= w.end_ts - DAY:
            return
        case = self._new_case()
        for _ in range(int(self.rng.integers(3, 10))):
            v = self._victim(t)
            if v is None:
                continue
            tt = t + int(self.rng.uniform(0, 3 * DAY))
            if tt >= w.end_ts:
                continue
            amt = self.rng.lognormal(np.log(3500), 0.6)
            self._ev(
                ts=tt,
                cust=v,
                ttype="MERCHANT_PAYMENT",
                merch=m,
                amount=amt,
                device=self._own_device(v, tt),
                scenario="suspicious_merchant",
                case=case,
                allow_topup=True,
            )

    def rapid_fund_movement(self) -> None:
        t = self._t0()
        v = self._victim(t)
        if v is None:
            return
        case = self._new_case()
        mule = self._mule(t)
        amount_in = self.rng.uniform(5_000, 40_000)
        self._ev(
            ts=t,
            cust=v,
            ttype="P2P",
            cp_cust=mule,
            amount=amount_in,
            device=self._own_device(v, t),
            scenario="rapid_fund_movement",
            case=case,
            allow_topup=True,
        )
        t2 = t + int(self.rng.uniform(2, 25) * 60)
        mule_dev = self.w.primary_device[mule] if self.rng.random() < 0.6 else self._attacker_device()
        ring = self.rings.get(self._ring_of(mule), np.array([mule]))
        others = [x for x in ring if x != mule and self.w.active_from[x] <= t]
        if others and self.rng.random() < 0.6:
            self._ev(
                ts=t2,
                cust=mule,
                ttype="P2P",
                cp_cust=int(self.rng.choice(others)),
                amount=amount_in * self.rng.uniform(0.9, 0.99),
                device=mule_dev,
                scenario="rapid_fund_movement",
                case=case,
            )
        else:
            d = int(self.w.home_district_idx[mule])
            self._ev(
                ts=t2,
                cust=mule,
                ttype="CASH_OUT",
                merch=self._agent_in(d),
                amount=amount_in * self.rng.uniform(0.9, 0.98),
                device=mule_dev,
                district=d,
                scenario="rapid_fund_movement",
                case=case,
            )

    def structuring(self) -> None:
        t = self._t0()
        actor = self._mule(t) if self.rng.random() < 0.6 else self._victim(t)
        if actor is None:
            return
        case = self._new_case()
        d = int(self.w.home_district_idx[actor])
        dev = self._own_device(actor, t) if not self.w.is_mule[actor] else self.w.primary_device[actor]
        n = int(self.rng.integers(3, 6))
        for _ in range(n):  # split deposits across agents
            t += int(self.rng.integers(600, 4 * 3600))
            self._ev(
                ts=t,
                cust=actor,
                ttype="CASH_IN",
                merch=self._agent_in(d),
                channel=CH_AGENT,
                amount=self.threshold * self.rng.uniform(0.88, 0.995),
                device=None,
                scenario="structuring",
                case=case,
            )
        for _ in range(int(self.rng.integers(2, 4))):
            t += int(self.rng.integers(300, 3 * 3600))
            self._ev(
                ts=t,
                cust=actor,
                ttype="P2P",
                cp_cust=self._mule(t),
                amount=self.threshold * self.rng.uniform(0.88, 0.995),
                device=dev,
                scenario="structuring",
                case=case,
            )

    def mule_collection(self) -> None:
        mule = self._mule()
        t = max(self._t0(), int(self.w.active_from[mule]) + DAY)
        if t >= self.w.end_ts - 6 * DAY:
            return
        case = self._new_case()
        for _ in range(int(self.rng.integers(5, 15))):
            tt = t + int(self.rng.uniform(0, 5 * DAY))
            v = self._victim(tt)
            if v is None:
                continue
            amt = self.rng.lognormal(np.log(2500), 0.6)
            self._ev(
                ts=tt,
                cust=v,
                ttype="P2P",
                cp_cust=mule,
                amount=amt,
                device=self._own_device(v, tt),
                scenario="mule_collection",
                case=case,
                allow_topup=True,
            )
        d = int(self.w.home_district_idx[mule])
        t_out = t + int(5 * DAY + self.rng.uniform(1, 12) * 3600)
        for _ in range(int(self.rng.integers(1, 3))):
            self._ev(
                ts=t_out,
                cust=mule,
                ttype="CASH_OUT",
                merch=self._agent_in(d),
                drain=0.95,
                device=self.w.primary_device[mule],
                district=d,
                scenario="mule_collection",
                case=case,
            )
            t_out += int(self.rng.integers(300, 7200))

    def failed_auth_then_success(self) -> None:
        t = self._t0()
        v = self._victim(t)
        if v is None:
            return
        case = self._new_case()
        dev = self._attacker_device()
        district = int(self.w.home_district_idx[v])
        attempts = int(self.rng.integers(2, 6))
        target_type, cp, merch = self._drain_targets(district, t)
        for i in range(attempts):
            t += int(self.rng.integers(10, 90))
            self._ev(
                ts=t,
                cust=v,
                ttype=target_type,
                cp_cust=cp,
                merch=merch,
                drain=self.rng.uniform(0.5, 0.9),
                device=dev,
                forced_fail="WRONG_PIN",
                pin_attempts=i + 1,
                scenario="failed_auth_then_success",
                case=case,
            )
        t += int(self.rng.integers(10, 120))
        self._ev(
            ts=t,
            cust=v,
            ttype=target_type,
            cp_cust=cp,
            merch=merch,
            drain=self.rng.uniform(0.5, 0.9),
            device=dev,
            pin_attempts=attempts + 1,
            scenario="failed_auth_then_success",
            case=case,
        )

    def unusual_recipient_scam(self) -> None:
        t = self._t0()
        v = self._victim(t)
        if v is None:
            return
        case = self._new_case()
        amt = self.median_amount[v] * self.rng.uniform(0.8, 4.0)
        dev = self._own_device(v, t)
        self._ev(
            ts=t,
            cust=v,
            ttype="P2P",
            cp_cust=self._mule(t),
            amount=amt,
            device=dev,
            channel=CH_APP if self.w.device_registry[dev]["device_type"] == "smartphone" else CH_USSD,
            scenario="unusual_recipient_scam",
            case=case,
            allow_topup=True,
        )

    def abnormal_nighttime(self) -> None:
        t = self._t0()
        v = self._victim(t, exclude_night=True)
        if v is None:
            return
        case = self._new_case()
        local_midnight = ((t + self.w.local_offset_s) // DAY) * DAY - self.w.local_offset_s
        t = int(local_midnight + self.rng.uniform(0.5, 4.5) * 3600)
        dev = self._own_device(v, t) if self.rng.random() < 0.6 else self._attacker_device()
        district = int(self.w.home_district_idx[v])
        for _ in range(int(self.rng.integers(2, 6))):
            ttype, cp, merch = self._drain_targets(district, t)
            self._ev(
                ts=t,
                cust=v,
                ttype=ttype,
                cp_cust=cp,
                merch=merch,
                drain=self.rng.uniform(0.3, 0.8),
                device=dev,
                scenario="abnormal_nighttime",
                case=case,
            )
            t += int(self.rng.integers(60, 1200))

    # -------------------------------------------------------------------- run
    def run(self, target_fraud_txns: int, weights: dict[str, float]) -> pd.DataFrame:
        handlers: dict[str, Callable[[], None]] = {s: getattr(self, s) for s in FRAUD_SCENARIOS}
        w = np.array([weights.get(s, 0.0) for s in FRAUD_SCENARIOS])
        denom = float((w * np.array([AVG_TXNS_PER_CASE[s] for s in FRAUD_SCENARIOS])).sum())
        n_cases = np.maximum(1, np.round(target_fraud_txns * w / denom)).astype(int)
        n_cases[w == 0] = 0
        plan = np.repeat(np.arange(len(FRAUD_SCENARIOS)), n_cases)
        self.rng.shuffle(plan)
        for si in plan:
            handlers[FRAUD_SCENARIOS[si]]()
        df = pd.DataFrame(self.events)
        if df.empty:
            return df
        df = df[(df["ts"] >= self.w.start_ts) & (df["ts"] < self.w.end_ts)].copy()
        # attach channel for feature-phone owners using their own device
        own_ussd = df["device"].map(
            lambda d: isinstance(d, str) and self.w.device_registry[d]["device_type"] == "feature_phone"
        )
        df.loc[own_ussd & (df["channel"] == CH_APP), "channel"] = CH_USSD
        return df

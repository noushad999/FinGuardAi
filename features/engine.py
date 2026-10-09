"""Point-in-time feature engine shared by training and real-time inference.

For every transaction the engine:

1. loads the state of the entities involved (customer wallet, receiving
   customer, device, merchant) from a :class:`~features.store.StateStore`,
2. computes features using *only* state accumulated from earlier events
   (the current event is never part of its own features),
3. optionally commits the event into the state.

Offline training replays the historical stream through exactly this code
(``features.offline``), so a feature value seen in training is, by construction,
the value the online service would have computed at that moment. This removes
the classic training/serving skew and label leakage from future data.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from common.geo import haversine_km
from features.definitions import FEATURE_NAMES
from features.reference import ReferenceData
from features.store import InMemoryStateStore, State, StateStore
from simulator.constants import TRANSACTION_TYPES

DAY = 86_400.0
WINDOWS = (60.0, 300.0, 3600.0, DAY)
NO_PREV_SECONDS = 1e7


@dataclass
class EngineConfig:
    local_utc_offset_hours: int = 6
    structuring_threshold: float = 25_000.0
    max_known_recipients: int = 300
    max_known_devices: int = 20
    max_recent_events: int = 400

    @classmethod
    def from_settings(cls, settings: dict) -> EngineConfig:
        sim, feat = settings["simulation"], settings["features"]
        return cls(
            local_utc_offset_hours=int(sim["local_utc_offset_hours"]),
            structuring_threshold=float(sim["structuring_threshold"]),
            max_known_recipients=int(feat["max_known_recipients"]),
            max_known_devices=int(feat["max_known_devices"]),
        )


def to_epoch_seconds(value: Any) -> float:
    """Accept epoch seconds, ISO-8601 strings, datetimes or pandas Timestamps."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    if isinstance(value, (str, datetime)) or hasattr(value, "dtype"):
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.timestamp()
    raise TypeError(f"Unsupported timestamp type: {type(value)!r}")


def _new_customer_state(ts: float) -> State:
    return {
        "n": 0,
        "n_fail": 0,
        "first_ts": ts,
        "last_ts": None,
        "last_lat": None,
        "last_lon": None,
        "amt_n": 0,
        "amt_mean": 0.0,
        "amt_m2": 0.0,
        "recent": [],  # [ts, amount, failed(0/1), wrong_pin(0/1), counterparty, device]
        "devices": {},
        "recipients": {},
        "districts": {},
        "hours": [0] * 24,
        "in_recent": [],  # [ts, amount]
        "in_senders": {},
        "in_n": 0,
        "last_in_ts": None,
        "last_in_amt": 0.0,
    }


def _new_device_state() -> State:
    return {"n": 0, "accounts": {}}


def _new_merchant_state() -> State:
    return {"n": 0, "customers": {}}


def _cap_dict(d: dict, key: str, cap: int) -> None:
    """Increment ``d[key]``; if the dict exceeds ``cap`` drop the least-used key."""
    d[key] = d.get(key, 0) + 1
    if len(d) > cap:
        victim = min((k for k in d if k != key), key=d.__getitem__)
        del d[victim]


class FeatureEngine:
    """Computes point-in-time features and maintains entity state."""

    def __init__(
        self,
        store: StateStore | None = None,
        reference: ReferenceData | None = None,
        config: EngineConfig | None = None,
    ) -> None:
        self.store = store if store is not None else InMemoryStateStore()
        self.reference = reference or ReferenceData()
        self.cfg = config or EngineConfig()
        self._offset = self.cfg.local_utc_offset_hours * 3600

    # ------------------------------------------------------------------ API
    def compute(self, txn: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
        """Return (features, context). ``context`` is passed to :meth:`commit`."""
        ev = self._normalise(txn)
        keys = [f"c:{ev['customer_id']}"]
        if ev["receiver_customer"]:
            keys.append(f"c:{ev['receiver_customer']}")
        if ev["device_id"]:
            keys.append(f"d:{ev['device_id']}")
        if ev["merchant_id"]:
            keys.append(f"m:{ev['merchant_id']}")
        states = self.store.get_many(keys)
        feats = self._features(ev, states)
        return feats, {"event": ev, "states": states}

    def commit(self, context: dict[str, Any]) -> None:
        """Fold the event into entity state (after the decision was taken)."""
        ev, states = context["event"], context["states"]
        updated = self._update(ev, states)
        self.store.put_many(updated)

    def process(self, txn: dict[str, Any], commit: bool = True) -> dict[str, float]:
        feats, ctx = self.compute(txn)
        if commit:
            self.commit(ctx)
        return feats

    def customer_state(self, customer_id: str) -> State | None:
        return self.store.get_many([f"c:{customer_id}"]).get(f"c:{customer_id}")

    # ------------------------------------------------------------ internals
    def _normalise(self, txn: dict[str, Any]) -> dict[str, Any]:
        ts = to_epoch_seconds(txn["event_time"])
        ttype = str(txn["transaction_type"])
        customer_id = str(txn["customer_id"])
        receiver = txn.get("receiver_id")
        receiver = receiver if isinstance(receiver, str) and receiver else None
        sender = txn.get("sender_id")
        receiver_customer = receiver if (ttype == "P2P" and receiver and receiver.startswith("CUS_")) else None
        counterparty = sender if ttype == "CASH_IN" else receiver
        meta = txn.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        device = txn.get("device_id")
        merchant = txn.get("merchant_id")
        status = txn.get("status") or "SUCCESS"
        failure = txn.get("failure_reason")
        balance = txn.get("balance_before")
        return {
            "ts": ts,
            "customer_id": customer_id,
            "type": ttype,
            "amount": float(txn["amount"]),
            "channel": str(txn.get("channel") or "APP"),
            "device_id": device if isinstance(device, str) and device else None,
            "merchant_id": merchant if isinstance(merchant, str) and merchant else None,
            "receiver_customer": receiver_customer,
            "counterparty": counterparty if isinstance(counterparty, str) else None,
            "district": txn.get("district"),
            "lat": _float_or_none(txn.get("latitude")),
            "lon": _float_or_none(txn.get("longitude")),
            "failed": status == "FAILED",
            "wrong_pin": failure == "WRONG_PIN",
            "balance_before": _float_or_none(balance),
            "pin_attempts": int(meta.get("pin_attempts", 1) or 1),
        }

    def _features(self, ev: dict[str, Any], states: dict[str, State | None]) -> dict[str, float]:
        cfg = self.cfg
        ts, amount, ttype = ev["ts"], ev["amount"], ev["type"]
        c = states.get(f"c:{ev['customer_id']}") or _new_customer_state(ts)
        f: dict[str, float] = {}
        is_debit = ttype != "CASH_IN"

        # ---- transaction level ----
        f["amount"] = amount
        f["log_amount"] = math.log1p(amount)
        for t in TRANSACTION_TYPES:
            f[f"type_{t.lower()}"] = 1.0 if ttype == t else 0.0
        ch = ev["channel"]
        f["channel_app"] = 1.0 if ch == "APP" else 0.0
        f["channel_ussd"] = 1.0 if ch == "USSD" else 0.0
        f["channel_agent"] = 1.0 if ch == "AGENT" else 0.0
        m_id = ev["merchant_id"]
        ref = self.reference
        if m_id:
            f["merchant_risk_level"] = float(ref.merchant_risk.get(m_id, 1))
            reg = ref.merchant_reg_ts.get(m_id)
            f["merchant_age_days"] = max(0.0, (ts - reg) / DAY) if reg is not None else 0.0
            ms = states.get(f"m:{m_id}") or _new_merchant_state()
            f["merchant_txn_count"] = float(ms["n"])
            f["merchant_unique_customers"] = float(len(ms["customers"]))
        else:
            f["merchant_risk_level"] = -1.0
            f["merchant_age_days"] = -1.0
            f["merchant_txn_count"] = -1.0
            f["merchant_unique_customers"] = -1.0
        thr = cfg.structuring_threshold
        f["is_near_threshold"] = 1.0 if 0.85 * thr <= amount < thr else 0.0
        bal = ev["balance_before"]
        f["amount_to_balance_ratio"] = min(50.0, amount / max(bal, 1.0)) if (is_debit and bal is not None) else 0.0
        f["pin_attempts"] = float(ev["pin_attempts"])

        # ---- behavioural (rolling windows over prior events) ----
        n = c["n"]
        f["n_prior_txns"] = float(n)
        reg = ref.customer_reg_ts.get(ev["customer_id"], c["first_ts"])
        f["account_age_days"] = max(0.0, (ts - reg) / DAY)
        cnt = [0, 0, 0, 0]
        sum_1h = sum_24h = 0.0
        fail_1h = pin_1h = 0
        rcpts: set[str] = set()
        devs: set[str] = set()
        for r_ts, r_amt, r_fail, r_pin, r_cp, r_dev in c["recent"]:
            age = ts - r_ts
            if age < 0 or age > DAY:
                continue
            cnt[3] += 1
            sum_24h += r_amt
            if r_cp:
                rcpts.add(r_cp)
            if r_dev:
                devs.add(r_dev)
            if age <= 3600:
                cnt[2] += 1
                sum_1h += r_amt
                fail_1h += r_fail
                pin_1h += r_pin
                if age <= 300:
                    cnt[1] += 1
                    if age <= 60:
                        cnt[0] += 1
        f["txn_count_1m"], f["txn_count_5m"], f["txn_count_1h"], f["txn_count_24h"] = map(float, cnt)
        f["amount_sum_1h"] = sum_1h
        f["amount_sum_24h"] = sum_24h
        a_n = c["amt_n"]
        mean_log = c["amt_mean"]
        std_log = math.sqrt(c["amt_m2"] / (a_n - 1)) if a_n > 1 else 0.0
        f["avg_amount_hist"] = math.expm1(mean_log) if a_n else 0.0
        f["amount_std_hist"] = std_log
        log_amt = f["log_amount"]
        f["amount_zscore"] = (log_amt - mean_log) / max(std_log, 0.5) if (a_n >= 3 and is_debit) else 0.0
        f["amount_to_avg_ratio"] = amount / max(f["avg_amount_hist"], 1.0) if (a_n >= 1 and is_debit) else 1.0
        f["unique_recipients_24h"] = float(len(rcpts))
        f["unique_devices_24h"] = float(len(devs))
        f["failed_ratio_hist"] = c["n_fail"] / n if n else 0.0
        f["failed_count_1h"] = float(fail_1h)
        f["wrong_pin_count_1h"] = float(pin_1h)
        days_active = max(1.0, (ts - c["first_ts"]) / DAY)
        daily_rate = n / days_active if n else 0.0
        f["velocity_ratio_24h"] = cnt[3] / max(daily_rate, 0.1)

        # ---- temporal ----
        local = ts + self._offset
        hour = int((local % DAY) // 3600)
        dow = int(((local // DAY) + 3) % 7)  # 1970-01-01 was Thursday -> Monday=0
        f["hour_local"] = float(hour)
        f["day_of_week"] = float(dow)
        f["is_weekend"] = 1.0 if dow in (4, 5) else 0.0
        f["is_night"] = 1.0 if hour < 5 else 0.0
        last_ts = c["last_ts"]
        # abs(): a late (out-of-order) event is as close in time to its neighbour as an in-order one
        gap = abs(ts - last_ts) if last_ts is not None else NO_PREV_SECONDS
        f["log_seconds_since_prev"] = math.log1p(max(gap, 0.0))
        hours = c["hours"]
        near = hours[hour] + hours[(hour - 1) % 24] + hours[(hour + 1) % 24]
        f["hour_frequency"] = (near + 0.125) / (sum(hours) + 1.0)

        # ---- device ----
        dev = ev["device_id"]
        if dev:
            ds = states.get(f"d:{dev}") or _new_device_state()
            f["device_missing"] = 0.0
            f["is_new_device"] = 0.0 if dev in c["devices"] else 1.0
            f["device_txn_count"] = float(ds["n"])
            accounts = ds["accounts"]
            f["accounts_per_device"] = float(len(accounts) + (0 if ev["customer_id"] in accounts else 1))
        else:
            f["device_missing"] = 1.0
            f["is_new_device"] = 0.0
            f["device_txn_count"] = 0.0
            f["accounts_per_device"] = 0.0
        f["customer_device_count"] = float(len(c["devices"]))

        # ---- geographic ----
        dist = ev["district"]
        districts = c["districts"]
        total_d = sum(districts.values())
        f["is_new_district"] = 1.0 if (total_d and dist not in districts) else 0.0
        f["district_frequency"] = districts.get(dist, 0) / total_d if total_d else 1.0
        if c["last_lat"] is not None and ev["lat"] is not None and last_ts is not None:
            km = haversine_km(c["last_lat"], c["last_lon"], ev["lat"], ev["lon"])
            f["distance_from_prev_km"] = km
            f["speed_from_prev_kmh"] = min(2000.0, km / max(gap / 3600.0, 1 / 60)) if km > 5 else 0.0
        else:
            f["distance_from_prev_km"] = 0.0
            f["speed_from_prev_kmh"] = 0.0

        # ---- network ----
        cp = ev["counterparty"]
        f["sender_out_degree"] = float(len(c["recipients"]))
        f["is_new_recipient"] = (
            1.0 if (ttype in ("P2P", "MERCHANT_PAYMENT") and cp and cp not in c["recipients"]) else 0.0
        )
        rc_id = ev["receiver_customer"]
        if rc_id:
            rc = states.get(f"c:{rc_id}") or _new_customer_state(ts)
            f["receiver_in_degree"] = float(len(rc["in_senders"]))
            f["receiver_inflow_count_24h"] = float(sum(1 for t_, _ in rc["in_recent"] if 0 <= ts - t_ <= DAY))
            rreg = ref.customer_reg_ts.get(rc_id, rc["first_ts"])
            f["receiver_account_age_days"] = max(0.0, (ts - rreg) / DAY)
            f["receiver_n_prior_txns"] = float(rc["n"])
        else:
            f["receiver_in_degree"] = -1.0
            f["receiver_inflow_count_24h"] = -1.0
            f["receiver_account_age_days"] = -1.0
            f["receiver_n_prior_txns"] = -1.0
        last_in = c["last_in_ts"]
        if last_in is not None and ts - last_in <= DAY:
            f["log_minutes_since_last_inflow"] = math.log1p(max(0.0, ts - last_in) / 60.0)
            f["outflow_to_last_inflow_ratio"] = amount / max(c["last_in_amt"], 1.0) if is_debit else 0.0
        else:
            f["log_minutes_since_last_inflow"] = math.log1p(DAY * 30 / 60.0)
            f["outflow_to_last_inflow_ratio"] = 0.0
        f["inflow_sum_24h"] = float(sum(a for t_, a in c["in_recent"] if 0 <= ts - t_ <= DAY))
        return f

    def _update(self, ev: dict[str, Any], states: dict[str, State | None]) -> dict[str, State]:
        cfg = self.cfg
        ts = ev["ts"]
        ck = f"c:{ev['customer_id']}"
        c = states.get(ck) or _new_customer_state(ts)
        failed = ev["failed"]
        ttype = ev["type"]
        cp = ev["counterparty"]
        dev = ev["device_id"]
        out: dict[str, State] = {}

        recent = [r for r in c["recent"] if ts - r[0] <= DAY]
        recent.append([ts, ev["amount"], 1 if failed else 0, 1 if ev["wrong_pin"] else 0, cp or "", dev or ""])
        c["recent"] = recent[-cfg.max_recent_events :]
        c["n"] += 1
        if failed:
            c["n_fail"] += 1
        if ttype != "CASH_IN":
            x = math.log1p(ev["amount"])
            c["amt_n"] += 1
            delta = x - c["amt_mean"]
            c["amt_mean"] += delta / c["amt_n"]
            c["amt_m2"] += delta * (x - c["amt_mean"])
        if dev:
            _cap_dict(c["devices"], dev, cfg.max_known_devices)
        if cp and ttype != "CASH_IN" and not failed:
            _cap_dict(c["recipients"], cp, cfg.max_known_recipients)
        if ev["district"]:
            _cap_dict(c["districts"], ev["district"], 64)
        hour = int(((ts + self._offset) % DAY) // 3600)
        c["hours"][hour] += 1
        c["last_ts"] = ts
        if ev["lat"] is not None:
            c["last_lat"], c["last_lon"] = ev["lat"], ev["lon"]
        if ttype == "CASH_IN" and not failed:
            c["in_recent"] = [r for r in c["in_recent"] if ts - r[0] <= DAY] + [[ts, ev["amount"]]]
            c["in_n"] += 1
            c["last_in_ts"], c["last_in_amt"] = ts, ev["amount"]
        out[ck] = c

        rc_id = ev["receiver_customer"]
        if rc_id and not failed:
            rk = f"c:{rc_id}"
            rc = states.get(rk) or _new_customer_state(ts)
            rc["in_recent"] = [r for r in rc["in_recent"] if ts - r[0] <= DAY] + [[ts, ev["amount"]]]
            rc["in_n"] += 1
            _cap_dict(rc["in_senders"], ev["customer_id"], cfg.max_known_recipients)
            rc["last_in_ts"], rc["last_in_amt"] = ts, ev["amount"]
            out[rk] = rc
        if dev:
            dk = f"d:{dev}"
            ds = states.get(dk) or _new_device_state()
            ds["n"] += 1
            _cap_dict(ds["accounts"], ev["customer_id"], 200)
            out[dk] = ds
        m_id = ev["merchant_id"]
        if m_id:
            mk = f"m:{m_id}"
            ms = states.get(mk) or _new_merchant_state()
            ms["n"] += 1
            _cap_dict(ms["customers"], ev["customer_id"], 500)
            out[mk] = ms
        return out


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def feature_vector(features: dict[str, float], names: list[str] | None = None) -> list[float]:
    """Order a feature dict according to the model's schema."""
    names = names or FEATURE_NAMES
    return [float(features[n]) for n in names]

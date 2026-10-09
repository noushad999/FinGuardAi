"""Chronological wallet ledger.

Runs once over all events in event-time order and resolves everything that
depends on account state:

* balance_before / balance_after of the customer wallet,
* insufficient balance, PIN errors, network timeouts, per-transaction limits,
* cash-out fees,
* realistic top-ups: when a customer lacks funds they usually cash-in shortly
  before paying (an extra CASH_IN event is inserted between their previous
  activity and the payment),
* balance-relative fraud drains (``drain`` = fraction of the current balance).

A plain Python loop is used deliberately: the logic is sequential by nature,
and ~1-2 µs per event keeps even the 10M preset tractable.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from simulator.constants import CASH_OUT_FEE_RATE, TRANSACTION_TYPES, TXN_LIMITS
from simulator.entities import DAY, World
from simulator.normal import CH_AGENT, T_IDX

FAIL_REASONS = ("", "INSUFFICIENT_BALANCE", "WRONG_PIN", "NETWORK_TIMEOUT", "LIMIT_EXCEEDED")


def run_ledger(world: World, ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve balances/failures. Returns (resolved events, inserted top-up events)."""
    rng = world.rng
    ev = ev.sort_values(["ts", "seq"], kind="stable").reset_index(drop=True)
    n = len(ev)
    ts = ev["ts"].to_numpy().tolist()
    cust = ev["cust"].to_numpy().tolist()
    ttype = ev["ttype"].to_numpy().tolist()
    cp = ev["cp_cust"].to_numpy().tolist()
    amount_in = ev["amount"].to_numpy().tolist()
    drain = ev["drain"].to_numpy().tolist()
    forced = ev["forced_fail"].tolist()
    allow_topup = ev["allow_topup"].to_numpy().tolist()
    is_normal = (ev["source"] == "normal").to_numpy().tolist()

    u_pin = rng.random(n).tolist()
    u_net = rng.random(n).tolist()
    u_top = rng.random(n).tolist()
    u_misc = rng.random(n).tolist()

    bal = world.initial_balance.astype(float).tolist()
    last_change = [-1] * world.n_customers
    churn = world.churn_ts.tolist()
    topup_prob = world.topup_prob.tolist()
    limits = [TXN_LIMITS[t] for t in TRANSACTION_TYPES]
    CASH_IN, CASH_OUT, P2P = T_IDX["CASH_IN"], T_IDX["CASH_OUT"], T_IDX["P2P"]

    status = [1] * n  # 1 success, 0 failed
    reason = [0] * n
    amount = [0.0] * n
    fee = [0.0] * n
    bb = [0.0] * n
    ba = [0.0] * n
    topups: list[tuple[int, int, float, float, float, int]] = []

    for i in range(n):
        c = cust[i]
        t = ts[i]
        ty = ttype[i]
        b = bal[c]
        p_net = 0.006
        if churn[c] > 0 and churn[c] - 30 * DAY < t:
            p_net = 0.05  # degraded experience precedes churn
        amt = amount_in[i]
        if ty == CASH_IN:
            amount[i] = amt
            bb[i] = b
            if u_net[i] < p_net:
                status[i], reason[i] = 0, 3
                ba[i] = b
            else:
                b += amt
                ba[i] = b
            bal[c] = b
            last_change[c] = t
            continue

        d = drain[i]
        if d == d:  # not NaN -> balance-relative drain
            amt = round(b * d, -1)
            if amt < 100:
                amt = float(round(1000 + 4000 * u_misc[i], -1))  # blind attempt on an empty wallet
        f = 0.0
        failed = 0
        if forced[i] == "WRONG_PIN" or is_normal[i] and u_pin[i] < 0.006:
            failed = 2
        elif u_net[i] < p_net:
            failed = 3
        elif amt > limits[ty]:
            failed = 4
        else:
            f = round(amt * CASH_OUT_FEE_RATE, 2) if ty == CASH_OUT else 0.0
            need = amt + f
            if need > b:
                if allow_topup[i] and u_top[i] < topup_prob[c]:
                    top = math.ceil((need - b + 2000 * u_misc[i]) / 500.0) * 500.0
                    gap = t - last_change[c] if last_change[c] >= 0 else 7200
                    dt = max(1, int(min(300 + 3300 * u_misc[i], gap * 0.5)))
                    topups.append((t - dt, c, top, b, b + top, i))
                    b += top
                elif ty == CASH_OUT and b >= 0.5 * need and b >= 100:
                    amt = math.floor(b / (1 + CASH_OUT_FEE_RATE) / 10.0) * 10.0
                    f = round(amt * CASH_OUT_FEE_RATE, 2)
                    need = amt + f
                else:
                    failed = 1
        amount[i] = amt
        bb[i] = b
        if failed:
            status[i], reason[i] = 0, failed
            fee[i] = 0.0
            ba[i] = b
        else:
            b -= amt + f
            fee[i] = f
            ba[i] = b
            if ty == P2P and cp[i] >= 0:
                r = cp[i]
                bal[r] += amt
                last_change[r] = t
        bal[c] = b
        last_change[c] = t

    ev["amount"] = np.round(amount, 2)
    ev["fee"] = fee
    ev["success"] = np.array(status, dtype=bool)
    ev["failure_reason"] = [FAIL_REASONS[r] or None for r in reason]
    ev["balance_before"] = np.round(bb, 2)
    ev["balance_after"] = np.round(ba, 2)

    if topups:
        tp = pd.DataFrame(topups, columns=["ts", "cust", "amount", "balance_before", "balance_after", "trigger"])
        trig = ev.loc[tp["trigger"].to_numpy()]
        tp["ttype"] = CASH_IN
        tp["cp_cust"] = -1
        tp["merch"] = world.home_agents[tp["cust"].to_numpy(), 0]
        tp["drain"] = np.nan
        tp["device"] = None
        tp["channel"] = CH_AGENT
        for col in ("district", "lat", "lon"):
            tp[col] = trig[col].to_numpy()
        tp["source"] = "topup"
        tp["fee"] = 0.0
        tp["success"] = True
        tp["failure_reason"] = None
        tp = tp.drop(columns=["trigger"])
    else:
        tp = pd.DataFrame()
    return ev, tp

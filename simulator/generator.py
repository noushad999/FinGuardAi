"""End-to-end synthetic dataset generation.

Outputs (Parquet, in ``paths.synthetic_dir``):

* ``customers.parquet``, ``devices.parquet``, ``merchants.parquet`` - entity tables
* ``transactions.parquet`` - the public event stream (what Kafka carries)
* ``transaction_labels.parquet`` - fraud ground truth + *when it was reported*
* ``customer_ground_truth.parquet`` / ``merchant_ground_truth.parquet`` - hidden
  truth for evaluation only (persona, mule ring, churn date, fraudulent merchant)
* ``generation_summary.json``

Labels are stored separately from events on purpose: in production they arrive
later from case management / chargebacks and must never leak into features.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import Settings, get_settings
from common.geo import DISTRICTS
from common.logging import get_logger
from simulator.constants import TRANSACTION_TYPES
from simulator.entities import DAY, World, build_world
from simulator.fraud import FraudInjector
from simulator.ledger import run_ledger
from simulator.normal import CH_APP, generate_normal_events
from simulator.personas import PERSONAS

log = get_logger(__name__)
CHANNEL_NAMES = np.array(["APP", "USSD", "AGENT"])
NETWORKS = np.array(["4G", "4G", "4G", "3G", "WIFI", "2G"])


@dataclass
class SyntheticDataset:
    customers: pd.DataFrame
    devices: pd.DataFrame
    merchants: pd.DataFrame
    transactions: pd.DataFrame
    labels: pd.DataFrame
    customer_truth: pd.DataFrame
    merchant_truth: pd.DataFrame
    summary: dict


def generate(settings: Settings | None = None) -> SyntheticDataset:
    """Generate a full synthetic dataset according to ``settings.simulation``."""
    settings = settings or get_settings()
    sim = settings["simulation"]
    t_start = time.time()
    n_txn = int(sim["n_transactions"])
    n_customers = max(300, int(n_txn / float(sim["transactions_per_customer"])))
    log.info("generation_started", extra={"n_transactions": n_txn, "n_customers": n_customers})

    world = build_world(
        n_customers=n_customers,
        start_date=sim["start_date"],
        days=int(sim["days"]),
        local_offset_hours=int(sim["local_utc_offset_hours"]),
        mule_fraction=float(sim["mule_fraction"]),
        seed=int(sim["seed"]),
    )
    fraud_rate = float(sim["target_fraud_rate"])
    # balance top-ups (inserted by the ledger) add roughly 20% on top of legitimate volume
    n_normal = int(n_txn * (1 - fraud_rate) / 1.2)
    normal = generate_normal_events(world, n_normal)
    log.info("normal_events_generated", extra={"n": len(normal)})

    injector = FraudInjector(world, normal, float(sim["structuring_threshold"]))
    fraud = injector.run(int(n_txn * fraud_rate), dict(sim["scenario_weights"]))
    log.info("fraud_events_generated", extra={"n": len(fraud), "cases": injector.case_seq})

    normal = normal.assign(
        is_fraud=False,
        is_ato=False,
        fraud_scenario=None,
        fraud_case_id=-1,
        forced_fail=None,
        pin_attempts=1,
        allow_topup=True,
    )
    events = pd.concat([normal, fraud], ignore_index=True)
    events["seq"] = np.arange(len(events))
    events, topups = run_ledger(world, events)
    if len(topups):
        topups = topups.assign(
            is_fraud=False,
            is_ato=False,
            fraud_scenario=None,
            fraud_case_id=-1,
            forced_fail=None,
            pin_attempts=1,
            allow_topup=True,
            seq=-1,
        )
        events = pd.concat([events, topups], ignore_index=True)
    events = events.sort_values(["ts", "seq"], kind="stable").reset_index(drop=True)
    log.info("ledger_resolved", extra={"n": len(events), "topups": len(topups)})

    ds = _assemble(world, events, settings)
    ds.summary["generation_seconds"] = round(time.time() - t_start, 1)
    return ds


def _assemble(world: World, ev: pd.DataFrame, settings: Settings) -> SyntheticDataset:
    rng = world.rng
    sim = settings["simulation"]
    n = len(ev)
    cust_ids = world.customer_ids
    merch_ids = world.merchant_ids
    ttype_names = np.array(TRANSACTION_TYPES)[ev["ttype"].to_numpy()]
    wallet = cust_ids[ev["cust"].to_numpy()]
    cp = ev["cp_cust"].to_numpy()
    merch = ev["merch"].to_numpy()
    merchant_id = np.where(merch >= 0, merch_ids[np.clip(merch, 0, None)], None)
    counterparty = np.where(cp >= 0, cust_ids[np.clip(cp, 0, None)], merchant_id)
    is_cash_in = ttype_names == "CASH_IN"
    sender = np.where(is_cash_in, merchant_id, wallet)
    receiver = np.where(is_cash_in, wallet, counterparty)
    channel = CHANNEL_NAMES[ev["channel"].to_numpy().astype(int)]
    pin = ev["pin_attempts"].to_numpy().astype(int)
    net = NETWORKS[rng.integers(0, len(NETWORKS), n)]
    app_minor = rng.integers(0, 6, n)
    metadata = [
        f'{{"app_version": "5.{m}.0", "network_type": "{nw}", "pin_attempts": {p}}}'
        if ch == "APP"
        else f'{{"network_type": "{nw}", "pin_attempts": {p}}}'
        for ch, nw, p, m in zip(channel, net, pin, app_minor)
    ]
    transactions = pd.DataFrame(
        {
            "transaction_id": [f"TXN{i + 1:011d}" for i in range(n)],
            "event_time": pd.to_datetime(ev["ts"].to_numpy(), unit="s", utc=True),
            "sender_id": sender,
            "receiver_id": receiver,
            "customer_id": wallet,
            "merchant_id": merchant_id,
            "amount": ev["amount"].to_numpy().astype(float),
            "fee": ev["fee"].to_numpy().astype(float),
            "currency": "BDT",
            "transaction_type": ttype_names,
            "channel": channel,
            "device_id": ev["device"].to_numpy(),
            "district": np.array([d.name for d in DISTRICTS])[ev["district"].to_numpy().astype(int)],
            "latitude": ev["lat"].to_numpy().round(5),
            "longitude": ev["lon"].to_numpy().round(5),
            "status": np.where(ev["success"].to_numpy(dtype=bool), "SUCCESS", "FAILED"),
            "failure_reason": ev["failure_reason"].to_numpy(),
            "balance_before": ev["balance_before"].to_numpy().astype(float),
            "balance_after": ev["balance_after"].to_numpy().astype(float),
            "metadata": metadata,
        }
    )
    for col in ("sender_id", "receiver_id", "merchant_id", "device_id", "failure_reason"):
        transactions[col] = transactions[col].astype(object).where(transactions[col].notna(), None)

    # ---- labels (ground truth + report time) ----
    is_fraud = ev["is_fraud"].to_numpy(dtype=bool)
    case = ev["fraud_case_id"].to_numpy().astype(np.int64)
    case_last = pd.Series(ev["ts"].to_numpy()[is_fraud]).groupby(case[is_fraud]).max()
    delay = rng.lognormal(np.log(float(sim["label_delay_median_days"]) * DAY), 0.9, len(case_last))
    never = rng.random(len(case_last)) < float(sim["label_never_reported_rate"])
    reported = pd.Series(np.where(never, np.nan, case_last.to_numpy() + delay), index=case_last.index)
    rep_ts = np.full(n, np.nan)
    rep_ts[is_fraud] = reported.reindex(case[is_fraud]).to_numpy()
    labels = pd.DataFrame(
        {
            "transaction_id": transactions["transaction_id"],
            "is_fraud": is_fraud,
            "is_ato": ev["is_ato"].to_numpy(dtype=bool),
            "fraud_scenario": ev["fraud_scenario"].where(ev["fraud_scenario"].notna(), None).to_numpy(),
            "fraud_case_id": np.where(case >= 0, [f"FC_{c:06d}" for c in case], None),
            "fraud_reported_at": pd.to_datetime(rep_ts, unit="s", utc=True),
        }
    )

    # ---- devices (observed) ----
    dev_obs = transactions.loc[transactions["device_id"].notna(), ["device_id", "customer_id", "event_time"]]
    grp = dev_obs.groupby("device_id")
    owner = grp["customer_id"].agg(lambda s: s.value_counts().index[0])
    devices = pd.DataFrame(
        {
            "device_id": owner.index,
            "customer_id": owner.to_numpy(),
            "first_seen": grp["event_time"].min().to_numpy(),
            "last_seen": grp["event_time"].max().to_numpy(),
        }
    )
    reg = world.device_registry
    devices["device_type"] = devices["device_id"].map(lambda d: reg[d]["device_type"])
    devices["os"] = devices["device_id"].map(lambda d: reg[d]["os"])
    devices["device_age_days"] = devices["device_id"].map(lambda d: reg[d]["device_age_days"]).astype(int)
    devices = devices[["device_id", "customer_id", "device_type", "os", "device_age_days", "first_seen", "last_seen"]]

    # ---- customers ----
    customers = world.customers.drop(columns=[c for c in world.customers.columns if c.startswith("_")]).copy()
    end = pd.Timestamp(world.end_ts, unit="s", tz="UTC")
    ok = transactions[transactions["status"] == "SUCCESS"]
    last_ok = ok.groupby("customer_id")["event_time"].max()
    recent = customers["customer_id"].map(last_ok)
    customers["account_status"] = np.where(recent >= end - pd.Timedelta(days=30), "active", "dormant")

    merchants = world.merchants.drop(columns=["is_fraudulent"]).copy()
    merchants = merchants[
        ["merchant_id", "merchant_type", "merchant_category", "district", "risk_category", "registration_date"]
    ]

    customer_truth = pd.DataFrame(
        {
            "customer_id": cust_ids,
            "persona": [PERSONAS[i].name for i in world.persona_idx],
            "is_mule": world.is_mule,
            "mule_ring_id": world.mule_ring,
            "night_shift": world.customers["_night_shift"].to_numpy(),
            "churn_date": pd.to_datetime(np.where(world.churn_ts > 0, world.churn_ts, np.nan), unit="s", utc=True),
        }
    )
    merchant_truth = world.merchants[["merchant_id", "is_fraudulent"]].copy()

    lab = labels[labels["is_fraud"]]
    summary = {
        "n_transactions": int(n),
        "n_customers": int(len(customers)),
        "n_merchants": int(len(merchants)),
        "n_devices": int(len(devices)),
        "period_start": pd.Timestamp(world.start_ts, unit="s", tz="UTC").isoformat(),
        "period_end": end.isoformat(),
        "fraud_transactions": int(is_fraud.sum()),
        "fraud_rate": round(float(is_fraud.mean()), 5),
        "ato_transactions": int(labels["is_ato"].sum()),
        "fraud_cases": int(lab["fraud_case_id"].nunique()),
        "fraud_by_scenario": {k: int(v) for k, v in lab["fraud_scenario"].value_counts().items()},
        "failed_rate": round(float((transactions["status"] == "FAILED").mean()), 5),
        "mule_accounts": int(world.is_mule.sum()),
        "churners": int((world.churn_ts > 0).sum()),
        "transaction_types": {k: int(v) for k, v in transactions["transaction_type"].value_counts().items()},
        "seed": int(sim["seed"]),
    }
    return SyntheticDataset(
        customers, devices, merchants, transactions, labels, customer_truth, merchant_truth, summary
    )


def write_dataset(ds: SyntheticDataset, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds.customers.to_parquet(out / "customers.parquet", index=False)
    ds.devices.to_parquet(out / "devices.parquet", index=False)
    ds.merchants.to_parquet(out / "merchants.parquet", index=False)
    ds.transactions.to_parquet(out / "transactions.parquet", index=False)
    ds.labels.to_parquet(out / "transaction_labels.parquet", index=False)
    ds.customer_truth.to_parquet(out / "customer_ground_truth.parquet", index=False)
    ds.merchant_truth.to_parquet(out / "merchant_ground_truth.parquet", index=False)
    with open(out / "generation_summary.json", "w", encoding="utf-8") as fh:
        json.dump(ds.summary, fh, indent=2)
    log.info(
        "dataset_written",
        extra={"path": str(out), **{k: v for k, v in ds.summary.items() if isinstance(v, (int, float, str))}},
    )
    return out


# re-exported for convenience in tests
__all__ = ["SyntheticDataset", "generate", "write_dataset", "CH_APP"]

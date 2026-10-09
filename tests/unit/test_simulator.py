"""Synthetic generator: realism properties, ground truth and determinism."""

import numpy as np
import pandas as pd

from simulator.constants import ATO_SCENARIOS, FRAUD_SCENARIOS, LABEL_COLUMNS, TRANSACTION_COLUMNS


def test_schema_and_label_separation(small_data):
    _, ds = small_data
    assert list(ds.transactions.columns) == list(TRANSACTION_COLUMNS)
    assert list(ds.labels.columns) == list(LABEL_COLUMNS)
    # labels never leak into the event stream
    assert not {"is_fraud", "is_ato", "fraud_scenario"} & set(ds.transactions.columns)
    assert ds.transactions["transaction_id"].is_unique
    assert len(ds.labels) == len(ds.transactions)


def test_fraud_is_rare_but_present_and_all_scenarios_generated(small_data):
    _, ds = small_data
    rate = ds.labels["is_fraud"].mean()
    assert 0.005 < rate < 0.08
    scen = set(ds.labels.loc[ds.labels["is_fraud"], "fraud_scenario"])
    assert scen == set(FRAUD_SCENARIOS)
    # every fraud row has an explainable scenario and a case id
    fraud = ds.labels[ds.labels["is_fraud"]]
    assert fraud["fraud_scenario"].notna().all() and fraud["fraud_case_id"].notna().all()
    # ATO flag only on ATO scenarios
    ato = ds.labels[ds.labels["is_ato"]]
    assert set(ato["fraud_scenario"]) <= ATO_SCENARIOS


def test_amounts_are_heavily_skewed_and_activity_varies(small_data):
    _, ds = small_data
    amt = ds.transactions["amount"]
    assert amt.mean() > 1.5 * amt.median()
    per_customer = ds.transactions.groupby("customer_id").size()
    assert per_customer.quantile(0.9) > 4 * per_customer.quantile(0.1)


def test_fraud_overlaps_with_normal_behaviour(small_data):
    """Fraud must not be trivially separable: legit traffic also has night activity, new devices, big amounts."""
    _, ds = small_data
    df = ds.transactions.merge(ds.labels, on="transaction_id")
    hour = (df["event_time"].dt.hour + 6) % 24
    legit = df[~df["is_fraud"]]
    assert (hour[legit.index] < 5).mean() > 0.01
    assert (legit["amount"] > df.loc[df["is_fraud"], "amount"].median()).mean() > 0.05
    # some fraud looks ordinary: modest amounts from the customer's own home district
    scam = df[df["fraud_scenario"] == "unusual_recipient_scam"]
    assert len(scam) == 0 or scam["amount"].median() < 20_000


def test_labels_arrive_late(small_data):
    _, ds = small_data
    f = ds.labels[ds.labels["is_fraud"]].merge(ds.transactions[["transaction_id", "event_time"]], on="transaction_id")
    reported = f.dropna(subset=["fraud_reported_at"])
    assert (reported["fraud_reported_at"] > reported["event_time"]).all()
    assert f["fraud_reported_at"].isna().any()  # some fraud is never reported


def test_generation_is_deterministic(tmp_path):
    from simulator.generator import generate
    from tests.conftest import make_settings

    a = generate(make_settings(tmp_path / "a", 3000, 100, seed=5))
    b = generate(make_settings(tmp_path / "b", 3000, 100, seed=5))
    pd.testing.assert_frame_equal(a.transactions, b.transactions)
    c = generate(make_settings(tmp_path / "c", 3000, 100, seed=6))
    assert not np.array_equal(a.transactions["amount"].to_numpy()[:100], c.transactions["amount"].to_numpy()[:100])


def test_generated_data_passes_quality_checks(small_data):
    from data_engineering.validation.quality import build_report

    _, ds = small_data
    rep = build_report(
        "t", ds.transactions, customers=ds.customers, devices=ds.devices, merchants=ds.merchants, labels=ds.labels
    )
    failed = [c.name for c in rep.checks if not c.passed]
    assert rep.passed, failed

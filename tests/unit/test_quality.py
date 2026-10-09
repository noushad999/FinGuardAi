"""Data-quality checks detect each class of defect."""

import numpy as np
import pandas as pd
import pytest

from data_engineering.validation.quality import build_report


@pytest.fixture()
def clean(small_data):
    _, ds = small_data
    return ds.transactions.head(2000).copy(), ds


def _failed(tx, ds, **kw):
    rep = build_report("t", tx, customers=ds.customers, devices=ds.devices, merchants=ds.merchants, **kw)
    return {c.name for c in rep.checks if not c.passed}, rep


def test_clean_sample_passes(clean):
    tx, ds = clean
    failed, rep = _failed(tx, ds)
    assert not failed and rep.passed


def test_detects_duplicates_nulls_amounts_timestamps(clean):
    tx, ds = clean
    bad = pd.concat([tx, tx.iloc[[0]]], ignore_index=True)
    bad.loc[1, "customer_id"] = None
    bad.loc[2, "amount"] = -10
    bad.loc[3, "event_time"] = pd.Timestamp("2099-01-01", tz="UTC")
    failed, rep = _failed(bad, ds)
    assert {"duplicate_transaction_id", "required_not_null", "invalid_amount", "invalid_timestamp"} <= failed
    assert not rep.passed


def test_detects_impossible_balances_and_referential_integrity(clean):
    tx, ds = clean
    bad = tx.copy()
    i = bad.index[bad["status"] == "SUCCESS"][0]
    bad.loc[i, "balance_after"] = bad.loc[i, "balance_after"] + 999
    bad.loc[bad.index[5], "balance_before"] = -1
    j = bad.index[bad["device_id"].notna()][0]
    bad.loc[j, "device_id"] = "DEV_UNKNOWN"
    bad.loc[bad.index[7], "customer_id"] = "CUS_9999999"
    failed, _ = _failed(bad, ds)
    assert {"ledger_arithmetic", "negative_balance", "unknown_device", "unknown_customer"} <= failed


def test_detects_schema_drift(clean):
    tx, ds = clean
    drifted = tx.drop(columns=["channel"]).assign(amount=tx["amount"].astype(str), surprise=1)
    rep = build_report("t", drifted)
    check = next(c for c in rep.checks if c.name == "schema_drift")
    assert not check.passed and any("channel" in s for s in check.sample)


def test_detects_bad_ids_and_enums(clean):
    tx, ds = clean
    bad = tx.copy()
    bad.loc[bad.index[0], "customer_id"] = "C-1"
    bad.loc[bad.index[1], "transaction_type"] = "TELEPORT"
    bad.loc[bad.index[2], "latitude"] = np.float64(51.5)
    failed, _ = _failed(bad, ds)
    assert {"invalid_customer_id_format", "invalid_enumeration", "geo_out_of_bounds"} <= failed

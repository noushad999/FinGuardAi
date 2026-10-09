"""Feature engine: window arithmetic, novelty flags, point-in-time correctness and online/offline parity."""

import json

import numpy as np
import pandas as pd
import pytest

from features.definitions import ATO_FEATURES, FEATURE_NAMES
from features.engine import FeatureEngine
from features.offline import build_transaction_features
from features.reference import ReferenceData
from features.store import InMemoryStateStore, RedisStateStore

T0 = 1_780_000_000.0  # arbitrary epoch seconds


def ev(i, dt, **kw):
    base = dict(
        transaction_id=f"T{i}",
        event_time=T0 + dt,
        customer_id="CUS_0000001",
        sender_id="CUS_0000001",
        receiver_id="CUS_0000002",
        transaction_type="P2P",
        amount=1000.0,
        channel="APP",
        device_id="DEV_A",
        district="Dhaka",
        latitude=23.81,
        longitude=90.41,
        status="SUCCESS",
        balance_before=10_000.0,
        metadata={"pin_attempts": 1},
    )
    base.update(kw)
    return base


def test_feature_names_unique_and_complete():
    eng = FeatureEngine()
    f = eng.process(ev(0, 0))
    assert list(f) == FEATURE_NAMES
    assert set(ATO_FEATURES) <= set(FEATURE_NAMES)
    assert all(np.isfinite(v) for v in f.values())


def test_velocity_windows_and_point_in_time():
    eng = FeatureEngine()
    times = [0, 30, 50, 200, 1000, 4000]
    feats = [eng.process(ev(i, t)) for i, t in enumerate(times)]
    # the first event never sees itself
    assert feats[0]["txn_count_24h"] == 0 and feats[0]["n_prior_txns"] == 0
    # event at t=50 sees t=0 and t=30 within 60 s
    assert feats[2]["txn_count_1m"] == 2
    # event at t=200: 3 prior within 5 min, 0 within the last minute
    assert feats[3]["txn_count_1m"] == 0 and feats[3]["txn_count_5m"] == 3
    # event at t=4000: everything before t=400 is older than 1 h
    assert feats[5]["txn_count_1h"] == 1 and feats[5]["txn_count_24h"] == 5
    assert feats[5]["amount_sum_24h"] == pytest.approx(5000)


def test_novelty_flags():
    eng = FeatureEngine()
    for i in range(5):
        eng.process(ev(i, i * 3600))
    f = eng.process(
        ev(
            9,
            6 * 3600,
            device_id="DEV_NEW",
            district="Sylhet",
            latitude=24.89,
            longitude=91.87,
            receiver_id="CUS_0000999",
            metadata='{"pin_attempts": 3}',
        )
    )
    assert f["is_new_device"] == 1 and f["is_new_district"] == 1 and f["is_new_recipient"] == 1
    assert f["pin_attempts"] == 3
    assert f["distance_from_prev_km"] > 150
    assert f["speed_from_prev_kmh"] == pytest.approx(f["distance_from_prev_km"] / 2)  # 2 h since previous
    known = eng.process(ev(10, 7 * 3600))
    assert known["is_new_device"] == 0 and known["is_new_recipient"] == 0


def test_shared_device_and_receiver_in_degree():
    eng = FeatureEngine()
    for i in range(4):
        eng.process(
            ev(
                i,
                i * 60,
                customer_id=f"CUS_000010{i}",
                sender_id=f"CUS_000010{i}",
                device_id="DEV_FARM",
                receiver_id="CUS_0000777",
            )
        )
    f = eng.process(
        ev(9, 600, customer_id="CUS_0000109", sender_id="CUS_0000109", device_id="DEV_FARM", receiver_id="CUS_0000777")
    )
    assert f["accounts_per_device"] == 5
    assert f["receiver_in_degree"] == 4 and f["receiver_inflow_count_24h"] == 4


def test_failed_attempts_are_counted_but_not_recipients():
    eng = FeatureEngine()
    for i in range(3):
        eng.process(ev(i, i * 20, status="FAILED", failure_reason="WRONG_PIN", receiver_id="CUS_0000555"))
    f = eng.process(ev(5, 100, receiver_id="CUS_0000555"))
    assert f["wrong_pin_count_1h"] == 3 and f["failed_count_1h"] == 3
    assert f["is_new_recipient"] == 1  # failed transfers did not establish the relationship


def test_compute_without_commit_does_not_mutate_state():
    eng = FeatureEngine()
    eng.process(ev(0, 0))
    before = json.dumps(eng.customer_state("CUS_0000001"), sort_keys=True)
    eng.compute(ev(1, 10))
    assert json.dumps(eng.customer_state("CUS_0000001"), sort_keys=True) == before


def test_rapid_pass_through_features():
    eng = FeatureEngine()
    eng.process(
        ev(0, 0, customer_id="CUS_0000002", sender_id="CUS_0000002", receiver_id="CUS_0000001", amount=20_000)
    )  # inflow to CUS_0000001
    f = eng.process(ev(1, 300, amount=19_000))
    assert f["outflow_to_last_inflow_ratio"] == pytest.approx(0.95)
    assert f["log_minutes_since_last_inflow"] == pytest.approx(np.log1p(5))


class FakeRedis:
    def __init__(self):
        self.kv = {}

    def mget(self, keys):
        return [self.kv.get(k) for k in keys]

    def pipeline(self, transaction=False):
        outer = self

        class P:
            def __init__(self):
                self.ops = []

            def set(self, k, v):
                self.ops.append((k, v))

            def execute(self):
                outer.kv.update(dict(self.ops))

        return P()


def test_redis_store_gives_identical_features_to_memory_store():
    events = [ev(i, i * 45, amount=500 + 100 * i, device_id="DEV_A" if i % 3 else "DEV_B") for i in range(12)]
    mem, red = FeatureEngine(InMemoryStateStore()), FeatureEngine(RedisStateStore("", client=FakeRedis()))
    for e in events:
        assert mem.process(e) == pytest.approx(red.process(e))


def test_offline_replay_equals_online_processing(small_data):
    """Training features (offline replay) == what the online engine computes event by event."""
    settings, ds = small_data
    tx = ds.transactions.head(3000)
    ref = ReferenceData.from_frames(ds.customers, ds.merchants)
    offline = build_transaction_features(tx, ref).set_index("transaction_id")
    online = FeatureEngine(InMemoryStateStore(), ref)
    for rec in tx.sort_values(["event_time", "transaction_id"]).head(400).to_dict("records"):
        rec = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in rec.items()}
        f = online.process(rec)
        row = offline.loc[rec["transaction_id"], FEATURE_NAMES].to_numpy(dtype=float)
        np.testing.assert_allclose(row, np.array([f[n] for n in FEATURE_NAMES], dtype=np.float32), rtol=1e-5)


def test_features_do_not_depend_on_future_events(small_data):
    """Leakage test: changing *future* transactions must not change past feature values."""
    _, ds = small_data
    ref = ReferenceData.from_frames(ds.customers, ds.merchants)
    tx = ds.transactions.head(2000).copy()
    cut = tx["event_time"].iloc[1000]
    a = build_transaction_features(tx, ref).set_index("transaction_id")
    fut = tx["event_time"] > cut
    tx.loc[fut, "amount"] = tx.loc[fut, "amount"] * 7
    tx.loc[fut, "device_id"] = "DEV_TAMPERED"
    b = build_transaction_features(tx, ref).set_index("transaction_id")
    past = tx.loc[~fut, "transaction_id"]
    pd.testing.assert_frame_equal(a.loc[past, FEATURE_NAMES], b.loc[past, FEATURE_NAMES])
    assert not a.loc[tx.loc[fut, "transaction_id"], "amount"].equals(b.loc[tx.loc[fut, "transaction_id"], "amount"])

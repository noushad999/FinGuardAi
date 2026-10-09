"""API contract tests (FastAPI TestClient, file-backed repository, in-memory state)."""

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.ml


@pytest.fixture(scope="module")
def client(small_env):
    from api.main import create_app
    from api.repository import FileRepository
    from features.store import InMemoryStateStore
    from services.risk_engine.scorer import RiskScorer

    settings, _, _ = small_env
    app = create_app(
        RiskScorer.from_settings(settings, store=InMemoryStateStore()),
        FileRepository(settings.path("synthetic_dir"), settings.path("processed_dir")),
    )
    with TestClient(app) as c:
        yield c


def payload(ds, **kw):
    cust = ds.customers["customer_id"].iloc[0]
    body = {
        "event_time": "2026-05-20T21:30:00Z",
        "customer_id": cust,
        "sender_id": cust,
        "receiver_id": ds.customers["customer_id"].iloc[1],
        "amount": 15000,
        "transaction_type": "P2P",
        "channel": "APP",
        "device_id": "DEV_TEST_1",
        "district": "Dhaka",
        "latitude": 23.8,
        "longitude": 90.4,
        "balance_before": 16000,
        "metadata": {"pin_attempts": 2},
    }
    body.update(kw)
    return body


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["models_loaded"] and body["model_versions"]["fraud"].startswith("fraud-xgb-v")


def test_score_contract(client, small_env):
    _, ds, _ = small_env
    r = client.post("/risk/score", json=payload(ds, transaction_id="TXN_API_TEST_1"))
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["transaction_id"] == "TXN_API_TEST_1"
    assert 0 <= b["fraud_probability"] <= 1 and 0 <= b["risk_score"] <= 100
    assert b["decision"] in ("APPROVE", "REVIEW", "BLOCK")
    assert b["model_version"].startswith("fraud-xgb") and b["policy_version"]
    assert "features" not in b and b["state_committed"] is False


def test_score_generates_id_and_commit_updates_state(client, small_env):
    _, ds, _ = small_env
    body = payload(ds)
    r = client.post("/risk/score?commit=true", json=body)
    assert r.status_code == 200 and r.json()["transaction_id"].startswith("API_")
    prof = client.get(f"/customer/{body['customer_id']}").json()
    assert prof["live_state"]["prior_transactions"] >= 1


@pytest.mark.parametrize(
    "patch",
    [
        {"amount": -5},
        {"customer_id": "bad"},
        {"transaction_type": "X"},
        {"transaction_type": "MERCHANT_PAYMENT", "merchant_id": None},
    ],
)
def test_score_validation_errors(client, small_env, patch):
    _, ds, _ = small_env
    r = client.post("/risk/score", json=payload(ds, **patch))
    assert r.status_code == 422 and r.json()["error"] == "validation_error"


def test_customer_profile(client, small_env):
    _, ds, _ = small_env
    cid = ds.customers["customer_id"].iloc[3]
    r = client.get(f"/customer/{cid}")
    assert r.status_code == 200
    assert r.json()["customer_id"] == cid and r.json()["risk_level"] in ("Low", "Medium", "High")
    assert client.get("/customer/CUS_9999999").status_code == 404


def test_transaction_lookup(client, small_env):
    _, ds, _ = small_env
    tid = ds.transactions["transaction_id"].iloc[10]
    r = client.get(f"/transaction/{tid}")
    assert r.status_code == 200
    assert r.json()["transaction"]["transaction_id"] == tid
    assert any(p["source"] == "batch" for p in r.json()["predictions"])
    assert client.get("/transaction/NOPE").status_code == 404


def test_alerts_model_info_and_metrics(client, small_env):
    _, ds, _ = small_env
    # a very suspicious transaction should land in the alert queue
    client.post(
        "/risk/score",
        json=payload(
            ds,
            amount=16000,
            balance_before=16000,
            device_id="DEV_EVIL",
            metadata={"pin_attempts": 5},
            transaction_id="TXN_ALERT_1",
        ),
    )
    alerts = client.get("/fraud/alerts?limit=10").json()
    assert alerts["count"] == len(alerts["alerts"])
    assert client.get("/fraud/alerts?decision=MAYBE").status_code == 422
    info = client.get("/model/info").json()
    assert {"fraud", "account_takeover", "isolation_forest"} <= set(info["models"])
    assert info["policy"]["thresholds"]["review"] < info["policy"]["thresholds"]["block"]
    m = client.get("/metrics").text
    assert "finguard_api_requests_total" in m and "finguard_decisions_total" in m

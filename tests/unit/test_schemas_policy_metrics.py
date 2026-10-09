import numpy as np
import pytest
from pydantic import ValidationError

from ingestion.schemas import TransactionEvent
from ml.common.metrics import calibration_report, classification_report, precision_recall_at_k, select_threshold
from services.risk_engine.policy import DecisionPolicy

VALID = {
    "transaction_id": "TXN0001",
    "event_time": "2026-05-01T10:00:00",
    "customer_id": "CUS_0000001",
    "receiver_id": "CUS_0000002",
    "amount": 100,
    "transaction_type": "P2P",
}


def test_event_schema_accepts_valid_and_normalises():
    e = TransactionEvent(**VALID, metadata='{"pin_attempts": 2}')
    assert e.event_time.tzinfo is not None
    assert e.metadata["pin_attempts"] == 2
    assert e.to_engine_dict()["event_time"] == pytest.approx(e.event_time.timestamp())


@pytest.mark.parametrize(
    "patch",
    [
        {"amount": -1},
        {"amount": 0},
        {"customer_id": "123"},
        {"transaction_type": "TELEPORT"},
        {"event_time": "not-a-date"},
        {"transaction_type": "MERCHANT_PAYMENT"},
        {"schema_version": "2.0"},
        {"status": "FAILED"},
        {"latitude": 55.0},
    ],
)
def test_event_schema_rejects_invalid(patch):
    with pytest.raises(ValidationError):
        TransactionEvent(**{**VALID, **patch})


POLICY = {
    "version": "t",
    "thresholds": {"review": 30, "block": 80},
    "rules": {"anomaly_review_percentile": 99.9, "max_single_amount": 100_000, "max_txn_per_minute": 5},
}


def test_policy_thresholds():
    p = DecisionPolicy(POLICY)
    assert p.decide(10).decision == "APPROVE"
    assert p.decide(30).decision == "REVIEW"
    assert p.decide(79).risk_level == "MEDIUM"
    assert p.decide(80).decision == "BLOCK"


def test_policy_rules_only_escalate():
    p = DecisionPolicy(POLICY)
    assert p.decide(5, anomaly_percentile=99.95).decision == "REVIEW"
    assert p.decide(5, amount=200_000).decision == "REVIEW"
    assert p.decide(5, txn_count_1m=5).decision == "REVIEW"
    # a model-driven BLOCK is never downgraded by rules
    assert p.decide(95, anomaly_percentile=1).decision == "BLOCK"


def test_policy_validates_config():
    with pytest.raises(ValueError):
        DecisionPolicy({"thresholds": {"review": 90, "block": 50}})


def test_ranking_metrics():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.1, 0.05, 0.0, 0.6])
    p, r = precision_recall_at_k(y, s, 3)
    assert p == pytest.approx(2 / 3) and r == pytest.approx(2 / 3)
    rep = classification_report(y, s, 0.65, ks=(3,))
    assert rep["confusion_matrix"] == {"tn": 6, "fp": 1, "fn": 1, "tp": 2}
    assert 0 < rep["pr_auc"] <= 1 and "accuracy" not in rep


def test_threshold_selection_meets_precision_target():
    rng = np.random.default_rng(0)
    y = rng.random(5000) < 0.05
    s = np.clip(y * 0.5 + rng.random(5000) * 0.6, 0, 1)
    t = select_threshold(y, s, target_precision=0.9)
    pred = s >= t["precision_threshold"]
    assert (y[pred].mean()) >= 0.9 - 1e-9


def test_calibration_report_perfectly_calibrated():
    rng = np.random.default_rng(1)
    p = rng.random(20000)
    y = rng.random(20000) < p
    c = calibration_report(y, p)
    assert c["ece"] < 0.02

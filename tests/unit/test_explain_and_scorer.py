import pytest

from ml.common.explain import EXPLANATION_NOTE, reasons_from_contributions

pytestmark = pytest.mark.ml


def test_reasons_are_ordered_and_readable():
    feats = {"amount_to_avg_ratio": 7.2, "is_new_device": 1.0, "is_new_recipient": 1.0, "hour_local": 3.0}
    contrib = {"amount_to_avg_ratio": 1.2, "is_new_device": 2.0, "is_new_recipient": 0.4, "hour_local": 0.9}
    r = reasons_from_contributions(feats, contrib, max_reasons=3)
    assert [x["feature"] for x in r] == ["is_new_device", "amount_to_avg_ratio", "is_new_recipient"]
    assert r[1]["reason"] == "Transaction amount 7.2x the customer's typical amount"


def test_negative_contributions_are_not_reasons():
    assert reasons_from_contributions({"is_new_device": 1.0}, {"is_new_device": -1.5}) == []


def test_explanations_avoid_causal_language():
    assert "not proof or cause" in EXPLANATION_NOTE
    assert "caused" not in EXPLANATION_NOTE.replace("cause of fraud", "")


def test_scorer_end_to_end(small_env):
    from features.store import InMemoryStateStore
    from services.risk_engine.scorer import RiskScorer

    settings, ds, _ = small_env
    scorer = RiskScorer.from_settings(settings, store=InMemoryStateStore())
    rows = ds.transactions.head(200).to_dict("records")
    out = [scorer.score({k: v for k, v in r.items()}, commit=True) for r in rows]
    for o in out:
        assert 0 <= o["risk_score"] <= 100
        assert o["decision"] in {"APPROVE", "REVIEW", "BLOCK"}
        assert o["risk_score"] == round(100 * max(o["fraud_probability"], o["account_takeover_probability"]))
        if o["decision"] != "APPROVE":
            assert o["risk_reasons"]
    assert scorer.engine.customer_state(rows[-1]["customer_id"]) is not None

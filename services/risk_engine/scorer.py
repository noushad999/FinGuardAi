"""Real-time risk scoring core used by both the FastAPI service and the Kafka worker.

    transaction ──> FeatureEngine (online state) ──> fraud XGB  ─┐
                                                   ──> ATO XGB   ├─> risk_score ──> DecisionPolicy ──> decision
                                                   ──> IForest   │   (ML)            (business)
                                                   ──> behavioural baseline ┘

The ML side outputs calibrated probabilities and SHAP-based reasons; the
policy converts them into APPROVE / REVIEW / BLOCK.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from common.config import Settings, get_settings
from common.logging import get_logger
from features.definitions import FEATURE_NAMES
from features.engine import EngineConfig, FeatureEngine
from features.reference import ReferenceData
from features.store import StateStore, build_store
from ml.anomaly_detection.behavioral import BehavioralAnomalyModel
from ml.anomaly_detection.isolation_forest import IsolationForestModel
from ml.common.explain import EXPLANATION_NOTE, reasons_from_contributions
from ml.common.xgb import XGBModel
from services.risk_engine.policy import DecisionPolicy

log = get_logger(__name__)


@dataclass
class ModelSet:
    fraud: XGBModel
    ato: XGBModel | None = None
    iforest: IsolationForestModel | None = None
    behavioral: BehavioralAnomalyModel | None = None

    def versions(self) -> dict[str, str | None]:
        return {
            "fraud": self.fraud.version,
            "account_takeover": self.ato.version if self.ato else None,
            "isolation_forest": self.iforest.version if self.iforest else None,
            "behavioral_anomaly": self.behavioral.version if self.behavioral else None,
        }


def load_models(models_dir: str | Path) -> ModelSet:
    d = Path(models_dir)
    fraud = XGBModel(d / "fraud_xgb", available_features=FEATURE_NAMES)
    ato = XGBModel(d / "ato_xgb", available_features=FEATURE_NAMES) if (d / "ato_xgb").exists() else None
    iforest = IsolationForestModel(d / "fraud_iforest") if (d / "fraud_iforest").exists() else None
    beh = BehavioralAnomalyModel.load(d / "behavioral_anomaly") if (d / "behavioral_anomaly").exists() else None
    return ModelSet(fraud, ato, iforest, beh)


def resolve_models_dir(settings: Settings) -> Path:
    """Local bundle dir, or pull the ``production`` aliases from MLflow first."""
    models_dir = settings.path("models_dir")
    if settings["api"]["model_source"] == "mlflow":
        from ml.common import registry

        for name in ("finguard-fraud-xgb", "finguard-ato-xgb", "finguard-fraud-iforest", "finguard-behavioral-anomaly"):
            try:
                registry.deploy(name, models_dir, alias="production")
            except Exception as exc:  # noqa: BLE001 - fall back to what is on disk
                log.warning("mlflow_model_pull_failed", extra={"model": name, "error": str(exc)})
    return models_dir


class RiskScorer:
    def __init__(self, models: ModelSet, engine: FeatureEngine, policy: DecisionPolicy) -> None:
        self.models = models
        self.engine = engine
        self.policy = policy

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, store: StateStore | None = None, reference: ReferenceData | None = None
    ) -> RiskScorer:
        settings = settings or get_settings()
        models = load_models(resolve_models_dir(settings))
        if reference is None:
            synth = settings.path("synthetic_dir")
            reference = ReferenceData.from_parquet(synth) if (synth / "customers.parquet").exists() else ReferenceData()
        store = store if store is not None else build_store(settings["redis"]["url"] or None)
        engine = FeatureEngine(store, reference, EngineConfig.from_settings(settings))
        log.info("risk_scorer_ready", extra={"models": models.versions(), "store": type(store).__name__})
        return cls(models, engine, DecisionPolicy())

    def score(self, txn: dict[str, Any], commit: bool = False, include_features: bool = False) -> dict[str, Any]:
        t0 = time.perf_counter()
        features, ctx = self.engine.compute(txn)
        m = self.models
        x = m.fraud.vector(features)
        fraud_p = float(m.fraud.predict_proba(x)[0])
        fraud_contrib = dict(zip(m.fraud.feature_names, m.fraud.contributions(x)[0][:-1].tolist()))

        ato_p, ato_contrib = 0.0, {}
        if m.ato is not None:
            xa = m.ato.vector(features)
            ato_p = float(m.ato.predict_proba(xa)[0])
            ato_contrib = dict(zip(m.ato.feature_names, m.ato.contributions(xa)[0][:-1].tolist()))
        anomaly_pct = float(m.iforest.percentile(x)[0]) if m.iforest is not None else None
        beh_pct, beh_reasons = None, []
        if m.behavioral is not None:
            beh_pct = float(m.behavioral.percentile(features)[0])
            beh_reasons = m.behavioral.explain(features)

        risk_score = int(round(100 * max(fraud_p, ato_p)))
        decision = self.policy.decide(
            risk_score, anomaly_percentile=anomaly_pct, amount=features["amount"], txn_count_1m=features["txn_count_1m"]
        )
        # Explain with the model that drives the score.
        driver, contrib = ("account_takeover", ato_contrib) if ato_p > fraud_p else ("fraud", fraud_contrib)
        details = reasons_from_contributions(features, contrib, self.policy.max_reasons)
        reasons = [d["reason"] for d in details] + decision.policy_reasons
        if not reasons and beh_reasons and decision.decision != "APPROVE":
            reasons = [r["reason"] for r in beh_reasons]
        ato_details = reasons_from_contributions(features, ato_contrib, self.policy.max_reasons) if ato_contrib else []

        if commit:
            self.engine.commit(ctx)
        result: dict[str, Any] = {
            "transaction_id": str(txn.get("transaction_id")),
            "customer_id": str(txn.get("customer_id")),
            "fraud_probability": round(fraud_p, 6),
            "account_takeover_probability": round(ato_p, 6),
            "account_takeover_risk_score": int(round(100 * ato_p)),
            "account_takeover_reasons": [d["reason"] for d in ato_details],
            "anomaly_percentile": None if anomaly_pct is None else round(anomaly_pct, 3),
            "behavioral_anomaly_percentile": None if beh_pct is None else round(beh_pct, 3),
            "risk_score": risk_score,
            "risk_level": decision.risk_level,
            "decision": decision.decision,
            "risk_reasons": reasons,
            "reason_details": details,
            "score_driver": driver,
            "explanation_note": EXPLANATION_NOTE,
            "model_version": m.fraud.version,
            "model_versions": m.versions(),
            "policy_version": self.policy.version,
            "scored_at": datetime.now(UTC).isoformat(),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
        }
        if include_features:
            result["features"] = {k: float(np.round(v, 6)) for k, v in features.items()}
        return result

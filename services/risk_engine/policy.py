"""Business decision policy - kept strictly separate from the ML models.

Models produce probabilities; this module maps them to actions using
``configs/policy.yaml``. Rules can only *escalate* a decision (e.g. an
unsupervised anomaly or a hard amount limit sends an otherwise-approved
transaction to review); they never downgrade a model-driven BLOCK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from common.config import load_policy

LEVELS = ("LOW", "MEDIUM", "HIGH")
DECISIONS = {"LOW": "APPROVE", "MEDIUM": "REVIEW", "HIGH": "BLOCK"}


@dataclass
class PolicyDecision:
    risk_level: str
    decision: str
    policy_reasons: list[str] = field(default_factory=list)


class DecisionPolicy:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or load_policy()
        self.version = str(cfg.get("version", "policy"))
        self.review = float(cfg["thresholds"]["review"])
        self.block = float(cfg["thresholds"]["block"])
        if not 0 <= self.review < self.block <= 100:
            raise ValueError("policy thresholds must satisfy 0 <= review < block <= 100")
        rules = cfg.get("rules", {})
        self.anomaly_pct = float(rules.get("anomaly_review_percentile", 101))
        self.max_amount = float(rules.get("max_single_amount", float("inf")))
        self.max_per_minute = float(rules.get("max_txn_per_minute", float("inf")))
        self.max_reasons = int(cfg.get("max_reasons", 4))
        self.config = cfg

    def level_for(self, risk_score: float) -> str:
        if risk_score >= self.block:
            return "HIGH"
        if risk_score >= self.review:
            return "MEDIUM"
        return "LOW"

    def decide(
        self,
        risk_score: float,
        *,
        anomaly_percentile: float | None = None,
        amount: float = 0.0,
        txn_count_1m: float = 0.0,
    ) -> PolicyDecision:
        level = self.level_for(risk_score)
        reasons: list[str] = []
        if level == "LOW":
            if anomaly_percentile is not None and anomaly_percentile >= self.anomaly_pct:
                reasons.append(
                    f"Unsupervised anomaly score in the top {100 - self.anomaly_pct:.1f}% "
                    "of training traffic (policy rule)"
                )
            if amount > self.max_amount:
                reasons.append(
                    f"Amount above the single-transaction review limit of BDT {self.max_amount:,.0f} (policy rule)"
                )
            if txn_count_1m + 1 > self.max_per_minute:
                reasons.append(f"More than {self.max_per_minute:.0f} attempts within one minute (policy rule)")
            if reasons:
                level = "MEDIUM"
        return PolicyDecision(level, DECISIONS[level], reasons)

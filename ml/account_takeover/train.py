"""Dedicated account-takeover (ATO) model.

ATO is a different question from "is this transaction fraudulent?": it asks
"is someone other than the owner operating this account?". Its label is
``is_ato`` (velocity attacks, new-device drains, impossible travel, PIN
brute-force, stolen-phone night activity) and its features are restricted to
signals of a *change of controller* (device, location, time-of-day, PIN
failures, amount relative to the customer's own baseline, new recipients) -
see ``features.definitions.ATO_FEATURES``.

Scam victims who send money themselves are fraud but *not* ATO, which is
exactly why the two models disagree on some cases and why the risk engine
uses both.
"""

from __future__ import annotations

from typing import Any

from common.config import Settings, get_settings, load_policy
from common.logging import get_logger
from features.definitions import ATO_FEATURES
from ml.common.bundle import sanitize, write_json
from ml.common.data import TrainingFrame, load_training_frame
from ml.common.supervised import fit_supervised
from ml.common.tracking import next_version, tracked_run

log = get_logger(__name__)
REGISTERED_NAME, PREFIX = "finguard-ato-xgb", "ato-xgb"


def train(tf: TrainingFrame, settings: Settings) -> dict[str, Any]:
    policy = load_policy()
    bundle = settings.path("models_dir") / "ato_xgb"
    version = next_version(REGISTERED_NAME, bundle, PREFIX)
    with tracked_run("ato-xgb", {"model_family": "account_takeover", "approach": "supervised"}) as run:
        res = fit_supervised(
            tf=tf,
            feature_names=ATO_FEATURES,
            label_col="is_ato",
            settings=settings,
            run=run,
            bundle_dir=bundle,
            name="ato_xgb",
            version=version,
            policy_review_score=policy["thresholds"]["review"],
            policy_block_score=policy["thresholds"]["block"],
        )
        run.register_bundle(bundle, REGISTERED_NAME, "Account takeover probability")
    reports = settings.path("reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    write_json(reports / "ato_xgb_metrics.json", sanitize(res["meta"].metrics))
    m = res["meta"].metrics["test"]["at_policy_review"]
    log.info(
        "ato_trained",
        extra={"version": version, "test_pr_auc": m["pr_auc"], "recall": m["recall"], "precision": m["precision"]},
    )
    return res


def run(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    tf = load_training_frame(settings)
    res = train(tf, settings)
    scores_path = settings.path("processed_dir") / "fraud_scores.parquet"
    if scores_path.exists():
        import pandas as pd

        scores = pd.read_parquet(scores_path)
        ato = res["scores"].rename(columns={"probability": "ato_probability"})[["transaction_id", "ato_probability"]]
        scores = scores.drop(columns=[c for c in ("ato_probability", "ato_model_version") if c in scores]).merge(
            ato, on="transaction_id", how="left"
        )
        scores["ato_model_version"] = res["meta"].version
        scores.to_parquet(scores_path, index=False)
    return res


if __name__ == "__main__":  # pragma: no cover
    run()

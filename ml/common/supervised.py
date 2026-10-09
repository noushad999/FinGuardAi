"""Supervised transaction-model training shared by the fraud and ATO models.

Steps (all on the temporal split, see ``ml.common.splits``):

1. Labels known at each cutoff (late fraud reports are *not* visible in training).
2. Class-imbalance strategy chosen on validation PR-AUC among
   ``scale_pos_weight`` in {1, sqrt(neg/pos), neg/pos}.
3. Platt calibration fitted on validation margins.
4. Threshold suggestions from validation (max-F1 and a precision target).
5. Final, untouched evaluation on the test period against ground truth,
   including per-fraud-scenario recall and SHAP global importance.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.logging import get_logger
from ml.common.bundle import BundleMetadata, PlattCalibrator, feature_reference, save_metadata, write_json
from ml.common.data import TrainingFrame
from ml.common.explain import global_importance
from ml.common.metrics import (
    calibration_report,
    classification_report,
    per_group_recall,
    select_threshold,
)
from ml.common.splits import known_labels
from ml.common.tracking import Run
from ml.common.xgb import contributions, margins, train_booster

log = get_logger(__name__)


def fit_supervised(
    *,
    tf: TrainingFrame,
    feature_names: list[str],
    label_col: str,
    settings: dict,
    run: Run,
    bundle_dir: Path,
    name: str,
    version: str,
    policy_review_score: float,
    policy_block_score: float,
) -> dict[str, Any]:
    tr_cfg = settings["training"]
    train, val, test = tf.part("train"), tf.part("validation"), tf.part("test")
    maturity = pd.Timedelta(days=float(tr_cfg.get("label_maturity_days", 0)))
    if tr_cfg.get("respect_label_delay", True) and maturity > pd.Timedelta(0):
        val = val[val["event_time"] < tf.val_end - maturity]
    X = {
        s: d[feature_names].to_numpy(dtype=np.float32)
        for s, d in (("train", train), ("validation", val), ("test", test))
    }
    if tr_cfg.get("respect_label_delay", True):
        y_tr = known_labels(train[label_col], train["fraud_reported_at"], tf.val_end)
        y_va = known_labels(val[label_col], val["fraud_reported_at"], tf.val_end)
    else:
        y_tr, y_va = train[label_col].astype(int).to_numpy(), val[label_col].astype(int).to_numpy()
    y_te = test[label_col].astype(int).to_numpy()
    label_stats = {
        "train_positives_ground_truth": int(train[label_col].sum()),
        "train_positives_known_at_cutoff": int(y_tr.sum()),
        "validation_positives_known": int(y_va.sum()),
        "test_positives": int(y_te.sum()),
        "rows": {"train": len(train), "validation_mature": len(val), "test": len(test)},
        "label_maturity_days": maturity.days,
    }
    log.info("training_labels", extra={"model": name, **{k: v for k, v in label_stats.items() if k != "rows"}})

    # ---- class imbalance experiment ----
    ratio = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    strategies = {"none": 1.0, "sqrt_ratio": math.sqrt(ratio), "full_ratio": ratio}
    params = dict(tr_cfg["xgb"])
    seed = int(tr_cfg.get("random_state", 42))
    experiments: dict[str, Any] = {}
    best_name, best_booster, best_ap = "", None, -1.0
    for sname, spw in strategies.items():
        booster = train_booster(X["train"], y_tr, X["validation"], y_va, params, feature_names, spw, seed)
        m_va = margins(booster, X["validation"], feature_names)
        rep = classification_report(y_va, m_va, threshold=0.0)
        experiments[sname] = {
            "scale_pos_weight": spw,
            "val_pr_auc": rep["pr_auc"],
            "val_roc_auc": rep["roc_auc"],
            "best_iteration": int(booster.best_iteration),
        }
        if rep["pr_auc"] > best_ap:
            best_name, best_booster, best_ap = sname, booster, rep["pr_auc"]
    assert best_booster is not None
    log.info("imbalance_strategy_selected", extra={"model": name, "strategy": best_name, "val_pr_auc": best_ap})

    # ---- calibration & thresholds (validation) ----
    m_va = margins(best_booster, X["validation"], feature_names)
    calibrator = PlattCalibrator.fit(m_va, y_va)
    p_va = calibrator.transform(m_va)
    thr = select_threshold(y_va, p_va, float(tr_cfg.get("target_precision", 0.8)))

    # ---- test evaluation (ground truth) ----
    m_te = margins(best_booster, X["test"], feature_names)
    p_te = calibrator.transform(m_te)
    review_p, block_p = policy_review_score / 100.0, policy_block_score / 100.0
    test_metrics = {
        "at_suggested_threshold": classification_report(
            y_te, p_te, thr.get("precision_threshold", thr["max_f1_threshold"])
        ),
        "at_policy_review": classification_report(y_te, p_te, review_p),
        "at_policy_block": classification_report(y_te, p_te, block_p),
        "calibration": calibration_report(y_te, p_te),
    }
    scen = test["fraud_scenario"].to_numpy(dtype=object)
    truth = test[label_col].to_numpy(dtype=bool)
    test_metrics["scenario_recall_at_policy_review"] = per_group_recall(truth, p_te >= review_p, scen)

    # ---- SHAP global importance (test sample) ----
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(test), size=min(5000, len(test)), replace=False)
    contrib = contributions(best_booster, X["test"][sample], feature_names)
    importance = global_importance(contrib, feature_names)

    # ---- bundle ----
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    best_booster.save_model(str(bundle_dir / "model.json"))
    calibrator.to_json(bundle_dir / "calibrator.json")
    write_json(bundle_dir / "feature_reference.json", feature_reference(train, feature_names))
    write_json(bundle_dir / "shap_importance.json", importance)
    meta = BundleMetadata(
        name=name,
        version=version,
        model_type="xgboost_binary_classifier",
        feature_names=feature_names,
        params={**params, "imbalance_strategy": best_name, "scale_pos_weight": strategies[best_name]},
        metrics={
            "test": test_metrics,
            "validation": {"pr_auc": best_ap, "thresholds": thr},
            "imbalance_experiments": experiments,
        },
        thresholds=thr,
        data={
            "dataset_version": tf.dataset_version,
            "label": label_col,
            "train_end": tf.train_end.isoformat(),
            "validation_end": tf.val_end.isoformat(),
            "label_policy": "labels reported before validation_end"
            if tr_cfg.get("respect_label_delay", True)
            else "ground truth",
            **label_stats,
        },
        extra={"best_iteration": int(best_booster.best_iteration), "calibration": "platt", "mlflow_run_id": run.run_id},
    )
    save_metadata(bundle_dir, meta)

    run.log_params(
        {
            "model": name,
            "version": version,
            "imbalance_strategy": best_name,
            **{f"xgb.{k}": v for k, v in params.items()},
            "dataset_version": tf.dataset_version,
            "n_features": len(feature_names),
        }
    )
    run.log_metrics(
        {k: v for k, v in test_metrics["at_suggested_threshold"].items() if isinstance(v, (int, float))}, prefix="test."
    )
    run.log_metrics(
        {"brier": test_metrics["calibration"]["brier"], "ece": test_metrics["calibration"]["ece"]},
        prefix="test.calibration.",
    )
    run.log_metrics({k: {"val_pr_auc": v["val_pr_auc"]} for k, v in experiments.items()}, prefix="imbalance.")

    all_X = tf.df[feature_names].to_numpy(dtype=np.float32)
    all_p = calibrator.transform(margins(best_booster, all_X, feature_names))
    scores = pd.DataFrame(
        {"transaction_id": tf.df["transaction_id"].to_numpy(), "probability": all_p, "split": tf.df["split"].to_numpy()}
    )
    return {"meta": meta, "test_prob": p_te, "test_truth": y_te, "scores": scores, "importance": importance}

"""Train and compare the three fraud-detection approaches.

* Model A - supervised XGBoost  (``models/fraud_xgb``)
* Model B - Isolation Forest    (``models/fraud_iforest``)
* Model C - behavioural anomaly (``models/behavioral_anomaly``)

All metrics are computed on the held-out *test period* and written to
``reports/fraud_model_comparison.{json,md}``. Nothing is hard-coded.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from common.config import Settings, get_settings, load_policy
from common.logging import get_logger
from features.definitions import FEATURE_NAMES
from ml.anomaly_detection.behavioral import REQUIRED_FEATURES, BehavioralAnomalyModel
from ml.anomaly_detection.isolation_forest import (
    PercentileMapper,
    anomaly_scores,
    save_bundle,
    train_isolation_forest,
)
from ml.common.bundle import BundleMetadata, sanitize, write_json
from ml.common.data import TrainingFrame, load_training_frame
from ml.common.metrics import classification_report, per_group_recall, precision_recall_at_k
from ml.common.supervised import fit_supervised
from ml.common.tracking import next_version, tracked_run

log = get_logger(__name__)

REGISTERED = {
    "fraud_xgb": ("finguard-fraud-xgb", "fraud-xgb"),
    "fraud_iforest": ("finguard-fraud-iforest", "iforest"),
    "behavioral_anomaly": ("finguard-behavioral-anomaly", "behavioral"),
}


def train_supervised(tf: TrainingFrame, settings: Settings) -> dict[str, Any]:
    policy = load_policy()
    bundle = settings.path("models_dir") / "fraud_xgb"
    reg_name, prefix = REGISTERED["fraud_xgb"]
    version = next_version(reg_name, bundle, prefix)
    with tracked_run("fraud-xgb", {"model_family": "fraud", "approach": "supervised"}) as run:
        res = fit_supervised(
            tf=tf,
            feature_names=FEATURE_NAMES,
            label_col="is_fraud",
            settings=settings,
            run=run,
            bundle_dir=bundle,
            name="fraud_xgb",
            version=version,
            policy_review_score=policy["thresholds"]["review"],
            policy_block_score=policy["thresholds"]["block"],
        )
        run.log_artifact(bundle / "metadata.json", "reports")
        run.register_bundle(bundle, reg_name, "Supervised XGBoost P(fraud | transaction)")
    log.info(
        "fraud_xgb_trained",
        extra={"version": version, "test_pr_auc": res["meta"].metrics["test"]["at_policy_review"]["pr_auc"]},
    )
    return res


def train_iforest(tf: TrainingFrame, settings: Settings) -> dict[str, Any]:
    cfg = settings["training"]
    seed = int(cfg.get("random_state", 42))
    train, test = tf.part("train"), tf.part("test")
    bundle = settings.path("models_dir") / "fraud_iforest"
    reg_name, prefix = REGISTERED["fraud_iforest"]
    version = next_version(reg_name, bundle, prefix)
    with tracked_run("fraud-iforest", {"model_family": "fraud", "approach": "unsupervised"}) as run:
        X_tr = train[FEATURE_NAMES].to_numpy(dtype=np.float32)
        model = train_isolation_forest(X_tr, dict(cfg["isolation_forest"]), seed)
        mapper = PercentileMapper.fit(anomaly_scores(model, X_tr))
        s_te = anomaly_scores(model, test[FEATURE_NAMES].to_numpy(dtype=np.float32))
        y_te = test["is_fraud"].astype(int).to_numpy()
        pct_thr = float(load_policy()["rules"]["anomaly_review_percentile"])
        thr_score = float(np.interp(pct_thr, mapper.levels, mapper.q))
        metrics = classification_report(y_te, s_te, thr_score)
        meta = BundleMetadata(
            name="fraud_iforest",
            version=version,
            model_type="isolation_forest",
            feature_names=FEATURE_NAMES,
            params=dict(cfg["isolation_forest"]),
            metrics={"test": metrics},
            thresholds={"review_percentile": pct_thr, "review_score": thr_score},
            data={
                "dataset_version": tf.dataset_version,
                "trained_on": "train split, no labels",
                "train_end": tf.train_end.isoformat(),
            },
            extra={"score_quantiles": mapper.q.tolist(), "mlflow_run_id": run.run_id},
        )
        save_bundle(bundle, model, meta)
        run.log_params({"model": "fraud_iforest", "version": version, **dict(cfg["isolation_forest"])})
        run.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, prefix="test.")
        run.register_bundle(bundle, reg_name, "Isolation Forest transaction anomaly score")
    all_s = anomaly_scores(model, tf.df[FEATURE_NAMES].to_numpy(dtype=np.float32))
    return {"meta": meta, "test_score": s_te, "all_percentile": mapper(all_s)}


def train_behavioral(tf: TrainingFrame, settings: Settings) -> dict[str, Any]:
    train, test = tf.part("train"), tf.part("test")
    bundle = settings.path("models_dir") / "behavioral_anomaly"
    reg_name, prefix = REGISTERED["behavioral_anomaly"]
    version = next_version(reg_name, bundle, prefix)
    with tracked_run("behavioral-anomaly", {"model_family": "fraud", "approach": "customer-baseline"}) as run:
        model = BehavioralAnomalyModel.fit(train, version)
        s_te = model.score_frame(test)
        y_te = test["is_fraud"].astype(int).to_numpy()
        thr = float(np.quantile(model.score_frame(train), 0.99))
        metrics = classification_report(y_te, s_te, thr)
        meta = BundleMetadata(
            name="behavioral_anomaly",
            version=version,
            model_type="customer_baseline_fisher",
            feature_names=REQUIRED_FEATURES,
            metrics={"test": metrics},
            thresholds={"top_1pct_score": thr},
            data={"dataset_version": tf.dataset_version, "trained_on": "train split, no labels"},
            extra={"mlflow_run_id": run.run_id},
        )
        model.save(bundle, meta)
        run.log_params({"model": "behavioral_anomaly", "version": version})
        run.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, prefix="test.")
        run.register_bundle(bundle, reg_name, "Customer-baseline behavioural anomaly score")
    return {"meta": meta, "test_score": s_te, "all_percentile": model.percentile(tf.df)}


def compare(tf: TrainingFrame, scores: dict[str, np.ndarray], alert_rate: float = 0.01) -> dict[str, Any]:
    """Compare approaches on the test period at an equal alert budget."""
    test = tf.part("test")
    y = test["is_fraud"].astype(int).to_numpy()
    scen = test["fraud_scenario"].to_numpy(dtype=object)
    k = max(1, int(round(alert_rate * len(test))))
    flagged: dict[str, np.ndarray] = {}
    table: dict[str, Any] = {}
    for name, s in scores.items():
        order = np.argsort(-s, kind="stable")[:k]
        f = np.zeros(len(s), dtype=bool)
        f[order] = True
        flagged[name] = f
        rep = classification_report(y, s, threshold=float(np.sort(s)[-k]))
        p, r = precision_recall_at_k(y, s, k)
        table[name] = {
            "pr_auc": rep["pr_auc"],
            "roc_auc": rep["roc_auc"],
            f"precision_at_{k}": p,
            f"recall_at_{k}": r,
            "precision_at_100": rep["precision_at_100"],
            "recall_at_500": rep["recall_at_500"],
            "scenario_recall_at_budget": per_group_recall(y, f, scen),
        }
    names = list(scores)
    overlap = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = (flagged[a] & flagged[b]).sum()
            union = (flagged[a] | flagged[b]).sum()
            caught_a = flagged[a] & (y == 1)
            caught_b = flagged[b] & (y == 1)
            overlap[f"{a}__{b}"] = {
                "alert_jaccard": float(inter / union) if union else 0.0,
                f"fraud_caught_only_by_{a}": int((caught_a & ~caught_b).sum()),
                f"fraud_caught_only_by_{b}": int((caught_b & ~caught_a).sum()),
            }
    union_all = np.logical_or.reduce(list(flagged.values()))
    return {
        "alert_budget": {"rate": alert_rate, "k": k, "test_rows": int(len(test)), "test_fraud": int(y.sum())},
        "models": table,
        "overlap": overlap,
        "union_of_all_alerts": {
            "alerts": int(union_all.sum()),
            "recall": float(union_all[y == 1].mean()) if y.any() else 0.0,
        },
    }


def write_comparison_report(result: dict[str, Any], sup_meta: BundleMetadata, path_md, path_json) -> None:
    write_json(path_json, sanitize(result))
    k = result["alert_budget"]["k"]
    lines = [
        "# Fraud model comparison (test period)",
        "",
        "_Generated by `python -m ml train-fraud`; all numbers are computed from the current synthetic dataset._",
        "",
        f"Test rows: {result['alert_budget']['test_rows']:,} - fraud: {result['alert_budget']['test_fraud']:,} "
        f"- equal alert budget: top {result['alert_budget']['rate']:.0%} = {k:,} alerts per model.",
        "",
        f"| Model | PR-AUC | ROC-AUC | Precision@{k} | Recall@{k} |",
        "|---|---|---|---|---|",
    ]
    for name, m in result["models"].items():
        lines.append(
            f"| {name} | {m['pr_auc']:.3f} | {m['roc_auc']:.3f} | {m[f'precision_at_{k}']:.3f} | "
            f"{m[f'recall_at_{k}']:.3f} |"
        )
    lines += ["", "## Recall per fraud scenario at the equal alert budget", ""]
    scen = sorted({s for m in result["models"].values() for s in m["scenario_recall_at_budget"]})
    lines.append("| Scenario | positives | " + " | ".join(result["models"]) + " |")
    lines.append("|---|---|" + "---|" * len(result["models"]))
    for s in scen:
        pos = next(m["scenario_recall_at_budget"].get(s, {}).get("positives", 0) for m in result["models"].values())
        vals = [f"{m['scenario_recall_at_budget'].get(s, {}).get('recall', 0):.2f}" for m in result["models"].values()]
        lines.append(f"| {s} | {pos} | " + " | ".join(vals) + " |")
    lines += ["", "## Alert overlap", ""]
    for pair, o in result["overlap"].items():
        lines.append(
            f"- `{pair}`: " + ", ".join(f"{k2}={v:.3f}" if isinstance(v, float) else f"{k2}={v}" for k2, v in o.items())
        )
    u = result["union_of_all_alerts"]
    lines += [f"- Union of all three alert sets: {u['alerts']:,} alerts, recall {u['recall']:.3f}", ""]
    t = sup_meta.metrics["test"]
    lines += [
        "## Supervised model at policy thresholds",
        "",
        "| Operating point | threshold | precision | recall | F1 | alert rate | FP | TP |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label in ("at_suggested_threshold", "at_policy_review", "at_policy_block"):
        m = t[label]
        cm = m["confusion_matrix"]
        lines.append(
            f"| {label} | {m['threshold']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['f1']:.3f} | {m['alert_rate']:.4f} | {cm['fp']} | {cm['tp']} |"
        )
    cal = t["calibration"]
    lines += ["", f"Calibration (test): Brier={cal['brier']:.4f}, ECE={cal['ece']:.4f}", ""]
    with open(path_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def run(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    tf = load_training_frame(settings)
    log.info("training_frame", extra={"rows": len(tf.df), "train_end": str(tf.train_end), "val_end": str(tf.val_end)})
    sup = train_supervised(tf, settings)
    iso = train_iforest(tf, settings)
    beh = train_behavioral(tf, settings)
    test_mask = (tf.df["split"] == "test").to_numpy()
    result = compare(
        tf,
        {
            "A_xgboost_supervised": sup["scores"]["probability"].to_numpy()[test_mask],
            "B_isolation_forest": iso["test_score"],
            "C_behavioral_baseline": beh["test_score"],
        },
    )
    reports = settings.path("reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    write_comparison_report(
        result, sup["meta"], reports / "fraud_model_comparison.md", reports / "fraud_model_comparison.json"
    )
    write_json(reports / "fraud_xgb_metrics.json", sanitize(sup["meta"].metrics))
    write_json(reports / "fraud_xgb_shap_importance.json", sup["importance"][:25])
    scores = sup["scores"].rename(columns={"probability": "fraud_probability"})
    scores["anomaly_percentile"] = iso["all_percentile"]
    scores["behavioral_percentile"] = beh["all_percentile"]
    scores["fraud_model_version"] = sup["meta"].version
    out = settings.path("processed_dir") / "fraud_scores.parquet"
    scores.to_parquet(out, index=False)
    return {"supervised": sup["meta"], "comparison": result, "scores_path": str(out)}


if __name__ == "__main__":  # pragma: no cover
    run()

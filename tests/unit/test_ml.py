"""ML tests: leakage safety, schema stability, prediction shape, model loading, sane evaluation."""

import json

import numpy as np
import pandas as pd
import pytest

from features.definitions import ATO_FEATURES, FEATURE_NAMES
from ml.common.bundle import BundleMetadata, SchemaMismatchError, check_schema, load_metadata
from ml.common.data import load_training_frame
from ml.common.splits import known_labels, temporal_boundaries
from ml.common.xgb import XGBModel

pytestmark = pytest.mark.ml


def test_temporal_split_has_no_overlap(small_env):
    settings, _, _ = small_env
    tf = load_training_frame(settings)
    tr, va, te = tf.part("train"), tf.part("validation"), tf.part("test")
    assert tr["event_time"].max() < va["event_time"].min() <= va["event_time"].max() < te["event_time"].min()


def test_training_labels_only_include_reports_known_at_cutoff():
    labels = pd.Series([True, True, True, False])
    reported = pd.Series(pd.to_datetime(["2026-01-05", "2026-02-10", None, None], utc=True))
    assert known_labels(labels, reported, pd.Timestamp("2026-02-01", tz="UTC")).tolist() == [1, 0, 0, 0]


def test_metadata_records_label_policy_and_lineage(small_env):
    settings, _, _ = small_env
    meta = load_metadata(settings.path("models_dir") / "fraud_xgb")
    assert meta.data["label_policy"].startswith("labels reported before")
    assert meta.data["dataset_version"].startswith("tx-")
    assert meta.data["train_end"] < meta.data["validation_end"]
    assert meta.data["train_positives_known_at_cutoff"] <= meta.data["train_positives_ground_truth"]


def test_feature_schema_is_stable(small_env):
    settings, _, _ = small_env
    assert load_metadata(settings.path("models_dir") / "fraud_xgb").feature_names == FEATURE_NAMES
    assert load_metadata(settings.path("models_dir") / "ato_xgb").feature_names == ATO_FEATURES


def test_schema_mismatch_fails_fast():
    meta = BundleMetadata(name="m", version="v1", model_type="x", feature_names=["amount", "not_a_feature"])
    with pytest.raises(SchemaMismatchError):
        check_schema(meta, FEATURE_NAMES)


def test_model_loads_and_predicts_probabilities_of_right_shape(small_env):
    settings, _, _ = small_env
    model = XGBModel(settings.path("models_dir") / "fraud_xgb", available_features=FEATURE_NAMES)
    feats = pd.read_parquet(settings.path("processed_dir") / "transaction_features.parquet").head(50)
    p = model.predict_proba(feats)
    assert p.shape == (50,) and np.all((p >= 0) & (p <= 1))
    contrib = model.contributions(feats)
    assert contrib.shape == (50, len(FEATURE_NAMES) + 1)
    # SHAP additivity: contributions sum to the raw margin
    np.testing.assert_allclose(contrib.sum(axis=1), model.predict_margin(feats), rtol=1e-3, atol=1e-3)


def test_supervised_model_beats_chance_and_reports_imbalance_metrics(small_env):
    settings, _, res = small_env
    t = res["fraud"]["supervised"].metrics["test"]["at_policy_review"]
    assert t["pr_auc"] > 3 * t["prevalence"]  # far better than a random ranker
    assert t["roc_auc"] > 0.8
    for key in ("precision", "recall", "f1", "confusion_matrix", "precision_at_100", "recall_at_100"):
        assert key in t
    report = json.loads((settings.path("reports_dir") / "fraud_model_comparison.json").read_text())
    assert set(report["models"]) == {"A_xgboost_supervised", "B_isolation_forest", "C_behavioral_baseline"}


def test_batch_scores_cover_every_transaction(small_env):
    settings, ds, _ = small_env
    s = pd.read_parquet(settings.path("processed_dir") / "fraud_scores.parquet")
    assert len(s) == len(ds.transactions) and s["transaction_id"].is_unique
    assert {"fraud_probability", "ato_probability", "anomaly_percentile", "behavioral_percentile"} <= set(s.columns)


def test_churn_splits_are_leakage_safe(small_env):
    _, _, res = small_env
    snaps = {k: [pd.Timestamp(x) for x in v] for k, v in res["churn"]["snapshots"].items()}
    horizon = pd.Timedelta(days=res["churn"]["definition"]["inactivity_days"])
    assert max(snaps["train"]) + horizon <= snaps["validation"][0]
    assert snaps["validation"][0] + horizon <= snaps["test"][0]
    assert 0 <= res["churn"]["test"]["brier"] <= 0.25


def test_segmentation_and_graph_outputs(small_env):
    settings, _, res = small_env
    seg = pd.read_parquet(settings.path("processed_dir") / "customer_segments.parquet")
    lo, hi = settings["segmentation"]["k_range"]
    assert lo <= seg["segment_id"].nunique() <= hi
    g = pd.read_parquet(settings.path("processed_dir") / "graph_customer_metrics.parquet")
    assert g["mule_score"].between(0, 1).all()
    assert res["graph"]["mule_score"]["roc_auc"] > 0.7


def test_customer_profiles_use_policy_levels(small_env):
    settings, _, _ = small_env
    p = pd.read_parquet(settings.path("processed_dir") / "customer_profiles.parquet")
    assert set(p["risk_level"]) <= {"Low", "Medium", "High"}
    assert p["customer_id"].is_unique


def test_temporal_boundaries_follow_config(small_env):
    settings, ds, _ = small_env
    tr_end, va_end = temporal_boundaries(ds.transactions["event_time"], settings)
    assert tr_end < va_end

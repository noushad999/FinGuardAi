"""Unsupervised customer segmentation (K-Means vs. HDBSCAN).

Behavioural features per customer over the last ``segmentation.window_days``:
frequency, spend level, monthly spend, merchant diversity, and the mix of
P2P / cash-out / cash-in / bill / recharge / merchant activity.

* K-Means: k chosen by silhouette score over ``segmentation.k_range``.
* HDBSCAN (scikit-learn implementation): density-based, finds its own number
  of clusters and labels outliers as noise.

Clusters are named automatically from their centroid profile, and - because
the data is synthetic - we can check how well they recover the hidden personas
(adjusted Rand index). Personas are never used for fitting.
"""

from __future__ import annotations

import shutil
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from common.config import Settings, get_settings
from common.logging import get_logger
from ml.common.bundle import BundleMetadata, sanitize, save_metadata, write_json
from ml.common.tracking import next_version, tracked_run

log = get_logger(__name__)
REGISTERED_NAME, PREFIX = "finguard-segmentation", "segment-kmeans"

SEGMENT_FEATURES = [
    "log_txn_per_active_day",
    "active_day_ratio",
    "log_avg_debit",
    "log_monthly_spend",
    "merchant_diversity",
    "p2p_ratio",
    "cash_out_ratio",
    "cash_in_ratio",
    "bill_ratio",
    "recharge_ratio",
    "merchant_ratio",
    "bank_ratio",
    "p2p_received_ratio",
]


def customer_behaviour(tx: pd.DataFrame, as_of: pd.Timestamp, window_days: int) -> pd.DataFrame:
    start = as_of - pd.Timedelta(days=window_days)
    w = tx[(tx["event_time"] >= start) & (tx["event_time"] < as_of) & (tx["status"] == "SUCCESS")]
    g = w.groupby("customer_id")
    n = g.size()
    days = g["event_time"].agg(lambda s: s.dt.floor("D").nunique())
    debit = w[w["transaction_type"] != "CASH_IN"]
    spend_types = ["MERCHANT_PAYMENT", "BILL_PAYMENT", "MOBILE_RECHARGE"]
    spend = w[w["transaction_type"].isin(spend_types)].groupby("customer_id")["amount"].sum()
    types = pd.crosstab(w["customer_id"], w["transaction_type"])
    received = w[(w["transaction_type"] == "P2P")].groupby("receiver_id").size()
    out = pd.DataFrame(index=n.index)
    out["n_txn"] = n
    out["log_txn_per_active_day"] = np.log1p(n / days.clip(lower=1))
    out["active_day_ratio"] = days / window_days
    out["log_avg_debit"] = np.log1p(debit.groupby("customer_id")["amount"].mean()).reindex(out.index).fillna(0)
    out["log_monthly_spend"] = np.log1p(spend.reindex(out.index).fillna(0) / (window_days / 30))
    mp = w[w["transaction_type"] == "MERCHANT_PAYMENT"]
    out["merchant_diversity"] = (
        np.log1p(mp.groupby("customer_id")["merchant_id"].nunique()).reindex(out.index).fillna(0)
    )
    for t, col in [
        ("P2P", "p2p_ratio"),
        ("CASH_OUT", "cash_out_ratio"),
        ("CASH_IN", "cash_in_ratio"),
        ("BILL_PAYMENT", "bill_ratio"),
        ("MOBILE_RECHARGE", "recharge_ratio"),
        ("MERCHANT_PAYMENT", "merchant_ratio"),
        ("BANK_TRANSFER", "bank_ratio"),
    ]:
        out[col] = types.get(t, 0) / n
    out["p2p_received_ratio"] = received.reindex(out.index).fillna(0) / (n + received.reindex(out.index).fillna(0))
    out["monthly_spend"] = spend.reindex(out.index).fillna(0) / (window_days / 30)
    return out.fillna(0)


# Archetype prototypes in (standardised) feature space, used only for naming.
ARCHETYPES: dict[str, dict[str, float]] = {
    "Frequent Digital User": {
        "merchant_ratio": 1.0,
        "log_txn_per_active_day": 0.6,
        "merchant_diversity": 0.8,
        "active_day_ratio": 0.8,
    },
    "Cash-Out Remittance Receiver": {"cash_out_ratio": 1.2, "p2p_received_ratio": 1.0, "active_day_ratio": -0.6},
    "High-Value Trader": {"log_avg_debit": 1.0, "cash_in_ratio": 0.6, "p2p_ratio": 0.4, "log_txn_per_active_day": 0.8},
    "Bill-Paying Salaried": {"bill_ratio": 1.2, "log_monthly_spend": 0.4},
    "Bank-Linked Saver": {"bank_ratio": 1.4, "log_avg_debit": 0.5},
    "Mobile-First Low-Value": {"recharge_ratio": 1.2, "log_avg_debit": -0.8},
    "Occasional User": {"active_day_ratio": -1.0, "log_txn_per_active_day": -0.6, "merchant_diversity": -0.6},
}


def name_clusters(centroids: pd.DataFrame) -> dict[int, str]:
    """Greedy one-to-one assignment of archetype names by cosine similarity."""
    sims = []
    for cid, row in centroids.iterrows():
        for name, proto in ARCHETYPES.items():
            v = np.array([proto.get(f, 0.0) for f in centroids.columns])
            c = row.to_numpy()
            sims.append((float(c @ v / (np.linalg.norm(c) * np.linalg.norm(v) + 1e-9)), cid, name))
    names: dict[int, str] = {}
    used: set[str] = set()
    for _, cid, name in sorted(sims, reverse=True):
        if cid in names or name in used:
            continue
        names[cid] = name
        used.add(name)
    for cid in centroids.index:
        names.setdefault(cid, f"Mixed Behaviour {cid}")
    return names


def run(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    synth, processed = settings.path("synthetic_dir"), settings.path("processed_dir")
    cfg = settings["segmentation"]
    seed = int(settings["training"]["random_state"])
    tx = pd.read_parquet(synth / "transactions.parquet")
    tx["event_time"] = pd.to_datetime(tx["event_time"], utc=True)
    as_of = tx["event_time"].max().ceil("D")
    beh = customer_behaviour(tx, as_of, int(cfg["window_days"]))
    beh = beh[beh["n_txn"] >= 3]  # too little activity to characterise
    scaler = StandardScaler().fit(beh[SEGMENT_FEATURES])
    X = scaler.transform(beh[SEGMENT_FEATURES])
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(X), size=min(5000, len(X)), replace=False)

    k_lo, k_hi = cfg["k_range"]
    sil: dict[int, float] = {}
    models: dict[int, KMeans] = {}
    for k in range(int(k_lo), int(k_hi) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        models[k] = km
        sil[k] = float(silhouette_score(X[sample], km.labels_[sample]))
    best_k = max(sil, key=sil.get)
    km = models[best_k]
    labels = km.labels_

    hdb = HDBSCAN(min_cluster_size=max(15, len(X) // 100), min_samples=10).fit(X)
    h_labels = hdb.labels_
    non_noise = h_labels >= 0
    h_sil = (
        float(silhouette_score(X[non_noise], h_labels[non_noise]))
        if len(set(h_labels[non_noise])) > 1
        else float("nan")
    )

    truth = pd.read_parquet(synth / "customer_ground_truth.parquet").set_index("customer_id").reindex(beh.index)
    centroids = pd.DataFrame(km.cluster_centers_, columns=SEGMENT_FEATURES)
    names = name_clusters(centroids)
    beh["segment_id"] = labels
    beh["segment_name"] = [names[c] for c in labels]
    beh["hdbscan_cluster"] = h_labels

    profile = (
        beh.groupby("segment_name")
        .agg(
            customers=("n_txn", "size"),
            avg_txn_90d=("n_txn", "mean"),
            avg_monthly_spend=("monthly_spend", "mean"),
            p2p_ratio=("p2p_ratio", "mean"),
            cash_out_ratio=("cash_out_ratio", "mean"),
            merchant_ratio=("merchant_ratio", "mean"),
            bill_ratio=("bill_ratio", "mean"),
            recharge_ratio=("recharge_ratio", "mean"),
            bank_ratio=("bank_ratio", "mean"),
        )
        .round(3)
    )
    persona_mix = pd.crosstab(beh["segment_name"], truth["persona"], normalize="index").round(2)
    evaluation = {
        "customers_segmented": int(len(beh)),
        "window_days": int(cfg["window_days"]),
        "kmeans": {
            "silhouette_by_k": sil,
            "best_k": best_k,
            "silhouette": sil[best_k],
            "ari_vs_hidden_persona": float(adjusted_rand_score(truth["persona"], labels)),
        },
        "hdbscan": {
            "n_clusters": int(len(set(h_labels[non_noise]))),
            "noise_share": float((~non_noise).mean()),
            "silhouette_non_noise": h_sil,
            "ari_vs_hidden_persona": float(adjusted_rand_score(truth["persona"], h_labels)),
        },
        "segments": profile.reset_index().to_dict("records"),
        "persona_mix_by_segment": persona_mix.reset_index().to_dict("records"),
        "choice": "K-Means is deployed: stable, assigns every customer, and cheap to score; HDBSCAN is used to "
        "sanity-check cluster structure and surface outliers.",
    }

    bundle = settings.path("models_dir") / "segmentation"
    version = next_version(REGISTERED_NAME, bundle, PREFIX)
    with tracked_run("segmentation", {"model_family": "customer_segmentation"}) as run:
        if bundle.exists():
            shutil.rmtree(bundle)
        bundle.mkdir(parents=True)
        joblib.dump({"scaler": scaler, "kmeans": km, "names": names}, bundle / "model.joblib")
        meta = BundleMetadata(
            name="segmentation",
            version=version,
            model_type="kmeans",
            feature_names=SEGMENT_FEATURES,
            params={"k": best_k},
            metrics=sanitize(
                {k: v for k, v in evaluation.items() if k in ("kmeans", "hdbscan", "customers_segmented")}
            ),
            data={"as_of": as_of.isoformat()},
            extra={"segment_names": names, "mlflow_run_id": run.run_id},
        )
        save_metadata(bundle, meta)
        run.log_params({"best_k": best_k, "window_days": cfg["window_days"]})
        run.log_metrics(
            {
                "kmeans_silhouette": sil[best_k],
                "kmeans_ari": evaluation["kmeans"]["ari_vs_hidden_persona"],
                "hdbscan_noise_share": evaluation["hdbscan"]["noise_share"],
            }
        )
        run.register_bundle(bundle, REGISTERED_NAME, "Customer segmentation (K-Means)")

    out = beh.reset_index()[
        ["customer_id", "segment_id", "segment_name", "hdbscan_cluster", "n_txn", "monthly_spend", *SEGMENT_FEATURES]
    ]
    out["segmentation_version"] = version
    out.to_parquet(processed / "customer_segments.parquet", index=False)
    write_json(settings.path("reports_dir") / "segmentation.json", sanitize(evaluation))
    log.info(
        "segmentation_done",
        extra={"k": best_k, "silhouette": sil[best_k], "ari": evaluation["kmeans"]["ari_vs_hidden_persona"]},
    )
    return evaluation


if __name__ == "__main__":  # pragma: no cover
    run()

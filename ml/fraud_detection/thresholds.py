"""Threshold analysis for the business decision policy.

The risk engine's ``risk_score = 100 * max(P(fraud), P(ATO))``. This report
shows, on the *validation* period (mature labels only) and on the test period,
what each candidate review/block threshold means operationally: alert rate
(queue size), precision (share of alerts that are real fraud) and recall.

Fraud operations choose thresholds from this table; they are then written to
``configs/policy.yaml``. Validation drives the choice, test confirms it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from common.config import Settings, get_settings, load_policy
from ml.common.bundle import sanitize, write_json
from ml.common.data import load_training_frame
from ml.common.metrics import threshold_metrics
from ml.common.splits import known_labels

# Operating targets agreed with fraud operations (illustrative): ~4 in 5 reviewed
# alerts should be real; blocks (customer friction, call-centre cost) need >= 92.5%.
REVIEW_PRECISION = 0.80
BLOCK_PRECISION = 0.925
GRID = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 85, 90, 95]


def analyse(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    tf = load_training_frame(settings)
    scores = pd.read_parquet(settings.path("processed_dir") / "fraud_scores.parquet")
    df = (
        tf.df[["transaction_id", "event_time", "split", "is_fraud", "fraud_reported_at", "transaction_id"]]
        .loc[:, lambda d: ~d.columns.duplicated()]
        .merge(scores[["transaction_id", "fraud_probability", "ato_probability"]], on="transaction_id")
    )
    df["risk_score"] = 100 * np.maximum(df["fraud_probability"], df["ato_probability"].fillna(0))
    maturity = pd.Timedelta(days=float(settings["training"].get("label_maturity_days", 0)))
    val = df[(df["split"] == "validation") & (df["event_time"] < tf.val_end - maturity)]
    test = df[df["split"] == "test"]
    y_val = known_labels(val["is_fraud"], val["fraud_reported_at"], tf.val_end)
    y_test = test["is_fraud"].astype(int).to_numpy()
    rows = []
    for t in GRID:
        v = threshold_metrics(y_val, val["risk_score"].to_numpy(), t)
        te = threshold_metrics(y_test, test["risk_score"].to_numpy(), t)
        rows.append(
            {
                "risk_score_threshold": t,
                "val_precision": v["precision"],
                "val_recall": v["recall"],
                "val_alert_rate": v["alert_rate"],
                "test_precision": te["precision"],
                "test_recall": te["recall"],
                "test_alert_rate": te["alert_rate"],
                "test_alerts_per_10k": te["alert_rate"] * 10_000,
            }
        )
    table = pd.DataFrame(rows)

    # suggestion: review = lowest threshold with val precision >= REVIEW_PRECISION,
    # block = lowest threshold with val precision >= BLOCK_PRECISION
    def lowest(p: float) -> int | None:
        ok = table[table["val_precision"] >= p]
        return int(ok["risk_score_threshold"].min()) if len(ok) else None

    result = {
        "suggested_review": lowest(REVIEW_PRECISION),
        "suggested_block": lowest(BLOCK_PRECISION),
        "current_policy": load_policy()["thresholds"],
        "table": table.to_dict("records"),
    }
    reports = settings.path("reports_dir")
    write_json(reports / "threshold_analysis.json", sanitize(result))
    lines = [
        "# Decision threshold analysis",
        "",
        "risk_score = 100 x max(P(fraud), P(ATO)). Validation uses mature labels only; test uses ground truth.",
        "",
        f"Suggested review threshold (val precision >= {REVIEW_PRECISION:.3f}): **{result['suggested_review']}**  ",
        f"Suggested block threshold (val precision >= {BLOCK_PRECISION:.3f}): **{result['suggested_block']}**  ",
        f"Current policy: review={result['current_policy']['review']}, block={result['current_policy']['block']}",
        "",
        "| threshold | val precision | val recall | val alert rate | test precision | test recall | "
        "test alerts / 10k txns |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['risk_score_threshold']} | {r['val_precision']:.3f} | {r['val_recall']:.3f} | "
            f"{r['val_alert_rate']:.4f} | {r['test_precision']:.3f} | {r['test_recall']:.3f} | "
            f"{r['test_alerts_per_10k']:.1f} |"
        )
    (reports / "threshold_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":  # pragma: no cover
    analyse()

"""Evaluation metrics for imbalanced classification and ranking.

Accuracy is deliberately absent: with ~1% positives a model that never flags
fraud is 99% "accurate". We report ranking quality (PR-AUC, ROC-AUC),
thresholded quality (precision/recall/F1/confusion matrix), alert-budget
metrics (precision@k / recall@k) and calibration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    k = min(k, len(scores))
    top = np.argsort(-np.asarray(scores), kind="stable")[:k]
    hits = int(y_true[top].sum())
    positives = int(y_true.sum())
    return hits / k if k else 0.0, hits / positives if positives else 0.0


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "alert_rate": float(pred.mean()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def classification_report(
    y_true: np.ndarray, scores: np.ndarray, threshold: float, ks: tuple[int, ...] = (50, 100, 500)
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "prevalence": float(y_true.mean()) if len(y_true) else 0.0,
        "pr_auc": float(average_precision_score(y_true, scores)) if y_true.any() else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, scores)) if 0 < y_true.sum() < len(y_true) else float("nan"),
    }
    out.update(threshold_metrics(y_true, scores, threshold))
    for k in ks:
        p, r = precision_recall_at_k(y_true, scores, k)
        out[f"precision_at_{k}"] = p
        out[f"recall_at_{k}"] = r
    return out


def select_threshold(y_true: np.ndarray, scores: np.ndarray, target_precision: float | None = None) -> dict[str, float]:
    """Pick a threshold on *validation* data.

    Returns the max-F1 threshold and, if requested, the lowest threshold whose
    precision reaches ``target_precision`` (maximising recall under a
    precision constraint - how fraud ops usually think about alert queues).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0)
    best = int(np.argmax(f1)) if len(f1) else 0
    result = {
        "max_f1_threshold": float(thresholds[best]) if len(thresholds) else 0.5,
        "max_f1": float(f1[best]) if len(f1) else 0.0,
    }
    if target_precision is not None and len(thresholds):
        ok = np.where(precision >= target_precision)[0]
        idx = int(ok[0]) if len(ok) else best
        result["target_precision"] = float(target_precision)
        result["precision_threshold"] = float(thresholds[idx])
        result["recall_at_precision_threshold"] = float(recall[idx])
    return result


def calibration_report(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    """Brier score, expected calibration error and a reliability table (quantile bins)."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.clip(np.asarray(probs, dtype=float), 0, 1)
    edges = np.unique(np.quantile(probs, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, probs, side="right") - 1, 0, max(len(edges) - 2, 0))
    rows = []
    ece = 0.0
    for b in range(max(len(edges) - 1, 1)):
        m = idx == b
        if not m.any():
            continue
        mean_p, frac = float(probs[m].mean()), float(y_true[m].mean())
        ece += m.mean() * abs(mean_p - frac)
        rows.append({"bin": b, "n": int(m.sum()), "mean_predicted": mean_p, "observed_rate": frac})
    return {"brier": float(brier_score_loss(y_true, probs)), "ece": float(ece), "reliability": rows}


def per_group_recall(y_true: np.ndarray, flagged: np.ndarray, groups: np.ndarray) -> dict[str, dict[str, float]]:
    """Recall per group (e.g. fraud scenario) for flagged positives."""
    out: dict[str, dict[str, float]] = {}
    y_true = np.asarray(y_true).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    for g in sorted({x for x in groups[y_true] if isinstance(x, str)}):
        m = y_true & (groups == g)
        out[g] = {"positives": int(m.sum()), "recall": float(flagged[m].mean()) if m.any() else 0.0}
    return out

"""Metric helpers for imbalanced classification (PRD §4.5)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def full_metric_suite(
    y_true: Iterable[int], y_pred: Iterable[int], y_proba: Iterable[float]
) -> dict[str, float]:
    """Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, Brier."""
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    y_proba = np.asarray(list(y_proba))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def precision_recall_thresholds(y_true: np.ndarray, y_proba: np.ndarray):
    return precision_recall_curve(y_true, y_proba)


def value_based_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    intervention_cost: float,
    purchase_value: float,
    min_recall: float,
) -> dict[str, float]:
    """Pick threshold maximising expected value at >= min_recall (PRD §4.5)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    best = {"threshold": 0.5, "expected_value": -1e18, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    n_pos = int(y_true.sum())
    for idx, t in enumerate(thresholds):
        rec = float(recall[idx])
        prec = float(precision[idx])
        if rec < min_recall:
            continue
        tp = rec * n_pos
        fp = (tp / prec) * (1 - prec) if prec > 0 else np.nan
        if np.isnan(fp):
            continue
        ev = tp * purchase_value - fp * intervention_cost
        if ev > best["expected_value"]:
            best = {
                "threshold": float(t),
                "expected_value": float(ev),
                "precision": prec,
                "recall": rec,
                "f1": float(2 * prec * rec / max(prec + rec, 1e-9)),
            }
    return best


def calibration_bucket_report(
    y_true: np.ndarray, y_proba: np.ndarray, n_buckets: int = 10
) -> dict[str, float]:
    """Reliability curve data for the dashboard (PRD §11 calibration stretch)."""
    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    out = {"n_buckets": n_buckets, "buckets": []}
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_proba >= lo) & (y_proba < hi if i < n_buckets - 1 else y_proba <= hi)
        n = int(mask.sum())
        if n == 0:
            out["buckets"].append(
                {"lo": float(lo), "hi": float(hi), "n": 0, "mean_pred": 0.0, "frac_pos": 0.0}
            )
        else:
            out["buckets"].append(
                {
                    "lo": float(lo),
                    "hi": float(hi),
                    "n": n,
                    "mean_pred": float(y_proba[mask].mean()),
                    "frac_pos": float(y_true[mask].mean()),
                }
            )
    return out

"""Metric and decision-threshold helpers shared by training and evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, proba, threshold: float, prefix: str = "") -> dict[str, float]:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return {f"{prefix}{k}": (float(v) if isinstance(v, float | np.floating) else v) for k, v in out.items()}


def expected_cost(y_true, proba, threshold: float, cost_fn: float, cost_fp: float) -> float:
    """Average misclassification cost per vehicle at a given threshold."""
    y_true = np.asarray(y_true)
    pred = np.asarray(proba) >= threshold
    fn = np.sum((y_true == 1) & ~pred)
    fp = np.sum((y_true == 0) & pred)
    return float((cost_fn * fn + cost_fp * fp) / len(y_true))


def cost_curve(y_true, proba, cost_fn: float, cost_fp: float, n: int = 199):
    thresholds = np.linspace(0.005, 0.995, n)
    costs = np.array([expected_cost(y_true, proba, t, cost_fn, cost_fp) for t in thresholds])
    return thresholds, costs


def optimal_threshold(y_true, proba, cost_fn: float, cost_fp: float) -> tuple[float, float]:
    """Threshold that minimises expected cost on the given (validation) fold."""
    thresholds, costs = cost_curve(y_true, proba, cost_fn, cost_fp)
    best = int(np.argmin(costs))
    return float(round(thresholds[best], 4)), float(costs[best])

"""
Shared Evaluation Metrics
==========================
All metrics used across the ARTEMIS benchmark suite.

Regression:   rmse, mae, rank_ic, weighted_r2, directional_accuracy
Classification: classification_metrics  (AUC, F1, accuracy)
Printing:     print_summary_table  (accepts list[dict] or dict[str,dict])
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation between predictions and actuals."""
    corr, _ = spearmanr(y_pred, y_true)
    return float(corr) if np.isfinite(corr) else 0.0


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted R² with weights = 1/y_true²  (matches RMSPE weighting).
    Robust to scale differences across instruments.
    """
    mask = np.abs(y_true) > 1e-12
    if mask.sum() < 2:
        return 0.0
    yt, yp = y_true[mask], y_pred[mask]
    w  = 1.0 / (yt ** 2 + 1e-16)
    wn = w / w.sum()
    ss_res = float(np.sum(wn * (yt - yp) ** 2))
    ss_tot = float(np.sum(wn * (yt - np.average(yt, weights=wn)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-16)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of samples where sign(pred) == sign(actual)."""
    return float(np.mean(np.sign(y_pred) == np.sign(y_true)))


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def _optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Search threshold in [0.05, 0.95] that maximises F1."""
    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.05, 0.95, 91):
        yb = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, yb, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """
    Full classification metric suite.

    Parameters
    ----------
    y_true     : (N,) ground-truth binary labels
    y_prob     : (N,) predicted probabilities for the positive class
    threshold  : decision threshold; if None, optimal F1 threshold is found

    Returns dict with keys: auc, f1, acc, precision, recall, pr_auc, threshold
    """
    if threshold is None:
        threshold = _optimal_threshold(y_true, y_prob)

    y_bin = (y_prob >= threshold).astype(int)

    try:
        auc   = float(roc_auc_score(y_true, y_prob))
        prauc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        auc = prauc = 0.5

    return {
        "auc":       round(auc,   4),
        "f1":        round(float(f1_score(y_true, y_bin, zero_division=0)), 4),
        "acc":       round(float(accuracy_score(y_true, y_bin)), 4),
        "precision": round(float(precision_score(y_true, y_bin, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_bin, zero_division=0)), 4),
        "pr_auc":    round(prauc, 4),
        "threshold": round(float(threshold), 4),
    }


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def print_summary_table(rows):
    """
    Print a formatted results table to stdout.

    Accepts either:
        list[dict]          — each dict has a 'Model' key plus metric keys
        dict[str, dict]     — model name → metrics dict  (legacy format)
    """
    # normalise to list[dict]
    if isinstance(rows, dict):
        rows = [{"Model": name, **m} for name, m in rows.items()]

    if not rows:
        print("  (no results)")
        return

    # determine columns (preserve insertion order)
    all_keys = list(rows[0].keys())
    col_widths = {k: max(len(str(k)), max(len(str(r.get(k, ""))) for r in rows))
                  for k in all_keys}

    def fmt_row(r):
        return "  " + "  ".join(str(r.get(k, "")).ljust(col_widths[k]) for k in all_keys)

    header = fmt_row({k: k for k in all_keys})
    sep    = "  " + "-" * (len(header) - 2)

    print(header)
    print(sep)
    for r in rows:
        print(fmt_row(r))
    print()

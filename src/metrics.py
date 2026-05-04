"""
Evaluation Metrics

All metrics used across KATS experiments:
- Primary: Recall_High (Recall for High-priority class)
- Secondary: Macro_F1, Cohen's Kappa
- Generalization: Jensen-Shannon Divergence (E2)
- Statistical: McNemar test (E1)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    recall_score, f1_score, cohen_kappa_score,
    confusion_matrix, precision_score,
)
from scipy.spatial.distance import jensenshannon
from statsmodels.stats.contingency_tables import mcnemar
from typing import Tuple


def recall_high(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall for High-priority class (label=2). Primary KATS metric."""
    return recall_score(y_true, y_pred, labels=[2], average='macro', zero_division=0)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1 score."""
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cohen's Kappa."""
    return cohen_kappa_score(y_true, y_pred)


def precision_high(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Precision for High-priority class (label=2)."""
    return precision_score(y_true, y_pred, labels=[2], average='macro', zero_division=0)


def full_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all primary metrics in one call."""
    return {
        'Recall_High':    round(recall_high(y_true, y_pred), 4),
        'Macro_F1':       round(macro_f1(y_true, y_pred), 4),
        'Kappa':          round(kappa(y_true, y_pred), 4),
        'Precision_High': round(precision_high(y_true, y_pred), 4),
    }


def mcnemar_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> Tuple[float, float]:
    """
    McNemar's test comparing two classifiers on the same test set.

    Returns
    -------
    (statistic, p_value)
        p < 0.05 indicates statistically significant difference.
    """
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)
    n01 = np.sum(correct_a & ~correct_b)  # A correct, B wrong
    n10 = np.sum(~correct_a & correct_b)  # A wrong, B correct
    table = np.array([[0, n01], [n10, 0]])  # 2x2 table
    result = mcnemar(table, exact=True)
    return result.statistic, result.pvalue


def jsd_feature(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    n_bins: int = 20,
) -> float:
    """
    Jensen-Shannon Divergence between two 1D continuous distributions.
    Used in E2 to measure feature distribution shift between KATS-SYN
    and real production datasets.

    Returns
    -------
    float in [0, 1]. 0 = identical distributions, 1 = maximally different.
    """
    all_vals = np.concatenate([dist_a, dist_b])
    bins     = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)
    p = np.histogram(dist_a, bins=bins, density=True)[0] + 1e-10
    q = np.histogram(dist_b, bins=bins, density=True)[0] + 1e-10
    p /= p.sum(); q /= q.sum()
    return round(float(jensenshannon(p, q)), 4)

"""
Baseline Methods (B1–B7)

All 7 baselines used in E1 comparison against KATS-Ensemble.
Rule-based baselines (B1–B3, B6–B7) require no training.
ML baselines (B4–B5) are trained on the same KATS-SYN split.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, f1_score, cohen_kappa_score
from typing import Union


# ── Rule-Based Baselines ─────────────────────────────────────────────────

def b1_criticality_rank(X: pd.DataFrame) -> np.ndarray:
    """
    B1: Sort by service_criticality only.
    Baseline representing current operational practice.
    """
    score = X['service_criticality'].values / 10.0
    return pd.qcut(pd.Series(score).rank(method='first'), q=3, labels=[0, 1, 2]).astype(int).values


def b2_rto_first(X: pd.DataFrame) -> np.ndarray:
    """
    B2: Sort by RTO (shortest RTO = most urgent = High).
    """
    score = 1.0 - X['rto_minutes'].clip(0, 1440).values / 1440.0
    return pd.qcut(pd.Series(score).rank(method='first'), q=3, labels=[0, 1, 2]).astype(int).values


def b3_composite_rule(X: pd.DataFrame) -> np.ndarray:
    """
    B3: Weighted composite rule — criticality + RTO + AZ risk.
    Represents best-practice manual triage.
    """
    sc  = X['service_criticality'].values / 10.0
    rt  = X['rto_minutes'].clip(0, 1440).values / 1440.0
    az  = X['az_risk_score'].values
    score = 0.5 * sc + 0.3 * (1 - rt) + 0.2 * az
    return pd.qcut(pd.Series(score).rank(method='first'), q=3, labels=[0, 1, 2]).astype(int).values


def b6_deadline_first(X: pd.DataFrame) -> np.ndarray:
    """
    B6: Earliest-deadline-first (EDF) scheduling.
    """
    score = 1.0 - X['rpo_minutes'].clip(0, 2880).values / 2880.0
    return pd.qcut(pd.Series(score).rank(method='first'), q=3, labels=[0, 1, 2]).astype(int).values


def b7_connectivity_rank(X: pd.DataFrame) -> np.ndarray:
    """
    B7: Rank by dependency_count + downstream_critical (graph centrality proxy).
    """
    score = (X['dependency_count'].values / 15.0 * 0.6 +
             X['downstream_critical'].values * 0.4)
    return pd.qcut(pd.Series(score).rank(method='first'), q=3, labels=[0, 1, 2]).astype(int).values


# ── ML Baselines ─────────────────────────────────────────────────────────

def build_b4_logreg(seed: int = 42) -> Pipeline:
    """B4: Logistic Regression baseline."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    LogisticRegression(max_iter=1000, random_state=seed,
                                      class_weight='balanced', n_jobs=-1)),
    ])


def build_b5_dectree(seed: int = 42) -> Pipeline:
    """B5: Decision Tree baseline."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    DecisionTreeClassifier(max_depth=10, random_state=seed,
                                          class_weight='balanced')),
    ])


# ── Evaluation Helper ─────────────────────────────────────────────────────

def evaluate(preds: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Compute Recall_High, Macro_F1, and Cohen's Kappa.
    Recall_High is the primary metric (High=2 label).
    """
    return {
        'Recall_High': round(recall_score(y_true, preds, labels=[2], average='macro', zero_division=0), 4),
        'Macro_F1':    round(f1_score(y_true, preds, average='macro', zero_division=0), 4),
        'Kappa':       round(cohen_kappa_score(y_true, preds), 4),
    }

"""
KATS-Ensemble Model Definition

Asymmetric-loss LightGBM + Calibrated Naive Bayes ensemble.
Trained with alpha-weighted loss to maximize Recall_High — the
operationally critical metric where missing a High-priority service
during a cyberattack has catastrophic consequences.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from typing import Optional

from .dataset import KATS_FEATURES


DEFAULT_ALPHA = 5  # Asymmetric loss ratio — chosen via E7 sensitivity analysis


def _make_asymmetric_weights(
    y_train: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    label_map: dict = None,
) -> np.ndarray:
    """
    Compute per-sample weights giving alpha× more weight to High-class errors.

    Parameters
    ----------
    y_train : array-like of int
        Encoded labels (2=High, 1=Medium, 0=Low).
    alpha : float
        Loss asymmetry ratio. E7 shows optimal tradeoff at alpha=5.
    """
    weights = np.ones(len(y_train), dtype=float)
    weights[y_train == 2] = alpha  # High class gets alpha× weight
    return weights


def build_kats_ensemble(
    alpha: float = DEFAULT_ALPHA,
    lgbm_params: Optional[dict] = None,
    seed: int = 42,
) -> Pipeline:
    """
    Build the KATS-Ensemble pipeline.

    Architecture:
        StandardScaler
        → VotingClassifier (soft voting):
            ├─ LGBMClassifier (asymmetric loss via sample_weight)
            └─ CalibratedClassifierCV(GaussianNB)

    Parameters
    ----------
    alpha : float
        Asymmetric loss ratio for High-class samples.
    lgbm_params : dict, optional
        Override default LightGBM hyperparameters.
    seed : int
        Random seed.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Untrained KATS-Ensemble pipeline.
    """
    default_lgbm = {
        'n_estimators':    400,
        'learning_rate':   0.05,
        'max_depth':       8,
        'num_leaves':      63,
        'min_child_samples': 20,
        'subsample':       0.8,
        'colsample_bytree': 0.8,
        'class_weight':    'balanced',
        'random_state':    seed,
        'n_jobs':          -1,
        'verbose':         -1,
    }
    if lgbm_params:
        default_lgbm.update(lgbm_params)

    lgbm = LGBMClassifier(**default_lgbm)
    nb   = CalibratedClassifierCV(GaussianNB(), cv=3, method='isotonic')

    ensemble = VotingClassifier(
        estimators=[('lgbm', lgbm), ('nb', nb)],
        voting='soft',
        weights=[0.85, 0.15],
    )

    pipe = Pipeline([
        ('scaler',   StandardScaler()),
        ('ensemble', ensemble),
    ])
    return pipe


def train_kats(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
) -> Pipeline:
    """
    Train the KATS-Ensemble with asymmetric sample weights.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (KATS_FEATURES columns).
    y_train : array-like
        Integer-encoded labels (2=High, 1=Medium, 0=Low).
    alpha : float
        Asymmetric loss ratio (default=5, per E7).

    Returns
    -------
    Trained Pipeline.
    """
    pipe   = build_kats_ensemble(alpha=alpha, seed=seed)
    y_arr  = np.asarray(y_train)
    weights = _make_asymmetric_weights(y_arr, alpha=alpha)

    # Pass sample_weight to LightGBM via fit_params
    pipe.fit(
        X_train, y_arr,
        ensemble__lgbm__sample_weight=weights,
    )
    return pipe

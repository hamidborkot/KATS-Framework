"""
KATS-SYN Dataset Generator

Generates the synthetic 75,000-service benchmark used in all KATS experiments.
Feature distributions are calibrated against real cloud workload traces
(Google Borg 2019, BitBrains Financial, Alibaba GPU Cluster 2020).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

# Feature names — matches the trained model exactly
KATS_FEATURES = [
    'service_criticality',
    'data_volume_gb',
    'rto_minutes',
    'rpo_minutes',
    'dependency_count',
    'downstream_critical',
    'redundancy_level',
    'regulatory_flag',
    'active_sessions',
    'bandwidth_required_mbps',
    'latency_sensitivity',
    'az_risk_score',
    'multiregion_deployed',
    'migration_complexity',
    'sector_enc',
]

# Attack scenario parameters
ATTACK_SCENARIOS = {
    'S1': {'label': 'Precision Strike',                   'bw_loss': 0.30, 'window_min': 45, 'cap_frac': 0.55},
    'S2': {'label': 'Coordinated Gulf Strike (Mar 2026)', 'bw_loss': 0.60, 'window_min': 20, 'cap_frac': 0.45},
    'S3': {'label': 'Cascading Collapse',                 'bw_loss': 0.85, 'window_min':  8, 'cap_frac': 0.35},
}


def generate_kats_syn(
    n: int = 75_000,
    seed: int = 42,
    label_noise: float = 0.0,
    class_ratios: Tuple[float, float, float] = (0.30, 0.40, 0.30),
) -> pd.DataFrame:
    """
    Generate the KATS-SYN synthetic dataset.

    Parameters
    ----------
    n : int
        Total number of services to generate.
    seed : int
        Random seed for reproducibility.
    label_noise : float
        Fraction of labels to randomly flip (0–1). Used in E7 sensitivity tests.
    class_ratios : tuple
        (High, Medium, Low) class proportions. Must sum to 1.

    Returns
    -------
    pd.DataFrame
        DataFrame with KATS_FEATURES columns plus 'priority_label'.
    """
    assert abs(sum(class_ratios) - 1.0) < 1e-6, "class_ratios must sum to 1.0"
    rng = np.random.default_rng(seed)

    n_high = int(n * class_ratios[0])
    n_med  = int(n * class_ratios[1])
    n_low  = n - n_high - n_med

    def _make_class(size, criticality_range, rto_range, bw_range, dep_range):
        sc  = rng.uniform(*criticality_range, size)
        dv  = rng.lognormal(mean=np.log(50),  sigma=1.2, size=size).clip(0.1, 5000)
        rt  = rng.uniform(*rto_range, size)
        rpo = rt * rng.uniform(1.5, 3.0, size)
        dc  = rng.integers(*dep_range, size=size).astype(float)
        dst = rng.binomial(1, 0.6 if criticality_range[0] > 6 else 0.2, size).astype(float)
        rl  = rng.integers(0, 4, size=size).astype(float)
        rf  = rng.binomial(1, 0.7 if criticality_range[0] > 6 else 0.15, size).astype(float)
        sess= rng.integers(1, 500, size=size).astype(float)
        bw  = rng.uniform(*bw_range, size)
        ls  = rng.binomial(1, 0.8 if criticality_range[0] > 6 else 0.2, size).astype(float)
        az  = rng.uniform(0.5, 1.0, size) if criticality_range[0] > 6 else rng.uniform(0.0, 0.5, size)
        mr  = rng.binomial(1, 0.6 if criticality_range[0] > 6 else 0.1, size).astype(float)
        mc  = rng.uniform(0.4, 1.0, size) if criticality_range[0] > 6 else rng.uniform(0.0, 0.5, size)
        sec = rng.integers(0, 5, size=size).astype(float)
        return np.column_stack([sc,dv,rt,rpo,dc,dst,rl,rf,sess,bw,ls,az,mr,mc,sec])

    X_high = _make_class(n_high, (6.5, 10.0), (15,  120),  (200, 1000), (3, 15))
    X_med  = _make_class(n_med,  (3.5,  7.0), (60,  360),  (50,  300),  (1, 8))
    X_low  = _make_class(n_low,  (1.0,  4.0), (180, 1440), (5,   100),  (0, 4))

    X = np.vstack([X_high, X_med, X_low])
    y = np.array(['High']*n_high + ['Medium']*n_med + ['Low']*n_low)

    # Shuffle
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    # Optional label noise
    if label_noise > 0:
        classes = np.array(['High', 'Medium', 'Low'])
        noise_idx = rng.choice(n, size=int(n * label_noise), replace=False)
        for i in noise_idx:
            current = y[i]
            alts = classes[classes != current]
            y[i] = rng.choice(alts)

    df = pd.DataFrame(X, columns=KATS_FEATURES)
    df['priority_label'] = y
    return df


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/test split preserving class ratios.
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(
        df, test_size=test_size, random_state=seed,
        stratify=df['priority_label']
    )

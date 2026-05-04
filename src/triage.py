"""
Attack Scenario Simulator

Simulates three cyberattack scenarios (S1–S3) and evaluates
the survivability of High-priority services under bandwidth constraints.

Reference scenario: S2 Coordinated Gulf Strike (March 2026)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from .dataset import ATTACK_SCENARIOS


def run_triage_scenario(
    scores: np.ndarray,
    df_services: pd.DataFrame,
    y_true: np.ndarray,
    bw_loss: float,
    window_min: float,
    cap_frac: float,
    high_first: bool = False,
) -> Dict:
    """
    Simulate migration triage under a single attack scenario.

    Parameters
    ----------
    scores : np.ndarray
        Model confidence scores for High class (shape: [n_services]).
    df_services : pd.DataFrame
        Service metadata including bandwidth_required_mbps and data_volume_gb.
    y_true : np.ndarray
        True priority labels (2=High, 1=Medium, 0=Low).
    bw_loss : float
        Fraction of bandwidth destroyed by attack (0–1).
    window_min : float
        Migration window in minutes.
    cap_frac : float
        Budget cap as fraction of remaining effective bandwidth.
    high_first : bool
        If True, prioritize High-confidence predictions first (KATS behavior).

    Returns
    -------
    dict with keys: survivability, rescued_high, total_high, services_migrated
    """
    bw_factor  = 1.0 - bw_loss
    eff_bw     = df_services['bandwidth_required_mbps'].clip(0.5).values * bw_factor
    data_vol   = df_services['data_volume_gb'].values

    # Transfer time in minutes
    xfer_time  = (data_vol * 1024 / eff_bw) / 60.0
    migratable = (xfer_time <= window_min) | (data_vol <= 2.0)

    is_high    = (y_true == 2)
    budget     = eff_bw[migratable].sum() * cap_frac
    mig_idx    = np.where(migratable)[0]

    # Ranking order
    if high_first:
        confident  = scores[mig_idx] >= 0.80
        order = np.concatenate([
            mig_idx[confident][np.argsort(-scores[mig_idx][confident])],
            mig_idx[~confident][np.argsort(-scores[mig_idx][~confident])],
        ])
    else:
        order = mig_idx[np.argsort(-scores[mig_idx])]

    bw_used = rescued = selected = 0
    for idx in order:
        bw_req = eff_bw[idx]
        if bw_used + bw_req > budget:
            continue
        bw_used  += bw_req
        selected += 1
        if is_high[idx]:
            rescued += 1

    total_high    = int(is_high.sum())
    survivability = rescued / max(total_high, 1)

    return {
        'survivability':    round(survivability, 4),
        'rescued_high':     rescued,
        'total_high':       total_high,
        'services_migrated': selected,
    }


def run_all_scenarios(
    model_scores_dict: Dict[str, np.ndarray],
    df_services: pd.DataFrame,
    y_true: np.ndarray,
    high_first_methods: Tuple[str, ...] = ('KATS-Ensemble',),
) -> pd.DataFrame:
    """
    Run all 3 attack scenarios for all provided methods.

    Parameters
    ----------
    model_scores_dict : dict
        {method_name: confidence_scores_array}
    df_services : pd.DataFrame
        Service metadata.
    y_true : np.ndarray
        True labels.
    high_first_methods : tuple
        Method names that use high-confidence-first ordering.

    Returns
    -------
    pd.DataFrame with columns: Scenario, Method, Survivability, ...
    """
    rows = []
    for sc_key, sc_params in ATTACK_SCENARIOS.items():
        for method, scores in model_scores_dict.items():
            result = run_triage_scenario(
                scores       = scores,
                df_services  = df_services,
                y_true       = y_true,
                bw_loss      = sc_params['bw_loss'],
                window_min   = sc_params['window_min'],
                cap_frac     = sc_params['cap_frac'],
                high_first   = (method in high_first_methods),
            )
            rows.append({
                'Scenario': sc_params['label'],
                'Method':   method,
                **result,
            })
    return pd.DataFrame(rows)

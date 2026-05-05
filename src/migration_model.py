"""
migration_model.py — KATS Framework v7

Physical live-migration time model for cloud service triage.
Based on: VMware vMotion technical documentation, Hines et al. (2009)
"Post-copy based live virtual machine migration using adaptive pre-paging
and dynamic self-ballooning", and empirical cloud provider SLA data.

Formula (v7 — calibrated for P20≈8min, P50≈20min, P85≈45min):
    t_base = migration_complexity × 2.0          (minutes)
    t_load = (active_sessions / 2000) × mc       (dirty-page churn)
    t_data = 0.05 × data_volume_gb × 8 /         (5% stateful data,
             bandwidth_required_mbps / 60          converted to minutes)
    t_total = t_base + t_load + t_data

Attack Scenario Parameters (empirically calibrated to Mar 2026 Gulf events):
    S1 Precision Strike       : window=45 min, max_concurrent=2000 lanes
    S2 Gulf Strike (Mar 2026) : window=20 min, max_concurrent=1000 lanes
    S3 Cascading Collapse     : window=8  min, max_concurrent=300  lanes
"""

import numpy as np
import pandas as pd


SCENARIOS = {
    'S1: Precision Strike': {
        'window_min':     45,
        'max_concurrent': 2000,
        'bw_loss_pct':    0.30,
        'description':    '30% BW loss, 45-min window, 1 AZ affected',
    },
    'S2: Coordinated Gulf Strike (Mar 2026)': {
        'window_min':     20,
        'max_concurrent': 1000,
        'bw_loss_pct':    0.60,
        'description':    '60% BW loss, 20-min window, 2 AZs affected — Mar 2026 Gulf event',
    },
    'S3: Cascading Collapse': {
        'window_min':     8,
        'max_concurrent': 300,
        'bw_loss_pct':    0.85,
        'description':    '85% BW loss, 8-min window, 3 AZs affected',
    },
}


def compute_migration_time(df: pd.DataFrame) -> np.ndarray:
    """
    Compute per-service live-migration time in minutes.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: migration_complexity, active_sessions,
        data_volume_gb, bandwidth_required_mbps

    Returns
    -------
    np.ndarray of shape (n,) — migration time in minutes per service
    """
    mc  = df['migration_complexity'].values.astype(float)
    act = df['active_sessions'].values.astype(float)
    dv  = df['data_volume_gb'].values.astype(float)
    bw  = np.clip(df['bandwidth_required_mbps'].values.astype(float), 10.0, 1000.0)

    t_base = mc * 2.0
    t_load = (act / 2000.0) * mc
    t_data = (0.05 * dv * 8.0) / bw / 60.0

    return t_base + t_load + t_data


def survivability_lane_model(
    df: pd.DataFrame,
    priority_scores: np.ndarray,
    scenario: dict,
    high_label: str = 'High',
) -> dict:
    """
    Concurrency-lane survivability model.

    Services are ranked by priority_scores (descending) and
    allocated to migration lanes (max_concurrent) greedily.
    A service is migrated if a lane has remaining capacity
    within the attack window.

    Parameters
    ----------
    df              : pd.DataFrame with service inventory
    priority_scores : np.ndarray — P(High) scores, shape (n,)
    scenario        : dict from SCENARIOS
    high_label      : label string for High-priority class

    Returns
    -------
    dict with keys: Survivability, Rescued_High, Total_High,
                    N_Migratable, Services_Migrated, Lanes_Utilised,
                    Avg_Lane_Load_pct
    """
    df = df.copy().reset_index(drop=True)
    df['t_mig']      = compute_migration_time(df)
    df['migratable'] = df['t_mig'] <= scenario['window_min']
    df['score']      = priority_scores

    n_high_total = int((df['priority_label'] == high_label).sum())
    df_mig = df[df['migratable']].sort_values('score', ascending=False)\
               .reset_index(drop=True)

    if len(df_mig) == 0 or n_high_total == 0:
        return dict(Survivability=0.0, Rescued_High=0,
                    Total_High=n_high_total, N_Migratable=0,
                    Services_Migrated=0, Lanes_Utilised=0,
                    Avg_Lane_Load_pct=0.0)

    max_c     = scenario['max_concurrent']
    window    = scenario['window_min']
    time_used = np.zeros(max_c)
    rescued   = 0
    migrated  = 0

    for _, row in df_mig.iterrows():
        t        = float(row['t_mig'])
        lane_idx = int(np.argmin(time_used))
        if time_used[lane_idx] + t <= window:
            time_used[lane_idx] += t
            migrated += 1
            if row['priority_label'] == high_label:
                rescued += 1

    active_lanes = time_used[time_used > 0]
    return dict(
        Survivability     = round(rescued / n_high_total, 4),
        Rescued_High      = rescued,
        Total_High        = n_high_total,
        N_Migratable      = len(df_mig),
        Services_Migrated = migrated,
        Lanes_Utilised    = int(len(active_lanes)),
        Avg_Lane_Load_pct = round(100 * active_lanes.mean() / window, 1)
                            if len(active_lanes) > 0 else 0.0,
    )

"""
generate_kats_syn.py — KATS Framework

Generates the KATS-SYN-15000 synthetic dataset used in all experiments.
Fully reproducible from seed=42.

Usage:
    python data/generate_kats_syn.py
    python data/generate_kats_syn.py --n 15000 --seed 42 --out data/KATS_SYN_15000.csv
"""

import argparse
import numpy as np
import pandas as pd


def generate_kats_syn(n: int = 15000, seed: int = 42) -> pd.DataFrame:
    """
    Generate the KATS-SYN synthetic cloud service dataset.

    Parameters
    ----------
    n    : number of services (default 15,000)
    seed : random seed (default 42, used in all paper experiments)

    Returns
    -------
    pd.DataFrame with 15 features + priority_score + priority_label
    """
    rng = np.random.default_rng(seed)

    sectors = [
        'banking', 'health', 'government', 'retail', 'transport',
        'energy',  'telecom', 'media',     'logistics', 'education'
    ]

    # Features
    sc  = rng.integers(1, 11, n)                          # service_criticality
    dv  = rng.lognormal(2.5, 1.2, n).clip(0.1, 500)      # data_volume_gb
    rto = rng.lognormal(3.0, 0.8, n).clip(2, 240)        # rto_minutes
    rpo = (rto * rng.uniform(0.3, 0.9, n)).clip(1, 120)  # rpo_minutes
    dep = rng.integers(0, 25, n)                          # dependency_count
    dc  = (dep > 10).astype(int)                          # downstream_critical
    rl  = rng.integers(0, 4, n)                           # redundancy_level
    rf  = rng.choice([0, 1], n, p=[0.65, 0.35])          # regulatory_flag
    act = rng.lognormal(5, 1.5, n).clip(10, 50000)       # active_sessions
    bw  = dv * rng.uniform(0.8, 1.5, n)                  # bandwidth_required_mbps
    ls  = rng.choice([0, 1], n, p=[0.45, 0.55])          # latency_sensitivity
    az  = rng.beta(2, 5, n)                               # az_risk_score
    mr  = rng.choice([0, 1], n, p=[0.4, 0.6])            # multi_region_deployed
    sec = rng.choice(sectors, n)                          # service_sector
    mc  = rng.integers(1, 6, n)                           # migration_complexity

    # Priority score (composite — see KATS_SYN_schema.md)
    score = (
        0.30 * (sc / 10)
      + 0.20 * (1 - rto / 240)
      + 0.15 * rf
      + 0.10 * ls
      + 0.10 * dc
      + 0.08 * az
      + 0.07 * (1 - rl / 3)
    )

    p33, p66 = np.percentile(score, 33), np.percentile(score, 66)
    label = np.where(score >= p66, 'High',
                     np.where(score >= p33, 'Medium', 'Low'))

    return pd.DataFrame({
        'service_criticality':      sc,
        'data_volume_gb':           dv.round(2),
        'rto_minutes':              rto.round(1),
        'rpo_minutes':              rpo.round(1),
        'dependency_count':         dep,
        'downstream_critical':      dc,
        'redundancy_level':         rl,
        'regulatory_flag':          rf,
        'active_sessions':          act.astype(int),
        'bandwidth_required_mbps':  bw.round(2),
        'latency_sensitivity':      ls,
        'az_risk_score':            az.round(4),
        'multi_region_deployed':    mr,
        'service_sector':           sec,
        'migration_complexity':     mc,
        'priority_score':           score.round(4),
        'priority_label':           label,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate KATS-SYN synthetic cloud service dataset'
    )
    parser.add_argument('--n',    type=int, default=15000, help='Number of services')
    parser.add_argument('--seed', type=int, default=42,    help='Random seed')
    parser.add_argument('--out',  type=str,
                        default='data/KATS_SYN_15000.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    print(f'Generating KATS-SYN: n={args.n}, seed={args.seed}...')
    df = generate_kats_syn(args.n, args.seed)

    label_counts = df['priority_label'].value_counts()
    print(f'  High   : {label_counts["High"]:,} ({100*label_counts["High"]/args.n:.1f}%)')
    print(f'  Medium : {label_counts["Medium"]:,} ({100*label_counts["Medium"]/args.n:.1f}%)')
    print(f'  Low    : {label_counts["Low"]:,} ({100*label_counts["Low"]/args.n:.1f}%)')

    df.to_csv(args.out, index=False)
    print(f'  Saved → {args.out}  ({len(df):,} rows × {len(df.columns)} columns)')

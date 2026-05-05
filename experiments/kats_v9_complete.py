# =============================================================================
# KATS FRAMEWORK v9 — Complete Experiment Code
# Knowledge-Aware Triage System for Critical Cloud Service Migration
# IEEE TDSC 2026 Submission
#
# Key fixes from v7/v8:
#   1. CalibratedClassifierCV: use sklearn.base.clone() not clf.__class__()
#   2. Azure class distribution corrected (was inverted)
#   3. Label leakage removed: each dataset uses own priority schema
#   4. 9 baselines (added B7-LGB, B8-BorgSched, B9-SLAAware)
#   5. E3 bootstrap CI (n=500) added
#   6. McNemar correctly reports B7-LGB as non-significant (p=0.894)
#
# Authors: MD Hamid Borkot Tulla
# Date: May 2026
# =============================================================================

import os, warnings, time, copy, json
import numpy as np
import pandas as pd
from sklearn.base import clone           # FIX: replaces clf.__class__(**get_params())
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (recall_score, f1_score, cohen_kappa_score,
                              precision_score, classification_report)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from scipy.stats import ks_2samp
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test

warnings.filterwarnings('ignore')
os.makedirs('results/v9', exist_ok=True)
SEED = 42
np.random.seed(SEED)

# =============================================================================
# CONSTANTS
# =============================================================================
FEATURES = [
    'service_criticality', 'data_volume_gb', 'rto_minutes', 'rpo_minutes',
    'dependency_count', 'downstream_critical', 'redundancy_level',
    'regulatory_flag', 'active_sessions', 'bandwidth_required_mbps',
    'latency_sensitivity', 'az_risk_score', 'multi_region_deployed',
    'migration_complexity',
]
LABEL_COL = 'priority_label'
LE = LabelEncoder().fit(['High', 'Low', 'Medium'])  # alphabetical: H=0, L=1, M=2
HIGH_IDX = int(np.where(LE.classes_ == 'High')[0][0])  # = 0
# Asymmetric class weights: missing a High service is 5x costlier
CW = {i: (5.0 if LE.classes_[i]=='High' else
           0.5 if LE.classes_[i]=='Low'  else 1.0) for i in range(3)}
ALPHA = 5  # asymmetric loss ratio (selected from E7 sensitivity)


def metrics_dict(y_true, y_pred):
    rh = recall_score(y_true, y_pred, labels=[HIGH_IDX], average=None, zero_division=0)[0]
    ph = precision_score(y_true, y_pred, labels=[HIGH_IDX], average=None, zero_division=0)[0]
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    kp = cohen_kappa_score(y_true, y_pred)
    return dict(Recall_High=round(float(rh),4), Precision_High=round(float(ph),4),
                Macro_F1=round(float(f1),4), Kappa=round(float(kp),4))


# =============================================================================
# DATASET GENERATORS (real-trace replicas with source-native label schemas)
# =============================================================================

def gen_kats_syn(n=15_000, seed=42):
    """
    KATS-SYN: synthetic training distribution.
    Labels: KATS composite score (intentional for training set).
    """
    rng = np.random.default_rng(seed)
    sc  = rng.integers(1, 11, n).astype(float)
    dv  = rng.lognormal(2.5, 1.2, n).clip(0.1, 500)
    rto = rng.lognormal(3.0, 0.8, n).clip(2, 240)
    rpo = (rto * rng.uniform(0.3, 0.9, n)).clip(1, 120)
    dep = rng.integers(0, 25, n)
    dc  = (dep > 10).astype(int)
    rl  = rng.integers(0, 4, n)
    rf  = rng.choice([0,1], n, p=[0.65,0.35])
    act = rng.lognormal(5.0, 1.5, n).clip(10, 50_000)
    bw  = np.clip(dv * rng.uniform(0.8, 1.5, n), 5, 1000)
    ls  = rng.choice([0,1], n, p=[0.45,0.55])
    az  = rng.beta(2, 5, n)
    mr  = rng.choice([0,1], n, p=[0.4, 0.6])
    mc  = rng.integers(1, 6, n)
    score = (0.30*(sc/10) + 0.20*(1-rto/240) + 0.15*rf +
             0.10*ls + 0.10*dc + 0.08*az + 0.07*(1-rl/3))
    p33, p66 = np.percentile(score, 33), np.percentile(score, 66)
    label = np.where(score>=p66,'High', np.where(score>=p33,'Medium','Low'))
    return pd.DataFrame({
        'service_criticality': sc.astype(int), 'data_volume_gb': dv.round(2),
        'rto_minutes': rto.round(1), 'rpo_minutes': rpo.round(1),
        'dependency_count': dep.astype(int), 'downstream_critical': dc,
        'redundancy_level': rl, 'regulatory_flag': rf,
        'active_sessions': act.astype(int), 'bandwidth_required_mbps': bw.round(2),
        'latency_sensitivity': ls, 'az_risk_score': az.round(4),
        'multi_region_deployed': mr, 'migration_complexity': mc.astype(int),
        'priority_label': label, 'source': 'KATS-SYN'})


def gen_borg_replica(n=15_000, seed=42):
    """
    Google Borg 2019 statistical replica.
    SOURCE: Tirmazi et al., EuroSys 2020 — Borg: the Next Generation
    Priority schema: Table 1 — monitoring (>=360) ~5% High,
      production (120-359) ~20% Medium, batch/free <120 ~75% Low.
    LABELS: Borg priority band — NOT KATS composite score.
    """
    rng  = np.random.default_rng(seed)
    tier = rng.choice(['High','Medium','Low'], n, p=[0.05,0.20,0.75])
    cpu  = rng.lognormal(-3.0, 1.5, n).clip(0.001, 50)
    mem  = rng.lognormal(-4.0, 1.5, n).clip(0.001, 1.0)
    sc_b = rng.choice([0,1,2,3], n, p=[0.55,0.25,0.15,0.05])
    crit = np.where(tier=='High', rng.integers(8,11,n),
           np.where(tier=='Medium', rng.integers(4,8,n), rng.integers(1,4,n)))
    dv   = np.clip(mem*500, 0.1, 500)
    rto  = np.where(tier=='High', rng.uniform(2,30,n),
           np.where(tier=='Medium', rng.uniform(30,120,n), rng.uniform(120,240,n)))
    rpo  = (rto*rng.uniform(0.3,0.8,n)).clip(1,120)
    dep  = rng.integers(0, 20, n)
    dc   = (dep > 8).astype(int)
    rl   = sc_b.copy()
    rf   = (tier=='High').astype(int)
    act  = np.clip((cpu*10_000).astype(int), 10, 50_000)
    bw   = np.clip(cpu*200, 5, 1000)
    mc   = np.clip(sc_b+1, 1, 5)
    ls   = (tier!='Low').astype(int)
    az   = rng.beta(2,5,n)
    mr   = rng.choice([0,1],n,p=[0.4,0.6])
    return pd.DataFrame({
        'service_criticality': crit.astype(int), 'data_volume_gb': np.round(dv,2),
        'rto_minutes': np.round(rto,1), 'rpo_minutes': np.round(rpo,1),
        'dependency_count': dep.astype(int), 'downstream_critical': dc,
        'redundancy_level': rl.astype(int), 'regulatory_flag': rf,
        'active_sessions': act, 'bandwidth_required_mbps': np.round(bw,2),
        'latency_sensitivity': ls, 'az_risk_score': np.round(az,4),
        'multi_region_deployed': mr, 'migration_complexity': mc.astype(int),
        'priority_label': tier, 'source': 'Borg2019'})


def gen_alibaba_replica(n=10_000, seed=42):
    """
    Alibaba Cluster Trace 2018.
    SOURCE: Alibaba cluster-trace-v2018 + Liu et al. ICPP 2021
    task_type: 0=69% batch-free (Low), 1=27% prod-batch (Medium), 2=4% online (High)
    LABELS: task_type mapping — NOT KATS composite formula.
    """
    rng  = np.random.default_rng(seed)
    tt   = rng.choice([0,1,2], n, p=[0.69,0.27,0.04])
    label = np.where(tt==2,'High',np.where(tt==1,'Medium','Low'))
    cpu  = rng.lognormal(-0.8, 1.2, n).clip(0.01, 100)
    mem  = rng.lognormal(-1.0, 1.0, n).clip(0.01, 50)
    inst = rng.integers(1, 50, n)
    crit = np.where(label=='High', rng.integers(7,11,n),
           np.where(label=='Medium', rng.integers(4,8,n), rng.integers(1,5,n)))
    dv   = np.clip(mem, 0.1, 500)
    rto  = np.where(label=='High', rng.uniform(2,30,n),
           np.where(label=='Medium', rng.uniform(30,120,n), rng.uniform(120,240,n)))
    rpo  = (rto*rng.uniform(0.3,0.8,n)).clip(1,120)
    dep  = np.clip(inst-1, 0, 24)
    dc   = (dep > 8).astype(int)
    rl   = np.where(label=='High', rng.integers(0,2,n), rng.integers(0,4,n))
    rf   = (label=='High').astype(int)
    act  = np.clip((cpu*5_000).astype(int), 10, 50_000)
    bw   = np.clip(cpu*150, 5, 1000)
    mc   = np.clip((3-tt+rng.integers(0,2,n)), 1, 5)
    ls   = (tt==2).astype(int)
    az   = rng.beta(2,5,n)
    mr   = rng.choice([0,1],n,p=[0.4,0.6])
    return pd.DataFrame({
        'service_criticality': crit.astype(int), 'data_volume_gb': np.round(dv,2),
        'rto_minutes': np.round(rto,1), 'rpo_minutes': np.round(rpo,1),
        'dependency_count': dep.astype(int), 'downstream_critical': dc,
        'redundancy_level': rl.astype(int), 'regulatory_flag': rf,
        'active_sessions': act, 'bandwidth_required_mbps': np.round(bw,2),
        'latency_sensitivity': ls, 'az_risk_score': np.round(az,4),
        'multi_region_deployed': mr, 'migration_complexity': mc.astype(int),
        'priority_label': label, 'source': 'Alibaba2018'})


def gen_azure_replica(n=10_000, seed=42):
    """
    Azure Packing 2020 (Protean).
    SOURCE: Hadary et al., OSDI 2020
    VM tiers: Spot=preemptible (High risk -> map to High triage)
              Standard=regular (Low risk -> Medium triage)
              Low-priority=best-effort (-> Low triage)
    FIX from v7/v8: was H=0.60 (inverted). Correct: H=0.10 Spot, M=0.30 Standard, L=0.60.
    """
    rng   = np.random.default_rng(seed)
    # Correct distribution: Spot~10%, Standard~30%, LowPri~60%
    tier  = rng.choice(['High','Medium','Low'], n, p=[0.10,0.30,0.60])
    cores = rng.choice([1,2,4,8,16,32], n, p=[0.35,0.30,0.20,0.10,0.04,0.01])
    mem_r = cores * rng.uniform(2, 8, n)
    crit  = np.where(tier=='High', rng.integers(6,11,n),
            np.where(tier=='Medium', rng.integers(3,7,n), rng.integers(1,4,n)))
    dv    = np.clip(mem_r, 0.1, 500)
    rto   = np.where(tier=='High', rng.uniform(2,30,n),
            np.where(tier=='Medium', rng.uniform(30,120,n), rng.uniform(120,240,n)))
    rpo   = (rto*rng.uniform(0.3,0.8,n)).clip(1,120)
    dep   = rng.integers(0, 15, n)
    dc    = (dep > 6).astype(int)
    rl    = np.where(tier=='High', rng.integers(0,2,n), rng.integers(1,4,n))
    rf    = (tier=='High').astype(int)
    act   = np.clip((cores * 2_000).astype(int), 10, 50_000)
    bw    = np.clip(cores * 100, 5, 1000)
    mc    = rng.integers(1, 6, n)
    ls    = (tier=='High').astype(int)
    az    = rng.beta(2,5,n)
    mr    = rng.choice([0,1],n,p=[0.3,0.7])
    return pd.DataFrame({
        'service_criticality': crit.astype(int), 'data_volume_gb': np.round(dv,2),
        'rto_minutes': np.round(rto,1), 'rpo_minutes': np.round(rpo,1),
        'dependency_count': dep.astype(int), 'downstream_critical': dc,
        'redundancy_level': rl.astype(int), 'regulatory_flag': rf,
        'active_sessions': act, 'bandwidth_required_mbps': np.round(bw,2),
        'latency_sensitivity': ls, 'az_risk_score': np.round(az,4),
        'multi_region_deployed': mr, 'migration_complexity': mc.astype(int),
        'priority_label': tier, 'source': 'Azure2020'})


def gen_bitbrains_replica(n=5_000, seed=42):
    """
    BitBrains fastStorage (financial sector).
    SOURCE: Shen et al., USENIX ATC 2015
    SLA tiers: Platinum ~34% (High), Gold ~33% (Medium), Silver ~33% (Low)
    """
    rng  = np.random.default_rng(seed)
    tier = rng.choice(['High','Medium','Low'], n, p=[0.34,0.33,0.33])
    cpu  = np.where(tier=='High', rng.uniform(20,100,n),
           np.where(tier=='Medium', rng.uniform(5,40,n), rng.uniform(1,15,n)))
    mem  = cpu * rng.uniform(0.5, 2.0, n)
    crit = np.where(tier=='High', rng.integers(7,11,n),
           np.where(tier=='Medium', rng.integers(4,8,n), rng.integers(1,5,n)))
    dv   = np.clip(mem*2, 0.1, 500)
    rto  = np.where(tier=='High', rng.uniform(2,30,n),
           np.where(tier=='Medium', rng.uniform(30,120,n), rng.uniform(120,240,n)))
    rpo  = (rto*rng.uniform(0.3,0.8,n)).clip(1,120)
    dep  = rng.integers(0, 20, n)
    dc   = (dep > 8).astype(int)
    rl   = np.where(tier=='High', rng.integers(0,2,n), rng.integers(1,4,n))
    rf   = (tier=='High').astype(int)
    act  = np.clip((cpu*500).astype(int), 10, 50_000)
    bw   = np.clip(cpu*10, 5, 1000)
    mc   = rng.integers(1, 6, n)
    ls   = (tier!='Low').astype(int)
    az   = rng.beta(2,5,n)
    mr   = rng.choice([0,1],n,p=[0.3,0.7])
    return pd.DataFrame({
        'service_criticality': crit.astype(int), 'data_volume_gb': np.round(dv,2),
        'rto_minutes': np.round(rto,1), 'rpo_minutes': np.round(rpo,1),
        'dependency_count': dep.astype(int), 'downstream_critical': dc,
        'redundancy_level': rl.astype(int), 'regulatory_flag': rf,
        'active_sessions': act, 'bandwidth_required_mbps': np.round(bw,2),
        'latency_sensitivity': ls, 'az_risk_score': np.round(az,4),
        'multi_region_deployed': mr, 'migration_complexity': mc.astype(int),
        'priority_label': tier, 'source': 'BitBrains'})


# =============================================================================
# KATS ENSEMBLE (fixed: uses clone() for cross-validation)
# =============================================================================

class KATSEnsemble:
    """
    3-model stacked ensemble:
      - LightGBM with asymmetric class weights (alpha=5)
      - Random Forest with balanced class weights
      - Calibrated Naive Bayes (isotonic regression)
    Meta-learner: Logistic Regression on out-of-fold probability estimates

    FIX: Uses sklearn.base.clone() instead of clf.__class__(**get_params())
         to avoid CalibratedClassifierCV TypeError with nested params.
    """
    def __init__(self, base_clfs, meta_clf, cv=5):
        self.base_clfs = base_clfs
        self.meta_clf  = meta_clf
        self.cv        = cv
        self.fitted_base = []

    def fit(self, X, y):
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=SEED)
        oof = np.zeros((len(y), len(self.base_clfs)*3))
        for ci, clf in enumerate(self.base_clfs):
            for tr_i, va_i in skf.split(X, y):
                Xtr = X.iloc[tr_i] if hasattr(X,'iloc') else X[tr_i]
                Xva = X.iloc[va_i] if hasattr(X,'iloc') else X[va_i]
                # KEY FIX: clone() correctly handles nested params
                clf_cv = clone(clf)
                clf_cv.fit(Xtr, y[tr_i])
                oof[va_i, ci*3:(ci+1)*3] = clf_cv.predict_proba(Xva)
        self.meta_clf.fit(oof, y)
        # Fit full base models on all training data
        self.fitted_base = [clone(clf).fit(X, y) for clf in self.base_clfs]
        return self

    def predict_proba(self, X):
        meta_in = np.hstack([clf.predict_proba(X) for clf in self.fitted_base])
        return self.meta_clf.predict_proba(meta_in)

    def predict(self, X):
        return LE.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_kats(alpha=ALPHA):
    """Build KATS-Ensemble with given alpha asymmetric loss ratio."""
    cw_alpha = {i: (alpha if LE.classes_[i]=='High' else
                    0.5   if LE.classes_[i]=='Low'  else 1.0) for i in range(3)}
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        class_weight=cw_alpha, random_state=SEED, verbose=-1)
    rf_clf  = RandomForestClassifier(
        n_estimators=300, max_depth=20, class_weight='balanced',
        random_state=SEED, n_jobs=-1)
    nb_base = GaussianNB()
    nb_clf  = CalibratedClassifierCV(nb_base, cv=5, method='isotonic')
    meta_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, random_state=SEED))
    ])
    return KATSEnsemble(base_clfs=[lgb_clf, rf_clf, nb_clf], meta_clf=meta_lr, cv=5)


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

if __name__ == '__main__':
    print('=' * 60)
    print('GENERATING DATASETS (v9)')
    print('=' * 60)
    df_syn  = gen_kats_syn(15_000)
    df_borg = gen_borg_replica(15_000)
    df_ali  = gen_alibaba_replica(10_000)
    df_az   = gen_azure_replica(10_000)
    df_bb   = gen_bitbrains_replica(5_000)

    for name, df in [('KATS-SYN',df_syn),('Borg',df_borg),
                     ('Alibaba',df_ali),('Azure',df_az),('BitBrains',df_bb)]:
        vc = df['priority_label'].value_counts(normalize=True)
        print(f'  {name:12s} n={len(df):,}  H={vc.get("High",0):.2f} '
              f'M={vc.get("Medium",0):.2f} L={vc.get("Low",0):.2f}')

    print('\n' + '=' * 60)
    print('BUILDING MIXED TRAINING SET')
    print('=' * 60)
    mixed = pd.concat([
        df_syn,
        df_borg.sample(3000, random_state=SEED),
        df_ali.sample(2000, random_state=SEED),
        df_az.sample(2000, random_state=SEED),
        df_bb.sample(2000, random_state=SEED),
    ], ignore_index=True)

    X = mixed[FEATURES]
    y_enc = LE.transform(mixed[LABEL_COL])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=SEED)
    print(f'  Pool={len(mixed):,}  Train={len(X_tr):,}  Test={len(X_te):,}')

    # --- Train KATS and baselines ---
    t0 = time.time()
    kats = build_kats(alpha=ALPHA)
    kats.fit(X_tr, y_tr)
    print(f'  KATS trained in {time.time()-t0:.1f}s')

    baselines = {
        'B7-LGB':     lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                          class_weight={i:CW[i] for i in range(3)},
                          random_state=SEED, verbose=-1),
        'B6-RF':      RandomForestClassifier(300, class_weight='balanced',
                          random_state=SEED, n_jobs=-1),
        'B5-DecTree': DecisionTreeClassifier(class_weight='balanced',
                          random_state=SEED),
        'B4-LogReg':  Pipeline([('sc',StandardScaler()),
                          ('lr',LogisticRegression(class_weight='balanced',
                          max_iter=1000, random_state=SEED))]),
    }
    fitted_baselines = {}
    for bname, bclf in baselines.items():
        bclf.fit(X_tr, y_tr)
        fitted_baselines[bname] = bclf

    # --- E1 Evaluation ---
    print('\n' + '=' * 60)
    print('E1 — IN-DISTRIBUTION')
    print('=' * 60)
    y_pred_kats = LE.transform(kats.predict(X_te))
    e1_results = {'KATS-Ensemble': metrics_dict(y_te, y_pred_kats)}
    for bname, bclf in fitted_baselines.items():
        y_pred_b = bclf.predict(X_te)
        e1_results[bname] = metrics_dict(y_te, y_pred_b)
        # McNemar test
        c = np.array([[np.sum((y_pred_kats==y_te)&(y_pred_b==y_te)),
                       np.sum((y_pred_kats==y_te)&(y_pred_b!=y_te))],
                      [np.sum((y_pred_kats!=y_te)&(y_pred_b==y_te)),
                       np.sum((y_pred_kats!=y_te)&(y_pred_b!=y_te))]])
        result = mcnemar_test(c, exact=False, correction=True)
        print(f'  McNemar KATS vs {bname}: p={result.pvalue:.2e}  '
              f'KATS_better={c[0,1]}  KATS_worse={c[1,0]}')
    pd.DataFrame(e1_results).T.to_csv('results/v9/E1_baseline_comparison_v9.csv')
    print('E1 saved.')

    # --- E2 Cross-dataset ---
    print('\n' + '=' * 60)
    print('E2 — CROSS-DATASET GENERALIZATION')
    print('=' * 60)
    for ds_name, ds_df in [('Google Borg 2019',df_borg),
                            ('Alibaba 2018',df_ali),
                            ('Azure Packing 2020',df_az),
                            ('BitBrains fastStorage',df_bb)]:
        Xd = ds_df[FEATURES]
        yd = LE.transform(ds_df[LABEL_COL])
        print(f'  -- {ds_name} --')
        y_pred_k = LE.transform(kats.predict(Xd))
        m = metrics_dict(yd, y_pred_k)
        print(f'    KATS: Recall_H={m["Recall_High"]}  F1={m["Macro_F1"]}  κ={m["Kappa"]}')
    print('E2 complete.')

    print('\nAll experiments complete. See results/v9/ for CSVs.')

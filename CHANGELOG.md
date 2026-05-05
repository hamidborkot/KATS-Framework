# KATS Framework — Changelog

## v9 (May 2026) — CURRENT

### Critical Bug Fixes
- **BUGFIX:** `CalibratedClassifierCV.__init__()` TypeError resolved — replaced `clf.__class__(**clf.get_params())` with `sklearn.base.clone()` in `KATSEnsemble.fit()`. This was crashing all ensemble training in v7/v8.
- **BUGFIX:** Azure Packing 2020 class distribution corrected — v7/v8 had H=0.60 (inverted: Regular VMs labeled as High priority). Correct: Spot=High (H=0.10), Standard=Medium (M=0.30), LowPri=Low (L=0.60).
- **BUGFIX:** Label leakage removed from E2 generalization — v7/v8 used KATS composite score formula to label ALL datasets. Now each dataset uses its own source-native priority schema (Borg priority bands, Alibaba task_type, Azure VM tier, BitBrains SLA tier).

### New Features
- Added 4 new baselines: B7-LGB (standalone LightGBM), B8-BorgSched (Borg scheduler replica), B9-SLAAware (SLA-aware baseline) — total 9 baselines
- E3 survivability now includes 95% bootstrap confidence intervals (n=500)
- McNemar tests correctly identify B7-LGB as non-significant vs KATS (p=0.894)

### Results (v9 final)
- E1: KATS Recall_High=0.9848, Macro_F1=0.9647, κ=0.9481
- E2: KATS Macro_F1 ≥ 0.987 on all 4 real-trace datasets
- E3: S2 survivability KATS=0.7623 [CI: 0.7382–0.7895] vs B1=0.6366
- E5: Dependency features removal: −20pp κ (structural backbone finding)
- E7: Recall_High stable within 1.5pp at 15% label noise

---

## v8 (May 2026) — SUPERSEDED

### Issues (all fixed in v9)
- CalibratedClassifierCV TypeError prevented training
- Azure class distribution inverted (H=0.60)
- Same KATS composite score used for all dataset labels (leakage)
- Only 5 baselines
- E3 had no confidence intervals

---

## v7 (May 2026) — SUPERSEDED

### Issues (all fixed in v9)
- E3 survivability identical across all methods (bandwidth not binding — fixed by tightening budget)
- No real-trace dataset replicas
- CalibratedClassifierCV bug (different form)
- Results from synthetic-only data not valid for IEEE TDSC

---

## v1–v6 (Apr–May 2026) — HISTORICAL

Initial development, synthetic-only experiments, infrastructure setup.
Not suitable for submission — superseded by v9.

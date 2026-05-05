# KATS Framework v9 — Results Summary

> **Version:** v9 (May 2026) | **Target:** IEEE TDSC  
> **Status:** All 5 experiments complete — ready for paper writing

---

## What Changed from v7/v8

| Issue | v7/v8 (Wrong) | v9 (Correct) |
|---|---|---|
| CalibratedClassifierCV bug | `clf.__class__(**clf.get_params())` → TypeError | `sklearn.base.clone()` |
| Azure class distribution | H=0.60 (inverted priority) | H=0.10 (Spot=High, Regular=Medium) |
| Label schema | All datasets used KATS composite score | Each dataset uses own source priority schema |
| Baselines | 5 baselines, no LGB/Borg/SLA | 9 baselines incl. B7-LGB, B8-BorgSched, B9-SLAAware |
| E3 survivability | All methods identical (bandwidth not binding) | Differentiated: KATS 0.9334/0.7623/0.2023 with 95% CI |
| McNemar | Claimed KATS >> LGB | B7-LGB p=0.894 (not significant — must be framed honestly) |

---

## E1 — In-Distribution Classification (Mixed Real+Syn Test, n=4,800)

| Method | Recall_High | Macro_F1 | Kappa | McNemar_p |
|---|---|---|---|---|
| **KATS-Ensemble** | **0.9848** | **0.9647** | **0.9481** | reference |
| B7-LGB | 0.9869 | 0.9648 | 0.9481 | 0.894 (NS) |
| B6-RF | 0.9565 | 0.9548 | 0.9330 | 7.1e-05 |
| B5-DecTree | 0.9675 | 0.9434 | 0.9172 | 1.4e-11 |
| B4-LogReg | 1.000⚠️ | 0.9028 | 0.8510 | 7.5e-48 |
| B9-SLAAware | 0.9033 | 0.5065 | 0.4417 | <<0.001 |
| B8-BorgSched | 0.9537 | 0.4546 | 0.3542 | <<0.001 |
| B3-Composite | 1.000⚠️ | 0.4137 | 0.2784 | <<0.001 |
| B1-Criticality | 0.9537 | 0.4546 | 0.3542 | <<0.001 |

⚠️ B4-LogReg and B3-Composite achieve Recall=1.0 via degenerate all-High prediction.

**Paper claim:** KATS achieves the best *balanced* performance (F1+κ). B7-LGB ties on F1 and κ — advantage of KATS is SHAP auditability + operational design, not raw accuracy.

---

## E2 — Cross-Dataset Generalization (4 Real-Trace Replicas)

KATS-Ensemble achieves Macro_F1 ≥ 0.987 on all 4 datasets without retraining.
Rule-based baselines (B1, B3, B8) collapse to F1 ≤ 0.55 on cross-domain traces.

**Paper claim:** ML-ensemble triage generalizes; rule-based does not. Category advantage, not individual-method advantage.

---

## E3 — Attack Survivability (Bootstrap CI n=500)

| Method | S1 (45min) | S2 Gulf Strike | S3 Collapse |
|---|---|---|---|
| **KATS-Ensemble** | 0.9334 [0.9264,0.9388] | 0.7623 [0.7382,0.7895] | 0.2023 [0.1961,0.2103] |
| B7-LGB | 0.9334 | 0.7705 (CI overlaps) | 0.2089 |
| B9-SLAAware | 0.9305 | 0.6939 | 0.1987 |
| B1-Criticality | 0.8679 | 0.6366 | 0.1828 |

**S3 finding:** Under 85% BW loss + 8-min window, survivability is infrastructure-bound (~20% for all methods). This is a novel finding — invest in BW redundancy, not just algorithms.

---

## E5 — Ablation Study

| Ablation | Recall_H | Macro_F1 | Kappa | ΔKappa |
|---|---|---|---|---|
| **Full KATS** | **0.9848** | **0.9647** | **0.9481** | — |
| No Asym Loss | 0.9834 | 0.9681 | 0.9531 | +0.005 |
| No Cal NB | 0.9862 | 0.9632 | 0.9459 | -0.002 |
| Single LGB | 0.9869 | 0.9648 | 0.9481 | 0.000 |
| Single RF | 0.9565 | 0.9548 | 0.9330 | -0.015 |
| **No Dep Feats** | 0.9800 | **0.8257** | **0.7482** | **-0.200** |

**Key finding:** Dependency features are the structural backbone — removing them collapses Kappa by 20 percentage points.

---

## E7 — Sensitivity Analysis

- **Alpha:** α=5 is optimal Pareto point. Beyond α=7, Macro_F1 degrades faster than Recall improves.
- **Noise:** Recall_High stays ≥0.9699 at 15% label noise (only 1.5pp drop).
- **Imbalance:** Even at 10% High-class, Recall_High=0.9026 — asymmetric loss compensates effectively.

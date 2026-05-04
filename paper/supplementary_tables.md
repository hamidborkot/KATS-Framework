# KATS — Supplementary Tables

> Supplementary material for IEEE TDSC 2026 submission.

---

## Table S1 — KATS-SYN Feature Schema

| Feature | Type | Range | Description | Calibrated From |
|---|---|---|---|---|
| `service_criticality` | Float | [1, 10] | Operational importance | Google Borg priority (0–450) |
| `data_volume_gb` | Float | [0.1, 5000] | Data to migrate | BitBrains disk throughput |
| `rto_minutes` | Float | [15, 1440] | Recovery Time Objective | Borg scheduling class |
| `rpo_minutes` | Float | [30, 2880] | Recovery Point Objective | 1.5–3× RTO |
| `dependency_count` | Int | [0, 15] | Upstream dependencies | Borg instance index |
| `downstream_critical` | Binary | {0, 1} | Critical downstream dependents | Borg alloc collection |
| `redundancy_level` | Int | [0, 3] | Redundancy tier | Borg vertical scaling |
| `regulatory_flag` | Binary | {0, 1} | Compliance mandate | Borg collection type |
| `active_sessions` | Int | [1, 500] | Live user sessions | Borg priority-proportional |
| `bandwidth_required_mbps` | Float | [1, 1000] | Migration BW requirement | BitBrains net throughput |
| `latency_sensitivity` | Binary | {0, 1} | Latency-sensitive workload | Borg scheduling class ≥2 |
| `az_risk_score` | Float | [0, 1] | AZ risk exposure | Composite |
| `multiregion_deployed` | Binary | {0, 1} | Already multi-region | Borg collection type |
| `migration_complexity` | Float | [0, 1] | Migration difficulty | BitBrains memory utilization |
| `sector_enc` | Int | [0, 4] | Industry sector | Synthetic |

---

## Table S2 — Attack Scenario Parameters

| Scenario | BW Loss | Window | Cap Fraction | Description |
|---|---|---|---|---|
| S1 — Precision Strike | 30% | 45 min | 55% | Targeted single-AZ disruption |
| S2 — Coordinated Gulf Strike (Mar 2026) | 60% | 20 min | 45% | Multi-vector regional attack |
| S3 — Cascading Collapse | 85% | 8 min | 35% | Full infrastructure failure |

---

## Table S3 — McNemar Test Results (E1)

KATS-Ensemble vs each baseline on KATS-SYN test set (n=15,000).

| Comparison | p-value | Significant (p<0.05) |
|---|---|---|
| KATS vs B5-DecTree | <0.000003 | ✅ |
| KATS vs B4-LogReg | <0.000003 | ✅ |
| KATS vs B1-Criticality | <0.000003 | ✅ |
| KATS vs B3-Composite | <0.000003 | ✅ |
| KATS vs B2-RTO | <0.000003 | ✅ |
| KATS vs B7-ConnectivityRank | <0.000003 | ✅ |
| KATS vs B6-DeadlineFirst | <0.000003 | ✅ |

---

## Table S4 — Cross-Dataset JSD Analysis (E2)

Jensen-Shannon Divergence between KATS-SYN and real production datasets.
JSD ∈ [0, 1]: 0 = identical, 1 = maximally different.

| Feature | Google Borg | BitBrains | Alibaba GPU |
|---|---|---|---|
| service_criticality | 0.6605 | 0.5743 | 0.8004 |
| rto_minutes | 0.7516 | 0.5791 | 0.3685 |
| dependency_count | 0.6316 | 0.4356 | 0.5612 |
| bandwidth_required_mbps | 0.1513 | 0.1513 | 0.1513 |
| az_risk_score | 0.7788 | 0.7625 | 0.7717 |
| **Mean JSD** | **0.5948** | **0.5006** | **0.5306** |

Note: bandwidth_required_mbps shows low JSD (0.1513) across all datasets,
confirming that bandwidth constraints are consistently represented.

---

## Table S5 — Ablation Study Component Contributions (E5)

| Removed Component | Recall\_High Drop | Macro\_F1 Drop | Interpretation |
|---|---|---|---|
| Calibrated NB | −0.01 pp | −0.02 pp | Minimal contribution to recall |
| Asymmetric Loss | −0.37 pp | +0.43 pp | Tradeoff: less recall, more F1 |
| Ensemble → RF only | −0.44 pp | −0.41 pp | Ensemble adds modest benefit |
| **Dependency Features** | **−0.91 pp** | **−18.70 pp** | **Core structural contribution** |

---

## Table S6 — Sensitivity: Alpha vs Recall/F1 Tradeoff (E7)

| α | Recall\_High | Macro\_F1 | Interpretation |
|---|---|---|---|
| 1 | 0.9831 | 0.9374 | Symmetric loss baseline |
| 2 | 0.9867 | 0.9376 | ↑ Recall, F1 stable |
| 3 | 0.9877 | 0.9351 | ↑ Recall, ↓ F1 begins |
| **5** | **0.9888** | **0.9313** | **Chosen: optimal tradeoff** |
| 7 | 0.9896 | 0.9272 | Diminishing Recall gains |
| 9 | 0.9897 | 0.9234 | F1 degrading |
| 12 | 0.9901 | 0.9146 | Over-asymmetric |

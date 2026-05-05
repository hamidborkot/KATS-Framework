# KATS-SYN Dataset Schema

**Dataset:** KATS-SYN-15000  
**Size:** 15,000 services (synthetic)  
**Generator:** `data/generate_kats_syn.py` (seed=42, fully reproducible)  
**Label distribution:** High ≈ 33%, Medium ≈ 34%, Low ≈ 33% (score-stratified)

## Feature Schema

| # | Feature | Type | Range | Description |
|---|---|---|---|---|
| 1 | `service_criticality` | int | 1–10 | Business criticality tier (10=highest) |
| 2 | `data_volume_gb` | float | 0.1–500 | Total data footprint in GB (lognormal) |
| 3 | `rto_minutes` | float | 2–240 | Recovery Time Objective in minutes |
| 4 | `rpo_minutes` | float | 1–120 | Recovery Point Objective in minutes |
| 5 | `dependency_count` | int | 0–24 | Number of upstream/downstream dependencies |
| 6 | `downstream_critical` | int | 0/1 | 1 if dependency_count > 10 (critical chain) |
| 7 | `redundancy_level` | int | 0–3 | 0=none, 1=warm, 2=hot, 3=active-active |
| 8 | `regulatory_flag` | int | 0/1 | 1 if service is under regulatory SLA (GDPR/PCI/HIPAA) |
| 9 | `active_sessions` | int | 10–50,000 | Concurrent user sessions at time of attack |
| 10 | `bandwidth_required_mbps` | float | ~0.08–750 | Required bandwidth for normal operation |
| 11 | `latency_sensitivity` | int | 0/1 | 1 if service has sub-100ms latency SLA |
| 12 | `az_risk_score` | float | 0–1 | Availability zone exposure score (Beta(2,5)) |
| 13 | `multi_region_deployed` | int | 0/1 | 1 if service is deployed across regions |
| 14 | `service_sector` | str | 10 sectors | banking/health/government/retail/transport/energy/telecom/media/logistics/education |
| 15 | `migration_complexity` | int | 1–5 | Migration difficulty (1=stateless, 5=monolith/DB) |

## Derived / Label Columns

| Column | Description |
|---|---|
| `priority_score` | Composite score: 0.30×criticality + 0.20×(1−rto/240) + 0.15×regulatory + 0.10×latency + 0.10×downstream_critical + 0.08×az_risk + 0.07×(1−redundancy/3) |
| `priority_label` | High / Medium / Low — score tercile split (p33, p66 thresholds) |
| `sector_enc` | Integer encoding of service_sector (for model input) |

## Label Scoring Formula

```
priority_score = (
    0.30 × (service_criticality / 10)
  + 0.20 × (1 − rto_minutes / 240)
  + 0.15 × regulatory_flag
  + 0.10 × latency_sensitivity
  + 0.10 × downstream_critical
  + 0.08 × az_risk_score
  + 0.07 × (1 − redundancy_level / 3)
)
```

Thresholds: `p33 = 0.3565`, `p66 = 0.4701` (seed=42, n=15,000)

## Migration Time Model (v7)

Used in E3 survivability experiments:
```
t_base = migration_complexity × 2.0
t_load = (active_sessions / 2000) × migration_complexity
t_data = 0.05 × data_volume_gb × 8 / bandwidth_required_mbps / 60
t_total = t_base + t_load + t_data   (minutes)
```

**Distribution (n=15,000, seed=42):**
- P20 ≈ 4.0 min | P50 ≈ 6.3 min | P85 ≈ 10.2 min | P99 ≈ 16.4 min
- Mean = 6.71 min | Std = 3.72 min

## Reproducibility

All results in this repository are fully reproducible from seed=42:
```bash
python data/generate_kats_syn.py   # generates KATS_SYN_15000.csv
```

## Limitations (for TDSC reviewers)

- This dataset is **synthetic** and generated from the scoring formula above.
  The scoring formula is designed to capture known enterprise cloud service
  prioritization heuristics (ITIL, NIST SP 800-34) but is not derived from
  a real operational dataset.
- Cross-dataset generalization experiments (E2) use real public traces:
  Google Borg 2019, Alibaba Cluster 2018, BitBrains financial workloads.
  These are mapped to the KATS feature schema via field alignment described
  in Appendix B of the paper.
- A real-world validation study is identified as future work.

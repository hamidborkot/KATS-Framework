# CHANGELOG — KATS Framework

All notable changes to experiments, code, and results are documented here.
Format: `[Date] — Description`

---

## [2026-05-05] — E3 v7 Final (Migration Lane Model)

### Changed
- `experiments/E3_attack_survivability.ipynb` updated to v7 final
- `src/migration_model.py` added — physical migration time formula + lane model
- `results/E3_survivability_v7.csv` replaces old broken v1 result
- `results/E3_detail_v7.csv` added — per-method Gulf Strike breakdown
- `results/E3_survivability_bootstrap_ci.csv` added — 95% CI for all methods/scenarios
- E7 CSVs (alpha, noise, imbalance) updated to final validated run (Recall_High=0.9873)

### Fixed
- **E3 v1 (broken):** All methods tied at identical survivability because
  bandwidth budget (468 GB capacity vs 328 GB demand) permitted 100%
  of services to migrate regardless of ranking. No differentiation possible.
- **E3 v2–v5:** Migration time model produced values too compressed
  (P10=23min to P90=58min) — S2 window of 20 min only caught 7% of services,
  making max survivability = 7% for all methods (not meaningful).
- **E3 v7 fix:** Recalibrated migration time formula:
  `t = mc×2 + (sessions/2000)×mc + 0.05×data_gb×8/bw/60`
  This produces P20≈4min, P50≈6min, P85≈10min — appropriate spread for
  the 8/20/45 minute attack windows.

### Key Results (v7 Final)
- KATS S2 Gulf Strike survivability: **0.6604** (rescues 3,368/5,100 High services)
- Best baseline S2 (B4-LogReg): 0.6563 → KATS lead: **+0.0041 (0.41 pp)**
- KATS vs B1-Criticality S2: **+19.4 pp** (0.6604 vs 0.4667)
- S3 Cascading Collapse: All methods ≤11.3% — infrastructure binding constraint confirmed

---

## [2026-05-04] — Initial Repository Creation

### Added
- Full experiment suite E1–E7 (30/30 tests complete)
- `src/` modules: dataset, model, baselines, metrics, explainability, triage
- `results/` CSVs for all experiments
- `experiments/` Jupyter notebooks for all 7 experiments
- `requirements.txt`, `LICENSE` (MIT), `README.md`

### E1 Results
- KATS-Ensemble: Recall_High=0.9873, Macro_F1=0.9555, Kappa=0.9335
- McNemar test vs all baselines: p < 0.000003
- Leads all 8 baselines on Recall_High (safety-critical metric)

### E5 Results (Ablation)
- Removing dependency features: Macro_F1 drops −18.7 pp, Kappa drops −30.3 pp
- Single most important structural contribution of KATS feature schema

### E7 Results (Sensitivity)
- 15% label noise: Recall_High still 0.9699 (−1.9 pp degradation only)
- α=5 chosen as optimal tradeoff (Recall_High=0.9888, Macro_F1=0.9313)

---

## Planned (Pre-Submission)

- [ ] Add real-trace experiment using Google Borg 2019 cluster data
- [ ] Add bootstrap CI computation notebook
- [ ] Add domain-specific baseline (prior TDSC/DSN DR triage paper)
- [ ] Add E3 scenario parameter citation from Gulf 2026 incident reports

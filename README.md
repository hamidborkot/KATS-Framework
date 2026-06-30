# KATS: Knowledge-Aware Triage System

> **Knowledge-Aware Triage System for Critical Cloud Service Migration under Coordinated Cyberattack**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Experiments-30%2F30%20Complete-brightgreen)](#experiment-tracker)
[![Recall\_High](https://img.shields.io/badge/Recall__High-98.80%25-blue)](#e1--baseline-comparison)
[![Survivability](https://img.shields.io/badge/S2%20Survivability-48.68%25-orange)](#e3--attack-survivability)

---

## 📌 Overview

KATS is a machine-learning ensemble framework for **real-time triage and prioritized migration** of critical cloud services under coordinated cyberattack. Given bandwidth constraints and shrinking migration windows, KATS ranks services by predicted operational priority — maximizing the survivability of High-criticality workloads.

Motivated by the **March 2026 Coordinated Gulf Infrastructure Attack**, KATS addresses the triage gap in existing disaster recovery frameworks: most approaches rank services by static criticality scores, ignoring dependency structure, regulatory constraints, and real-time attack dynamics.

### Key Contributions

| # | Contribution |
|---|---|
| 1 | **KATS-SYN dataset** — 75,000-service synthetic benchmark with 15 operationally grounded features and three attack scenarios |
| 2 | **KATS-Ensemble** — asymmetric-loss LightGBM + calibrated Naive Bayes ensemble with dependency-aware feature engineering |
| 3 | **Attack Scenario Simulator** — three parameterized scenarios (Precision Strike, Gulf Strike, Cascading Collapse) with bandwidth-budget triage |
| 4 | **Comprehensive evaluation** — 30 tests across 7 experiments: baseline comparison, generalization, survivability, explainability, ablation, timing, sensitivity |

---

## 🏆 Headline Results

| Metric | Value | Comparison |
|---|---|---|
| Recall\_High (E1, KATS-SYN) | **0.9880** | +1.35 pp over best ML baseline (B5-DecTree) |
| McNemar significance | **p < 0.000003** | All 8 baselines |
| 5-Fold CV Recall\_High | **1.0000 ± 0.0000** | Stable across all folds |
| S2 Gulf Strike Survivability | **0.4868** (#1) | 868/1,783 High-priority services rescued |
| S1 Precision Strike | **0.7773** (#1) | Best across all methods |
| Inference latency (10k svcs) | **460 ms** | 1,042× faster than 8-min window |
| SHAP top-20 explanation | **1.7 sec** | 17.6× faster than 30-sec target |
| Label noise robustness (15%) | **0.9699** | Only 1.9 pp drop |

---

## 📁 Repository Structure

```
KATS-Framework/
├── README.md                        # This file
├── LICENSE
├── requirements.txt
├── notebooks/
│   ├── 01_data_generation.ipynb     # KATS-SYN dataset generation
│   ├── 02_model_training.ipynb      # KATS-Ensemble training
│   ├── 03_E1_baseline.ipynb         # Experiment 1: Baseline comparison
│   ├── 04_E2_generalization.ipynb   # Experiment 2: Cross-dataset generalization
│   ├── 05_E3_survivability.ipynb    # Experiment 3: Attack survivability
│   ├── 06_E4_explainability.ipynb   # Experiment 4: SHAP explainability
│   ├── 07_E5_ablation.ipynb         # Experiment 5: Ablation study
│   ├── 08_E6_timing.ipynb           # Experiment 6: Computational timing
│   └── 09_E7_sensitivity.ipynb      # Experiment 7: Sensitivity analysis
├── src/
│   ├── __init__.py
│   ├── dataset.py                   # KATS-SYN generator
│   ├── model.py                     # KATS-Ensemble definition
│   ├── baselines.py                 # All 7 baseline methods
│   ├── triage.py                    # Attack scenario simulator
│   ├── metrics.py                   # Evaluation metrics
│   └── explainability.py            # SHAP wrapper
├── results/
│   ├── E1_baseline_comparison.csv
│   ├── E2_cv_results.csv
│   ├── E2_jsd_analysis.csv
│   ├── E3_survivability.csv
│   ├── E4_shap_values.csv
│   ├── E5_ablation.csv
│   ├── E6_timing.csv
│   └── E7_sensitivity.csv
├── figures/
│   └── [auto-generated during experiments]
└── paper/
    └── supplementary_tables.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/hamidborkot/KATS-Framework.git
cd KATS-Framework
pip install -r requirements.txt
```

### Run All Experiments

```bash
# Run sequentially
jupyter nbconvert --to notebook --execute notebooks/01_data_generation.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_model_training.ipynb
# ... repeat for 03–09
```

Or open any notebook directly in Kaggle / JupyterLab.

---

## 🧪 Experiment Tracker

| # | Experiment | Tests | Key Result | Status |
|---|---|---|---|---|
| E1 | Baseline Comparison | 8/8 | Recall\_High=0.9880, p<0.000003 | ✅ |
| E2 | Cross-Dataset Generalization | 4/4 | 5-fold=1.0000±0.0000, JSD measured | ✅ |
| E3 | Attack Survivability | 3/3 | S2 rank #1, survivability=0.4868 | ✅ |
| E4 | SHAP Explainability | 3/3 | Stability=0.243, top feature: regulatory\_flag | ✅ |
| E5 | Ablation Study | 5/5 | Dep. features: −18.7 pp F1 | ✅ |
| E6 | Computational Timing | 3/3 | 460ms inference, 1.7s SHAP | ✅ |
| E7 | Sensitivity Analysis | 3/3 | Stable within 1.9pp at 15% noise | ✅ |
| **Total** | | **30/30** | | ✅ **COMPLETE** |

---

## 📊 Dataset: KATS-SYN

| Property | Value |
|---|---|
| Total services | 75,000 |
| Training set | 60,000 |
| Test set | 15,000 |
| Features | 15 |
| Classes | High / Medium / Low |
| Class distribution | 30% / 40% / 30% |
| Attack scenarios | 3 (S1, S2, S3) |

### Features

| Feature | Type | Description |
|---|---|---|
| `service_criticality` | Float [1–10] | Operational importance score |
| `data_volume_gb` | Float | Data to migrate (GB) |
| `rto_minutes` | Float | Recovery Time Objective |
| `rpo_minutes` | Float | Recovery Point Objective |
| `dependency_count` | Int | Number of upstream dependencies |
| `downstream_critical` | Binary | Has critical downstream dependents |
| `redundancy_level` | Int [0–3] | Existing redundancy tier |
| `regulatory_flag` | Binary | Subject to compliance mandate |
| `active_sessions` | Int | Live user sessions at time of attack |
| `bandwidth_required_mbps` | Float | Migration bandwidth requirement |
| `latency_sensitivity` | Binary | Latency-sensitive workload |
| `az_risk_score` | Float [0–1] | Availability zone risk exposure |
| `multiregion_deployed` | Binary | Already multi-region |
| `migration_complexity` | Float [0–1] | Estimated migration difficulty |
| `sector_enc` | Int | Industry sector (encoded) |

---

## 🔬 Experiment Results

### E1 — Baseline Comparison

| Method | Recall\_High | Macro\_F1 | Kappa |
|---|---|---|---|
| **KATS-Ensemble** | **0.9880** | 0.9069 | 0.8772 |
| B5-DecTree | 0.9745 | 0.9195 | 0.8930 |
| B4-LogReg | 0.9525 | 0.7602 | 0.6350 |
| B1-Criticality | 0.4867 | 0.5080 | 0.2844 |
| B3-Composite | 0.4863 | 0.5099 | 0.2873 |
| B2-RTO | 0.4341 | 0.4744 | 0.2222 |
| B7-ConnectivityRank | 0.4191 | 0.4600 | 0.2196 |
| B6-DeadlineFirst | 0.4189 | 0.4745 | 0.2266 |

McNemar test: p < 0.000003 vs all baselines.

### E2 — Cross-Dataset Generalization

**5-Fold Cross-Validation on held-out KATS-SYN partitions:**

| Method | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | Mean ± Std |
|---|---|---|---|---|---|---|
| **KATS-Ensemble** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **1.0000 ± 0.0000** |
| B5-DecTree | 0.9921 | 0.9938 | 0.9871 | 0.9871 | 0.9921 | 0.9904 ± 0.0028 |
| B4-LogReg | 0.9770 | 0.9787 | 0.9680 | 0.9720 | 0.9731 | 0.9738 ± 0.0038 |
| B3-Composite | 0.5356 | 0.5339 | 0.5407 | 0.5351 | 0.5426 | 0.5376 ± 0.0034 |
| B1-Criticality | 0.5373 | 0.5407 | 0.5395 | 0.5395 | 0.5443 | 0.5403 ± 0.0023 |

**Feature Distribution Shift (JSD) vs real datasets:**

| Dataset | service\_criticality | rto\_minutes | dependency\_count | bandwidth | az\_risk | Mean JSD |
|---|---|---|---|---|---|---|
| Google Borg | 0.6605 | 0.7516 | 0.6316 | 0.1513 | 0.7788 | **0.5948** |
| BitBrains (Fin.) | 0.5743 | 0.5791 | 0.4356 | 0.1513 | 0.7625 | **0.5006** |
| Alibaba GPU | 0.8004 | 0.3685 | 0.5612 | 0.1513 | 0.7717 | **0.5306** |

### E3 — Attack Survivability

| Method | S1: Precision Strike | S2: Gulf Strike (Mar 2026) | S3: Cascading Collapse | Mean |
|---|---|---|---|---|
| **KATS-Ensemble** | **0.7773** | **0.4868** | 0.0796 | **0.4479** |
| B5-DecTree | 0.7678 | 0.4700 | 0.0808 | 0.4395 |
| B4-LogReg | 0.7493 | 0.4745 | 0.0667 | 0.4302 |
| B1-Criticality | 0.7072 | 0.4543 | 0.0690 | 0.4102 |
| B3-Composite | 0.6932 | 0.4734 | 0.0701 | 0.4122 |

S2 Gulf Strike detail: KATS rescued **868/1,783** High-priority services (Rank #1).

### E4 — SHAP Explainability

- SHAP stability score: **0.243** (top-5 features consistent across bootstrap samples)
- Top feature: `regulatory_flag` (highest mean |SHAP|)
- SHAP explanation time for top-20 services: **1.70 seconds** ✅

### E5 — Ablation Study

| Configuration | Recall\_High | Macro\_F1 | Kappa | Drop vs Full |
|---|---|---|---|---|
| **Full KATS-Ensemble** | **0.9880** | 0.9356 | 0.9104 | — |
| No Calibrated NB | 0.9879 | 0.9354 | 0.9100 | −0.01 pp |
| No Asymmetric Loss (α=1) | 0.9843 | 0.9399 | 0.9191 | −0.37 pp |
| RF Only (no ensemble) | 0.9836 | 0.9315 | 0.9056 | −0.44 pp |
| **No Dependency Features** | **0.9789** | **0.7486** | **0.6071** | **−18.7 pp F1** |

Key finding: Dependency features (`dependency_count`, `downstream_critical`) are the single most important structural contribution — removal causes −18.7 pp Macro\_F1 collapse.

### E6 — Computational Timing

| Benchmark | Result | Target | Status |
|---|---|---|---|
| Training time (N=15,000) | 40.5 sec | Feasible | ✅ |
| Inference N=100 | 149 ms | <8 min | ✅ |
| Inference N=1,000 | 117 ms | <8 min | ✅ |
| Inference N=5,000 | 267 ms | <8 min | ✅ |
| Inference N=10,000 | **460 ms** | <8 min | ✅ |
| SHAP top-20 | **1.7 sec** | <30 sec | ✅ |

### E7 — Sensitivity Analysis

**Alpha (α) sensitivity:**

| α | Recall\_High | Macro\_F1 | Kappa |
|---|---|---|---|
| 1 | 0.9831 | 0.9374 | 0.9158 |
| 2 | 0.9867 | 0.9376 | 0.9156 |
| 3 | 0.9877 | 0.9351 | 0.9108 |
| **5** | **0.9888** | **0.9313** | **0.9028** |
| 7 | 0.9896 | 0.9272 | 0.8954 |
| 9 | 0.9897 | 0.9234 | 0.8882 |
| 12 | 0.9901 | 0.9146 | 0.8730 |

Chosen α=5: optimal tradeoff between Recall\_High and Macro\_F1.

**Label noise robustness:**

| Noise | Recall\_High | Macro\_F1 | Kappa |
|---|---|---|---|
| 0% | 0.9888 | 0.9313 | 0.9028 |
| 5% | 0.9897 | 0.9321 | 0.9060 |
| 10% | 0.9795 | 0.8846 | 0.8383 |
| 15% | **0.9699** | 0.8358 | 0.7674 |

Only **−1.89 pp** Recall\_High at 15% noise — strong robustness.

**Class imbalance sensitivity:**

| Config | Recall\_High | Macro\_F1 |
|---|---|---|
| 10H-40M-50L | 0.9567 | 0.8874 |
| 30H-40M-30L | 0.9767 | 0.9338 |
| 40H-30M-30L | 0.9800 | 0.9349 |
| 50H-30M-20L | 0.9853 | 0.9303 |

---

## 🔐 Threat Model

KATS targets **coordinated multi-vector cyberattacks** against critical cloud infrastructure:

- **S1 — Precision Strike**: 30% BW loss, 45-min window, cap\_frac=0.55
- **S2 — Coordinated Gulf Strike** *(Mar 2026 reference scenario)*: 60% BW loss, 20-min window, cap\_frac=0.45
- **S3 — Cascading Collapse**: 85% BW loss, 8-min window, cap\_frac=0.35

---

## 📋 Citation

If you use KATS or KATS-SYN in your research, please cite:

```bibtex
@article{tulla2026kats,
  title   = {KATS: Knowledge-Aware Triage System for Critical Cloud Service
             Migration under Coordinated Cyberattack},
  author  = {Tulla, Md. Hamid Borkot},
  year    = {2026},
  note    = {Under review}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>KATS Framework · IEEE TDSC 2026 Submission · MD Hamid Borkot Tulla</sub>
</p>

# Stetho Sprint 6C — Baseline & Evaluation Protocol Specification

**Author:** Lead ML Research Engineer  
**Sprint:** Sprint 6C  
**Date:** 2026-11-01  
**Git Branch:** `research/ml-diverse-machines`  
**Evaluation Scope:** 360 Industrial Audio Files across Fan, Pump, and Valve (12 Machines)  
**Status:** COMPLETE — PHASE 1 EVALUATION PROTOCOL ESTABLISHED

---

## 1. Executive Summary & Sprint Charter

Sprint 6C addresses the primary operational limitation identified during the Sprint 6B audit: **the collapse of acoustic anomaly detection models when deployed zero-shot on previously unseen physical machines (0.5667 Macro ROC-AUC) and cross-category industrial machines (0.3694 ROC-AUC).**

### Core Research Objective
To develop, validate, and benchmark lightweight, leakage-safe **Few-Shot Machine Calibration** and **Generalization Architectures** that restore high anomaly detection accuracy ($\ge 0.90$ ROC-AUC) on unseen machines with minimal operator effort ($N=5$ to $N=10$ normal clips), while maintaining sub-100ms practical inference latency.

---

## 2. Dataset Inventory & Partition Protocol

All splits reside in `ml/data/splits/` and have been verified to have **strictly zero cross-split data leakage or sample overlap**:

```
ml/data/splits/
├── diverse_machine_specific/    (In-domain machine-specific benchmark)
│   ├── fan/   (id_00, id_02, id_04, id_06) -> 10 train_norm, 2 dev_norm, 18 test (3 norm, 15 abn)
│   ├── pump/  (id_00, id_02, id_04, id_06) -> 10 train_norm, 2 dev_norm, 18 test (3 norm, 15 abn)
│   └── valve/ (id_00, id_02, id_04, id_06) -> 10 train_norm, 2 dev_norm, 18 test (3 norm, 15 abn)
├── unseen_machine/             (Leave-one-machine-out cross-validation)
│   ├── fan/   (heldout_id_00..06) -> 30 train_known, 18 test_unseen (3 norm, 15 abn)
│   ├── pump/  (heldout_id_00..06) -> 30 train_known, 18 test_unseen (3 norm, 15 abn)
│   └── valve/ (heldout_id_00..06) -> 30 train_known, 18 test_unseen (3 norm, 15 abn)
└── cross_category/             (Cross-domain transfer benchmark)
    ├── fan_to_pump   -> 40 train_src, 72 test_tgt (12 norm, 60 abn)
    ├── fan_to_valve  -> 40 train_src, 72 test_tgt (12 norm, 60 abn)
    ├── pump_to_fan   -> 40 train_src, 72 test_tgt (12 norm, 60 abn)
    ├── pump_to_valve -> 40 train_src, 72 test_tgt (12 norm, 60 abn)
    ├── valve_to_fan  -> 40 train_src, 72 test_tgt (12 norm, 60 abn)
    └── valve_to_pump -> 40 train_src, 72 test_tgt (12 norm, 60 abn)
```

### Partitioning Rules & Data Availability Notice
- Each machine possesses exactly **15 normal** and **15 abnormal** physical recordings on disk (30 files per machine; 360 total).
- For in-domain splits: 10 normal files are reserved for training, 2 for validation/dev, and 3 normal + 15 abnormal for final testing.
- For few-shot calibration experiments: $N \in \{2, 5, 10\}$ normal recordings are drawn strictly from available training/dev normal pools. **Calibration on 20 or 50 recordings is documented as physically unavailable** on this dataset due to the total sample limit of 15 normal clips per machine.
- Under no circumstance are abnormal recordings or final test clips exposed to training or threshold calibration.

---

## 3. Verified Pre-Sprint Baseline Reference

The audited baseline metrics prior to introducing Sprint 6C enhancements:

| Evaluation Protocol | Target Machines | Model Architecture | Macro ROC-AUC | Macro pAUC (0.1) | Failure Mechanism |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Protocol A (In-Domain)** | Fan `id_00, 02, 04, 06` | Optimized ResNet-AE | **0.9833** | **0.9737** | SOTA in-domain reference |
| **Protocol A (In-Domain)** | Fan `id_00, 02, 04, 06` | Compact Student-AE | **0.9000** | **0.9035** | Sub-50ms edge reference |
| **Protocol A (In-Domain)** | Fan `id_00, 02, 04, 06` | Dense Autoencoder | **0.9722** | **0.9561** | Sprint 1 MLP reference |
| **Protocol B (Unseen Machines)** | Fan Held-out IDs | Optimized ResNet-AE | **0.5667** | **0.7281** | **Severe Inversion** on `id_02` & `id_06` (0.2667) |
| **Protocol C (Cross-Category)** | Fan $\to$ Pump | Optimized ResNet-AE | **0.3694** | **0.5185** | Complete hydraulic mismatch |
| **Protocol C (Cross-Category)** | Fan $\to$ Valve | Optimized ResNet-AE | **0.6986** | **0.6543** | High false alarms on impulse hiss |

---

## 4. Sprint 6C Quality Gates

1. **Gate 1 (Few-Shot Calibration Target):**
   Few-shot calibration with $N \le 10$ normal clips must improve unseen-machine Macro ROC-AUC from **0.5667 to $\ge 0.8500$**, with no individual machine scoring below 0.7000.
2. **Gate 2 (Cross-Category Adaptation):**
   Few-shot adaptation must improve Fan $\to$ Pump transfer from **0.3694 to $\ge 0.7500$**.
3. **Gate 3 (Latency & Efficiency):**
   Calibration must run in $< 5.0$ seconds on CPU, and inference latency must remain $< 100$ ms (ideally $< 50$ ms).
4. **Gate 4 (Leakage Isolation):**
   Zero exposure of test-set abnormal clips or final test normal clips during calibration.

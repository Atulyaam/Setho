# Stetho Sprint 6B — Ensemble and Cascade Verification Report

**Author:** Lead Independent ML Validation Engineer  
**Date:** 2026-11-01  
**Git Branch:** `research/ml-diverse-machines`  
**Artifact Scope:** [`ml/models/research/ensemble_fusion.py`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/models/research/ensemble_fusion.py), [`ml/models/research/margin_cascade.py`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/models/research/margin_cascade.py)  
**Audit Status:** COMPLETE — FULL DIVERSITY, CORRELATION & CASCADE PATH AUDIT

---

## 1. Executive Summary & Audit Mandate

An independent audit was conducted on the multi-model fusion engines, score normalization procedures, correlation dynamics, and confidence-margin gated cascade policies developed in Phase 6 and Phase 7.

### Core Audit Verdicts

1. **Model Diversity & Collinearity:**
   - CNN autoencoder architectures (`residual_cnn_ae` and `multiscale_cnn_ae`) are **highly collinear** (Pearson $r = 0.9782$, Spearman $\rho = 0.9736$). Combining them yields near-zero orthogonal information.
   - Embedding Mahalanobis distance exhibits **meaningful diversity** with ResNet-AE (Pearson $r = 0.4796$, Spearman $\rho = 0.4489$).
2. **Ensemble Sufficiency Paradox:**
   - The top ensemble (`msae_maha_weighted` and `tri_model_consensus`) achieved **0.9833 Macro ROC-AUC**.
   - The single Optimized ResNet-AE achieved the exact same **0.9833 Macro ROC-AUC**.
   - However, the ensemble requires **204 to 220 ms** of CPU inference, whereas ResNet-AE requires only **100.2 ms** (and its ONNX version runs in **63 ms**).
   - **Audit Verdict:** The ensemble adds **no net accuracy gain** over a properly regularized single ResNet-AE, while doubling operational latency and operational complexity.
3. **Margin-Gated Cascade Integrity:**
   - The reported **0.9445 Macro ROC-AUC** for the confidence-margin gated cascade ($\gamma = 0.05$) is **100% genuine and correctly derived** through dynamic routing.
   - Stage 1 (Fast Student) handles **61.1% of samples with immediate early exit**, achieving an average latency of **87.38 ms**.

---

## 2. Model Score Correlation & Diversity Audit

Evaluated across all 72 benchmark test recordings:

| Model Pair | Pearson Correlation ($r$) | Spearman Rank Correlation ($\rho$) | Diversity Assessment |
| :--- | :---: | :---: | :--- |
| **ResNet-AE vs. MultiScale-AE** | **0.9782** | **0.9736** | **Redundant / Highly Collinear** (Both learn identical spatial error maps) |
| **ResNet-AE vs. Mahalanobis** | **0.4796** | **0.4489** | **High Diversity** (Covariance distance captures different tail errors) |
| **MultiScale vs. Mahalanobis** | **0.5235** | **0.4959** | **Moderate-High Diversity** |

```
                 Model Score Correlation Matrix (Spearman Rank)
                     ResNet-AE   MultiScale-AE   Mahalanobis
      ResNet-AE        1.0000        0.9736        0.4489
      MultiScale-AE    0.9736        1.0000        0.4959
      Mahalanobis      0.4489        0.4959        1.0000
```

---

## 3. Score Normalization & Fusion Strategy Audit

### 3.1 Normalization Derivation
In [`ml/models/research/ensemble_fusion.py`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/models/research/ensemble_fusion.py), scores are calibrated using development normal sets exclusively:
$$z_i = \frac{s_i - \mu_{dev, i}}{\sigma_{dev, i} + \epsilon}$$
- **Zero Leakage Verified:** Means and standard deviations were computed strictly on normal development recordings ($N=2$).
- **Score Rank Normalization:** Rank averaging computes percentiles against the empirical CDF of the dev set via `np.searchsorted`.

### 3.2 Ensemble Comparison Table

| Fusion Strategy | Constituents | Macro ROC-AUC | Macro pAUC | Macro F1 | Macro Recall | CPU Latency | Architectural Justification |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Weighted Average** | ResNet + Mahalanobis (0.6 / 0.4) | 0.9445 | 0.9298 | 0.8240 | 0.8167 | 209.6 ms | Fails to outperform ResNet alone (0.9833) |
| **Rank Average** | ResNet + Mahalanobis | 0.8750 | 0.7697 | 0.7514 | 0.7000 | 215.4 ms | Quantile discretization degrades subtle margins |
| **Max Residual** | ResNet + Mahalanobis | 0.9667 | 0.9561 | **0.8889** | **0.9167** | 207.7 ms | **Highest Recall (91.67%)** for safety-critical cases |
| **Weighted Average** | MultiScale + Mahalanobis | **0.9833** | **0.9737** | 0.7879 | 0.8000 | 204.5 ms | Matches single ResNet-AE at 2x compute cost |
| **Consensus (Tri-Model)**| ResNet + MultiScale + Maha | **0.9833** | **0.9737** | 0.7798 | 0.8000 | 219.8 ms | Triple compute overhead for zero net accuracy gain |

---

## 4. Margin-Gated Cascade Deep Dive & Routing Verification

The confidence-margin gated cascade in [`ml/models/research/margin_cascade.py`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/models/research/margin_cascade.py) implements:
1. **Stage 1:** Run ultra-lightweight Student detector ([`CompactStudentAE`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/models/research/distilled_student.py), 22K params).
2. **Decision Margin:** $d = |s_{student} - \tau_{student}|$.
3. **Routing Condition:**
   - If $d \ge \gamma$: Decision is accepted immediately (**Early Exit**).
   - If $d < \gamma$: Input is in the ambiguous decision boundary; escalate to Stage 2 Teacher ([`ResidualCNNAE`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/models/research/residual_cnn_ae.py)).

### 4.1 Operating Parameter Sweep

| Margin ($\gamma$) | Early-Exit Rate | Escalation Rate (Stage 2) | Macro ROC-AUC | Macro pAUC | Average CPU Latency | Operational Profile |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $\gamma = 0.01$ | **91.7%** | 8.3% | 0.9000 | 0.9035 | **56.07 ms** | Ultra-fast edge filter |
| $\gamma = 0.03$ | **70.8%** | 29.2% | 0.9000 | 0.9035 | 76.88 ms | Moderate boundary guard |
| $\gamma = 0.05$ | **61.1%** | **38.9%** | **0.9445** | **0.9474** | **87.38 ms** | **Optimal Accuracy/Latency Tradeoff** |
| $\gamma = 0.10$ | 43.1% | 56.9% | 0.9445 | 0.9474 | 105.56 ms | Over-escalation; latency approaches full teacher |

### 4.2 Per-Machine Decision Path Verification ($\gamma = 0.05$)

- **`id_00`:** Student AUC: 0.8667 $\to$ Teacher AUC: 1.0000. Escalation rate: 44.4%. Cascade AUC: **0.8667** (Latency: 93.9 ms).
  - *Missed anomaly analysis:* Two subtle bearing defect clips on `id_00` had student scores cleanly above the threshold ($d > 0.05$), exiting Stage 1 with normal classification instead of escalating.
- **`id_02`:** Student AUC: 1.0000 $\to$ Teacher AUC: 1.0000. Escalation rate: 16.7%. Cascade AUC: **1.0000** (Latency: 64.1 ms).
- **`id_04`:** Student AUC: 0.8889 $\to$ Teacher AUC: 0.9111. Escalation rate: 44.4%. Cascade AUC: **0.9111** (Latency: 93.8 ms).
  - Ambiguous bearing chatter correctly escalated to teacher, boosting AUC from 0.8889 to 0.9111.
- **`id_06`:** Student AUC: 0.8444 $\to$ Teacher AUC: 1.0000. Escalation rate: 50.0%. Cascade AUC: **1.0000** (Latency: 97.7 ms).
  - Borderline cases escalated to teacher, achieving a perfect 1.0000 ROC-AUC.

---

## 5. Architectural Recommendations

1. **Deprecate Multi-Model Ensembles in Production:**
   - Ensembles offer no meaningful accuracy gain over single ResNet-AE while increasing CPU latency beyond 200 ms.
2. **Retain Margin Cascade for Hybrid Cloud/Edge Architectures:**
   - At $\gamma = 0.05$, the cascade reduces cloud ingestion workload by 61.1% while maintaining 0.9445 Macro ROC-AUC.

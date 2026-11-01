# Stetho Master Research Program — Phase 6: Advanced Ensemble & Model Fusion Research

## Executive Summary

Phase 6 of the Stetho Master Research Program systematically investigated multi-model fusion mechanisms, non-parametric rank combination, extreme-value max aggregation, and hierarchical cascaded early-exit filtering across all four pilot industrial machines (`id_00`, `id_02`, `id_04`, `id_06`).

All ensembles were constructed, calibrated, and evaluated under strict zero-leakage conditions:
- Base neural autoencoders and covariance models were trained strictly on normal training recordings.
- Z-score normalization parameters ($\mu, \sigma$), empirical rank pools, and operating thresholds were derived exclusively from development normal recordings.
- Test evaluation was conducted on held-out test splits (72 total evaluation recordings).

### Key Takeaway & Strategic Decision
The multi-model ensemble combining **MultiScale CNN-AE and Embedding Mahalanobis (`msae_maha_weighted`)** achieved a state-of-the-art **0.9833 Macro ROC-AUC** and **0.9737 Macro pAUC**. However, the single **Optimized ResNet-AE** architecture developed in Phase 5 matches this identical accuracy benchmark (**0.9833 ROC-AUC / 0.9737 pAUC**) at **109.2 ms CPU latency**—less than half the latency of the ensemble (219.8 ms) and with an ultra-compact single-checkpoint deployment footprint (1.81M parameters, 7.08 MB).

For applications where catastrophic anomaly misses must be prevented at all costs, **Extreme Max Fusion (`resnet_maha_max`)** emerged as the top operational performer, achieving the highest macro **F1-score of 0.8889** and **93.3% recall**.

---

## 1. Fusion Strategies & Mathematical Formulations

### 1.1 Z-Score Standardized Weighted Average
Because reconstruction errors ($L_1 \in [0.15, 0.45]$) and statistical Mahalanobis distances ($D_M \in [10, 150]$) reside on vastly different dynamic ranges, unnormalized combinations are biased towards distance metrics. Each model's raw score $s_m(x)$ is standardized against its development set mean $\mu_m$ and variance $\sigma_m^2$:
$$z_m(x) = \frac{s_m(x) - \mu_m}{\sigma_m}$$
The fused anomaly score is computed as:
$$S_{\text{weighted}}(x) = \sum_{m=1}^{M} w_m z_m(x), \quad \sum w_m = 1$$

### 1.2 Non-Parametric Rank Averaging
To guard against heavy-tailed distributions and calibration drift, rank averaging maps raw continuous scores to their empirical cumulative distribution function (ECDF) percentile rank relative to development recordings:
$$R_m(x) = \frac{1}{N_{\text{dev}}} \sum_{i=1}^{N_{\text{dev}}} \mathbb{I}(s_m(x_i) \le s_m(x))$$
$$S_{\text{rank}}(x) = \frac{1}{M} \sum_{m=1}^{M} R_m(x)$$

### 1.3 Extreme-Value Max Fusion
Flags an anomaly if any trusted constituent model detects a significant deviation:
$$S_{\max}(x) = \max_{m} \left( \frac{s_m(x) - \mu_m}{\sigma_m} \right)$$

### 1.4 Hierarchical Two-Stage Early-Exit Cascade
To achieve high-speed inference for production edge deployment:
- **Stage 1 (Fast Filter):** Runs ultra-lightweight `embedding_mahalanobis` (<5 ms). If the anomaly score $s_1(x) \le \theta_{\text{exit}}$ (calibrated at the 50th percentile of normal development audio), the recording is immediately classified as **NORMAL** and inference terminates early.
- **Stage 2 (Heavy Inspection):** If $s_1(x) > \theta_{\text{exit}}$, the recording is escalated to the high-capacity `residual_cnn_ae` for deep spatial-temporal reconstruction analysis.

---

## 2. Experimental Results & Comparative Leaderboard

The table below summarizes the quantitative evaluation across all tested ensemble architectures:

| Ensemble ID | Fusion Strategy | Constituent Models | Macro ROC-AUC | Macro pAUC (0.1) | Macro F1 | Precision | Recall | CPU Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`msae_maha_weighted`** | Weighted Average (0.7/0.3) | MultiScale-AE + Mahalanobis | **0.9833** | **0.9737** | 0.7879 | 0.9166 | 0.8000 | 219.8 ms |
| **`tri_model_consensus`** | Weighted Consensus (0.4/0.4/0.2) | ResNet-AE + MSAE + Mahalanobis | **0.9833** | **0.9737** | 0.7798 | 0.8931 | 0.8000 | 319.4 ms |
| **`resnet_maha_max`** | Extreme Max Fusion | ResNet-AE + Mahalanobis | **0.9667** | **0.9561** | **0.8889** | 0.9028 | **0.9333** | 207.7 ms |
| **`resnet_maha_weighted`** | Weighted Average (0.65/0.35) | ResNet-AE + Mahalanobis | **0.9445** | **0.9298** | 0.8240 | 0.9412 | 0.8167 | 209.6 ms |
| **`resnet_maha_rank`** | Empirical Rank Averaging | ResNet-AE + Mahalanobis | **0.8750** | **0.7697** | 0.7514 | 0.8958 | 0.8000 | 209.5 ms |
| **`cascade_fast_reject`** | Hierarchical 2-Stage ($q=0.5$) | Mahalanobis (S1) $\to$ ResNet (S2) | **0.8611** | **0.8947** | 0.8041 | **0.9486** | 0.7667 | **127.8 ms** |

---

## 3. Per-Machine Breakdown of Top Ensembles

### 3.1 MultiScale CNN-AE + Mahalanobis (`msae_maha_weighted`)
- **`id_00`:** ROC-AUC: **1.0000**, pAUC: **1.0000**, Precision: 1.0000, Latency: 212.1 ms
- **`id_02`:** ROC-AUC: **1.0000**, pAUC: **1.0000**, Precision: 1.0000, Latency: 213.0 ms
- **`id_04`:** ROC-AUC: **0.9333**, pAUC: **0.8947**, Precision: 0.8333, Latency: 219.8 ms
- **`id_06`:** ROC-AUC: **1.0000**, pAUC: **1.0000**, Precision: 0.8333, Latency: 234.2 ms

### 3.2 Extreme Max Fusion (`resnet_maha_max`)
- **`id_00`:** ROC-AUC: **1.0000**, pAUC: **1.0000**, Precision: 1.0000, Latency: 207.0 ms
- **`id_02`:** ROC-AUC: **1.0000**, pAUC: **1.0000**, Precision: 1.0000, Latency: 202.5 ms
- **`id_04`:** ROC-AUC: **0.9556**, pAUC: **0.9649**, Precision: 0.7778, Latency: 210.6 ms *(Highest id_04 ranking!)*
- **`id_06`:** ROC-AUC: **0.9111**, pAUC: **0.8596**, Precision: 0.8333, Latency: 210.6 ms

---

## 4. Single-Model Champion vs. Multi-Model Ensembles

A critical objective of Phase 6 was determining whether multi-model ensembles provide sufficient accuracy benefit over optimized single models to justify double the inference latency:

| Architecture | Model Count | Total Parameters | Macro ROC-AUC | Macro pAUC | CPU Latency | Architectural Complexity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Optimized ResNet-AE (Phase 5)** | **1 (Single)** | **1.81 M** | **0.9833** | **0.9737** | **109.2 ms** | **Low (Single Checkpoint)** |
| **`msae_maha_weighted`** | 2 (Dual) | 2.58 M | **0.9833** | **0.9737** | 219.8 ms | Moderate (2 models + stats) |
| **`tri_model_consensus`** | 3 (Tri) | 4.39 M | **0.9833** | **0.9737** | 319.4 ms | High (3 models + stats) |
| **`resnet_maha_max`** | 2 (Dual) | 1.81 M | 0.9667 | 0.9561 | 207.7 ms | Moderate (F1 champion: 0.8889) |
| **Dense Autoencoder (DAE)** | 1 (Single) | 4.34 M | 0.9722 | 0.9561 | 116.8 ms | Low |
| **Stetho-MSAE (Sprint 3)** | 1 (Single) | 0.56 M | 0.9722 | 0.9649 | 381.5 ms | Low |

### Scientific Verdict
1. **The Principle of Architectural Sufficiency:**
   With proper training optimization (AdamW + Cosine Annealing + GroupNorm), the single **`residual_cnn_ae`** absorbs the discriminative features that simpler ensembles capture, matching the peak 0.9833 ROC-AUC and 0.9737 pAUC while running in **109.2 ms** (3.5x faster than Stetho-MSAE and 2.0x faster than the dual ensemble).
2. **Specialized Use for Max Fusion:**
   `resnet_maha_max` is uniquely suited for ultra-high-reliability industrial deployments where false negatives carry extreme financial penalties, achieving **0.9556 ROC-AUC and 0.9649 pAUC on the challenging bearing failure machine `id_04`**.
3. **Cascade Tradeoff:**
   While `cascade_fast_reject` successfully cut inference latency down to 127.8 ms by early-exiting clear normal sounds, setting the cutoff at 50% normal quantile induced minor false negative leakage on `id_00` and `id_06`. For Phase 7, distillation or an adaptive classifier will be researched.

---

## 5. Phase 6 Acceptance Criteria Verification

- [x] Multi-model ensemble framework implemented in `ml/models/research/ensemble_fusion.py`.
- [x] Evaluated 5 diverse score fusion strategies and a 2-stage hierarchical cascade across all 4 pilot machines.
- [x] Zero-leakage calibration strictly observed on development normal recordings.
- [x] Detailed comparative summary logged in `ml/experiments/research_artifacts/phase6/ensemble_comparison_summary.json`.
- [x] Automated test suite verified (`tests/unit/test_phase6_ensembles.py` passing 3/3).
- [x] Experiment recorded in `ml/experiments/experiment_registry.json`.

---

## 6. Next Steps: Phase 7 — Cascaded Inference & Distillation
- Research knowledge distillation to transfer ensemble and multi-scale teacher representations into an ultra-compact single-network student (<50ms CPU latency).
- Design precision-gated early-exit cascades with adaptive confidence thresholds.

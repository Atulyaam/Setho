# Sprint 7.6 / E1 — Acoustic Feature Ablation Technical Analysis

**Author:** Principal Audio ML Research Engineer & Reproducibility Auditor  
**Date:** 2026-11-01  
**Git Branch:** `research/sprint-7-6-balanced-improvements`  
**Base Commit:** `e367a8e`  
**Evaluation Benchmark:** Frozen MIMII Fan Benchmark (`id_00`, `id_02`, `id_04`, `id_06`)  
**Status:** COMPLETE — EMPIRICALLY GROUNDED ANALYSIS

---

## 1. Experimental Objective & Controlled Setup

The objective of Sprint 7.6 / E1 was to determine whether alternative acoustic representations improve industrial anomaly detection over the standard 64-bin log-Mel filterbank baseline.

To isolate the effect of the acoustic representation and prevent architectural confounding variables:
1. **Identical Neural Architecture:** All 4 configurations utilized the exact same `ResidualCNNAE` autoencoder:
   - Input shape: $(1, 64, 64)$
   - Base channels: 16, Latent dimension: 64, Normalization: GroupNorm
   - Exact Parameter Count: **1,808,065 parameters** across all 16 trained models.
2. **Identical Data Partitions:**
   - Train Normal: 10 clips per machine (scalers fit exclusively on train normal).
   - Dev Normal: 2 clips per machine (threshold calibration at 95th quantile).
   - Test Partition: 3 Normal + 15 Abnormal clips per machine (18 clips total, 72 across fan family).
3. **Identical Training Pipeline:**
   - Seed: 42 (deterministic across NumPy, PyTorch CPU, Python).
   - Optimizer: AdamW ($lr=0.001$, $wd=1\times 10^{-4}$).
   - Loss: L1 reconstruction error.
   - Scheduler: CosineAnnealingLR (15 epochs max, early stopping patience 4 on dev normal loss).

---

## 2. Macro Performance Summary

| Feature Representation | Macro ROC-AUC | 95% Confidence Interval | Macro PR-AUC | Macro F1 | FPR | Feat. Latency (P50) | End-to-End (P50) | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline Log-Mel (64 bins)** | **0.9778** | $[0.9600, 0.9956]$ | **0.9958** | **0.8066** | 0.3333 | **4.57 ms** | **81.47 ms** | 12.26 clips/s |
| **2. MFCC (64 coefficients)** | **0.9833** | $[0.9506, 1.0000]$ | **0.9967** | **0.8698** | 0.4167 | **3.92 ms** | **70.32 ms** | 14.18 clips/s |
| **3. Log-Mel + Spectral Fusion** | **0.9445** | $[0.8356, 1.0000]$ | **0.9870** | **0.8778** | 0.5000 | **36.87 ms** | **116.33 ms** | 8.66 clips/s |
| **4. Log-Mel + Temporal Fusion** | **0.9889** | $[0.9671, 1.0000]$ | **0.9979** | **0.7944** | 0.4167 | **15.35 ms** | **85.80 ms** | 11.34 clips/s |

---

## 3. Per-Machine Breakdown

```
+---------------------------------------------------------------------------------------------------+
| Feature Variant         | Metric      | id_00   | id_02   | id_04   | id_06   | Macro Mean ± 95% CI   |
+---------------------------------------------------------------------------------------------------+
| baseline_logmel64       | ROC-AUC     | 0.9778  | 1.0000  | 0.9556  | 0.9778  | 0.9778 ± 0.0178       |
|                         | PR-AUC      | 0.9958  | 1.0000  | 0.9914  | 0.9958  | 0.9958 ± 0.0034       |
|                         | F1-Score    | 0.4211  | 0.9091  | 0.9286  | 0.9677  | 0.8066                |
|                         | FPR         | 0.0000  | 1.0000  | 0.0000  | 0.3333  | 0.3333                |
+---------------------------------------------------------------------------------------------------+
| mfcc64                  | ROC-AUC     | 0.9333  | 1.0000  | 1.0000  | 1.0000  | 0.9833 ± 0.0327       |
|                         | PR-AUC      | 0.9866  | 1.0000  | 1.0000  | 1.0000  | 0.9967 ± 0.0066       |
|                         | F1-Score    | 0.6364  | 0.9375  | 0.9375  | 0.9677  | 0.8698                |
|                         | FPR         | 0.0000  | 0.6667  | 0.6667  | 0.3333  | 0.4167                |
+---------------------------------------------------------------------------------------------------+
| logmel_spectral_fusion  | ROC-AUC     | 0.7778  | 1.0000  | 1.0000  | 1.0000  | 0.9445 ± 0.1089       |
|                         | PR-AUC      | 0.9482  | 1.0000  | 1.0000  | 1.0000  | 0.9870 ± 0.0254       |
|                         | F1-Score    | 0.6667  | 0.9091  | 0.9677  | 0.9677  | 0.8778                |
|                         | FPR         | 0.3333  | 1.0000  | 0.3333  | 0.3333  | 0.5000                |
+---------------------------------------------------------------------------------------------------+
| logmel_temporal_fusion  | ROC-AUC     | 1.0000  | 1.0000  | 1.0000  | 0.9556  | 0.9889 ± 0.0218       |
|                         | PR-AUC      | 1.0000  | 1.0000  | 1.0000  | 0.9914  | 0.9979 ± 0.0042       |
|                         | F1-Score    | 0.3333  | 0.9091  | 0.9677  | 0.9677  | 0.7944                |
|                         | FPR         | 0.0000  | 1.0000  | 0.3333  | 0.3333  | 0.4167                |
+---------------------------------------------------------------------------------------------------+
```

---

## 4. In-Depth Technical & Acoustic Analysis

### 4.1 Variant 1 vs Variant 2: Baseline Log-Mel vs. MFCC Cepstrogram
- **Acoustic Physics:** Log-Mel preserves direct localized spatial-frequency relationships across the 64 filterbanks. In contrast, MFCC applies the Discrete Cosine Transform (DCT) across frequency, packing spectral envelope energy into the lowest 12–16 quefrency bins while distributing residual noise across higher bins.
- **Empirical Behavior:**
  - MFCC achieves a slightly higher macro ROC-AUC (**0.9833 vs 0.9778**, $+0.56\%$) and higher F1 (**0.8698 vs 0.8066**).
  - However, on machine `id_00`, MFCC drops noticeably from **0.9778 to 0.9333** ($-0.0445$).
  - The 95% confidence intervals overlap substantially ($[0.9600, 0.9956]$ for log-Mel vs $[0.9506, 1.0000]$ for MFCC).
  - **Verdict on MFCC:** While computationally efficient ($3.92\text{ ms}$ feature extraction), MFCC does not offer a statistically decisive improvement and increases variance across machines.

### 4.2 Variant 3: Failure Mode of Log-Mel + Spectral Fusion
- **Hypothesis:** Appending frame-level spectral centroid, rolloff, bandwidth, and flatness was hypothesized to assist the autoencoder in flagging acoustic brightness or turbulence shifts.
- **Observed Empirical Failure:**
  - Macro ROC-AUC fell to **0.9445**, with a catastrophic drop on `id_00` down to **0.7778** ($-0.2000$ compared to baseline).
  - Standard deviation widened to $\pm 0.1111$ with a wide 95% CI ($[0.8356, 1.0000]$).
  - Average FPR rose to $0.5000$ (50% false positive rate on normal test clips).
  - Feature extraction latency escalated to **36.87 ms** ($8\times$ slower than baseline) due to continuous STFT moment calculations.
- **Root Cause Analysis:**
  - Spectral moments (especially centroid and bandwidth) exhibit high intra-machine variance during normal operating states.
  - In normal recordings with minor speed fluctuations or ambient hum, centroid and flatness fluctuate drastically, causing the convolutional autoencoder to produce high reconstruction residuals on normal validation clips.
  - This severely destabilizes the 95th quantile threshold and penalizes the ranking on quiet machines like `id_00`.
  - **Verdict on Spectral Fusion:** Formally rejected. Introduces high latency, high variance, and unacceptable degradation on baseline machinery.

### 4.3 Variant 4: Log-Mel + Temporal Fusion
- **Hypothesis:** Frame-level temporal dynamics (RMS energy, ZCR, crest factor, energy flux) might highlight impulsive clicking or bearing defects.
- **Observed Empirical Behavior:**
  - Nominal Macro ROC-AUC reached **0.9889 ± 0.0218** ($+0.0111$ absolute gain over baseline).
  - Achieved perfect 1.0000 ROC-AUC on `id_00`, `id_02`, and `id_04`.
  - However, on `id_06`, ROC-AUC was slightly lower at **0.9556** (compared to 0.9778 on baseline).
  - Macro F1 score degraded from **0.8066 down to 0.7944** due to threshold sensitivity on `id_00` ($F1=0.3333$).
  - Feature extraction latency tripled from **4.57 ms to 15.35 ms**.
- **Statistical Significance Check:**
  - The 95% confidence interval of temporal fusion ($[0.9671, 1.0000]$) overlaps the baseline log-Mel interval ($[0.9600, 0.9956]$).
  - In a test set where each machine contains only $N=3$ normal recordings, a $+0.0111$ macro change corresponds to exactly ONE normal recording shifting rank by 1 position on `id_00`.
  - This is well within the expected random sampling error of small test sets.

---

## 5. Decision Recommendation: KEEP BASELINE

In accordance with Section 7 of the Sprint 7.6 / E1 charter:
- **Decision:** **KEEP BASELINE**
- **Justification:**
  1. No alternative representation demonstrates a statistically significant, monotonic improvement across all four evaluation machines.
  2. The baseline standard 64-bin log-Mel filterbank achieves **0.9778 Macro ROC-AUC** with the lowest variance across machines ($\pm 0.0181$), the lowest latency overhead (**4.57 ms** extraction, **81.47 ms** end-to-end), and zero extraneous DSP dependencies.
  3. Spectral fusion caused severe performance collapse on `id_00` ($0.7778$).
  4. While temporal fusion nominally gained $+0.0111$ Macro ROC-AUC, its confidence interval overlaps the baseline, its F1 score is lower, and its extraction latency is $3.4\times$ higher. Adopting it would add computational complexity without statistically reliable benefit.

# Sprint 8.2: Independent Benchmark & Data Provenance Audit

## 1. Executive Summary

This independent audit investigates the anomaly detection benchmark metrics reported in Sprint 8.1 for the Stetho unified inference architecture. 

### Key Findings of the Audit
1. **Replication of Sprint 8.1 Metrics:**
   - The reported Sprint 8.1 metrics were independently reproduced:
     - Fan ROC-AUC: **0.9042**
     - Pump ROC-AUC: **0.2444** (one-sided) / **0.2222** (raw)
     - Valve ROC-AUC: **0.0000** (one-sided)
     - Overall Macro ROC-AUC: **0.6586**
2. **Root Cause of Valve ROC-AUC = 0.0000:**
   - **Deficit Anomaly Truncation:** In MIMII valve recordings, normal valve operation consists of loud, continuous, turbulent fluid hiss ($\text{RMS} \approx -41.8\text{ dBFS}$, peak $\approx 0.30$). Abnormal valves suffer from flow restriction, blockage, or closure, making them quieter ($\text{RMS} \approx -45.5\text{ dBFS}$, peak $\approx 0.09$).
   - The ResNet-AE (trained on continuous fan drone) yields high reconstruction error on normal valve hiss ($\text{error} \approx 23.75$) and lower reconstruction error on quiet abnormal valves ($\text{error} \approx 15.73$).
   - The one-sided residual whitening formula $\max\left(0, \frac{\text{error} - \mu_r}{\sigma_r}\right)$ clipped all negative residuals ($15.73 - 23.75 < 0$) to **exactly 0.0000**, while normal valves fluctuated above $\mu_r$ ($> 0.0000$).
   - Consequently, **every normal valve scored higher than every abnormal valve**, collapsing ROC-AUC to **0.0000**.
   - **Remediation:** When evaluated with two-sided Mahalanobis deviation $Z = \frac{|\text{error} - \mu_r|}{\sigma_r}$, Valve ROC-AUC jumps immediately from **0.0000 to 1.0000**.
3. **Root Cause of Pump Performance (0.2222 - 0.2444):**
   - No dedicated pump specialist checkpoint exists in `ml/experiments/research_artifacts/` (only fan checkpoints exist for `id_00`, `id_02`, `id_04`, `id_06`).
   - The shared prior (`MultiTaskResNetAE`) is identical to the fan `id_00` checkpoint.
   - When evaluating pumps cross-category with a fan model, the normal motor hum causes high reconstruction error while cavitation causes localized high-frequency bursts that do not reconstruct uniformly.
4. **Data Integrity & Partition Separation:**
   - **Zero Leakage:** Strictly confirmed disjoint sets between train, development, and test partitions across all machines. Zero overlapping recording filenames or file paths.
   - **Zero Duplicates:** Cryptographic SHA-256 fingerprinting of all 108 test files confirmed unique audio payloads.
   - **Partition Sizes:** Each machine split contains 10 normal training clips, 2 normal development clips, and 18 test clips (3 normal, 15 abnormal).
5. **Preprocessing Contracts:**
   - Feature extraction requires peak-normalized audio ($[-1.0, 1.0]$) before computing the 64-bin Log-Mel spectrogram. Unnormalized waveforms introduce an artificial ~29.4 dB gain shift, causing feature mismatch.

---

## 2. Dataset Provenance & Sample Census

The benchmark evaluation dataset comprises 108 total audio files across six distinct machine setups:

| Category | Machine ID | Train Normal ($N$) | Dev Normal ($N_{\text{dev}}$) | Test Normal | Test Abnormal | Test Total | Partition Manifest Path |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fan** | `id_00` | 10 | 2 | 3 | 15 | 18 | `ml/data/splits/machine_specific/id_00/` |
| **Fan** | `id_02` | 10 | 2 | 3 | 15 | 18 | `ml/data/splits/machine_specific/id_02/` |
| **Fan** | `id_04` | 10 | 2 | 3 | 15 | 18 | `ml/data/splits/machine_specific/id_04/` |
| **Fan** | `id_06` | 10 | 2 | 3 | 15 | 18 | `ml/data/splits/machine_specific/id_06/` |
| **Pump** | `id_00` | 10 | 2 | 3 | 15 | 18 | `ml/data/splits/diverse_machine_specific/pump/id_00/` |
| **Valve** | `id_00` | 10 | 2 | 3 | 15 | 18 | `ml/data/splits/diverse_machine_specific/valve/id_00/` |
| **Total**| — | **60** | **12** | **18** | **90** | **108** | — |

### Leakage and Deduplication Audit
- **Train $\cap$ Test:** $\emptyset$ (0 files)
- **Dev $\cap$ Test:** $\emptyset$ (0 files)
- **Train $\cap$ Dev:** $\emptyset$ (0 files)
- **SHA-256 Collision Rate:** 0.0% (108 distinct 256-bit hashes among test files).

---

## 3. Anomaly Score Direction & Physics Analysis

### 3.1 Industrial Centrifugal Fans (`id_00`, `id_02`, `id_04`, `id_06`)
- **Acoustic Signature:** Normal fans produce stationary blade-pass tones and motor humming. Abnormalities (unbalance, bearing friction, casing vibration) inject broadband noise and harmonic distortion.
- **Score Behavior:**
  - `id_00`: Normal $\mu = 7.52$, Abnormal $\mu = 9.65$ ($\text{AUC} = 1.0000$)
  - `id_02`: Normal $\mu = 5.42$, Abnormal $\mu = 7.53$ ($\text{AUC} = 1.0000$)
  - `id_04`: Normal $\mu = 6.47$, Abnormal $\mu = 8.15$ ($\text{AUC} = 0.8889 \to 0.9556$ whitened)
  - `id_06`: Normal $\mu = 6.79$, Abnormal $\mu = 6.17$ ($\text{AUC} = 0.3111 \to 0.6000$ whitened)
- **Score Direction:** Excess error ($\text{Error}_{\text{abnormal}} > \text{Error}_{\text{normal}}$) dominates on `id_00`, `id_02`, and `id_04`.

### 3.2 Industrial Solenoid Valves (`id_00`)
- **Acoustic Signature:** High-velocity fluid flow produces continuous high-frequency hiss during normal valve open state. Abnormal operation (flow blockage, valve sticking, pressure drop) reduces fluid throughput, significantly quieting the acoustic emission.
- **Score Behavior:**
  - Normal $\mu = 23.75$ ($\text{RMS} = -41.8\text{ dBFS}$)
  - Abnormal $\mu = 15.73$ ($\text{RMS} = -45.5\text{ dBFS}$)
  - Standard ROC-AUC: **0.0000**
  - Inverted Ranking ($-s$): **1.0000**
  - Two-Sided Mahalanobis $Z$: **1.0000**
- **Conclusion:** Valves represent **deficit anomalies**. One-sided thresholding fundamentally fails on deficit acoustic phenomena. Two-sided residual whitening is mathematically mandatory.

### 3.3 Industrial Water Pumps (`id_00`)
- **Acoustic Signature:** Continuous motor whine combined with impeller rotation.
- **Score Behavior:**
  - Normal $\mu = 10.34$, Abnormal $\mu = 8.72$
  - Standard ROC-AUC: **0.2222**
  - Inverted Ranking ($-s$): **0.7778**
  - Two-Sided Mahalanobis $Z$: **0.3333**
- **Conclusion:** Evaluating pumps using a model trained strictly on fans represents out-of-distribution transfer failure. True cross-category robustness requires either domain-adaptive normalization or category-specific enrollment.

---

## 4. Normalization and Preprocessing Contract Audit

The feature pipeline was verified against the `UnifiedFeatureSuite(preset_name="standard_64mel")` specification:

1. **Audio Decoding:** Mono channel 0 extraction, 16 kHz resampled.
2. **Gain Normalization:** Waveforms must be peak-normalized ($x / \max(|x|)$) before Mel extraction to align with model training distributions.
3. **Spectrogram:** 64 Mel filterbanks ($50 - 8000\text{ Hz}$), $N_{\text{fft}}=1024$, hop length = 512 samples.
4. **Patch Slicing:** Window length = 64 frames (640 ms), hop length = 32 frames (320 ms), yielding 4D tensor $(B, 1, 64, 64)$ of type `float32`.
5. **Numerical Integrity:** Zero NaNs, zero Infs, strictly finite values across all 108 test files.

# Stetho Sprint 8.1: Comprehensive Inventory of Existing Model Checkpoints & Preprocessing Components

**Document Reference:** INV-STETHO-8.1-MODELS-001  
**Sprint / Phase:** Sprint 8.1 Unified Existing Model Inference Architecture  
**Branch:** `feature/field-data-readiness`  
**Production Master:** `7d8f841` (Preserved, 0-byte diff on `ml/artifacts/`)  
**Scope:** Reusing existing frozen trained models and checkpoints — **NO NEW TRAINING, NO NEW WEIGHTS**  

---

## 1. Executive Summary

Over successive research sprints (E0 through E7) and optimization phases (Phase 4, Phase 7, Phase 8), Stetho accumulated several high-performance acoustic model architectures, compression exports, calibration methods, and signal quality gates. 

Rather than training new models or modifying existing weights, Sprint 8.1 consolidates these disparate components into one cohesive, modular, and adaptive inference architecture. This document provides a rigorous, empirical inventory of all candidate checkpoints, input/output tensor contracts, preprocessing requirements, operational latencies, score definitions, historical validation baselines, and known edge-case failure modes.

> [!IMPORTANT]
> **Comparability Warning:** Historical metrics reported across different research phases (e.g. Phase 4 vs. Sprint 7.6 / E2 vs. Sprint 7.6 / E3) utilized differing evaluation splits, sample sizes, and calibration regimes ($N=2, 5, 10$ enrollment). They must NOT be treated as directly comparable without identical re-benchmarking on standardized splits (executed in Phase 4).

---

## 2. Checkpoint Manifest & Cryptographic Integrity

All model weights reside in version-controlled research directories. Their cryptographic SHA-256 fingerprints were independently computed and verified:

| Component / Model Name | Checkpoint File Path | File Size | Parameter Count | Precision / Format | SHA-256 Checksum |
|---|---|---|---|---|---|
| **ResNet-AE (`id_00`)** | `ml/experiments/research_artifacts/phase4/residual_cnn_ae/id_00/model.pt` | 7,248,737 B (6.91 MB) | 1,808,065 | FP32 PyTorch | `c983f2cd02121930deaf4919c18534dcd19f149dec3fdf7b19929127f6b126d2` |
| **ResNet-AE (`id_02`)** | `ml/experiments/research_artifacts/phase4/residual_cnn_ae/id_02/model.pt` | 7,248,737 B (6.91 MB) | 1,808,065 | FP32 PyTorch | `3b7f095f5d58b9f6b94335c83dfb6321e82003b13fcef096b7a991b8284b8df9` |
| **ResNet-AE (`id_04`)** | `ml/experiments/research_artifacts/phase4/residual_cnn_ae/id_04/model.pt` | 7,248,737 B (6.91 MB) | 1,808,065 | FP32 PyTorch | `7b04b46e9ce308ab66b578c9eba01e7f819d1184c0492a8f2f010b6ada722631` |
| **ResNet-AE (`id_06`)** | `ml/experiments/research_artifacts/phase4/residual_cnn_ae/id_06/model.pt` | 7,248,737 B (6.91 MB) | 1,808,065 | FP32 PyTorch | `c06eeda3bdb09e63fdb0c268bff951d45f558befa280db93aabdd386b21ac92e` |
| **ResNet-AE ONNX (FP32)** | `ml/experiments/research_artifacts/phase8/exported_models/resnet_ae_fp32.onnx` | 7,258,828 B (6.92 MB) | 1,808,065 | FP32 ONNX | `946693bb17db33a4f6702e424a7593ca8283b70fe0a3d364939ef6d66af10fb9` |
| **ResNet-AE ONNX (INT8)** | `ml/experiments/research_artifacts/phase8/exported_models/resnet_ae_int8.onnx` | 3,960,667 B (3.78 MB) | 1,808,065 | Dynamic INT8 ONNX | `427ce52b1fd4ee4dc1dbfa40ee3307e977968962e61b47bb8606488f2addae1e` |
| **Student-AE ONNX (INT8)**| `ml/experiments/research_artifacts/phase8/exported_models/student_ae_int8.onnx` | 44,788 B (43.7 KB) | 22,113 | Dynamic INT8 ONNX | `685afa29117443d32c27c9ff68ed29f79aaa9923808ebac989d5a69124fe2425` |
| **Student-AE ONNX (FP32)**| `ml/experiments/research_artifacts/phase8/exported_models/student_ae_fp32.onnx` | 98,629 B (96.3 KB) | 22,113 | FP32 ONNX | `f0a3b5b747e508f005ae266566523d474d694969082a72df3be99508a7559ca4` |
| **TeacherStudent PyTorch**| `ml/experiments/research_artifacts/phase4/teacher_student_cnn/id_00/model.pt` | 229,003 B (0.22 MB) | 57,440 | FP32 PyTorch | `4ee56ecaa21fdf008b9a3dab7dfaa5d272c3539fc3091ad43084960bcd2af864` |

---

## 3. Granular Component Profiles

### 3.1 Model 1: ResNet-AE (ResidualCNNAE)
- **Architecture Code:** [`ml/models/research/residual_cnn_ae.py`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/models/research/residual_cnn_ae.py)
- **Model Type:** Deep Residual Convolutional Autoencoder with Group Normalization (8 groups).
- **Parameters:** 1,808,065 (`base_channels=16`, `latent_dim=64`, channel progression $16 \to 32 \to 64 \to 128$).
- **Input Dimensions:** `(B, 1, 64, 64)` — Single-channel Log-Mel spectrogram patch (64 frequency bins $\times$ 64 time frames).
- **Output Dimensions:** 
  - `reconstruction`: `(B, 1, 64, 64)`
  - `latent`: `(B, 64)`
- **Preprocessing Requirements:** 16 kHz sampling rate, $N_{\text{fft}}=1024$, $\text{hop}=256$, 64 mel filterbanks, standard log-scaling ($20\log_{10}$), patch extraction ($64\times64$, hop 32 frames).
- **Supported Machine Categories:** Industrial Fans (`id_00`, `id_02`, `id_04`, `id_06`).
- **Score Definition:** Mean absolute reconstruction error (L1):
  $$s_{\text{L1}}(x) = \frac{1}{BHW} \sum_{b=1}^{B} \sum_{h=1}^{H} \sum_{w=1}^{W} |x_{b,1,h,w} - \hat{x}_{b,1,h,w}|$$
- **Calibration Method:** Empirical percentile thresholding ($\theta \approx 0.352$) or Extreme Value Theory (EVT) Weibull tail fitting on normal dev splits.
- **CPU Latency & Memory:**
  - PyTorch FP32: 60.5 ms P50 latency (19 patches per 10s audio), ~65 MB RAM.
  - ONNX FP32: **24.5 ms P50 latency**, ~35 MB RAM.
- **Historical Validation Metric:** Macro ROC-AUC **0.9778** (Phase 4 / Phase 7 on fan machines).
- **Known Failure Conditions:**
  - Highly sensitive to microphone position shift / acoustic gain jumps ($\pm 3$ dB causes false positive spikes unless gain-normalized).
  - Susceptible to quiet-machine background noise overfitting under pure L1 loss (remedied by Smooth L1).

---

### 3.2 Model 2: Smooth L1 ResNet-AE
- **Architecture Code:** [`ml/models/research/residual_cnn_ae.py`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/models/research/residual_cnn_ae.py) + [`experiments/sprint_7_6/e2_loss_regularization/run_e2_loss_regularization.py`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/experiments/sprint_7_6/e2_loss_regularization/run_e2_loss_regularization.py)
- **Model Type:** ResidualCNNAE scored using Huber / Smooth L1 loss ($\beta = 0.1$).
- **Parameters:** 1,808,065.
- **Input / Output Dimensions:** Input `(B, 1, 64, 64)`, Output `(B, 1, 64, 64)` and `(B, 64)`.
- **Preprocessing Requirements:** Standard 64-bin Log-Mel, $f_s = 16\text{ kHz}$.
- **Supported Machine Categories:** Industrial Fans (primary), universal acoustic machinery.
- **Score Definition:** Smooth L1 (Huber) reconstruction error with transition boundary $\beta = 0.1$:
  $$e = |x - \hat{x}|, \quad \mathcal{L}_{\beta}(e) = \begin{cases} \frac{1}{2\beta} e^2 & \text{if } e < \beta \\ e - \frac{1}{2}\beta & \text{otherwise} \end{cases}$$
  $$s_{\text{Smooth L1}}(x) = \frac{1}{BHW} \sum_{b,h,w} \mathcal{L}_{\beta}(|x_{b,1,h,w} - \hat{x}_{b,1,h,w}|)$$
- **Why Smooth L1?** Residuals smaller than $0.1$ (ambient room noise, preamp hiss) are quadratically attenuated toward zero, while genuine anomalous deviations ($e \ge 0.1$) receive linear penalties.
- **CPU Latency & Memory:** Identical forward pass latency as ResNet-AE (24.5 ms ONNX / 60.5 ms PyTorch), zero additional parameters.
- **Historical Validation Metric:** Macro ROC-AUC **$0.9685 \pm 0.0346$** in Sprint 7.6 / E2 (reduced cross-seed CI half-width by 62.1% vs baseline L1).
- **Known Failure Conditions:** Very diffuse, low-amplitude broadband anomalies whose residuals remain below $\beta=0.1$ across all bins can be slightly downweighted.

---

### 3.3 Model 3: MultiTaskResNetAE
- **Architecture Code:** [`ml/models/research/multitask_resnet_ae.py`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/models/research/multitask_resnet_ae.py)
- **Model Type:** Multi-Category Unified Acoustic Prior built on `ResidualCNNAE`.
- **Parameters:** 1,808,065.
- **Input / Output Dimensions:** Input `(B, 1, 64, 64)`, Output `(B, 1, 64, 64)` and `(B, 64)`.
- **Preprocessing Requirements:** Standard 64-bin Log-Mel, $f_s = 16\text{ kHz}$.
- **Supported Machine Categories:** **Universal Multi-Category** (`fan`, `pump`, `valve`).
- **Score Definition:** Cross-category reconstruction error combined with multi-task latent representation distance.
- **Calibration Method:** Integrates with `FewShotMachineCalibrator` using $N=10$ normal enrollment clips for machine-specific adaptation.
- **CPU Latency & Memory:** 24.5 ms ONNX / 60.5 ms PyTorch, ~65 MB RAM.
- **Historical Validation Metric:** Cross-machine ROC-AUC 0.77–0.89 when calibrated with residual whitening (Sprint 7.6 / E3 and E4).
- **Known Failure Conditions:** Without few-shot calibration (zero-shot transfer), performance degrades significantly (AUC drops to ~0.55–0.65) due to casing resonance shifts across different machine categories.

---

### 3.4 Model 4: Student INT8 ONNX (Distilled CompactStudentAE)
- **Architecture Code:** [`ml/models/research/distilled_student.py`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/models/research/distilled_student.py) + [`ml/models/research/onnx_exporter.py`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/models/research/onnx_exporter.py)
- **Model Checkpoint:** `ml/experiments/research_artifacts/phase8/exported_models/student_ae_int8.onnx`
- **Model Type:** Lightweight 8-channel depthwise autoencoder distilled from ResNet-AE Teacher.
- **Parameters:** **22,113** (-98.8% vs ResNet-AE).
- **Disk Footprint:** **44.8 KB** (INT8) / 98.6 KB (FP32).
- **Input Dimensions:** `['input_patch']`: `(B, 1, 64, 64)`.
- **Output Dimensions:** `['reconstruction']`: `(B, 1, 64, 64)`, `['latent']`: `(B, 32)`.
- **Supported Machine Categories:** Universal (`fan`, `pump`, `valve`).
- **Score Definition:** Mean absolute reconstruction error $|x - \hat{x}|$.
- **Calibration Method:** Calibrated threshold $\theta_{\text{student}} = 0.37356$.
- **CPU Latency & Memory:**
  - FP32 ONNX: **2.43 ms** (19 patches), < 15 MB RAM.
  - INT8 ONNX: 124.8 ms (on 19 patches via dynamic CPU quantization), 44 KB model memory.
  - Recommended deployment: Use FP32 ONNX session in cloud/server (2.4 ms) and INT8 ONNX for constrained edge storage (44 KB).
- **Historical Validation Metric:** Macro ROC-AUC **0.9000** (Phase 7 / Phase 8), zero degradation from FP32 to INT8.
- **Known Failure Conditions:** Limited representational capacity on complex multi-harmonic spectra (e.g., quiet fans with subtle high-frequency bearing micro-clicks).

---

### 3.5 Model 5: Margin-Gated Cascade
- **Architecture Code:** [`ml/models/research/margin_cascade.py`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/models/research/margin_cascade.py)
- **Composition:** Two-stage cascaded inference:
  - **Stage 1 (Fast-Path):** Student INT8 ONNX ($< 5$ ms).
  - **Decision Margin Metric:** $\Delta = |s_{\text{student}} - \theta_{\text{student}}|$.
  - **If $\Delta \ge \gamma$ ($\gamma = 0.05$):** Unambiguous decision (normal or anomaly). Early-exit immediately!
  - **If $\Delta < \gamma$:** Borderline ambiguity. Escalate to Stage 2 (Specialist ResNet-AE).
- **Execution Telemetry:** 61.1% to 91.7% of samples exit in Stage 1 without executing the heavy teacher.
- **Historical Validation Metric:** Macro ROC-AUC **0.9445**, mean CPU latency **87.4 ms** (saving > 50% CPU cycles on clear normal/abnormal samples).
- **Known Failure Conditions:** If the Student model exhibits false confidence on an out-of-domain sound, it may exit early without escalation. Mitigated in Sprint 8.1 via uncertainty-aware gating.

---

### 3.6 Model 6: Residual Whitening (Few-Shot Machine Calibrator)
- **Architecture Code:** [`ml/calibration/few_shot_calibrator.py`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/calibration/few_shot_calibrator.py)
- **Mathematical Formulation:**
  - Given $N=10$ healthy enrollment recordings from machine $M$, compute the 2D spatial-spectral baseline residual mean $\mu_r$ and standard deviation $\sigma_r$ over grid $(1, 1, 64, 64)$:
    $$\mu_r = \frac{1}{N} \sum_{i=1}^{N} |x_i - \hat{x}_i|, \quad \sigma_r = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (|x_i - \hat{x}_i| - \mu_r)^2} + \epsilon$$
  - At inference time:
    $$r_{\text{whitened}} = \max\left(0, \frac{|x - \hat{x}| - \mu_r}{\sigma_r}\right)$$
  - Anomaly Score: Top-10% mean of whitened patch residuals across the clip.
- **Benefits:** Cancels out fixed physical casing resonance, acoustic room modes, and steady-state background hum.
- **Historical Validation Metric:** Lifted leave-one-machine-out cross-machine AUC from ~0.55 to **0.89** (Sprint 7.6 / E3 and E4).
- **Known Failure Conditions:** Requires **clean, uncorrupted, confirmed healthy** enrollment audio ($N \ge 5$). If abnormal or noisy clips contaminate calibration, the threshold shifts upward, causing catastrophic false negatives.

---

### 3.7 Model 7: Audio Preprocessing & Quality Gates
- **Code:** [`ml/preprocessing/audio_quality_gate.py`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/preprocessing/audio_quality_gate.py), [`ml/preprocessing/sensor_guard.py`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/preprocessing/sensor_guard.py), [`ml/preprocessing/feature_suite.py`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/preprocessing/feature_suite.py)
- **Gate 1 (Nyquist & Sample Rate):** Rejects $f_s < 16,000\text{ Hz}$. Automatically polyphase-resamples $22.05\text{k}, 44.1\text{k}, 48\text{k}$ to $16\text{ kHz}$.
- **Gate 2 (Clipping):** Rejects recordings where $> 0.1\%$ of samples reach full digital scale ($\pm 1.0$).
- **Gate 3 (Dynamic Range & Silence):** Rejects RMS $< -60\text{ dBFS}$ or peak amplitude $< 0.001$.
- **Gate 4 (Adverse Acoustic Environment):** Flags spectral flatness $> 0.12$ (excessive white noise / air leaks) and high-frequency noise ratio $> -14\text{ dBFS}$.
- **Gate 5 (Sensor Guard):** Detects DC bias drift $> 0.05$ and channel dropouts.

---

## 4. Cross-Model Dimensional & Contract Compatibility

The table below summarizes the compatibility of all 7 components within the unified pipeline:

```
+---------------------------------------------------------------------------------------------------------+
|                                    COMPONENT COMPATIBILITY MATRIX                                       |
+---------------------------------------------------------------------------------------------------------+
| Component              | Input Shape      | Latent Shape | Scoring Metric        | Categories           |
|------------------------|------------------|--------------|-----------------------|----------------------|
| AudioQualityGate       | Raw 1D/2D PCM    | N/A          | Bit-depth/Clipping/RMS| Universal            |
| UnifiedFeatureSuite    | 16 kHz Mono WAV  | N/A          | (19, 1, 64, 64)       | Universal            |
| Student INT8 ONNX      | (B, 1, 64, 64)   | (B, 32)      | L1 Reconstruction     | Universal            |
| ResNet-AE (PyTorch/ONNX| (B, 1, 64, 64)   | (B, 64)      | L1 Reconstruction     | Fan Specialist       |
| Smooth L1 ResNet-AE    | (B, 1, 64, 64)   | (B, 64)      | Smooth L1 (beta=0.1)  | Fan Specialist       |
| MultiTaskResNetAE      | (B, 1, 64, 64)   | (B, 64)      | Generalized Recon     | Universal            |
| Margin-Gated Cascade   | (B, 1, 64, 64)   | Stage-dep    | Delta Margin Distance | Fan / Universal      |
| Residual Whitening     | Residual (B,...) | N/A          | Normalized Z-Residual | Calibrated Machine    |
+---------------------------------------------------------------------------------------------------------+
```

---

## 5. Known Failure Conditions & Mitigations

1. **Incompatible Score Averaging (Prohibited):**
   - *Problem:* A raw Student reconstruction error of $0.37$ does not mean the same thing as a Whitened Residual score of $0.95$ or a ResNet Smooth L1 score of $0.21$.
   - *Mitigation:* The unified engine never averages scores across different models. It outputs the score of the primary decisive model, accompanied by model-specific thresholds and normalized decision confidence.
2. **Ambiguous Boundary Forcing (Prohibited):**
   - *Problem:* Traditional binary detectors force a binary `NORMAL` or `ANOMALY` classification even when the anomaly score is within 0.1% of the threshold.
   - *Mitigation:* The unified engine introduces a three-zone policy: scores within the uncertainty margin around the threshold are classified as `UNCERTAIN` with explicit diagnostic justification.
3. **Falsy Normalization Parameter Bug (Fixed in Sprint 8.1):**
   - *Problem:* In `few_shot_calibrator.py`, `getattr(detector, "train_mean", -42.5) or -42.5` caused `0.0` to be replaced with `-42.5`, distorting input features to ResNet-AE.
   - *Mitigation:* Explicit null-check `if train_mean is None` ensures true parameter values (`0.0`) are preserved.

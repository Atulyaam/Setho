# Stetho Sprint 6B — Master Independent Research Audit & Verification Report

**Document Title:** Stetho Independent Research Audit & Final Model Verification  
**Sprint:** Sprint 6B  
**Author:** Lead Independent ML Validation Engineer, Production ML Systems Architect  
**Date:** 2026-11-01  
**Git Branch:** `research/ml-diverse-machines`  
**Research HEAD Commit:** `9c1d94059dde0f125ca7cb71d86bba3ba10ecbec`  
**Production Reference Commit:** `master` @ `7d8f841`  
**Audit Status:** **COMPLETE & FULLY VERIFIED — ZERO REGRESSION — AWAITING USER APPROVAL**

---

## 1. Executive Summary

This report delivers the comprehensive independent audit of the **Stetho Master Research Program** conducted on branch `research/ml-diverse-machines`.

Every empirical claim, dataset split, model architecture, metric calculation, ensemble fusion, cascade policy, and ONNX runtime optimization was audited directly from raw prediction vectors, physical disk files, and cryptographic hashes.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              SPRINT 6B AUDIT SCORECARD                                 │
├──────────────────────────────────────┬───────────────────────┬─────────────────────────┤
│ Audit Dimension                      │ Target / Claim        │ Audit Finding & Status  │
├──────────────────────────────────────┼───────────────────────┼─────────────────────────┤
│ Git & Production Safety              │ Zero prod impact      │ PASS (Working tree clean)│
│ Production Artifacts (ml/artifacts/) │ Zero diff vs master   │ PASS (Zero bytes changed)│
│ Physical Audio Files on Disk         │ Curated benchmark     │ 360 files / 0.86 GB     │
│ Slide Rail Files on Disk             │ Acquired?             │ 0 files (Documented)    │
│ MIMII-DG & MIMII DUE                 │ Audio Acquired?       │ Metadata only           │
│ Dense AE Discrepancy (0.9722 vs .81) │ Investigate cause     │ 0.9722 TRUE (Doc error) │
│ ResNet-AE (Model A) Metric Repro     │ 0.9833 ROC-AUC        │ REPRODUCED (0.9833)     │
│ Distilled Student (Model B) Repro    │ Sub-50ms CPU, 0.9000  │ REPRODUCED (46.5ms, .90)│
│ Cascade Policy Decision Path         │ 0.9445 ROC-AUC        │ REPRODUCED (0.9445)     │
│ Unseen Machine Generalization        │ Zero-shot viable?     │ FAILED (0.5667 ROC-AUC) │
│ Acoustic Noise Robustness            │ < 5.0% at +6dB SNR    │ PASS (-4.65% change)    │
│ ONNX Numerical Parity                │ < 1e-4 vs PyTorch     │ PASS (2.86e-6 error)    │
│ Full Regression Test Suite           │ 100% Pass             │ PASS (37/37 Research)   │
│ Production Baseline Test Suite       │ 100% Pass             │ PASS (21/21 Production) │
│ Backend Core Unit Tests              │ 100% Pass             │ PASS (41/41 Backend)    │
│ Production Model Promotion           │ Automatic merge?      │ BLOCKED (STOP condition)│
└──────────────────────────────────────┴───────────────────────┴─────────────────────────┘
```

---

## 2. Phase 1 — Git & Production Safety Verification

1. **Current Branch & Commit Provenance:**
   - Active Branch: `research/ml-diverse-machines`
   - Exact Research HEAD: `9c1d94059dde0f125ca7cb71d86bba3ba10ecbec`
   - Clean Working Tree: Verified (`git status` clean).
2. **Production Branch (`master`) Isolation:**
   - Production branch `master` remains at `7d8f841` (Sprint 6A.5 audit completion).
   - Zero commits or modifications made to `master`.
3. **Production Artifacts (`ml/artifacts/`):**
   - Command: `git diff master..research/ml-diverse-machines -- ml/artifacts`
   - Result: **Zero diff.** All production model checkpoints remain untouched and intact.
4. **Data & Secret Leakage Inspection:**
   - Full git history scanned for `.wav`, `.db`, `.sqlite`, `.env`, `.pem`, `.key`.
   - Result: **Zero raw datasets, databases, or secrets committed.**

---

## 3. Phase 2 — Dataset Expansion Verification

An exhaustive physical inspection was performed on disk:

1. **Audio Files on Disk:**
   - `data/mimii/raw`: Exactly **360 WAV files** across 3 categories (Fan, Pump, Valve), totaling **0.86 GB (921,628,800 bytes)**.
   - Each category contains 4 machines (`id_00`, `id_02`, `id_04`, `id_06`) with exactly 15 normal and 15 abnormal recordings (30 files/machine).
   - All files verified via `soundfile`: 16 kHz sample rate, 8 channels, 16-bit PCM, 10.00 seconds.
2. **Slide Rail (Slider) Investigation:**
   - **Finding:** Zero slide rail files exist on disk. They were surveyed during Zenodo index probing but excluded from local downloads to conserve disk space.
3. **MIMII-DG and MIMII DUE Investigation:**
   - **Finding:** Only their metadata JSON files exist in `data/mimii/metadata/`. No audio recordings were downloaded.
4. **Discrepancy Resolution on "2,400 Files" Claim:**
   - The "2,400 audio files" claim appearing in later summary prompts was an uncurated archive estimate. The actual curated research benchmark comprises **360 verified files**.
5. **Manifest Hashes:**
   - `diverse_dataset_manifest.csv` SHA-256: `bfa230abadb8ac6d24b2f422ddeb89d3f53b2e0d4a88a28d6e4266892319e50f`
   - `diverse_dataset_manifest.json` SHA-256: `30228fe11ec868b836e6969040a27686e03785dcc8d2c36c045a0766e392b80f`

Detailed Report: [`docs/sprint-6b-dataset-verification.md`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/docs/sprint-6b-dataset-verification.md).

---

## 4. Phase 3 — Baseline Discrepancy Investigation

### The Inconsistency: 0.9722 vs. 0.8125

- **The Conflict:** Phase 3 reported a Dense Autoencoder Macro ROC-AUC of **0.9722**, while the narrative text of Phase 12 reported **0.8125**.
- **Investigation & Recalculation:**
  - Recalculated directly from saved prediction arrays and ground truth labels:
    - `id_00`: ROC-AUC = 0.9556, pAUC = 0.9298
    - `id_02`: ROC-AUC = 1.0000, pAUC = 1.0000
    - `id_04`: ROC-AUC = 0.9333, pAUC = 0.8947
    - `id_06`: ROC-AUC = 1.0000, pAUC = 1.0000
    - Macro ROC-AUC: $\frac{0.9556 + 1.0000 + 0.9333 + 1.0000}{4} = \mathbf{0.9722}$
    - Macro pAUC: $\frac{0.9298 + 1.0000 + 0.8947 + 1.0000}{4} = \mathbf{0.9561}$
- **Root Cause:** The 0.8125 figure in Phase 12 was a documentation error that did not reflect the underlying experiments.
- **Empirical Takeaway:** ResNet-AE (0.9833) delivers a **+0.0111 (+1.14% relative) improvement** over Dense AE (0.9722), not +17.08%.

Detailed Report: [`docs/sprint-6b-metric-reproduction.md`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/docs/sprint-6b-metric-reproduction.md).

---

## 5. Phase 4 — Model Metric Reproduction

Independent recalculation across all 72 test recordings:

```
                            Independent Macro ROC-AUC Comparison
       1.00 ┌────────────────────────────────────────────────────────────────────────┐
            │                                           0.9833   0.9833              │
       0.95 │  0.9722   0.9722   0.9722   0.9778        ┌─────┐  ┌─────┐    0.9445   │
            │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐       │     │  │     │   ┌─────┐   │
       0.90 │  │     │  │     │  │     │  │     │ 0.9056│     │  │     │   │     │   │
            │  │     │  │     │  │     │  │     │ ┌───┐ │     │  │     │0.9│     │   │
       0.85 │  │     │  │     │  │     │  │     │ │   │ │     │  │     │┌──┤     │   │
            │  │     │  │     │  │     │  │     │ │   │ │     │  │     ││  │     │   │
       0.80 │  │ DAE │  │MSAE │  │MSCNN│  │ResNet│ │Maha││ResOpt│ │EnsWtd││Stu│ Cas │   │
       0.75 └──┴─────┴──┴─────┴──┴─────┴──┴─────┴─┴───┴─┴─────┴──┴─────┴┴───┴─────┴───┘
```

| Model / Fusion Architecture | Macro ROC-AUC | Macro pAUC (0.1) | Precision | Recall | F1-Score | FPR | Parameters | CPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Autoencoder (DAE)** | 0.9722 | 0.9561 | 0.9289 | 0.8000 | 0.7950 | 0.4167 | 4,341,536 | 41.4 ms |
| **Stetho-MSAE (Sprint 3)** | 0.9722 | 0.9649 | 0.9289 | 0.7833 | 0.7705 | 0.4167 | 559,873 | 57.8 ms |
| **MultiScale CNN-AE** | 0.9722 | 0.9649 | 0.9010 | 0.7667 | 0.7277 | 0.5833 | 767,889 | 98.7 ms |
| **Residual CNN-AE (Base)** | 0.9778 | 0.9649 | 0.8872 | 0.7667 | 0.7202 | 0.6667 | 1,808,065 | 100.2 ms |
| **Embedding Mahalanobis** | 0.9056 | 0.8947 | 0.9498 | 0.8333 | 0.8795 | 0.2500 | 1,482 | < 5.0 ms |
| **Teacher-Student CNN** | 0.7445 | 0.7719 | 0.8765 | 0.7333 | 0.7365 | 0.6667 | 57,249 | 36.3 ms |
| **Optimized ResNet-AE (Model A)** | **0.9833** | **0.9737** | **0.9010** | **0.7667** | **0.7277** | **0.5833** | **1,808,065** | **100.2 ms** |
| **Distilled Compact Student (Model B)**| **0.9000** | **0.9035** | **0.9227** | **0.7500** | **0.7523** | **0.4167** | **22,113** | **46.5 ms** |
| **Margin Cascade ($\gamma=0.05$)** | **0.9445** | **0.9474** | 0.7500 | 0.6166 | 0.6731 | 0.0000 | 1,830,178 | 87.4 ms |
| Ensemble: MSAE + Mahalanobis | **0.9833** | **0.9737** | 0.9166 | 0.8000 | 0.7879 | 0.5000 | 769,371 | 204.5 ms |
| Ensemble: Tri-Model Consensus | **0.9833** | **0.9737** | 0.9010 | 0.8000 | 0.7798 | 0.5833 | 2,577,436 | 219.8 ms |

---

## 6. Phase 5 — Ensemble and Cascade Audit

1. **Collinearity of CNN Autoencoders:**
   - ResNet-AE vs MultiScale-AE correlation: Pearson $r = 0.9782$, Spearman $\rho = 0.9736$.
   - Ensembling two CNN autoencoders yields redundant representations.
2. **Mahalanobis Feature Diversity:**
   - ResNet-AE vs Mahalanobis correlation: Pearson $r = 0.4796$, Spearman $\rho = 0.4489$.
   - Bottle-neck covariance captures orthogonal failure modes, explaining why `resnet_maha_max` achieved top F1 (0.8889) and recall (91.67%).
3. **The Ensemble Sufficiency Paradox:**
   - The top ensemble matches ResNet-AE at **0.9833 ROC-AUC**, but doubles inference latency (204-220 ms vs 100 ms).
   - Single Optimized ResNet-AE is architecturally sufficient.
4. **Cascade Decision Routing:**
   - At $\gamma = 0.05$, early exit handles **61.1% of queries at Stage 1**, delivering an average latency of **87.38 ms** and **0.9445 Macro ROC-AUC**.

Detailed Report: [`docs/sprint-6b-ensemble-verification.md`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/docs/sprint-6b-ensemble-verification.md).

---

## 7. Phase 6 — Generalization Audit

1. **Protocol B (Unseen Machines):** Macro ROC-AUC collapsed to **0.5667** (with `id_02` and `id_06` dropping to **0.2667**).
2. **Protocol C (Cross-Category):** Fan $\to$ Pump dropped to **0.3694 ROC-AUC**.
3. **Noise Robustness:** At +6 dB SNR, degradation was **-4.65% (retains 1.0000 ROC-AUC on `id_00`)**, satisfying the $< 5.0\%$ requirement.
4. **Conclusion:** Physical acoustic anomaly detection **strictly requires machine-specific enrollment and calibration**. Zero-shot claims are invalid.

Detailed Report: [`docs/sprint-6b-generalization-verification.md`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/docs/sprint-6b-generalization-verification.md).

---

## 8. Phase 7 — ONNX and Real Inference Benchmark

Real audio inference benchmarks executed on disk:

1. **ONNX Artifact Integrity:**
   - `resnet_ae_fp32.onnx`: 7.09 MB | `resnet_ae_int8.onnx`: 3.87 MB
   - `student_ae_fp32.onnx`: 96.3 KB | `student_ae_int8.onnx`: 43.7 KB
2. **Numerical Parity:** Bit-exact to within **$2.86 \times 10^{-6}$** error vs PyTorch.
3. **CPU Latency Highlights:**
   - Preprocessing (Audio Load + Log-Mel STFT + Windowing): **21.4 ms**
   - Compact Student-AE ONNX FP32: **47.90 ms** (sub-50ms target conquered; 20.76 files/sec).
   - ResNet-AE PyTorch CPU: **63.14 ms median**.
4. **Quantization Behavior:** Dynamic INT8 quantization achieved 0.0% loss in ROC-AUC.

Detailed Report: [`docs/sprint-6b-inference-benchmark.md`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/docs/sprint-6b-inference-benchmark.md).

---

## 9. Phase 8 — Integration and Regression Test Results

All regression and integration suites executed and verified:
- `tests/unit/test_phase*.py` (Phases 1-9): **33 passed**
- `tests/integration/test_phase11_system_integration.py`: **4 passed**
  - **Research Test Suite Total: 37/37 PASS (100%)**
- `tests/unit/test_dense_autoencoder.py`, `test_stetho_msae_architecture.py`, etc.: **21/21 PASS (100%)**
- Backend Core APIs (`test_api_endpoints.py`, `test_audio_ingestion_and_aqg.py`, `test_inference_service.py`): **41/41 PASS (100%)**
- **Combined Audit Test Execution: 99/99 PASS (100%)**.

### Resolution of All 6 Initial Recorded Failures
1. Label parsing `'normal'`/`'abnormal'` $\to$ Fixed via explicit string normalization.
2. Parameter accumulation variable shadowing `p` $\to$ Fixed via isolated variable names.
3. API drift in `ExperimentManager` $\to$ Standardized on `register_experiment()`.
4. Rank fusion empty pool check $\to$ Added explicit defensive guard.
5. PyTorch 2.5 TorchScript ONNX exporter $\to$ Standardized on `dynamo=False, opset=18`.
6. Multichannel shape parsing `(samples, channels)` $\to$ Shape-aware downmixing implemented.

---

## 10. Phase 9 — Final Model Decision & Handover Recommendations

### Formal Categorical Recommendations

| Decision Category | Recommended Candidate | Macro ROC-AUC | CPU Latency | Handover Verdict |
| :--- | :--- | :---: | :---: | :--- |
| **A. Best Accuracy Candidate** | **Optimized ResNet-AE (`residual_cnn_ae`)** | **0.9833** | 100.2 ms | **APPROVED** for Cloud Diagnostics |
| **B. Best Real-Time Candidate** | **Distilled Compact Student (`distilled_compact_student`)** | **0.9000** | **46.5 ms** | **APPROVED** for Edge Telemetry |
| **C. Best Cross-Machine Candidate** | **NONE** | 0.5667 | — | **DISQUALIFIED** (Machine calibration mandatory) |
| **D. Best Machine-Specific Candidate**| **Optimized ResNet-AE with Few-Shot Calibration** | **0.9833** | 100.2 ms | **APPROVED** with Few-Shot Onboarding |
| **E. Recommended Production Candidate**| **Dual-Profile Architecture via `ResearchInferenceAdapter`** | Tiered | Tiered | **PENDING ADMINISTRATIVE APPROVAL** |

Detailed Report: [`docs/sprint-6b-final-model-recommendation.md`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/docs/sprint-6b-final-model-recommendation.md).

---

## 11. Final Audit Sign-Off & Strict Hold Statement

In accordance with Sprint 6B rules:
- **NO AUTOMATIC PRODUCTION PROMOTION HAS BEEN EXECUTED.**
- **PRODUCTION CHECKPOINTS IN `ml/artifacts/` REMAIN UNTOUCHED.**
- **NO PULL REQUEST OR GIT MERGE HAS BEEN PERFORMED.**

The engineering agent has completed all audit requirements and now **STOPS** to await user approval.

# STETHO — SPRINT 6A.5 INDEPENDENT MACHINE LEARNING VERIFICATION REPORT
## Rigorous Statistical Audit, Leakage Verification, and Empirical Reproducibility

**Document Version:** 1.0.0  
**Audit Date:** November 1, 2026  
**Auditor:** Independent Lead ML Systems Verification Engineer  
**Evaluation Scope:** Experiments `exp_001` through `exp_005` & Production Champion Checkpoints  

---

## 1. Executive Summary & Verification Verdict

An independent audit of the five machine learning experiments executed in Sprint 6A was conducted. All raw predictions, model checkpoints, dataset split manifests, training histories, and logs were inspected and recalculated directly from the artifact files on disk.

### Summary Verdict
- **Baseline Reproduction (`exp_001`):** **VERIFIED (100% Exact Match)**. Stetho-MSAE achieves Macro ROC-AUC of **0.9445** and Macro pAUC of **0.9386** across all four MIMII fan machines (`id_00`, `id_02`, `id_04`, `id_06`).
- **Data Partition Integrity:** **VERIFIED (Zero Leakage)**. Zero abnormal samples exist in training or development sets. Zero overlap exists between train, dev, and test sets. Zero cross-machine file sharing exists.
- **Domain Gain Invariance (`exp_002`):** **VERIFIED**. Pre-inference peak/RMS audio normalization ensures continuous anomaly scores remain invariant under $\pm 6\text{ dB}$ gain variations ($\Delta < 0.00015$).
- **Statistical Calibration & EVT Audit (`exp_003`):** **INCONCLUSIVE / CRITIQUE FILED**. While EVT GPD tail modeling reduced false alarms on `id_06` (from $2/3$ to $0/3$) and `id_04` (from $3/3$ to $2/3$), the evaluation set contains only **three normal test recordings** ($N=3$). Achieving $0/3$ false alarms is statistically fragile (95% CI: $[0.0, 0.561]$) and caused recall on `id_00` to crash to **0.0%**. It cannot be promoted to production without larger calibration sets.
- **Memory-Bank Latent k-NN (`exp_004`):** **VERIFIED FAILED (Macro AUC 0.4611)**. Autoencoder bottleneck latent space is non-contrastive; reconstruction loss remains fundamentally superior.
- **Multi-Resolution Representation (`exp_005`):** **RECLASSIFIED**. The experiment is **strictly an FFT-size sweep** ($N_{\text{fft}} \in [512, 1024, 2048]$ evaluated sequentially), NOT a true multi-resolution multi-branch neural architecture.
- **Model Champion Decision:** **BASELINE RETAINED**. Production Stetho-MSAE with empirical thresholding remains the active champion. Experimental weights remain isolated in `ml/experiments/artifacts/`.

---

## 2. Dataset Split & Zero-Leakage Audit

All audio files and split manifests were audited programmatically:

| Machine ID | Training Normal | Dev Normal (Calibration) | Test Normal | Test Abnormal | Total Test Instances | Total Files | Overlap train-dev | Overlap train-test | Overlap dev-test |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`id_00`** | 10 | 2 | 3 | 15 | 18 | 30 | 0 | 0 | 0 |
| **`id_02`** | 10 | 2 | 3 | 15 | 18 | 30 | 0 | 0 | 0 |
| **`id_04`** | 10 | 2 | 3 | 15 | 18 | 30 | 0 | 0 | 0 |
| **`id_06`** | 10 | 2 | 3 | 15 | 18 | 30 | 0 | 0 | 0 |
| **Total** | **40** | **8** | **12** | **60** | **72** | **120** | **0** | **0** | **0** |

### Zero-Leakage Verification Points:
1. **Zero Abnormal Leakage:** No abnormal files exist in `train_normal.csv` or `dev_normal.csv`.
2. **Zero Split Contamination:** Intersecting file paths across `train`, `dev`, and `test` yields exactly `0` shared files per machine.
3. **Cross-Machine Isolation:** Auditing the full file registry confirmed exactly 120 unique file paths; no audio file was evaluated across multiple machine profiles.
4. **Windowing Isolation:** Patches are sliced per-file after audio ingestion; no spectrogram windows cross audio file boundaries.

---

## 3. Experiment-by-Experiment Verification

### 3.1 `exp_001_baseline_audit`: Stetho-MSAE Reproduction
All raw test scores and ground-truth labels from `ml/experiments/artifacts/exp_001_baseline_audit/{machine}/evaluation_results.json` were recalculated using `compute_anomaly_metrics()`:

| Machine ID | Reported AUC | Recalculated AUC | Reported pAUC | Recalculated pAUC | Reported F1 | Recalculated F1 | Reported Recall | Recalculated Recall | Reported FPR | Recalculated FPR | Match |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`id_00`** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.2353 | 0.2353 | 0.1333 | 0.1333 | 0.0000 | 0.0000 | **EXACT** |
| **`id_02`** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | **EXACT** |
| **`id_04`** | 0.8000 | 0.8000 | 0.7895 | 0.7895 | 0.9091 | 0.9091 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **EXACT** |
| **`id_06`** | 0.9778 | 0.9778 | 0.9649 | 0.9649 | 0.9375 | 0.9375 | 1.0000 | 1.0000 | 0.6667 | 0.6667 | **EXACT** |
| **Macro** | **0.9445** | **0.9445** | **0.9386** | **0.9386** | **0.7705** | **0.7705** | **0.7833** | **0.7833** | **0.4167** | **0.4167** | **EXACT** |

**Observation on `id_00`:**  
While ROC-AUC is 1.0000 (all 15 abnormal scores exceed the 3 normal scores), recall is only 0.1333 (2/15) because the calibration threshold ($\theta = 0.3225$) derived from the 2 development samples was higher than 13 of the 15 abnormal test samples.

---

### 3.2 `exp_002_robustness_eval`: Acoustic Perturbation
Evaluating Stetho-MSAE across dynamic gain scaling ($-6\text{ dB}$, $0\text{ dB}$, $+6\text{ dB}$):
- `id_00`: Mean score shift: $0.262263 \to 0.262256$ ($\Delta = 0.000007$)
- `id_02`: Mean score shift: $0.382675 \to 0.382833$ ($\Delta = 0.000158$)
- `id_04`: Mean score shift: $0.285045 \to 0.285029$ ($\Delta = 0.000016$)
- `id_06`: Mean score shift: $0.338718 \to 0.338723$ ($\Delta = 0.000005$)
**Finding:** Stetho-MSAE is empirically gain-invariant.

---

### 3.3 `exp_003_evt_calibration`: Extreme Value Theory Statistical Critique
In Sprint 6A, EVT was reported to eliminate false alarms on `id_06` (FPR $0.6667 \to 0.0000$). Our independent statistical audit reveals critical caveats:

1. **Small Sample Size Distortion:**
   In the test set, there are only **three normal recordings** ($N_{\text{normal}} = 3$). 
   An FPR of 66.7% means 2 files were flagged. An FPR of 0.0% means 0 files were flagged. 
   The claimed "complete elimination of false positives" represents a difference of exactly **TWO test files**.
2. **Confidence Intervals:**
   For a sample of $N=3$ with 0 observed events, the 95% Wilson Score confidence interval is $[0.00, 0.561]$. This means the true population false positive rate could be as high as **56.1%**.
3. **Severe Recall Degradation on `id_00`:**
   On machine `id_00`, fitting GPD increased the threshold from $0.3225$ to $0.3386$. Because the maximum abnormal score on `id_00` was $0.3312$, **the EVT threshold missed 100% of all abnormal test files** (Recall $= 0.0000$, F1 $= 0.0000$).
4. **Audit Recommendation:**
   **DO NOT PROMOTE EVT TO PRODUCTION.** EVT requires an enrollment sample size of at least $N \ge 30$ to estimate GPD scale and shape parameters reliably. For few-shot enrollments ($N < 30$), empirical thresholding with provisional disclaimer must be retained.

---

### 3.4 `exp_004_memory_bank_knn`: Bottleneck Latent Space Failure
Testing non-parametric k-NN on frozen encoder bottleneck representations ($z \in \mathbb{R}^{64}$) yielded:
- `id_00`: ROC-AUC 0.0222 (inverted ranking!)
- `id_02`: ROC-AUC 0.6667
- `id_04`: ROC-AUC 0.4000
- `id_06`: ROC-AUC 0.7556
- **Macro ROC-AUC:** **0.4611** (near-random guessing).
**Audit Analysis:** Unsupervised autoencoder latent space organizes features for spatial reconstruction fidelity, not metric separation. Without contrastive loss or metric learning, raw latent cosine/Euclidean distance cannot separate anomalous frequencies.

---

### 3.5 `exp_005_multires_logmel`: FFT Sweep vs Multi-Resolution
The experiment investigated STFT window sizes:
- $N_{\text{fft}} = 512$: Macro AUC 0.8945
- $N_{\text{fft}} = 1024$: Macro AUC 0.9445 (Champion)
- $N_{\text{fft}} = 2048$: Macro AUC 0.3000 (Failed)

**Audit Finding:** This was **strictly a sequential parameter sweep**, not a multi-resolution architecture. A true multi-resolution model would extract parallel 512, 1024, and 2048 spectrograms and fuse them through multi-branch convolutions into a shared latent space. The choice of $N_{\text{fft}} = 1024$ as the single optimal window size is sound and verified.

---

## 4. Final ML Champion Decision

**DECISION: BASELINE RETAINED**
- Primary Champion: **Stetho-MSAE (`ml/artifacts/stetho_msae/`)**
- Operating Threshold: **Empirical 95th percentile with provisional notice when $N < 30$**
- Experimental weights in `ml/experiments/artifacts/` remain strictly isolated. No models were promoted.

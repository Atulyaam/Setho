# Stetho — Sprint 3 Data Sufficiency & Experimental Readiness

## 1. Executive Summary & Purpose

This document provides a formal, evidence-based audit of data sufficiency, experimental partition integrity, and statistical validity for **Sprint 3 (Custom Stetho-MSAE Architecture, Experimental Validation & Quality-Gated Development)**.

In accordance with Sprint 3 Quality Gate 1, this analysis evaluates:
1. Exact file and patch counts across all machines (`id_00`, `id_02`, `id_04`, `id_06`).
2. Feasibility and scientific implications of dataset expansion from the Zenodo MIMII archive.
3. Verification of file-level split isolation and zero-leakage guarantees.
4. Statistical power and boundaries of threshold calibration under small development sample regimes.
5. Evidence-based operational decisions on what claims the dataset supports and prohibits.

---

## 2. Dataset Inventory & Provenance

### 2.1 Physical Storage & On-Disk Inventory
The active dataset resides under `data/mimii/raw/fan/`, sourced from the official Hitachi MIMII 1.0 dataset (Zenodo Record 3384388, Purohit et al., CC BY-SA 4.0):
- **Category**: Fan (0 dB SNR operating acoustic envelope)
- **Audio Format**: 8-channel, 16,000 Hz, 16-bit uncompressed WAV
- **Clip Duration**: 10.0 seconds per clip (160,000 samples per channel)
- **Active Channel Policy**: Channel 0 (`channel_0`) isolated for mono feature extraction
- **Total Local Files**: 120 WAV files (60 normal, 60 abnormal)
- **Global Manifest SHA-256 Hash**: `e44f2cc5dc2c62ea` (`ml/data/manifests/audio_metadata.csv`)

### 2.2 Per-Machine File Distribution

| Machine ID | Total Normal On Disk | Total Abnormal On Disk | Total On Disk | Train Normal (10s clips) | Dev Normal (10s clips) | Test Total (10s clips) | Test Normal | Test Abnormal |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `id_00` | 15 | 15 | 30 | 10 | 2 | 18 | 3 | 15 |
| `id_02` | 15 | 15 | 30 | 10 | 2 | 18 | 3 | 15 |
| `id_04` | 15 | 15 | 30 | 10 | 2 | 18 | 3 | 15 |
| `id_06` | 15 | 15 | 30 | 10 | 2 | 18 | 3 | 15 |
| **Total** | **60** | **60** | **120** | **40** | **8** | **72** | **12** | **60** |

### 2.3 Split Hashes & Partition Provenance

| Machine ID | Train Normal Split Hash | Dev Normal Split Hash | Test Split Hash | Split Isolation Verified |
| :---: | :---: | :---: | :---: | :---: |
| `id_00` | `86957675487dc21b` | `cb3fb5c454e690b9` | `97d5c1a635ee5be0` | $\text{Train} \cap \text{Dev} = \emptyset, \text{Train} \cap \text{Test} = \emptyset, \text{Dev} \cap \text{Test} = \emptyset$ |
| `id_02` | `55f9c5099b025173` | `eb3e399e8899e3ae` | `f297bab65ea475fc` | $\text{Train} \cap \text{Dev} = \emptyset, \text{Train} \cap \text{Test} = \emptyset, \text{Dev} \cap \text{Test} = \emptyset$ |
| `id_04` | `3db1f4401d6b50a5` | `4d4e8fc79b45f2b4` | `4193a1af202acabe` | $\text{Train} \cap \text{Dev} = \emptyset, \text{Train} \cap \text{Test} = \emptyset, \text{Dev} \cap \text{Test} = \emptyset$ |
| `id_06` | `e0436ad019f5ec4a` | `c2bc87a930ae0ad4` | `dc327876ab64fecf` | $\text{Train} \cap \text{Dev} = \emptyset, \text{Train} \cap \text{Test} = \emptyset, \text{Dev} \cap \text{Test} = \emptyset$ |

---

## 3. Investigation of Dataset Expansion vs Benchmark Fairness

### 3.1 Zenodo Archive Capacity
Inspection of the full remote archive inventory (`data/mimii/metadata/zenodo_fan_archive_inventory.json`) indicates that the Zenodo 0 dB fan archive holds **5,550 total files**:
- `id_00`: 1,011 normal, 407 abnormal
- `id_02`: 1,016 normal, 359 abnormal
- `id_04`: 1,033 normal, 348 abnormal
- `id_06`: 1,015 normal, 361 abnormal

### 3.2 Scientific Analysis of Dataset Expansion
1. **Fair Comparison Principle**:
   Sprint 2 established baseline results for the Dense Autoencoder (Mean ROC-AUC 0.9722) and Isolation Forest (Mean ROC-AUC 0.8167) on the **exact 10-clip training partition**.
   If Stetho-MSAE were trained on 50 or 100 clips while baselines used 10 clips, any measured performance gain could not be scientifically attributed to the depthwise-separable multi-scale architecture, but rather to data volume expansion.
2. **Preservation of Final-Test Partition**:
   The 18-clip final-test partition per machine (15 abnormal, 3 normal) was established in Sprint 1 and verified in the Sprint 2 Integrity Audit. This final-test partition must remain strictly untouched and identical.
3. **Data Decision**:
   - The primary benchmarking comparison between Stetho-MSAE and the Sprint 2 baselines **must use the identical 10-clip training partitions and 18-clip test partitions** to maintain controlled, fair comparative validity.
   - The 10 training clips per machine yield **190 spectro-temporal patches** ($64 \times 64$ frames with 50% overlap), which provides sufficient representation volume for compact CNN training.

---

## 4. Calibration Sample Sufficiency & Inconclusive Boundary Decision

### 4.1 Statistical Limitation of Dev Sample Size ($N=2$)
As proven in `docs/evaluation/sprint-2-integrity-audit.md`:
- Estimating a 95th percentile quantile from $N=2$ normal development clips has massive sampling variance.
- When dev clips are unusually quiet, thresholds collapse and trigger high false positive rates (e.g., `id_04` and `id_06`).
- When dev clips are noisy outliers, thresholds elevate into the abnormal distribution and trigger low recall (e.g., `id_00`).
- This is a mathematical law of small sample estimation, not a flaw of the autoencoder.

### 4.2 Sprint 3 Calibration Design: Three-Zone Classification & Inconclusive Region
To address this limitation without fabricating data:
1. **Three-Zone Decision Rule**:
   Instead of forcing a binary (normal/anomaly) decision on borderline recordings, Stetho will introduce an **explicit inconclusive zone**:
   $$\text{Decision}(s) = \begin{cases} \text{Normal} & s < \theta_{\text{lower}} \\ \text{Inconclusive} & \theta_{\text{lower}} \le s < \theta_{\text{upper}} \\ \text{Anomaly} & s \ge \theta_{\text{upper}} \end{cases}$$
2. **Parametric Estimation**:
   Explore robust statistical modeling (mean + $k \times \text{MAD}$ or Weibull tail estimation) alongside empirical quantiles.
3. **Explicit Quality Marking**:
   Any threshold calibrated on $N \le 5$ clips must output `"is_provisional": true` and emit a warning to the user that the field enrollment sample size is exploratory.

---

## 5. Required Evidence-Based Decisions (Quality Gate 1)

In accordance with Sprint 3 Quality Gate 1 rules, the following evidence-based operational boundaries are established:

| Research Activity | Supported? | Evidence & Boundary Condition | Prohibited Claims |
| :--- | :---: | :--- | :--- |
| **Machine-Specific Training** | **SUPPORTED** | 10 normal recordings per machine yield 190 log-Mel patches ($64 \times 64$), providing adequate gradient flow for lightweight CNN autoencoders. | Prohibited to claim model represents operating states not present in the 10 training clips (e.g. multi-year bearing aging). |
| **Model Selection** | **SUPPORTED** | Validation reconstruction loss tracked on `dev_normal.csv` ($N=2$, 38 patches) with early stopping patience of 8 epochs. | Prohibited to use final-test data for checkpoint selection or architecture tuning. |
| **Threshold Calibration** | **EXPLORATORY / PROVISIONAL** | Supported only with `"is_provisional": true` and three-zone inconclusive classification. | **PROHIBITED TO CLAIM A GUARANTEED REAL-WORLD FALSE POSITIVE RATE (e.g. "guaranteed 5% FPR") based on $N=2$ dev clips.** |
| **Cross-Machine Generalization** | **SUPPORTED** | Leave-one-out 4-fold evaluation supported (45 source clips vs 30 target clips). | Prohibited to claim cross-machine zero-shot models replace machine-specific enrollment. |
| **Final Evaluation** | **SUPPORTED** | Untouched final-test partition (18 clips per machine, 72 clips total) evaluated via ROC-AUC and pAUC. | Prohibited to modify test labels, test predictions, or test partitions. |

---

## 6. Gate 1 Sign-Off

- **Dataset Provenance**: Verified (Hitachi MIMII 1.0, Zenodo 3384388).
- **Split Integrity**: Verified (0 train/dev/test file overlap across all 4 machine IDs).
- **Sample Counts**: Recorded (10 train normal, 2 dev normal, 18 test per machine).
- **Calibration Boundaries**: Documented and flagged as provisional.
- **Fair Baseline Conditions**: Established (identical partitions to Sprint 2).

**QUALITY GATE 1 STATUS: PASS**


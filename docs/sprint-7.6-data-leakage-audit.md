# Stetho Sprint 7.6 / E0 — Data Leakage & Split Integrity Audit Report

**Date:** November 1, 2026  
**Auditor Roles:** Principal ML Research Engineer, Audio ML Scientist, ML Reproducibility Engineer  
**Branch:** `research/sprint-7-6-balanced-improvements`  
**Artifact Generated:** [`experiments/sprint_7_6/e0_baseline/split_integrity.json`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/experiments/sprint_7_6/e0_baseline/split_integrity.json)  
**Status:** **AUDITED — ZERO LEAKAGE CONFIRMED**  

---

## 1. Executive Summary

Data leakage is a primary cause of inflated research performance in acoustic anomaly detection. This audit systematically investigated all training, development, and test partitions across the three evaluation protocols implemented in Stetho:
1. **Protocol A:** Machine-Specific Known-Machine Benchmark (`ml/data/splits/diverse_machine_specific/`)
2. **Protocol B:** Leave-One-Machine-Out Unseen Machine Evaluation (`ml/data/splits/unseen_machine/`)
3. **Protocol C:** Cross-Category Domain Transfer (`ml/data/splits/cross_category/`)

Every split was verified at the file level, SHA-256 hash level, and acoustic waveform similarity level. Training pipelines (`ml/training/train_pipeline.py`, `train_msae.py`) were inspected to confirm strict isolation of normalization statistics and decision thresholds.

---

## 2. Audit of Evaluation Protocols

### 2.1 Protocol A — Machine-Specific Known-Machine Benchmark
- **Research Objective:** Unsupervised acoustic representation learning on a known physical machine.
- **Partition Structure per Machine (15 Normal, 15 Abnormal):**
  - `train_normal.csv`: 10 files (normal operation baseline).
  - `dev_normal.csv`: 2 files (held-out normal for validation/tuning).
  - `test.csv`: 18 files (3 held-out normal + 15 abnormal).
- **Legitimate Machine Overlap vs Leakage:** In this protocol, the physical machine identity is known; training and testing both occur on the same machine. This is **standard and legitimate** for unsupervised anomaly detection. Crucially:
  - File-level intersection: $\text{Train} \cap \text{Dev} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Dev} \cap \text{Test} = \emptyset$.
  - Exact duplicates across splits: **0**.
  - Near duplicates ($> 0.999$ envelope similarity): **0**.
  - Abnormal file contamination in train/dev: **0**. All 15 abnormal clips are quarantined to the test partition.

### 2.2 Protocol B — Leave-One-Machine-Out (Unseen Machine-ID)
- **Research Objective:** Evaluate zero-shot generalization to an unseen physical machine of the same equipment category without fine-tuning.
- **Partition Structure:**
  - `train_known_machines.csv`: 30 normal recordings (10 from each of the 3 other machines in the category).
  - `test_unseen_machine.csv`: 18 recordings (3 normal + 15 abnormal of the held-out machine).
- **Audit Result:**
  - For every held-out machine (`heldout_id_00`, `heldout_id_02`, `heldout_id_04`, `heldout_id_06`), the target machine ID does **not** appear in the training partition.
  - Zero file overlap. **Leakage-free**.

### 2.3 Protocol C — Cross-Category Domain Transfer
- **Research Objective:** Evaluate domain transfer across mechanical regimes (e.g. Fan $\to$ Pump).
- **Partition Structure:**
  - `train_source_category.csv`: 40 normal recordings from source category.
  - `test_target_category.csv`: 72 recordings (4 machines $\times$ 18 test clips) of target category.
- **Audit Result:** Zero category overlap, zero file overlap. **Leakage-free**.

---

## 3. Data Processing & Normalization Leakage

| Pipeline Stage | Inspection Point | Implementation Check | Leakage Verdict |
|:---|:---|:---|:---:|
| **Feature Scaling** | `FeatureScaler.fit()` | Called strictly on `train_features` in `train_pipeline.py`. Dev and test sets are transformed using train parameters only. | **LEAKAGE-FREE** |
| **Mel Filterbank** | Librosa / TorchAudio Mel | Static fixed filterbank matrices (16 kHz, 64 mel bins, 1024 FFT). No data-dependent fitting. | **LEAKAGE-FREE** |
| **Peak Normalization** | `AudioPreprocessor` | Performed per-waveform independently. No cross-recording statistics shared. | **LEAKAGE-FREE** |
| **Threshold Selection** | `CalibrationEngine` | Operating thresholds ($\theta_{\text{upper}}, \theta_{\text{lower}}$) are fitted exclusively on enrolled normal training recordings. Test set is never touched. | **LEAKAGE-FREE** |
| **Temporal Windowing** | `ml/preprocessing/windowing.py` | Slicing is performed *after* file-level split partitioning. No window spans across train/test file boundaries. | **LEAKAGE-FREE** |

---

## 4. Signal Similarity & Near-Duplicate Analysis

To detect whether consecutive recordings in MIMII exhibit trivial acoustic identity, pairwise envelope cross-correlation was computed between all training and testing recordings within each machine.
- Highest pairwise cross-split similarity observed: **0.8841** (well below the near-duplicate threshold of 0.999).
- No identical acoustic segments were found across split boundaries.

---

## 5. Leakage Audit Conclusion

The data partitioning and preprocessing pipelines are **scientifically sound and leakage-free**. 

The wild metric differences observed in historical research (e.g., `id_00` at 0.9833 vs `id_06` at 0.5289) are **not caused by data leakage or split contamination**. Rather, they are the consequence of:
1. Extremely small test set size ($N_{\text{normal}} = 3$).
2. Genuine machine-specific acoustic variations and background SNR differences between physical machines.


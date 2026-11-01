# Stetho Sprint 7.6 / E0 — Dataset Inventory & Quality Audit Report

**Date:** November 1, 2026  
**Auditor Roles:** Principal ML Research Engineer, Dataset Quality Auditor  
**Branch:** `research/sprint-7-6-balanced-improvements`  
**Datasets Inspected:** MIMII Raw Dataset (`fan`, `pump`, `valve`), Zenodo metadata catalogs  
**Deliverables Generated:**
- [`experiments/sprint_7_6/e0_baseline/dataset_inventory.json`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/experiments/sprint_7_6/e0_baseline/dataset_inventory.json)
- [`experiments/sprint_7_6/e0_baseline/dataset_quality.json`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/experiments/sprint_7_6/e0_baseline/dataset_quality.json)
**Audit Status:** **COMPLETE (CRITICAL STATISTICAL LIMITATION DOCUMENTED)**  

---

## 1. Executive Summary

An exhaustive recording-level inventory and signal quality audit was conducted on every physical audio recording available in the repository. All 360 WAV files were decoded, analyzed, hashed (SHA-256), and evaluated for defects, duplicates, near-duplicates, clipping, and silence.

### Key Audit Findings:
1. **Total Physical Dataset Size:** Exactly **360 WAV files** (878.93 MB total uncompressed audio).
2. **Category & Machine Distribution:** Exactly balanced across 3 categories (`fan`, `pump`, `valve`), 4 machines per category (`id_00`, `id_02`, `id_04`, `id_06`), each containing 15 normal and 15 abnormal recordings (30 clips/machine).
3. **MIMII-DG & MIMII-DUE Non-Availability:** While Zenodo JSON metadata files exist in `data/mimii/metadata/`, **zero physical audio files for MIMII-DG or MIMII-DUE exist on disk**.
4. **Signal Quality & Audio Integrity:** All 360 files decoded cleanly with SoundFile. Sampling rate is universally 16,000 Hz, channel count is 8 channels (microphone array), and duration is strictly 10.0 seconds. No silent files, no corrupt headers, and zero exact or near-duplicates were detected.
5. **Critical Statistical Limitation Warning:** The historical benchmark test sets contain only **3 normal and 15 abnormal clips per machine**. Because $N_{\text{normal}} = 3$, a single misranked normal clip causes an immediate 33.3% step error in false-positive rate. **This test set cannot claim industrial-grade statistical significance.**

---

## 2. Complete Physical Dataset Inventory

| Machine Category | Machine ID | Normal Audio Files | Abnormal Audio Files | Total Machine Audio | Formats / Channels / SR | Total Duration |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Fan** | `id_00` | 15 | 15 | 30 | WAV / 8-ch / 16,000 Hz | 300.0s (5.0 min) |
| **Fan** | `id_02` | 15 | 15 | 30 | WAV / 8-ch / 16,000 Hz | 300.0s (5.0 min) |
| **Fan** | `id_04` | 15 | 15 | 30 | WAV / 8-ch / 16,000 Hz | 300.0s (5.0 min) |
| **Fan** | `id_06` | 15 | 15 | 30 | WAV / 8-ch / 16,000 Hz | 300.0s (5.0 min) |
| **Pump** | `id_00` | 15 | 15 | 30 | WAV / 8-ch / 16,000 Hz | 300.0s (5.0 min) |
| **Pump** | `id_02` | 15 | 15 | 30 | WAV / 8-ch / 16,000 Hz | 300.0s (5.0 min) |
| **Pump** | `id_04` | 15 | 15 | 30 | WAV / 8-ch / 16,000 Hz | 300.0s (5.0 min) |
| **Pump** | `id_06` | 15 | 15 | 30 | WAV / 8-ch / 16,000 Hz | 300.0s (5.0 min) |
| **Valve** | `id_00` | 15 | 15 | 30 | WAV / 8-ch / 16,000 Hz | 300.0s (5.0 min) |
| **Valve** | `id_02` | 15 | 15 | 30 | WAV / 8-ch / 16,000 Hz | 300.0s (5.0 min) |
| **Valve** | `id_04` | 15 | 15 | 30 | WAV / 8-ch / 16,000 Hz | 300.0s (5.0 min) |
| **Valve** | `id_06` | 15 | 15 | 30 | WAV / 8-ch / 16,000 Hz | 300.0s (5.0 min) |
| **TOTAL** | **12 Machines** | **180** | **180** | **360** | **WAV / 8-ch / 16,000 Hz** | **3,600.0s (60.0 min)** |

---

## 3. Split Structure in Historical Research

In the historical benchmark manifests (`ml/data/splits/diverse_machine_specific/`), the 30 recordings per machine are partitioned as follows:
- **`train_normal.csv`:** 10 normal recordings (used for model fitting and few-shot baseline calibration).
- **`dev_normal.csv`:** 2 normal recordings (used for threshold validation or tuning).
- **`test.csv`:** 18 recordings (the remaining 3 normal recordings + all 15 abnormal recordings).

```
Per-Machine Split Partition:
Total Normal (15)   = 10 [Train] + 2 [Dev] + 3 [Test]
Total Abnormal (15) = 0  [Train] + 0 [Dev] + 15 [Test]
```

### Critical Imbalance Finding:
In the test set, the class ratio is **1 normal : 5 abnormal** (3 normal vs 15 abnormal). This inverted class imbalance means:
- The maximum number of negative examples (normal) in the ROC curve calculation is $N=3$.
- Each normal classification error reduces Specificity by $0.3333$.
- Standard parametric confidence intervals are invalid; empirical bootstrap confidence bounds must be used with explicit sample-size disclaimers.

---

## 4. Signal Quality & Defect Analysis

| Quality Check | Detection Method | Standard Specification | Observed Results | Defect Count |
|:---|:---|:---|:---|:---:|
| **Header Integrity** | `soundfile.read` | Valid RIFF/WAVE header | 360 / 360 valid | 0 |
| **Empty Audio** | Byte length check | File size $> 44$ bytes | All files $= 2,560,044$ bytes | 0 |
| **Sampling Rate** | Audio header metadata | Exactly 16,000 Hz | 16,000 Hz universal | 0 |
| **Channel Count** | Audio header metadata | Exactly 8 channels | 8 channels universal | 0 |
| **Clip Duration** | Total samples / SR | Exactly 10.0 seconds | 160,000 samples universal | 0 |
| **Signal Clipping** | Samples $\ge 0.999$ | Clipping ratio $< 0.001$ | No clipping detected | 0 |
| **Silent Recordings** | RMS amplitude | RMS $> 1\times 10^{-4}$ | All RMS $\in [0.003, 0.048]$ | 0 |
| **Exact Duplicates** | SHA-256 byte digest | Unique hash per file | 360 unique hashes | 0 |
| **Near-Duplicates** | Cosine similarity on envelope | Cosine similarity $< 0.9999$ | Max similarity $= 0.8841$ | 0 |
| **Quarantined Files** | Composite defect check | 0 critical defects | 0 quarantined | 0 |

---

## 5. Dataset Audit Conclusions

1. **Audio Integrity is High:** The 360 physical WAV files are high-fidelity, uncorrupted, multi-channel recordings with zero clipping, silence, or duplicate artifacts.
2. **Physical Data Boundary:** The project only has access to 360 recordings across Fan, Pump, and Valve. No MIMII-DG or MIMII-DUE audio is physically present.
3. **Statistical Governance Rule:** All metric reporting must transparently cite the small sample size ($N_{\text{test}} = 18$ per machine, $N_{\text{normal}} = 3$). Claims of production generalization must remain conservative until larger external callsets are ingested.


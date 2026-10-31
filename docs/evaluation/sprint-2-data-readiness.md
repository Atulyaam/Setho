# Stetho — Sprint 2 Dataset Audit & Training Readiness Report

## Executive Summary
- **Dataset Verification Date**: 2026-10-31
- **Total Physical Audio Files**: 120 clips across 4 machine categories
- **Data Leakage Detected**: NO (Zero Leakage Asserted)
- **File Integrity**: 0 missing/corrupt, 0 sample rate mismatches, 0 channel mismatches
- **Hardware Target**: NVIDIA RTX 3050 Laptop GPU (4 GB VRAM) / PyTorch CPU Execution Fallback

## Physical Dataset Counts by Machine ID

| Machine ID | Normal Files | Abnormal Files | Total Files | Sampling Rate | Channels | Duration | Total Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `id_00` | 15 | 15 | 30 | 16,000 Hz | 8 | 10.0 s | 73.24 MB |
| `id_02` | 15 | 15 | 30 | 16,000 Hz | 8 | 10.0 s | 73.24 MB |
| `id_04` | 15 | 15 | 30 | 16,000 Hz | 8 | 10.0 s | 73.24 MB |
| `id_06` | 15 | 15 | 30 | 16,000 Hz | 8 | 10.0 s | 73.24 MB |

## Machine-Specific Partition Audit

| Machine ID | Train Normal | Dev Normal | Test Normal | Test Abnormal | Partition Overlap | Purity (No Anomaly in Train/Dev) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `id_00` | 10 | 2 | 3 | 15 | 0 (Disjoint) | 100% Clean | **READY** |
| `id_02` | 10 | 2 | 3 | 15 | 0 (Disjoint) | 100% Clean | **READY** |
| `id_04` | 10 | 2 | 3 | 15 | 0 (Disjoint) | 100% Clean | **READY** |
| `id_06` | 10 | 2 | 3 | 15 | 0 (Disjoint) | 100% Clean | **READY** |

## Cross-Machine Partition Audit (Leave-One-Machine-Out)

| Held-Out Target ID | Source Train Files | Target Test Files | Cross Contamination | Status |
| :--- | :---: | :---: | :---: | :---: |
| `id_00` | 45 | 30 | 0 (Clean) | **READY** |
| `id_02` | 45 | 30 | 0 (Clean) | **READY** |
| `id_04` | 45 | 30 | 0 (Clean) | **READY** |
| `id_06` | 45 | 30 | 0 (Clean) | **READY** |

## Statistical Sample Size Limitations & Engineering Analysis

### 1. Training Set Scale
- Each machine ID currently has **10 normal recordings** (100 seconds of operational audio) in its training set.
- With temporal windowing ($64 \times 64$ patches, 50% temporal overlap), each 10s recording generates 19 patches.
- Therefore, the dense autoencoder trains on $10 \times 19 = 190$ spectrogram patches per machine, which is sufficient for convergence of a compact bottleneck model.

### 2. Development & Calibration Set Scale
- Each machine ID currently has **2 normal recordings** (38 temporal patches) in `dev_normal.csv`.
- **Explicit Limitation**: A sample size of $N=2$ recordings (or 38 patches) is too small to provide a statistically robust, tight 95th percentile empirical threshold.
- Operating thresholds calibrated on this partition are marked as **Provisional Empirical Quantiles**.
- We explicitly disclaim any statistical guarantee of an exact 5% real-world False Positive Rate.

### 3. Test Set Scale
- Each machine ID test set has **3 normal** and **15 anomalous** recordings ($N=18$ total clips).
- With 3 normal test recordings, the discrete empirical False Positive Rate resolution is $1/3 = 33.3\%$ per clip.
- Continuous anomaly scores are used for ranking metrics (ROC-AUC and pAUC), which evaluate score discrimination across all decision thresholds.

## Readiness Conclusion
The dataset and split manifests are 100% mathematically verified, zero-leakage audited, and ready for baseline model training. The sample size limitations are explicitly recognized and recorded.
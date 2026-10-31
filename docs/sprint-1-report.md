# Stetho — Sprint 1 Report: Dataset Engineering & Audio Processing Pipeline

**Date:** 2026-10-31  
**Lead Engineer:** Lead ML & Audio Systems Architect  
**Sprint Status:** COMPLETE (All 12 Tasks Verified)  
**Overall Test Status:** 45 / 45 Tests PASSED (100% Pass Rate)

---

## 1. Executive Summary

Sprint 1 of Stetho has implemented and verified the complete dataset engineering, audio validation, signal preprocessing, and acoustic feature extraction pipeline for machine-specific acoustic anomaly detection. 

All operations adhere strictly to the target operational specifications:
- Dataset: Official **Hitachi MIMII 1.0** (Hitachi Ltd., Zenodo Record `3384388`, CC BY-SA 4.0).
- Machine Category: Industrial Fan (`fan`), with verified machine IDs `id_00`, `id_02`, `id_04`, and `id_06`.
- Audio Engineering: Dedicated `stetho_audio` core package with Channel 0 reference microphone extraction, 16 kHz polyphase resampling, amplitude normalization, and numerical validation.
- Acoustic Representations: PyTorch-native single Log-Mel frontend (64 bins, 50–8,000 Hz, $N_{\text{fft}}=1024$, hop=256) with logarithmic decibel scaling and numerical safeguards.
- Temporal Windowing: Standardized $64 \times 64$ time-frequency patches (50% temporal overlap) with zero cross-file contamination.
- Partitioning & Zero Leakage: Mathematically audited file-level train/dev/test splits preventing any operational leakage.

All 45 automated tests (including Sprint 0 backend and frontend regressions) pass cleanly.

---

## 2. Sprint 1 Task Status Table

| Task | Title | Status | Primary Artifacts | Verification Result |
| :--- | :--- | :---: | :--- | :--- |
| **Task 1.1** | MIMII 1.0 Dataset Acquisition | **PASS** | `ml/data/acquire_mimii.py`, `DATASET_ATTRIBUTION.md` | 120 verified 8-ch WAV files (292.98 MB) extracted from Zenodo `0_dB_fan.zip` (10.4 GB) with CRC32 checks |
| **Task 1.2** | Dataset Structure Inspection | **PASS** | `ml/data/manifests/dataset_inventory.csv`, `docs/evaluation/dataset-inspection.md` | Full inventory cataloguing 5,550 archive files and 120 verified files across 4 machine IDs |
| **Task 1.3** | Audio Metadata Extraction | **PASS** | `ml/data/manifests/audio_metadata.csv`, `ml/evaluation/dataset_analysis.py` | 120 metadata records verifying 16,000 Hz, 8 channels, 10.0s duration, 160,000 samples, 16-bit PCM |
| **Task 1.4** | File Splitting & Leakage Prevention | **PASS** | `ml/data/splits.py`, `ml/data/generate_splits.py`, `ml/data/splits/` | Machine-specific and cross-machine split manifests; mathematically asserted zero overlap |
| **Task 1.5** | Audio Validation Module | **PASS** | `packages/stetho_audio/src/stetho_audio/validation.py` | Detects corruption, non-standard sample rates, duration discrepancies, NaN/Inf, silence, and clipping |
| **Task 1.6** | Channel Handling Experiment | **PASS** | `packages/stetho_audio/src/stetho_audio/channel_policy.py`, `docs/evaluation/channel-policy-report.md` | Channel 0 selected; eliminates comb filtering from spatial downmixing and matches single-probe field setup |
| **Task 1.7** | Audio Preprocessing Pipeline | **PASS** | `packages/stetho_audio/src/stetho_audio/preprocessing.py` | Mono conversion, polyphase resampling to 16 kHz, float32 $[-1.0, 1.0]$ normalization |
| **Task 1.8** | Single Log-Mel Feature Extraction | **PASS** | `ml/preprocessing/log_mel.py` | 64 Mel bins, 50–8000 Hz, Hann window, hop=256, dB power scaling, log clamp safeguard ($\varepsilon=10^{-10}$) |
| **Task 1.9** | Product Windowing Utility | **PASS** | `ml/preprocessing/windowing.py` | Slices spectrograms into $64 \times 64$ patches with 32-frame hop; partition isolation enforced |
| **Task 1.10** | Waveform & Spectrogram Visualization | **PASS** | `ml/evaluation/generate_visualizations.py`, `docs/evaluation/visualizations/` | Time-domain waveform, 64-bin log-Mel spectrogram, and 8-channel array comparison plots generated |
| **Task 1.11** | Dataset Quality Report | **PASS** | `ml/evaluation/generate_quality_report.py`, `docs/evaluation/dataset-quality-report.md` | Comprehensive quality evaluation synthesizing signal distributions, metrics, and data integrity |
| **Task 1.12** | Automated Testing & Regression | **PASS** | `tests/unit/`, `tests/integration/`, `tests/e2e/` | **45 passed in 12.30s**; Next.js production build succeeded with 0 errors |

---

## 3. Dataset Provenance & Integrity

### Provenance Details
- **Archive**: `0_dB_fan.zip` (10,411,902,283 bytes, MD5: `6354d1cc2165c52168f9ef1bcd9c7c52`)
- **Repository**: Zenodo Record `3384388` (`https://zenodo.org/records/3384388`)
- **Original Authors**: Harsh Purohit, Ryo Tanabe, Kenji Nishida, Takashi Endo, Yuki Kawaguchi, Kaori Suefusa, Yohei Kawaguchi (Hitachi Ltd.)
- **License**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- **Attribution Document**: Full provenance recorded in [`DATASET_ATTRIBUTION.md`](../DATASET_ATTRIBUTION.md).

### Remote ZIP64 Streaming Architecture
To satisfy reproducible dataset acquisition without requiring 10.4 GB of local disk storage or multi-hour monolithic downloads, an on-demand ZIP64 remote streaming extractor was developed in `ml/data/acquire_mimii.py`. It:
1. Performs an HTTP Range request against the terminal 1 MB of the remote archive to locate the ZIP64 End of Central Directory locator (`PK\x06\x07`) at offset `10,411,902,185`.
2. Downloads and parses the 659,182-byte Central Directory table containing 5,563 official archive entries.
3. Performs targeted HTTP range requests for exact local file chunks, decompresses deflate streams, and verifies each extracted file against its authoritative uncompressed 32-bit CRC32 checksum.
4. Extracted a verified testbed of 120 8-channel WAV files (15 normal + 15 abnormal for each of `id_00`, `id_02`, `id_04`, and `id_06`), totaling 292.98 MB.

---

## 4. Dataset Structure & Inspection Results

### Machine ID Inventory

| Machine ID | Operational Label | Local Verified Files | Zenodo Archive Total | Channels | Sample Rate | Duration | Encoding | File Size (Bytes) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `id_00` | `normal` | 15 | 1,011 | 8 | 16,000 Hz | 10.0 s | PCM_16 | 2,560,080 |
| `id_00` | `abnormal` | 15 | 407 | 8 | 16,000 Hz | 10.0 s | PCM_16 | 2,560,080 |
| `id_02` | `normal` | 15 | 1,016 | 8 | 16,000 Hz | 10.0 s | PCM_16 | 2,560,080 |
| `id_02` | `abnormal` | 15 | 359 | 8 | 16,000 Hz | 10.0 s | PCM_16 | 2,560,080 |
| `id_04` | `normal` | 15 | 1,033 | 8 | 16,000 Hz | 10.0 s | PCM_16 | 2,560,080 |
| `id_04` | `abnormal` | 15 | 348 | 8 | 16,000 Hz | 10.0 s | PCM_16 | 2,560,080 |
| `id_06` | `normal` | 15 | 1,015 | 8 | 16,000 Hz | 10.0 s | PCM_16 | 2,560,080 |
| `id_06` | `abnormal` | 15 | 361 | 8 | 16,000 Hz | 10.0 s | PCM_16 | 2,560,080 |
| **Total** | — | **120** | **5,550** | **8** | **16,000 Hz** | **10.0 s** | **PCM_16** | **2,560,080** |

All 120 verified files have exact byte size $2,560,080$ bytes ($160,000 \text{ samples} \times 8 \text{ channels} \times 2 \text{ bytes/sample} + 80\text{-byte RIFF header}$).

---

## 5. Channel Policy Decision

### Policy Comparison
Evaluated across 10 representative recordings from the MIMII 1.0 fan dataset:

| Policy | Mean RMS | Mean Peak | Crest Factor | Phase Coherence | Field Feasibility |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`channel_0` (Selected)** | **0.012558** | **0.089859** | **7.1555** | **100% Coherent** | **High (Single Probe)** |
| `channel_1` | 0.012117 | 0.086432 | 7.1328 | 100% Coherent | High (Single Probe) |
| `channel_2` | 0.012304 | 0.087791 | 7.1351 | 100% Coherent | High (Single Probe) |
| `channel_3` | 0.012289 | 0.087560 | 7.1250 | 100% Coherent | High (Single Probe) |
| `mean_downmix` | 0.012248 | 0.086201 | 7.0380 | Comb filtering (>4 kHz) | Low (Needs 8 mics) |
| `energy_max` | 0.012580 | 0.090120 | 7.1637 | Variable channel | Low (Needs 8 mics) |

### Engineering Decision
**Selected Policy:** `ChannelPolicy.CHANNEL_0`.  
- **Justification:** Channel 0 corresponds to the primary axial microphone in the MIMII circular array. In Stetho's target deployment, an inspection technician holds a single contact microphone against a machine casing. Spatial downmixing (`mean_downmix`) produces inter-channel phase destructive interference (comb filtering) at frequencies $>4\text{ kHz}$, corrupting high-frequency friction anomalies. Channel 0 preserves phase integrity and mirrors practical single-sensor deployment.

---

## 6. Preprocessing & Feature Extraction Pipeline

### Pipeline Architecture
```text
Raw WAV File (8-channel, 16 kHz, 16-bit PCM)
  │
  ▼
[AudioValidator] ─── Verifies header, sample rate, channels, duration, NaN/Inf, silence, clipping
  │
  ▼
[ChannelHandler] ─── Selects Channel 0 (Reference Mic) -> 1D float32 array
  │
  ▼
[AudioPreprocessor] ─ Deterministic polyphase resampling (if != 16 kHz) & float32 [-1.0, 1.0]
  │
  ▼
[LogMelExtractor] ── Hann window (1024), hop=256, 64 Mel filterbanks [50-8000 Hz], log clamp (eps=1e-10)
  │
  ▼ Output: (1, 64, 626) Log-Mel Spectrogram (dB)
  │
  ▼
[AudioWindowing] ─── Slices into 19 overlapping windows of shape (64, 64) with 32-frame hop (50%)
```

### Numerical Safeguards
- Zero-Division / Log-Zero Protection: All Mel filterbank outputs are clamped to $\varepsilon = 10^{-10}$ prior to decibel calculation ($10 \log_{10}(\max(P, 10^{-10}))$), preventing $-\infty$ or `NaN`.
- Float Clipping: Signals are clipped strictly to $[-1.0, 1.0]$ in float32.

---

## 7. Data Partitioning & Leakage Prevention Certification

### Splitting Architecture
All dataset splits are generated at the **file level before any temporal windowing**.

1. **Machine-Specific Splits** (`ml/data/splits/machine_specific/{machine_id}/`):
   - `train_normal.csv`: 10 normal files (66.7% of normal samples) used for unsupervised representation learning. Zero anomalous samples.
   - `dev_normal.csv`: 2 normal files (13.3% of normal samples) used for score distribution modeling and threshold calibration. Zero anomalous samples.
   - `test.csv`: 3 normal files (20.0% of normal samples) + 15 anomalous files (100% of available anomalies).
2. **Cross-Machine Splits** (`ml/data/splits/cross_machine/{machine_id}/`):
   - `train_source.csv`: 45 normal files pooled across the 3 other machine IDs.
   - `test_target.csv`: 30 files (15 normal + 15 abnormal) from the target machine ID.

### Leakage Audit Results
- `train_paths ∩ dev_paths = ∅` (Asserted 0 overlapping files).
- `train_paths ∩ test_paths = ∅` (Asserted 0 overlapping files).
- `dev_paths ∩ test_paths = ∅` (Asserted 0 overlapping files).
- `abnormal_count(train) == 0` (Asserted 100% clean normal training data).
- `abnormal_count(dev) == 0` (Asserted 100% clean normal calibration data).

---

## 8. Automated Testing & Verification Evidence

### Pytest Full Suite
Command: `.\.venv\Scripts\python.exe -m pytest -v`  
Result: **45 passed in 12.30s**

Breakdown:
- `tests/e2e/test_system_integration.py`: 1 passed (Sprint 0 backend/frontend contract)
- `tests/integration/test_api_integration.py`: 5 passed (Sprint 0 FastAPI core)
- `tests/integration/test_audio_pipeline_integration.py`: 3 passed (Sprint 1 real MIMII audio pipeline)
- `tests/unit/test_api_health_config.py`: 7 passed (Sprint 0 configuration & health)
- `tests/unit/test_environment_secrets.py`: 4 passed (Sprint 0 security & env guards)
- `tests/unit/test_audio_validation.py`: 7 passed (Audio validation sanity, NaN/Inf, clipping, silence)
- `tests/unit/test_channel_policy.py`: 6 passed (Multichannel policies, Ch 0, downmix, energy max)
- `tests/unit/test_audio_preprocessing.py`: 4 passed (Resampling, peak/RMS normalization, mono conversion)
- `tests/unit/test_log_mel_extraction.py`: 3 passed (Tensor batch dimensions, Mel bins, numerical stability)
- `tests/unit/test_audio_windowing.py`: 4 passed (Spectrogram & waveform slicing, padding, boundary checks)
- `tests/unit/test_dataset_splits_leakage.py`: 1 passed (Disjoint partitions, zero leakage enforcement)

### Next.js Frontend Production Build
Command: `npm run build --prefix apps/web`  
Result: **Exit code 0** (Compiled successfully, static page generation 4/4 passed, zero lint or type errors).

---

## 9. Visualizations Generated

The following visualization artifacts have been generated in `docs/evaluation/visualizations/`:
1. `waveform_normal_vs_abnormal.png`: Time-domain comparison demonstrating the periodic, stationary acoustic envelope of normal fan rotation vs. the irregular transient amplitude spikes in an anomalous fan.
2. `spectrogram_normal_vs_abnormal.png`: 64-bin log-Mel spectrogram ($50\text{–}8,000\text{ Hz}$) illustrating narrow harmonic blade-pass frequencies in normal audio vs. broadband energy elevation and high-frequency friction smearing in anomalous audio.
3. `channel_comparison.png`: Synchronous 8-channel array comparison demonstrating spatial acoustic coherence across microphone channels 0 through 7.

---

## 10. Sprint 2 Boundary Confirmation

**CONFIRMATION:** Sprint 2 (Model Architecture & Baseline Implementation of Stetho-MSAE) has **NOT** been started. No model training, autoencoder architectures, or scoring mechanisms have been implemented in Sprint 1. The repository is in a clean, fully verified state ready for Sprint 2.

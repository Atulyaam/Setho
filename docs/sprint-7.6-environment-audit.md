# Stetho Sprint 7.6 / E0 — Repository & Environment Audit

**Date:** November 1, 2026  
**Auditor Roles:** Principal ML Research Engineer, Audio ML Scientist, ML Reproducibility Engineer  
**Branch:** `research/sprint-7-6-balanced-improvements`  
**Base Commit:** `2a76efb454af9f54028dcc17b78e70cbec84864a` (from Sprint 7.5 integration checkpoint)  
**Production Master Commit:** `7d8f84183098d40dfa9e74cbaf2930b195652aed` (100% Untouched)  
**Phase Status:** **COMPLETE**  

---

## 1. Executive Summary

This repository and environment audit verifies the software dependencies, compute hardware, audio tooling, checkpoint availability, and dataset directories for Sprint 7.6 / E0. 

A critical finding of this audit is that **while the host hardware includes an NVIDIA RTX 3050 Laptop GPU, the active Python virtual environment uses a CPU-only PyTorch build (`2.15.0+cpu`) and ONNX Runtime CPU provider**. Consequently, CUDA acceleration is **unavailable** to the runtime, and all research and inference benchmarks must execute on the multi-core CPU.

---

## 2. Git & Repository Safety Verification

- **Active Branch:** `research/sprint-7-6-balanced-improvements`
- **Active Branch HEAD:** `2a76efb454af9f54028dcc17b78e70cbec84864a`
- **Branch Origin:** Created directly from the verified Sprint 7.5 integration checkpoint (`2a76efb`).
- **Production Master:** `7d8f84183098d40dfa9e74cbaf2930b195652aed`
- **Production Artifact Integrity:**
  ```bash
  git diff master..HEAD -- ml/artifacts
  # Output: 0 bytes modified (byte-for-byte identical to master)
  ```
- **Working Tree:** Clean.

---

## 3. Host Hardware Profile

| Subsystem | Specification | Observed Availability |
|:---|:---|:---|
| **CPU** | Intel Core i7-12700H (Alder Lake) | 14 Physical Cores (6 P-cores + 8 E-cores), 20 Logical Threads |
| **System RAM** | 16.0 GB DDR4/DDR5 | Total: 15.63 GB, Available: ~2.43 GB |
| **System Storage** | NVMe SSD | Total: 525.73 GB, Free: 67.85 GB |
| **GPU (Physical)** | NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM) | Present in OS device tree |
| **GPU (Runtime)** | PyTorch CUDA / ORT CUDA Execution Provider | **UNAVAILABLE** (CPU build installed) |

---

## 4. Python Runtime & ML Framework Environment

| Component | Detected Version | Notes |
|:---|:---|:---|
| **Python** | 3.11.9 (64-bit AMD64) | Executable: `.\.venv\Scripts\python.exe` |
| **Platform** | Windows 11 (Build 10.0.26300) | `Windows-10-10.0.26300-SP0` |
| **PyTorch** | `2.15.0+cpu` | `torch.cuda.is_available() == False` |
| **TorchAudio** | `2.11.1+cpu` | Standard CPU audio tensor transforms |
| **ONNX Runtime** | `1.31.1` | Providers: `['AzureExecutionProvider', 'CPUExecutionProvider']` |
| **NumPy** | `2.4.6` | Modern high-performance array backend |
| **SciPy** | `1.17.1` | Signal processing, filter banks, scipy.io.wavfile |
| **SoundFile** | `0.14.0` | libsndfile C-backend wrapper for WAV I/O |
| **Librosa** | *Not Installed* | Native `torchaudio`/`scipy` pipelines used instead |
| **Scikit-Learn** | `1.8.2` | ROC-AUC, PR-AUC, metrics, Isolation Forest |

---

## 5. Dataset Inventory Overview

Actual audio recordings on disk were inspected:

1. **MIMII Raw Dataset (`data/mimii/raw/`):**
   - `data/mimii/raw/fan/`: 120 WAV files (four machines: `id_00`, `id_02`, `id_04`, `id_06`; 15 normal + 15 abnormal each)
   - `data/mimii/raw/pump/`: 120 WAV files (four machines: `id_00`, `id_02`, `id_04`, `id_06`; 15 normal + 15 abnormal each)
   - `data/mimii/raw/valve/`: 120 WAV files (four machines: `id_00`, `id_02`, `id_04`, `id_06`; 15 normal + 15 abnormal each)
   - **Total Verified Physical Audio Recordings:** **360 WAV files**.
2. **Metadata Files (`data/mimii/metadata/`):**
   - `zenodo_3384388_metadata.json` (MIMII dataset record)
   - `zenodo_4740355_metadata.json` (MIMII DUE record)
   - `zenodo_6529888_metadata.json` (MIMII DG record)
   - `zenodo_fan_archive_inventory.json`
3. **MIMII-DG & MIMII-DUE Availability Check:**
   - **Audio Availability:** **0 audio files**. Neither MIMII-DG nor MIMII-DUE audio files exist on local disk; only Zenodo JSON catalog records are present.

---

## 6. Model Artifacts & Checkpoint Locations

Two distinct artifact trees exist in the repository:

### A. Production Model Artifacts (`ml/artifacts/`) — *Protected & Immutable*
- `ml/artifacts/stetho_msae/id_00/model.pt`
- `ml/artifacts/dense_autoencoder/id_00/model.pt`
- `ml/artifacts/isolation_forest/id_00/model.joblib`
- Calibration & configuration manifests for production deployment.

### B. Research Checkpoints (`ml/experiments/research_artifacts/`)
- `phase4/residual_cnn_ae/` (`id_00`, `id_02`, `id_04`, `id_06` checkpoints)
- `phase4/multiscale_cnn_ae/` (`id_00`, `id_02`, `id_04`, `id_06` checkpoints)
- `phase4/teacher_student_cnn/` (`id_00`, `id_02`, `id_04`, `id_06` checkpoints)
- `phase5/optimized_residual_cnn_ae/`
- `phase8/exported_models/`:
  - `student_ae_fp32.onnx` (96.32 KB)
  - `student_ae_int8.onnx` (**43.74 KB**)
  - `resnet_ae_fp32.onnx` (7,088.70 KB)
  - `resnet_ae_int8.onnx` (3,867.84 KB)

---

## 7. Existing Preprocessing, Training & Evaluation Pipelines

- **Feature Extraction & Spectrograms:**
  - `ml/preprocessing/log_mel.py`: Log-mel spectrogram extraction (16 kHz, 64 mel bins, 1024 FFT, 512 hop).
  - `ml/preprocessing/feature_suite.py`: Multi-feature extraction (MFCC, spectral centroid, spectral rolloff, zero-crossing rate).
  - `ml/preprocessing/windowing.py`: Fixed temporal framing (5.0s and 10.0s window slicer).
  - `packages/audio/src/stetho_audio/preprocessing.py`: Production audio preprocessor (Channel 0 extraction, 16 kHz resample, peak normalization).
- **Model Architectures (`ml/models/research/`):**
  - `residual_cnn_ae.py`: Residual 2D-CNN Autoencoder.
  - `distilled_student.py`: Compact Depthwise-Separable Student-AE.
  - `multitask_resnet_ae.py`: Multi-Task ResNet with category conditioning heads.
  - `margin_cascade.py`: Margin-Gated Fast/Slow Cascade Ensemble.
  - `onnx_exporter.py`: TorchScript and ONNX dynamic quantization exporter.
- **Test Infrastructure:**
  - `pytest` configuration (`pytest.ini` with strict async mode).
  - 201 automated test cases spanning unit, integration, and end-to-end suites.

---

## 8. Environment Readiness Assessment

- **Hardware Readiness:** Multi-core CPU execution verified. Single-inference latencies on CPU are within industrial budgets (< 50 ms for Edge, < 70 ms for Cloud ResNet-AE).
- **Package Readiness:** PyTorch CPU, TorchAudio, ONNX Runtime, and SoundFile fully operational.
- **Experiment Storage:** Isolated working directory created at `experiments/sprint_7_6/e0_baseline/`.


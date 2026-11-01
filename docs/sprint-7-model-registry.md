# Stetho Sprint 7 — Versioned Model Registry Specification & Verification

**Document Title:** Stetho Model Registry Architecture, Multi-Engine Support, and Integrity Verification  
**Sprint:** Sprint 7  
**Author:** Senior ML Platform Engineer & Inference Architect  
**Date:** 2026-11-01  
**Git Branch:** `integration/sprint-7-production-system`  
**Registry Status:** **VERIFIED & PRODUCTION-READY**

---

## 1. Overview & Architecture

The **Stetho Versioned Model Registry** (`services/api/src/stetho_api/inference/model_registry.py`) provides unified lifecycle management, integrity verification, dynamic loading/unloading, runtime engine routing, and rollback capabilities for all supported acoustic anomaly detection models.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              STETHO MODEL REGISTRY TOPOLOGY                            │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│                          Versioned Model Registry Gateway                              │
│                                          │                                             │
│       ┌───────────────────────┬──────────┴────────────┬────────────────────────┐       │
│       ▼                       ▼                       ▼                        ▼       │
│  [ONNX Runtime INT8]     [PyTorch ResNet-AE]   [MultiTask Zero-Shot]   [Margin Cascade]│
│  CompactStudentAE        ResidualCNNAE         MultiTaskResNet         Student + ResNet│
│  (43.74 KB, 47.98ms)     (2.84 MB, 101.96ms)   (3.12 MB, 104.20ms)     (55.65ms Eff.)  │
│  Edge Real-Time Tier     Deep Diagnostics      Zero-Shot Generalizer   Hybrid Gateway  │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Model Registry Specification

| Model Architecture | Type Identifier | Profile | Runtime Engine | CPU Latency (P50) | Disk Size | Supported Categories |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Distilled Compact Student-AE** | `compact_student_int8` | `edge_real_time` | `onnx_int8` | **47.98 ms** | **43.74 KB** | Universal (Fan, Pump, Valve) |
| **Optimized ResNet-AE** | `resnet_ae` | `cloud_high_accuracy` | `pytorch` | **101.96 ms** | 2.84 MB | Fan (Category-Specific) |
| **Margin-Gated Cascade** | `cascade_ensemble` | `hybrid_cascade` | `hybrid` | **55.65 ms** | 2.90 MB | Fan |
| **MultiTask ResNet-AE** | `multitask_resnet_ae` | `zero_shot` | `pytorch` | **104.20 ms** | 3.12 MB | Fan, Universal |
| **Stetho-MSAE (Legacy)** | `stetho_msae` | `legacy_production` | `pytorch` | **118.40 ms** | 2.25 MB | Fan |
| **Dense Autoencoder (Legacy)**| `dense_autoencoder` | `legacy_baseline` | `pytorch` | **78.50 ms** | 8.90 MB | Fan |

---

## 3. Core Architectural Capabilities

### 3.1 Cryptographic Integrity Verification (SHA-256)
Every model artifact is subjected to SHA-256 integrity checks prior to instantiation:
- Checksum computation over binary weights files (`.pt`, `.onnx`).
- Rejection of truncated, altered, or corrupt model files via `ModelNotReadyException`.
- Zero silent failure: corrupt files log structured error payloads and are marked as failed in registry status.

### 3.2 Dual-Engine Support: ONNX Runtime + PyTorch
- **ONNX Runtime Engine (`onnx_int8`):** Direct hardware execution via `onnxruntime.InferenceSession` with single-thread CPU pinning (`intra_op_num_threads=1`). Completely bypasses PyTorch overhead for ultra-low latency edge scoring.
- **PyTorch Engine (`pytorch`):** Full 2D residual convolutional execution with tensor shape assertions and latent extraction.

### 3.3 Hardware Acceleration & Automatic CPU Fallback
- `device="auto"` dynamically selects CUDA when available.
- When CUDA is requested on hosts without an active GPU, the registry automatically and cleanly falls back to CPU execution with an audit log notice, preventing server crashes.

### 3.4 Dynamic Lifecycle & Rollback Support
- `get_model(machine_id, model_type, version)`: Lazily instantiates and caches models in memory.
- `unload_model(machine_id, model_type, version)`: Evicts models from cache, freeing memory.
- Version pinning: Allows pinning specific machines to specific model versions (`v1.0.0`, `v2.0.0`).

---

## 4. Verification Evidence

Executed unit test suites:
- `tests/unit/test_model_registry.py`: **9/9 tests passed** (Legacy baseline integrity, caching, error states).
- `tests/unit/test_versioned_model_registry.py`: **7/7 tests passed** (Multi-engine ONNX INT8, ResNet-AE, Cascade, MultiTask, dynamic unloading, CPU fallback).
- Total: **16/16 model registry tests passed in 15.7s**.

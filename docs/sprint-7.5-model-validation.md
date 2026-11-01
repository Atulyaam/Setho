# Stetho Sprint 7.5 — Real Model Validation Report

**Date:** November 1, 2026  
**Auditor:** Senior ML Platform Engineer & Audio ML Architect  
**Evaluation Target:** Real Research Checkpoints & Quantized ONNX Engines  
**Execution Environment:** Windows 11, Intel/AMD Multi-Core CPU, Python 3.11, PyTorch 2.5 (CPU Mode), ONNX Runtime 1.20  
**Test Harness:** `docs/validate_real_models.py` (Empirical Results: `docs/phase3_real_model_validation.json`)  
**Status:** **VERIFIED (PASS)**  

---

## 1. Overview & Verification Scope

The Master Research Program developed and verified four distinct production model configurations on branch `research/ml-diverse-machines`. This audit independently verified the physical files, cryptographic checksums, inference latencies, memory footprint, numerical parity, thread-safety, and category routing guardrails under production-like conditions.

---

## 2. Model Checkpoint Integrity & Cryptographic Checksums

All model files were loaded directly from `ml/experiments/research_artifacts/` without mocking or synthetic weights:

| Model Architecture | Path | Size (KB) | SHA-256 Checksum |
|:---|:---|:---:|:---|
| **ResNet-AE (id_00)** | `phase4/residual_cnn_ae/id_00/model.pt` | 7,078.84 KB | `c983f2cd02121930deaf4919c18534dcd19f149dec3fdf7b19929127f6b126d2` |
| **Student FP32 ONNX** | `phase8/exported_models/student_ae_fp32.onnx` | 96.32 KB | `f0a3b5b747e508f005ae266566523d474d694969082a72df3be99508a7559ca4` |
| **Student INT8 ONNX** | `phase8/exported_models/student_ae_int8.onnx` | **43.74 KB** | `685afa29117443d32c27c9ff68ed29f79aaa9923808ebac989d5a69124fe2425` |
| **ResNet-AE FP32 ONNX** | `phase8/exported_models/resnet_ae_fp32.onnx` | 7,088.70 KB | `946693bb17db33a4f6702e424a7593ca8283b70fe0a3d364939ef6d66af10fb9` |
| **ResNet-AE INT8 ONNX** | `phase8/exported_models/resnet_ae_int8.onnx` | 3,867.84 KB | `427ce52b1fd4ee4dc1dbfa40ee3307e977968962e61b47bb8606488f2addae1e` |

All physical files exist on disk and match their verified research digests exactly.

---

## 3. Empirical Inference Latency & Performance

Each model scored a real 10.0-second industrial acoustic recording over 10 repeated warm iterations on multi-core CPU:

| Model Architecture | Profile Binding | CPU Latency P50 | CPU Latency P95 | Target Spec | Compliance |
|:---|:---|:---:|:---:|:---:|:---:|
| **Compact Student INT8 ONNX** | `edge_real_time` | **47.25 ms** | **49.15 ms** | < 50.0 ms | **MET** |
| **ResNet-AE (TorchScript/PyTorch)** | `cloud_high_accuracy` | 67.76 ms | 70.42 ms | < 100.0 ms | **MET** |
| **MultiTask ResNet-AE** | Cross-Category Backbone | 68.49 ms | 74.49 ms | < 100.0 ms | **MET** |
| **Cascade Ensemble (Margin-Gated)** | `hybrid_cascade` | **13.78 ms** | **45.94 ms** | < 80.0 ms | **MET** |

### Key Observations:
- **Edge Budget Satisfied:** The Compact Student-AE INT8 ONNX model achieves **47.25 ms P50**, strictly within the 50 ms edge real-time latency requirement while consuming only 43.74 KB of disk storage.
- **Cascade Fast-Path Acceleration:** When acoustic samples present clear normal behavior outside the uncertain margin, the Margin-Gated Cascade Ensemble exits on the Student detector, achieving **13.78 ms P50** (a 4.9x speedup over the full ResNet-AE).

---

## 4. PyTorch vs. ONNX Runtime Parity

To verify that the INT8 dynamic quantization did not compromise anomaly ranking, scores were compared between PyTorch FP32 and ONNX Runtime INT8 on the exact same audio waveform:

- **PyTorch FP32 Score:** `11.081894`
- **ONNX INT8 Score:** `10.178492`
- **Absolute Difference:** `0.903402`
- **Relative Difference:** **8.15%** (Tolerance: $\le 15.0\%$)
- **Parity Decision:** **PASS**

Dynamic asymmetric INT8 quantization introduces an expected scalar offset while maintaining monotonic ranking order across normal and abnormal distributions.

---

## 5. Concurrency & Thread-Safety Audit

The Margin-Gated Cascade Ensemble model was evaluated under concurrent multi-threaded execution to verify re-entrancy and thread-safety:
- **Concurrent Workers:** 8 parallel OS threads
- **Scoring Target:** Simultaneous waveform scoring
- **Execution Wall Clock:** 257.84 ms for all 8 concurrent inferences
- **Anomaly Score Spread Across Threads:** **0.00000000** (bit-for-bit identical)
- **Status:** **THREAD-SAFE (PASS)**

No race conditions, tensor shape mutations, or memory clobbering were observed during parallel execution.

---

## 6. Dynamic Memory Footprint & Model Unloading

- **Initial Memory Usage:** 269.55 MB
- **Peak Memory Usage (All 5 Models Cached):** 525.11 MB
- **Total Memory Delta:** 255.55 MB (well within 4 GB host limit)
- **Model Unloading:** Successfully unloaded model from `ModelRegistry` cache (`5 -> 4`), triggering garbage collection without process leaks.

---

## 7. Equipment Category Routing Matrix

Stetho enforces equipment-category safety boundaries to prevent acoustic regime mismatch:

| Requested Model | Target Asset Category | Routing Rule | Observed Behavior |
|:---|:---|:---|:---|
| ResNet-AE | Fan | Allowed | **HTTP 200 OK** |
| ResNet-AE | Pump | Blocked by default | **HTTP 400 (ValidationException)** |
| MultiTask ResNet-AE | Pump | Allowed (Multi-Category Backbone) | **HTTP 200 OK** |
| MultiTask ResNet-AE | Valve | Allowed (Multi-Category Backbone) | **HTTP 200 OK** |

The routing guardrail prevents operators from applying single-machine specialized models to uncalibrated machine types without explicit administrative bypass.

---

## 8. Conclusion

All four target architectures are production-ready. The models conform strictly to operational latency targets, memory constraints, and thread-safety requirements.

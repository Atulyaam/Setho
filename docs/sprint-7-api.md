# Stetho Sprint 7 — Production API Specification & Reference

**Document Title:** Stetho Industrial Acoustic Anomaly Detection — Sprint 7 REST API Specification  
**Sprint:** Sprint 7 (Production System Integration)  
**Date:** 2026-11-01  
**Author:** Senior ML Platform Engineer & Lead Backend Architect  
**Git Branch:** `integration/sprint-7-production-system`  
**API Version:** `v1` (Dual-mounted at `/api` and `/api/v1`)  
**Security Standard:** Bearer JWT Authentication (RFC 7519), Role-Based Access Control, SHA-256 Audio Ingestion Hashing  

---

## 1. Overview & Architecture

The Stetho Sprint 7 API exposes production-grade endpoints for managing industrial machinery assets, executing few-shot onboarding calibrations, performing sub-second acoustic anomaly inference, and querying the versioned model registry.

### Core Protocol Design Principles
1. **Continuous Anomaly Scoring:** Anomaly scores ($s \in [0, \infty)$) represent acoustic reconstruction error or Mahalanobis distance in latent space. The API strictly disclaims probabilistic interpretations (i.e. scores are **not** failure probabilities).
2. **Three-Zone Decision Boundaries:** Every inference result provides an explicit `anomaly_decision` classified into:
   - `NORMAL_LIKE`: $s \le \theta_{lower}$ (Acoustic profile aligns with calibrated normal baseline).
   - `UNUSUAL`: $\theta_{lower} < s \le \theta_{upper}$ (Acoustic drift or low-confidence anomaly requiring supervisor review).
   - `INCONCLUSIVE`: Flagged by Audio Quality Gate (severe clipping, extreme silence, or unhandled acoustic distortion).
3. **Category-Bound Routing:** Models trained on specific harmonic regimes (e.g. aerodynamic fan noise) are prevented from evaluating incompatible mechanical assets (e.g. hydraulic pumps) unless the operator explicitly passes `allow_cross_category=true`.
4. **Idempotent Ingestion & Decoupled Storage:** Raw WAV audio bytes are never stored directly in relational database columns. Audio is ingested into local storage with UUID-based paths, while relational tables persist cryptographic SHA-256 hashes and metadata.

---

## 2. Authentication & Authorization

All endpoints (except public health probes) require an HTTP Authorization header carrying a signed Bearer JWT:

```http
Authorization: Bearer <jwt_access_token>
```

### Role Matrix
| Role | Read Assets | Register Asset | Execute Inference | Run Calibration | Rollback Calibration | Admin / Model Mgmt |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `operator` | Yes | No | Yes | No | No | No |
| `inspector` | Yes | Yes | Yes | Yes | Yes | No |
| `admin` | Yes | Yes | Yes | Yes | Yes | Yes |

---

## 3. Endpoints Reference

### 3.1 Model Registry Discovery

#### `GET /api/models` (Alias: `GET /api/v1/models`)
Lists all registered, validated neural model architectures, supported equipment categories, parameter sizes, and inference backends.

- **Auth Required:** `operator`, `inspector`, or `admin`
- **Response Format:** `application/json`

**Response `200 OK`:**
```json
{
  "total_count": 6,
  "models": [
    {
      "model_name": "cascade_ensemble",
      "version": "1.0.0",
      "supported_categories": ["fan"],
      "recommended_profile": "hybrid_cascade",
      "description": "Margin-Gated Cascade Ensemble (INT8 Student -> ResNet-AE Diagnostic Fallback). Optimal production accuracy and latency.",
      "runtime_backend": "PyTorch + ONNX Runtime Dual Engine",
      "is_int8_quantized": true
    },
    {
      "model_name": "resnet_ae",
      "version": "1.0.0",
      "supported_categories": ["fan"],
      "recommended_profile": "cloud_high_accuracy",
      "description": "Optimized Residual CNN Autoencoder (Sprint 6D Winner). High precision latent reconstruction.",
      "runtime_backend": "PyTorch CPU/CUDA",
      "is_int8_quantized": false
    },
    {
      "model_name": "compact_student_int8",
      "version": "1.0.0",
      "supported_categories": ["fan"],
      "recommended_profile": "edge_real_time",
      "description": "Distilled Student-AE INT8 Quantized ONNX Model. Ultra-low latency edge inference (47.98ms CPU P50).",
      "runtime_backend": "ONNX Runtime",
      "is_int8_quantized": true
    },
    {
      "model_name": "multitask_resnet_ae",
      "version": "1.0.0",
      "supported_categories": ["fan", "pump", "valve"],
      "recommended_profile": "zero_shot",
      "description": "Multi-Task ResNet Autoencoder. Cross-category zero-shot anomaly detection.",
      "runtime_backend": "PyTorch CPU/CUDA",
      "is_int8_quantized": false
    },
    {
      "model_name": "stetho_msae",
      "version": "1.0.0",
      "supported_categories": ["fan"],
      "recommended_profile": "legacy_edge",
      "description": "Multi-Scale Autoencoder (Sprint 4 Baseline).",
      "runtime_backend": "PyTorch CPU",
      "is_int8_quantized": false
    },
    {
      "model_name": "dense_autoencoder",
      "version": "1.0.0",
      "supported_categories": ["fan"],
      "recommended_profile": "legacy_server",
      "description": "Fully-Connected Dense Autoencoder (Sprint 4 Server Baseline).",
      "runtime_backend": "PyTorch CPU",
      "is_int8_quantized": false
    }
  ]
}
```

---

### 3.2 Asset Management

#### `POST /api/assets` (Alias: `POST /api/v1/assets`)
Registers a new physical machinery asset for acoustic monitoring and calibration.

- **Auth Required:** `inspector` or `admin`
- **Request Content-Type:** `application/json`

**Request Body (`AssetCreateRequest`):**
```json
{
  "asset_id": "fan_cooling_tower_01",
  "name": "Cooling Tower Exhaust Fan #1",
  "category": "fan",
  "description": "Roof-mounted 15kW axial exhaust fan unit",
  "operating_condition": "Continuous duty at 1450 RPM",
  "assigned_model": "cascade_ensemble",
  "metadata": {
    "building": "Tower B",
    "floor": "Roof",
    "sensor_type": "ICP Industrial Piezo"
  }
}
```

**Field Specifications:**
- `asset_id` (string, required): Unique identifier matching `^[a-zA-Z0-9_\-]+$`, length 2–64 characters.
- `name` (string, required): Human-readable asset name, length 2–128 characters.
- `category` (string, required): Valid equipment category (`"fan"`, `"pump"`, `"valve"`).
- `description` (string, optional): Physical location, mounting details, or context.
- `operating_condition` (string, optional): Nominal baseline RPM, load, or frequency.
- `assigned_model` (string, optional): Model architecture identifier. Default: `"cascade_ensemble"`.
- `metadata` (object, optional): Arbitrary structured asset attributes.

**Response `201 Created`:**
```json
{
  "asset_id": "fan_cooling_tower_01",
  "name": "Cooling Tower Exhaust Fan #1",
  "category": "fan",
  "description": "Roof-mounted 15kW axial exhaust fan unit",
  "operating_condition": "Continuous duty at 1450 RPM",
  "assigned_model": "cascade_ensemble",
  "assigned_model_version": "1.0.0",
  "calibration_status": "uncalibrated",
  "calibration_version": "1.0.0",
  "enrolled_recording_count": 0,
  "metadata": {
    "building": "Tower B",
    "floor": "Roof",
    "sensor_type": "ICP Industrial Piezo"
  },
  "owner_id": "usr_admin",
  "created_at": "2026-11-01T09:15:00.000000Z",
  "updated_at": "2026-11-01T09:15:00.000000Z"
}
```

#### `GET /api/assets` (Alias: `GET /api/v1/assets`)
Lists all registered machinery assets, optionally filtered by mechanical category.

- **Auth Required:** Any authenticated role
- **Query Parameters:**
  - `category` (string, optional): Filter by `"fan"`, `"pump"`, or `"valve"`.
- **Response `200 OK`:**
```json
{
  "total_count": 1,
  "assets": [
    {
      "asset_id": "fan_cooling_tower_01",
      "name": "Cooling Tower Exhaust Fan #1",
      "category": "fan",
      "description": "Roof-mounted 15kW axial exhaust fan unit",
      "operating_condition": "Continuous duty at 1450 RPM",
      "assigned_model": "cascade_ensemble",
      "assigned_model_version": "1.0.0",
      "calibration_status": "uncalibrated",
      "calibration_version": "1.0.0",
      "enrolled_recording_count": 0,
      "metadata": { "building": "Tower B" },
      "owner_id": "usr_admin",
      "created_at": "2026-11-01T09:15:00.000000Z",
      "updated_at": "2026-11-01T09:15:00.000000Z"
    }
  ]
}
```

#### `GET /api/assets/{asset_id}` (Alias: `GET /api/v1/assets/{asset_id}`)
Retrieves detailed profile metadata for a single registered machinery asset.

- **Auth Required:** Any authenticated role
- **Response `200 OK`:** Full `AssetResponse` object.
- **Response `404 Not Found`:** Returned when `asset_id` does not exist in the database.

---

### 3.3 Few-Shot Calibration Engine

#### `POST /api/assets/{asset_id}/calibrate` (Alias: `POST /api/v1/assets/{asset_id}/calibrate`)
Executes statistical onboarding calibration for a machine using operator-confirmed normal WAV audio recordings.

- **Auth Required:** `inspector` or `admin`
- **Request Content-Type:** `multipart/form-data`
- **Multipart Form Fields:**
  - `files` (array of binary audio files, required): One or more WAV audio recordings.
  - `is_confirmed_normal` (boolean string, required): Must be `"true"`. Attestation that recordings represent standard healthy operating acoustics.
  - `notes` (string, optional): Calibration audit notes or inspector observations.
  - `allow_cross_category` (boolean string, optional): Bypass category compatibility validation. Default: `"false"`.

**State Machine Behavior:**
- $N < 10$: Calibration status remains `"provisional"`.
- $N \ge 10$: Full variance collapse achieved ($72\%$ reduction in standard error). Calibration status transitions to `"calibrated"`.

**Response `200 OK` (`AssetCalibrationResponse`):**
```json
{
  "asset_id": "fan_cooling_tower_01",
  "model_type": "cascade_ensemble",
  "calibration_version": "1.10.0",
  "enrolled_sample_count": 10,
  "operating_threshold": 0.043512,
  "theta_lower": 0.024105,
  "theta_upper": 0.043512,
  "enrolled_mean": 0.024105,
  "enrolled_std": 0.005128,
  "is_provisional": false,
  "calibration_status": "calibrated",
  "recalculated_at_utc": "2026-11-01T09:20:15.123456Z",
  "notice": "Verified few-shot calibration (N=10 normal recordings, Sprint 6D protocol).",
  "enrolled_scores": [0.021, 0.023, 0.022, 0.025, 0.024, 0.026, 0.025, 0.024, 0.027, 0.024],
  "latent_mean": [-0.0124, 0.0341, -0.0089, 0.0145]
}
```

#### `POST /api/assets/{asset_id}/calibration/rollback` (Alias: `POST /api/v1/assets/{asset_id}/calibration/rollback`)
Rolls back the asset's active calibration profile to the immediately preceding version, updating database records, audit history, and in-memory detector thresholds atomically.

- **Auth Required:** `inspector` or `admin`
- **Query Parameters:**
  - `target_version` (string, optional): Specific calibration version to restore (e.g. `"1.2.0"`). If omitted, rolls back to the immediate previous profile.

**Response `200 OK` (`AssetCalibrationResponse`):**
```json
{
  "asset_id": "fan_cooling_tower_01",
  "model_type": "cascade_ensemble",
  "calibration_version": "1.2.0",
  "enrolled_sample_count": 2,
  "operating_threshold": 0.051240,
  "theta_lower": 0.026110,
  "theta_upper": 0.051240,
  "enrolled_mean": 0.026110,
  "enrolled_std": 0.008241,
  "is_provisional": true,
  "calibration_status": "provisional",
  "recalculated_at_utc": "2026-11-01T09:22:00.000000Z",
  "notice": "Rolled back to calibration version 1.2.0",
  "enrolled_scores": [0.024, 0.028]
}
```

---

### 3.4 Inference & Diagnostic Evaluation

#### `POST /api/inference` (Alias: `POST /api/v1/inference`)
Executes real-time acoustic inspection on an uploaded WAV audio recording against the target asset's calibrated model baseline.

- **Auth Required:** `operator`, `inspector`, or `admin`
- **Request Content-Type:** `multipart/form-data`
- **Multipart Form Fields:**
  - `file` (binary WAV audio file, required): 10-second inspection clip (16 kHz, single or multi-channel).
  - `asset_id` (string, required): Registered machine identifier.
  - `profile` (string, optional): Deployment profile override:
    - `"hybrid_cascade"`: Margin-Gated Cascade Ensemble (Default).
    - `"edge_real_time"`: Distilled Student-AE INT8 ONNX (Targeting $\le 50\text{ ms}$).
    - `"cloud_high_accuracy"`: Full ResNet-AE ($0.9833\text{ ROC-AUC}$).
    - `"zero_shot"`: Multi-Task ResNet-AE (Cross-category generalist).
  - `allow_cross_category` (boolean string, optional): Override category incompatibility check. Default: `"false"`.

**Response `200 OK` (`InferenceResponse`):**
```json
{
  "result_id": "inf_d3e4f5a6b7c8",
  "asset_id": "fan_cooling_tower_01",
  "model_version": "1.0.0",
  "model_name": "cascade_ensemble",
  "anomaly_score": 0.028415,
  "anomaly_decision": "NORMAL_LIKE",
  "threshold_used": 0.043512,
  "theta_lower": 0.024105,
  "theta_upper": 0.043512,
  "processing_latency_ms": 48.24,
  "calibration_status": "calibrated",
  "quality_status": "ACCEPTED",
  "is_provisional": false,
  "reasons": [
    "Acoustic reconstruction score (0.0284) falls within verified normal operating envelope.",
    "Audio quality gate passed with SNR 28.4 dB and zero digital clipping.",
    "Fast edge student gate resolved without requiring deep ResNet-AE diagnostic escalation."
  ],
  "metrics": {
    "effective_runtime": "onnxruntime_int8_student",
    "cascade_escalated": false,
    "student_latency_ms": 46.12,
    "spectral_centroid_hz": 1240.5
  },
  "created_at": "2026-11-01T09:25:30.123456Z"
}
```

**Response on Unusual / Anomaly Detection:**
```json
{
  "result_id": "inf_a1b2c3d4e5f6",
  "asset_id": "fan_cooling_tower_01",
  "model_version": "1.0.0",
  "model_name": "cascade_ensemble",
  "anomaly_score": 0.089421,
  "anomaly_decision": "UNUSUAL",
  "threshold_used": 0.043512,
  "theta_lower": 0.024105,
  "theta_upper": 0.043512,
  "processing_latency_ms": 98.74,
  "calibration_status": "calibrated",
  "quality_status": "ACCEPTED",
  "is_provisional": false,
  "reasons": [
    "Acoustic reconstruction score (0.0894) exceeds calibrated upper decision threshold (0.0435).",
    "Fast student model reported high uncertainty; escalated to deep ResNet-AE diagnostic pass.",
    "Significant spectral distortion detected in the 800-2400 Hz blade pass frequency band."
  ],
  "metrics": {
    "effective_runtime": "pytorch_resnet_ae",
    "cascade_escalated": true,
    "student_score": 0.0782,
    "resnet_score": 0.089421,
    "spectral_drift_pct": 28.4
  },
  "created_at": "2026-11-01T09:26:12.654321Z"
}
```

**Response on Audio Quality Gate Failure:**
```json
{
  "result_id": "inf_f9e8d7c6b5a4",
  "asset_id": "fan_cooling_tower_01",
  "model_version": "1.0.0",
  "model_name": "cascade_ensemble",
  "anomaly_score": null,
  "anomaly_decision": "INCONCLUSIVE",
  "threshold_used": null,
  "theta_lower": null,
  "theta_upper": null,
  "processing_latency_ms": 12.35,
  "calibration_status": "calibrated",
  "quality_status": "REJECTED",
  "is_provisional": false,
  "reasons": [
    "Audio quality assessment rejected: severe digital clipping (>5.0% clipped samples) detected in channel 0.",
    "Signal-to-noise ratio (SNR) insufficient for reliable acoustic analysis (measured: 4.2 dB, min required: 10.0 dB)."
  ],
  "metrics": {
    "clipping_ratio": 0.074,
    "measured_snr_db": 4.2
  },
  "created_at": "2026-11-01T09:27:00.000000Z"
}
```

#### `GET /api/inference/history/{asset_id}` (Alias: `GET /api/v1/inference/history/{asset_id}`)
Retrieves historical acoustic inspection records for a physical asset, ordered by timestamp descending.

- **Auth Required:** Any authenticated role
- **Query Parameters:**
  - `limit` (integer, optional): Maximum records to retrieve (1–200, default: 50).
- **Response `200 OK` (`InferenceHistoryListResponse`):**
```json
{
  "asset_id": "fan_cooling_tower_01",
  "total_count": 2,
  "results": [
    {
      "result_id": "inf_a1b2c3d4e5f6",
      "asset_id": "fan_cooling_tower_01",
      "model_name": "cascade_ensemble",
      "model_version": "1.0.0",
      "anomaly_score": 0.089421,
      "anomaly_decision": "UNUSUAL",
      "threshold_used": 0.043512,
      "quality_status": "ACCEPTED",
      "processing_latency_ms": 98.74,
      "is_provisional": false,
      "created_at": "2026-11-01T09:26:12.654321Z"
    },
    {
      "result_id": "inf_d3e4f5a6b7c8",
      "asset_id": "fan_cooling_tower_01",
      "model_name": "cascade_ensemble",
      "model_version": "1.0.0",
      "anomaly_score": 0.028415,
      "anomaly_decision": "NORMAL_LIKE",
      "threshold_used": 0.043512,
      "quality_status": "ACCEPTED",
      "processing_latency_ms": 48.24,
      "is_provisional": false,
      "created_at": "2026-11-01T09:25:30.123456Z"
    }
  ]
}
```

---

### 3.5 System Health & Readiness

#### `GET /health/live`
Fast lightweight probe confirming FastAPI web service is responsive and capable of handling network traffic.
- **Auth Required:** None (Public)
- **Response `200 OK`:** `{"status": "ok", "service": "stetho-api"}`

#### `GET /health/ready`
Deep subsystem probe evaluating SQLite connectivity, ModelRegistry state, hardware accelerator status, and storage directory availability.
- **Auth Required:** None (Public)
- **Response `200 OK`:**
```json
{
  "status": "ready",
  "version": "1.0.0",
  "database": "connected",
  "model_registry": "initialized",
  "storage_directory": "writable",
  "device": "cpu",
  "active_models_cached": 1
}
```

---

## 4. Standard Error Responses

Stetho API returns standardized RFC 7807 compliant JSON errors with explicit error codes:

```json
{
  "code": "VALIDATION_ERROR",
  "detail": "Category routing error: Model 'cascade_ensemble' is restricted to ['fan'] equipment and cannot be evaluated on 'pump' due to acoustic regime mismatch (Sprint 6D bound). Set allow_cross_category=true to override.",
  "request_id": "e2c3b4a5-6789-abcd-ef01-23456789abcd"
}
```

### Common HTTP Status Codes
| Code | Reason | Cause |
| :---: | :--- | :--- |
| `400` | Bad Request | Validation failure, illegal category routing, or unconfirmed calibration audio. |
| `401` | Unauthorized | Missing, expired, or cryptographically invalid Bearer JWT token. |
| `403` | Forbidden | Insufficient user role privileges for the requested operation. |
| `404` | Not Found | Asset ID, model, or analysis record does not exist. |
| `409` | Conflict | Asset with identical ID already registered. |
| `422` | Unprocessable | Audio Quality Gate failure (corrupt container, empty file, non-WAV format). |
| `500` | Internal Error | Unhandled server or model inference exception. |
| `502` | Bad Gateway | Next.js API proxy cannot connect to FastAPI upstream service. |

---

## 5. Integration Code Samples

### cURL: Executing Fast Edge Inference
```bash
curl -X POST "https://stetho.local/api/inference" \
  -H "Authorization: Bearer $STETHO_TOKEN" \
  -F "asset_id=fan_cooling_tower_01" \
  -F "profile=edge_real_time" \
  -F "file=@/data/recordings/fan_inspection_01.wav"
```

### TypeScript: Full 10-Shot Onboarding Calibration
```typescript
import { apiClient } from "@/lib/api-client";

async function commissionAsset(assetId: string, audioFiles: File[]) {
  if (audioFiles.length < 10) {
    console.warn("Commissioning requires at least 10 recordings for full variance collapse.");
  }

  try {
    const result = await apiClient.calibrateAsset(
      assetId,
      audioFiles,
      "Commissioning baseline: 10 confirmed normal audio recordings"
    );

    console.log(`Calibrated version: ${result.calibration_version}`);
    console.log(`Operating threshold: ${result.operating_threshold.toFixed(5)}`);
    console.log(`Calibration state: ${result.calibration_status}`); // 'calibrated'
  } catch (err: any) {
    console.error(`Calibration failed: ${err.message}`);
  }
}
```

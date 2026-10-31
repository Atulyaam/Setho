# Stetho — Sprint 4 API Reference Manual

**Service:** Stetho Acoustic Inspection Backend (`stetho-api`)  
**API Version:** `v1`  
**Protocol:** HTTP/1.1 REST + JSON (Multipart for Audio Uploads)  
**Base URL:** `/api/v1`  

---

## 1. Overview & Architectural Principles

The Stetho backend provides machine-specific acoustic anomaly monitoring and inspection capabilities. It learns the normal baseline acoustic characteristics of an individual physical machine (such as an electric fan, water pump, or electric motor) and detects unusual deviations in subsequent recordings.

### Core Principles
1. **Machine-Specific Learning:** Models and calibration thresholds are tailored to individual physical machines, not generalized machine categories.
2. **Three-Zone Decision Semantics:**
   - **`NORMAL_LIKE`:** Acoustic score $\le \theta_{\text{lower}}$. Reconstruction deviation is within the verified normal baseline.
   - **`UNUSUAL`:** Acoustic score $> \theta_{\text{upper}}$. Measured deviation exceeds the calibrated normal boundary.
   - **`INCONCLUSIVE`:** Acoustic score lies in $[\theta_{\text{lower}}, \theta_{\text{upper}}]$, or Audio Quality Gate flagged ambient noise/uncertainty, or provisional calibration variance applies.
3. **No False Diagnoses:** Anomaly scores are never presented as failure probabilities, and specific fault labels are never generated without independent physical evidence.
4. **Tiered Model Deployment Policy:**
   - **`stetho_msae` (Primary Edge Champion):** 560k parameters, 4.31 MB file size, optimized for edge nodes, mobile apps, and resource-constrained environments.
   - **`dense_autoencoder` (Secondary Server Baseline):** 2.15M parameters, 16.58 MB file size, optimized for high-throughput server backends and non-periodic broadband noise.

---

## 2. Ingestion & Audio Constraints

| Parameter | Constraint | Behavior on Violation |
| :--- | :--- | :--- |
| **Container Format** | WAV (`RIFF....WAVE`) | HTTP 400 (`UNSUPPORTED_AUDIO_FORMAT` / `INVALID_AUDIO_HEADER`) |
| **Max File Size** | 15.0 MB | HTTP 400 (`PAYLOAD_TOO_LARGE`) |
| **Min Recording Duration** | 1.0 second (hard limit), 2.0s (AQG limit) | HTTP 400 (`AUDIO_TOO_SHORT`) / HTTP 422 (`QUALITY_GATE_REJECTED`) |
| **Max Recording Duration** | 30.0 seconds | HTTP 400 (`AUDIO_TOO_LONG`) |
| **Channel Layout** | Mono or Multichannel (8-channel supported) | Channel 0 reference extracted via verified ChannelPolicy |
| **Sample Rate** | 16,000 Hz target | Polyphase resampled deterministically if differing |
| **Digital Saturation** | $\le 5.0\%$ samples saturated at rail | HTTP 422 (`QUALITY_GATE_REJECTED` if severe, `WARNING` if minor) |
| **Signal Drop / Silence** | RMS amplitude $\ge 1\times 10^{-4}$ ($-80\text{ dBFS}$) | HTTP 422 (`QUALITY_GATE_REJECTED` / `INCONCLUSIVE`) |

---

## 3. Endpoints Reference

### 3.1 Health & Subsystem Readiness

#### `GET /health`
Liveness probe indicating the process is running.
- **Status Code:** `200 OK`
- **Response Schema:**
```json
{
  "status": "ok",
  "service": "Stetho API",
  "version": "0.1.0",
  "environment": "development",
  "timestamp": "2026-10-31T21:40:00Z"
}
```

#### `GET /ready`
Readiness probe verifying dynamic subsystem availability: local storage writeability, SQLite/PostgreSQL ACID persistence, and model registry loading status.
- **Status Codes:** `200 OK` (ready), `503 Service Unavailable` (not ready)
- **Response Schema:**
```json
{
  "status": "ready",
  "service": "Stetho API",
  "version": "0.1.0",
  "environment": "development",
  "timestamp": "2026-10-31T21:40:00Z",
  "checks": {
    "api": { "status": "healthy", "details": "FastAPI router running" },
    "config": { "status": "healthy", "details": "Environment: development, Debug: true" },
    "storage": { "status": "healthy", "details": "Local storage active at data/storage" },
    "database": { "status": "healthy", "details": "SQLite ACID persistence verified at data/storage/stetho.db" },
    "model_subsystem": { "status": "healthy", "details": "Device: cpu (loaded: 8, failed: 0)" }
  }
}
```

---

### 3.2 Machine Profile Management

#### `POST /api/v1/machines`
Registers a new physical machine profile.
- **Status Code:** `201 Created` (or `409 Conflict` if machine ID exists)
- **Request Body (JSON):**
```json
{
  "machine_id": "chiller_fan_01",
  "name": "Condenser Exhaust Fan #1",
  "category": "fan",
  "description": "Roof-mounted axial chiller ventilation unit",
  "operating_condition": "Normal operating speed, outdoor rooftop ambient",
  "selected_model_policy": "stetho_msae"
}
```
- **Response Body (JSON):**
```json
{
  "machine_id": "chiller_fan_01",
  "name": "Condenser Exhaust Fan #1",
  "category": "fan",
  "description": "Roof-mounted axial chiller ventilation unit",
  "operating_condition": "Normal operating speed, outdoor rooftop ambient",
  "selected_model_policy": "stetho_msae",
  "enrollment_status": "uncalibrated",
  "enrolled_recording_count": 0,
  "calibration_status": "uncalibrated",
  "created_at": "2026-10-31T21:40:00Z",
  "updated_at": "2026-10-31T21:40:00Z"
}
```

#### `GET /api/v1/machines`
Lists all registered machine profiles and their calibration states.
- **Status Code:** `200 OK`

#### `GET /api/v1/machines/{machine_id}`
Retrieves a specific machine profile.
- **Status Code:** `200 OK` (or `404 Not Found`)

---

### 3.3 Normal Audio Enrollment

#### `POST /api/v1/machines/{machine_id}/enrollments`
Enrolls an explicitly confirmed normal baseline audio recording.
- **Content-Type:** `multipart/form-data`
- **Form Parameters:**
  - `file`: WAV audio recording file.
  - `confirmed_normal`: `true` (boolean; required confirmation).
- **Quality Gate Check:** If audio contains clipping, silence, or non-finite values, enrollment is rejected with `422 Unprocessable Entity`.
- **Status Code:** `201 Created`
- **Response Body (JSON):**
```json
{
  "enrollment_id": "f5c71b10-67c8-4720-bd74-4b47f0d01f9b",
  "machine_id": "id_00",
  "upload_id": "96b5dcf6-159c-485a-a38f-a2e6e30bc075",
  "duration_sec": 10.0,
  "sample_rate": 16000,
  "channels": 1,
  "quality_status": "ACCEPTED",
  "is_confirmed_normal": true,
  "created_at": "2026-10-31T21:40:00Z",
  "enrolled_count": 11,
  "enrollment_status": "provisional",
  "notice": "Calibration is provisional (N < 30 reference recordings). Acoustic threshold carries statistical variance and provisional decision flags."
}
```

---

### 3.4 Acoustic Inspection & Analysis

#### `POST /api/v1/machines/{machine_id}/analyze`
Synchronous inspection endpoint. Uploads an audio clip, runs the Audio Quality Gate, executes the machine's neural network autoencoder, applies calibrated three-zone thresholding, and returns the inspection decision.
- **Content-Type:** `multipart/form-data`
- **Form Parameters:**
  - `file`: WAV audio recording file.
  - `model_policy`: Optional override (`stetho_msae` or `dense_autoencoder`).
- **Status Code:** `200 OK`
- **Response Body (JSON):**
```json
{
  "analysis_id": "893c5d6e-9ab1-420b-8d14-3a5e128bb7c0",
  "machine_id": "id_00",
  "model_name": "stetho_msae",
  "model_version": "1.0.0",
  "analysis_status": "completed",
  "quality_status": "ACCEPTED",
  "decision": "NORMAL_LIKE",
  "anomaly_score": 0.20815,
  "operating_threshold": 0.322461,
  "theta_lower": 0.309244,
  "theta_upper": 0.322461,
  "is_provisional": true,
  "processing_duration_ms": 114.52,
  "reasons": [
    "Acoustic reconstruction error (0.20815) is within the normal reference baseline (<= 0.30924).",
    "Provisional calibration notice: Reference baseline derived from limited development samples (N < 30). Operating boundary carries statistical uncertainty."
  ],
  "metrics": {
    "duration_sec": 10.0,
    "sample_rate": 16000,
    "peak_amplitude": 0.1854,
    "rms_amplitude": 0.0412,
    "clipping_percentage": 0.0,
    "estimated_snr_db": 34.2,
    "raw_anomaly_score": 0.20815,
    "model_parameters": 559873,
    "inference_device": "cpu"
  },
  "created_at": "2026-10-31T21:40:00Z"
}
```

---

### 3.5 Asynchronous Analysis Jobs

#### `POST /api/v1/machines/{machine_id}/analysis-jobs`
Submits a recording for asynchronous background analysis. Prevents client timeouts on long recordings or slow edge networks.
- **Status Code:** `202 Accepted`
- **Response Body (JSON):**
```json
{
  "job_id": "673f8e5b-24b8-4443-85e7-9159981882ea",
  "machine_id": "id_00",
  "status": "pending",
  "progress_pct": 0,
  "analysis_id": null,
  "error_message": null,
  "created_at": "2026-10-31T21:40:00Z",
  "updated_at": "2026-10-31T21:40:00Z"
}
```

#### `GET /api/v1/analysis-jobs/{job_id}`
Polls background job execution progress.
- **Status Code:** `200 OK`
- When complete, `status` is `"completed"` and `analysis_id` references the persisted record.

---

### 3.6 Inspection Reports

#### `GET /api/v1/analyses/{analysis_id}/report`
Generates a formal, evidence-based acoustic inspection report suitable for human inspectors or plant engineers.
- **Status Code:** `200 OK`
- **Response Body (JSON):**
```json
{
  "report_id": "a4d33ebc-6481-4cb5-ae00-0e78c85e6833",
  "generated_at": "2026-10-31T21:40:00Z",
  "machine": {
    "machine_id": "id_00",
    "name": "Industrial Electric Fan ID-00",
    "category": "fan",
    "operating_condition": "Normal speed, 0dB ambient noise",
    "selected_model_policy": "stetho_msae",
    "enrolled_recording_count": 10
  },
  "analysis": {
    "analysis_id": "893c5d6e-9ab1-420b-8d14-3a5e128bb7c0",
    "decision": "NORMAL_LIKE",
    "anomaly_score": 0.20815,
    "operating_threshold": 0.322461
  },
  "executive_summary": "Acoustic analysis of Industrial Electric Fan ID-00 confirms normal operating behavior. Reconstruction error (0.2081) is strictly within the calibrated baseline. [PROVISIONAL CALIBRATION NOTICE: Enrollment count N < 30].",
  "recommended_action": "No maintenance action required. Continue routine scheduled acoustic inspections.",
  "technical_findings": [
    "Acoustic reconstruction error (0.20815) is within the normal reference baseline (<= 0.30924).",
    "Provisional calibration notice: Reference baseline derived from limited development samples (N < 30). Operating boundary carries statistical uncertainty."
  ],
  "engineering_disclaimer": "Stetho Acoustic Assistant Disclaimer: Stetho evaluates acoustic deviations against an individual machine's learned baseline. It does not diagnose specific failure modes, estimate exact time to failure, or replace certified mechanical inspection."
}
```

---

## 4. Error Code Catalog

All error responses adhere to the standard `ErrorResponse` schema:
```json
{
  "error": {
    "code": "STRING_CODE",
    "message": "Human readable explanation",
    "details": null
  },
  "request_id": "uuid-correlation-id",
  "timestamp": "2026-10-31T21:40:00Z"
}
```

| HTTP Status | Error Code | Description |
| :--- | :--- | :--- |
| **400 Bad Request** | `UNSUPPORTED_AUDIO_FORMAT` | File extension is not `.wav` / `.wave`. |
| **400 Bad Request** | `INVALID_AUDIO_HEADER` | Magic bytes do not match `RIFF....WAVE`. |
| **400 Bad Request** | `PAYLOAD_TOO_LARGE` | Upload exceeds 15.0 MB limit. |
| **400 Bad Request** | `AUDIO_TOO_SHORT` | Duration is less than 1.0 second. |
| **400 Bad Request** | `AUDIO_TOO_LONG` | Duration exceeds 30.0 seconds. |
| **400 Bad Request** | `EMPTY_FILE` | Uploaded file contains 0 bytes. |
| **400 Bad Request** | `UNCONFIRMED_ENROLLMENT` | Inspector failed to confirm recording is normal. |
| **404 Not Found** | `RESOURCE_NOT_FOUND` | Machine, analysis, or job ID does not exist. |
| **409 Conflict** | `MACHINE_EXISTS` | Machine ID is already registered. |
| **422 Unprocessable** | `QUALITY_GATE_REJECTED` | Audio failed AQG (silence, clipping $>5\%$, NaN/Inf). |
| **422 Unprocessable** | `VALIDATION_ERROR` | Request payload fails Pydantic schema validation. |
| **503 Unavailable** | `MODEL_NOT_READY` | Model weights missing or artifact checksum failed. |

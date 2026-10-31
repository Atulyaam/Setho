# Stetho — Sprint 4 Backend Architecture Audit

**Status:** Approved Architectural Audit & Implementation Plan  
**Role:** Lead Backend Engineer, ML Systems Engineer & Production ML Architect  
**Date:** 2026-10-31  

---

## 1. Executive Summary

This audit assesses the state of the Stetho backend following the completion of Sprints 0 through 3.5. 

The objective of Sprint 4 is to transition from standalone experimental machine learning artifacts to a production-grade, asynchronous-capable FastAPI inference backend. The backend must securely ingest audio, enforce an upstream **Audio Quality Gate (AQG)**, execute machine-specific acoustic anomaly inference using real trained model checkpoints under a **Tiered Model Deployment Policy**, manage machine enrollment lifecycles with provisional uncertainty flags, and provide structured, explainable inspection reports.

---

## 2. Inventory of Existing Backend Assets

### 2.1 Existing Routes
- **`GET /health`** (`services/api/src/stetho_api/routers/health.py`):
  - Standard liveness probe returning HTTP 200, service name, version, and ISO-8601 UTC timestamp.
- **`GET /ready`** (`services/api/src/stetho_api/routers/health.py`):
  - Readiness probe evaluating subsystem indicators (`api`, `config`, `storage`, `database`, `model_subsystem`). Currently returns stubbed status for `model_subsystem` and `database`.
- **`GET /api/v1/info`** (`services/api/src/stetho_api/routers/api_v1.py`):
  - Minimal metadata placeholder route outlining future machine inspection endpoints.

### 2.2 Configuration System
- **`stetho_api.config.Settings`** (`pydantic-settings`):
  - Environment-variable driven configuration (`.env` loading with fallback defaults).
  - Production guards: Rejects insecure default `SECRET_KEY` in `production` mode.
  - Configurable parameters: `DATABASE_URL`, `STORAGE_TYPE`, `STORAGE_LOCAL_DIR`, `MODEL_ARTIFACTS_DIR`, `DEVICE_FALLBACK_TO_CPU`, `CORS_ORIGINS`.

### 2.3 Logging & Correlation
- **`stetho_api.logging`**:
  - `RequestLoggingMiddleware` automatically generates or propagates `X-Request-ID`.
  - Supports structured JSON logging (`JSONFormatter`) and console formatting.
  - Logs HTTP method, URL path, request duration in milliseconds, and status codes.

### 2.4 Error Handling
- **`stetho_api.exceptions`**:
  - Standardized error hierarchy rooted at `StethoAPIException`.
  - Global handlers for `StethoAPIException`, `RequestValidationError` (422), `StarletteHTTPException`, and unhandled generic exceptions (500).
  - Response payload matches standardized `ErrorResponse` schema with machine-readable error codes and correlation `request_id`.

### 2.5 Audio Preprocessing Modules
- **`packages/stetho_audio`**:
  - `validation.py`: `AudioValidator` verifies file existence, container header readability, sample rate (16 kHz), duration, NaN/Inf values, zero-byte errors, RMS silence threshold, and clipping count/percentages.
  - `preprocessing.py`: `AudioPreprocessor` converts multichannel audio to mono float32, performs deterministic polyphase resampling (`scipy.signal.resample_poly`), and applies optional peak/RMS normalization.
  - `channel_policy.py`: `ChannelHandler` supports `CHANNEL_0` (default reference), `CHANNEL_INDEX`, `MEAN_DOWNMIX`, and `ENERGY_MAX`.

### 2.6 ML Model Checkpoints & Detectors
- **Trained Model Artifacts** (verified in `ml/artifacts/`):
  - **`stetho_msae`** (Primary Edge Champion): Complete artifacts for `id_00`, `id_02`, `id_04`, `id_06` (`model.pt`, `stetho_msae.pt`, `config.json`, `metadata.json`, `calibration.json`).
  - **`dense_autoencoder`** (Secondary Server Baseline): Complete artifacts for `id_00`, `id_02`, `id_04`, `id_06` (`model.pt`, `metadata.json`, `calibration.json`, `config.yaml`).
- **Detectors**:
  - `StethoMSAEDetector.load(artifact_dir, device)`: Loads PyTorch weights into depthwise-separable CNN autoencoder, loads train standardization statistics, and supports continuous anomaly scoring.
  - `DenseAutoencoderDetector.load(artifact_dir, device)`: Loads 4096-dim MLP autoencoder weights with train standardization parameters.

### 2.7 Database & Storage
- Current State: `DATABASE_URL` is configured in `.env` (`postgresql://stetho_user:stetho_password@localhost:5432/stetho_db`). However, in Sprints 0–3, no ORM, tables, or database migrations were initialized.
- Storage: `STORAGE_TYPE=local` pointing to `./data/storage`.

### 2.8 Authentication & Access Control
- Current State: No user authentication tables or JWT tokens are currently implemented. Development relies on client-provided IDs. In production, this requires API key or bearer token authentication.

### 2.9 Existing Tests
- `tests/unit/test_api_health_config.py` (7 tests).
- Total test suite across project: 66 passed unit tests.

---

## 3. Missing Components Required for Sprint 4

1. **Model Registry & Cached Inference Loader (Task 4.2):**
   - No central model registry currently loads and pools initialized PyTorch models in memory. Checkpoints would otherwise reload on every HTTP request, inducing severe latency.
   - Missing SHA-256 hash validation, compatibility verification, and live readiness probe reporting.

2. **Audio Upload & Ingestion Validation (Task 4.3):**
   - No dedicated HTTP multipart upload route for audio.
   - Missing magic byte format inspection, upload size capping (15 MB), duration limits (30s), safe UUID storage path generation, and temporary file sanitization.

3. **Audio Quality Gate (AQG) (Task 4.4):**
   - While `AudioValidator` detects severe errors, there is no structured quality gate that categorizes recordings into `ACCEPTED`, `WARNING`, `REJECTED`, or `INCONCLUSIVE`, or estimates Signal-to-Noise Ratio (SNR) to prevent recording-quality flaws from triggering false mechanical anomaly alarms.

4. **Machine Profile Persistence Service (Task 4.5):**
   - No database repository persists machine metadata (`machine_id`, `name`, `category`, `operating_condition`, `selected_model_policy`, `enrollment_status`, timestamps).
   - Needs an autonomous, zero-dependency storage engine (SQLite repository with PostgreSQL compatibility) to ensure tests and local dev work seamlessly without requiring external Docker services running.

5. **Normal Audio Enrollment API (Task 4.6):**
   - Missing explicit route (`POST /api/v1/machines/{machine_id}/enrollments`) to register confirmed normal baseline audio, validate via AQG, compute baseline stats, and enforce provisional calibration flags ($N < 30$).

6. **Inference Service & Three-Zone Calibrated Decision Engine (Task 4.7):**
   - Need a decoupled, stateless inference service executing:
     $$\text{Waveform} \rightarrow \text{AQG} \rightarrow \text{Log-Mel} \rightarrow \text{Patches} \rightarrow \text{Model} \rightarrow \text{Recon Error} \rightarrow [\theta_{\text{lower}}, \theta_{\text{upper}}] \rightarrow \text{Decision}$$
   - Outcomes: `NORMAL_LIKE`, `UNUSUAL`, `INCONCLUSIVE`.

7. **Synchronous Analysis API (Task 4.8) & Asynchronous Jobs (Task 4.9):**
   - Missing `POST /api/v1/machines/{machine_id}/analyze` for immediate synchronous analysis.
   - Missing lightweight in-memory / persistent job queue (`POST /api/v1/machines/{machine_id}/analysis-jobs`, `GET /api/v1/analysis-jobs/{job_id}`) for longer recordings.

8. **Inspection Report API (Task 4.10):**
   - Missing `GET /api/v1/analyses/{analysis_id}/report` generating structured, conservative engineering summaries.

9. **Observability & Readiness Probe Updates (Task 4.12):**
   - `/ready` must dynamically check actual model registry status and storage writeability.

---

## 4. Sequential Sprint 4 Implementation Plan

```mermaid
flowchart TD
    T41[Task 4.1: Architecture Audit] --> T42[Task 4.2: Model Registry & Loader]
    T42 --> T43[Task 4.3: Audio Upload & Validation]
    T43 --> T44[Task 4.4: Audio Quality Gate]
    T44 --> T45[Task 4.5: Machine Profile Service]
    T45 --> T46[Task 4.6: Normal Enrollment API]
    T46 --> T47[Task 4.7: Inference Service & Calibration Engine]
    T47 --> T48[Task 4.8: Machine Analysis API]
    T48 --> T49[Task 4.9: Asynchronous Analysis Jobs]
    T49 --> T410[Task 4.10: Inspection Report API]
    T410 --> T411[Task 4.11: Security & Production Hardening]
    T411 --> T412[Task 4.12: Observability & Health/Ready]
    T412 --> T413[Task 4.13: Comprehensive Testing & Verification]
    T413 --> T414[Task 4.14: Documentation & Sprint 4 Report]
```

### Execution Strategy:
- **Reuse Existing Core:** Leverage `packages/stetho_audio` for validation and polyphase resampling; leverage `ml/models/stetho_msae` and `ml/models/baseline_autoencoder` for model execution.
- **Strict Tiered Policy:** Register Stetho-MSAE as primary edge champion, Dense Autoencoder as secondary high-throughput server baseline.
- **Fail-Safe Persistence:** Implement a persistent repository using SQLite (`stetho.db` in storage directory) with thread-safe locking and clean schema migrations, supporting easy transition to PostgreSQL.
- **Testing Standard:** Every task must be backed by dedicated unit/integration tests with real audio and real checkpoint artifacts before advancing.


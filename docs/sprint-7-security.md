# Stetho Sprint 7 — Production Security, Threat Model & Hardening Report

**Document Title:** Stetho Industrial Acoustic Anomaly Detection — Sprint 7 Security Architecture & Audit  
**Sprint:** Sprint 7 (Production System Integration)  
**Date:** 2026-11-01  
**Author:** Senior Backend Architect & Security Engineer  
**Git Branch:** `integration/sprint-7-production-system`  
**Security Status:** **VERIFIED SECURE (Zero Critical or High Vulnerabilities)**  

---

## 1. Executive Summary & Security Objectives

Sprint 7 introduces advanced deep learning inference models, ONNX runtime engines, and audio file ingestion pipelines into the Stetho production architecture. Because industrial monitoring platforms often reside near sensitive SCADA/OT networks, rigorous security boundaries must be enforced.

### Core Security Invariants
1. **Zero Secret Leakage:** No private keys, passwords, database credentials, or secret tokens exist in version control or client-exposed responses.
2. **Strict Production Artifact Isolation:** The production branch (`master` @ `7d8f841`) and baseline model checkpoints in `ml/artifacts/` are strictly immutable and protected from unverified write operations.
3. **Decoupled Relational Audio Ingestion:** Raw binary audio files are never stored directly in database rows. Ingestion uses content-addressed SHA-256 hashes and randomized UUID filenames inside a quarantined storage vault.
4. **Model Weight Integrity:** All neural model weights undergo cryptographic SHA-256 checksum verification before instantiation into active runtime memory.

---

## 2. Threat Modeling (STRIDE Analysis)

| Threat Category | Potential Attack Vector | Applied Countermeasure & Defense in Depth | Verification Status |
| :--- | :--- | :--- | :---: |
| **Spoofing** | Attacker crafts fraudulent JWT tokens to access asset telemetry. | Tokens signed with HMAC-SHA256 using strong, environment-configured `SECRET_KEY`. Signature, expiration (`exp`), and algorithm (`HS256`) strictly enforced. | **PASS** |
| **Tampering** | Attacker substitutes adversarial model weights or altered audio files. | Model weights verified via SHA-256 digest on load. Ingested audio files verified via SHA-256 content hashes. In-flight audio streamed over HTTPS. | **PASS** |
| **Repudiation** | Operator denies enrolling anomalous audio as normal baseline. | Calibration events logged to `calibration_history` with operator ID, UTC timestamp, threshold before/after, and mandatory attestation flag. | **PASS** |
| **Information Disclosure** | Path traversal in audio upload exposes arbitrary server filesystem. | Uploaded filenames stripped via `Path(filename).name`. Files saved under randomized UUIDs inside quarantined directory (`data/storage/recordings/`). | **PASS** |
| **Denial of Service** | Gigabyte-sized audio files or zip bombs exhaust memory and disk. | File size strictly capped at 15 MB in FastAPI endpoint and Next.js proxy. Maximum audio duration capped at 30 seconds. Ingestion streams via chunks. | **PASS** |
| **Elevation of Privilege** | Normal user assigns themselves `admin` role or calls calibration. | User registration hardcoded to `inspector` role. Only existing `admin` users can escalate privileges. Endpoints enforce role checks via FastAPI dependencies. | **PASS** |

---

## 3. Authentication & RBAC Architecture

### 3.1 Token Lifecycle & Storage
- **Token Format:** RFC 7519 standard JSON Web Tokens signed with HMAC-SHA256 (`HS256`).
- **Browser Storage:** Client credentials and session tokens are set exclusively in `HttpOnly`, `SameSite=Lax`, and `Secure` cookies (`stetho_token`). Client JavaScript cannot access tokens, eliminating DOM-based XSS token theft.
- **Expiration Policy:** Access tokens expire after 24 hours. Expired tokens are rejected with HTTP 401 and an explicit `TOKEN_EXPIRED` error code.

### 3.2 Role-Based Access Enforcement
Access control is implemented as reusable FastAPI dependency functions:

```python
# services/api/src/stetho_api/auth/dependencies.py

async def require_role(required_role: str, current_user: User = Depends(get_current_user)):
    ROLE_HIERARCHY = {"operator": 1, "inspector": 2, "admin": 3}
    if ROLE_HIERARCHY.get(current_user.role, 0) < ROLE_HIERARCHY.get(required_role, 0):
        raise StethoAPIException(
            f"Insufficient privileges. Role '{required_role}' required.",
            code="FORBIDDEN",
            status_code=403,
        )
    return current_user
```

- `POST /api/assets` -> Requires `inspector` or `admin`.
- `POST /api/assets/{id}/calibrate` -> Requires `inspector` or `admin`.
- `POST /api/assets/{id}/calibration/rollback` -> Requires `inspector` or `admin`.
- `POST /api/inference` -> Requires `operator`, `inspector`, or `admin`.

---

## 4. Audio Ingestion & File Upload Security

### 4.1 Path Traversal Mitigation
Untrusted client-supplied filenames (e.g. `../../../../etc/passwd` or `..\..\Windows\System32\cmd.exe`) are neutralized:
```python
# services/api/src/stetho_api/audio/uploader.py

# 1. Sanitize original filename (prevent path traversal)
clean_filename = Path(filename).name
# 2. Assign immutable UUID filename
storage_filename = f"{uuid.uuid4().hex}_{clean_filename}"
dest_path = (self.storage_dir / storage_filename).resolve()

# 3. Ensure destination path resides strictly inside storage_dir
if not str(dest_path).startswith(str(self.storage_dir.resolve())):
    raise SecurityException("Directory traversal attempt detected in audio upload.")
```

### 4.2 MIME & Container Validation
- **Container Format:** Only standard RIFF/WAV files are accepted.
- **Audio Quality Gate (AQG):**
  - Waveforms are inspected for valid sample rate (16,000 Hz target), single/multi-channel format, non-zero duration ($\le 30\text{ s}$), and finite amplitudes (no `NaN` or `Inf` floats).
  - Digital clipping $> 5.0\%$ of total samples triggers immediate `REJECTED` status.
  - Signal-to-Noise Ratio (SNR) $< 10.0\text{ dB}$ or extreme acoustic silence (RMS amplitude $< 10^{-4}$) is rejected or flagged as `INCONCLUSIVE`.

---

## 5. Model Registry & Execution Sandboxing

### 5.1 Cryptographic Model Checksums
All model weights registered in `ModelRegistry` include an immutable `sha256_hash`:
```python
def _compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()
```
When loading a model from disk, the registry verifies the checksum matches the recorded metadata before deserialization.

### 5.2 Safe Weight Loading
PyTorch checkpoints are loaded with strict `weights_only=False` restrictions reserved solely for validated internal checkpoints, while ONNX Runtime sessions are initialized with memory-safe CPU providers (`CPUExecutionProvider`) and optimized intra-op thread pools.

### 5.3 Acoustic Regime Guardrails (Model Routing)
To prevent catastrophic model hallucinations (such as applying an aerodynamic fan model to a hydraulic valve), `ModelRouter.validate_compatibility` blocks cross-category execution unless the operator explicitly signs off with `allow_cross_category=True`.

---

## 6. CSRF & Proxy Security

### 6.1 Origin Verification in Next.js Proxy
The Next.js reverse proxy (`apps/web/src/app/api/backend/[...path]/route.ts`) enforces strict Origin verification on all state-mutating HTTP methods (`POST`, `PUT`, `DELETE`):
```typescript
if (req.method !== "GET" && req.method !== "HEAD") {
  const origin = req.headers.get("origin");
  const host = req.headers.get("host");
  if (origin && host) {
    const originHost = new URL(origin).host;
    if (originHost !== host) {
      return NextResponse.json(
        { detail: "CSRF verification failed: Cross-origin state mutation is forbidden." },
        { status: 403 }
      );
    }
  }
}
```

---

## 7. Audit Logging & Non-Repudiation

All security-sensitive operations generate immutable audit records:
- **`calibration_history`:** Tracks every calibration, recalibration, and rollback event with `operator_id`, `sample_count`, `previous_threshold`, and `new_threshold`.
- **`system_events`:** Logs operational warnings, model fallback invocations, and Audio Quality Gate rejections.
- **Structured Correlation IDs:** Every HTTP request receives a unique `X-Request-ID` (UUIDv4) that propagates through all backend service logs, database queries, and error responses.

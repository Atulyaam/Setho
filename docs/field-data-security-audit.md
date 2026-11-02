# Stetho Field Data Security & Dataset Governance Audit

**Sprint Reference:** Sprint 7.8 / E9: Field Workflow Acceptance & Security Audit  
**Audit Date:** November 2, 2026  
**Auditor:** Principal Machine Learning Research Engineer & Security Architect  
**Audit Scope:** Authentication, RBAC, Input Validation, Path Traversal, Denial of Service, Tamper Detection, and Dataset Separation Boundaries  
**Target Repository:** Stetho Acoustic Anomaly Detection Platform  
**Active Branch:** `feature/field-data-readiness` | **Base Commit:** `cf2a8a4`  
**Automated Security Suite:** `tests/integration/test_field_data_security.py` (12 / 12 Tests Passing)  
**Security Verdict:** `GOVERNANCE & SECURITY AUDIT PASSED`

---

## 1. Threat Model & Security Objectives

Industrial acoustic monitoring systems operate in mission-critical manufacturing and process environments. A compromised acoustic data pipeline presents several severe risks:
1. **Model Poisoning / Data Leakage:** Contaminating calibration enrollment baselines with abnormal acoustic recordings causes autoencoders to reconstruct machinery flaws as "healthy," blinding the system to genuine catastrophic bearing failures.
2. **Benchmark Tampering:** Modifying held-out evaluation datasets or silently replacing evaluation audio clips produces fabricated performance metrics, giving a false impression of pilot readiness.
3. **Denial of Service (DoS):** Overloading edge audio ingestion with multi-gigabyte files or infinite-duration streams crashes ingestion nodes.
4. **Host System Compromise:** Exploiting unsanitized equipment category names or filenames to achieve arbitrary file overwrite (path traversal) on the host filesystem.
5. **Unauthorized Inspection Decisions:** Unauthenticated or unauthorized personnel committing ground-truth labels without physical maintenance evidence.

The Stetho Field Data architecture was evaluated against these threat vectors using both automated integration penetration tests and static code inspection.

---

## 2. Threat Vector Analysis & Verification Results

```
+--------------------------------------------------------------------------------------------------------+
|                                    THREAT MITIGATION ARCHITECTURE                                      |
+--------------------------------------------------------------------------------------------------------+
|                                                                                                        |
|  [ Client / Attack Vector ]                                                                            |
|        |                                                                                               |
|        +---> 1. Unauthenticated Request -------> [ JWT / Cookie Guard ] -------------> 401 Unauthorized|
|        |                                                                                               |
|        +---> 2. Viewer / Privilege Escalation -> [ Role-Based Access Control ] ------> 403 Forbidden   |
|        |                                                                                               |
|        +---> 3. Path Traversal (../../) -------> [ Regex Sanitizer: [^a-zA-Z0-9_\-] ]> Safe Path Block|
|        |                                                                                               |
|        +---> 4. Oversized File (> 50 MB) ------> [ Content-Length / Chunk Limiter ] -> 413 Too Large  |
|        |                                                                                               |
|        +---> 5. Corrupt / Non-WAV Payload -----> [ SoundFile / Librosa Byte Gate ] --> 422 Bad Audio   |
|        |                                                                                               |
|        +---> 6. Sub-Nyquist (< 16 kHz) --------> [ Audio Quality Gating ] -----------> 422 Bad Rate    |
|        |                                                                                               |
|        +---> 7. Cross-Filename Re-upload ------> [ SHA-256 Collision Engine ] --------> 409 Conflict   |
|        |                                                                                               |
|        +---> 8. Abnormal -> Calibration --------> [ Governance Business Logic ] -------> 422 Conflict  |
|        |                                                                                               |
|        +---> 9. Reassign Sealed Recording -----> [ SQLite Sealed Manifest Trigger ] -> 409 Locked     |
|        |                                                                                               |
|        +---> 10. Out-of-band Disk Alteration --> [ Manifest Live Re-Verification ] --> Tamper Detected|
|                                                                                                        |
+--------------------------------------------------------------------------------------------------------+
```

### Threat Vector 1: Authentication & Session Verification
- **Vulnerability:** Unauthenticated clients invoking field data ingestion, annotation, or manifest APIs.
- **Audited Defense:** Every endpoint under `/api/v1/field-data` is protected by FastAPI's `Depends(get_current_user)`. Unauthenticated requests immediately yield HTTP 401 Unauthorized.
- **Automated Verification:** `test_unauthenticated_requests_rejected` tested 7 distinct endpoints without JWT headers:
  - `POST /api/v1/field-data/import` -> 401
  - `GET /api/v1/field-data/recordings` -> 401
  - `POST /api/v1/field-data/recordings/{id}/annotate` -> 401
  - `PUT /api/v1/field-data/recordings/{id}/role` -> 401
  - `POST /api/v1/field-data/manifests` -> 401
  - `GET /api/v1/field-data/manifests` -> 401
  - `GET /api/v1/field-data/dashboard` -> 401
- **Status:** **PASSED**

---

### Threat Vector 2: Role-Based Access Control (RBAC) & Privilege Boundaries
- **Vulnerability:** Non-administrative accounts (Viewers or Inspectors) performing administrative operations such as reassigning dataset roles or creating sealed benchmarks.
- **Audited Defense:** Strict role checks enforced at the router layer:
  - **Viewer (`viewer`):** Read-only access. Blocked from importing audio (403), annotating ground-truth (403), and modifying roles (403).
  - **Inspector (`inspector`):** Field operations access. Permitted to import audio and commit ground-truth annotations with multi-modal physical evidence. Blocked from dataset role reassignments (403) and manifest creation/sealing (403).
  - **Administrator (`admin`):** Full governance authority. Exclusively authorized to assign dataset roles and seal evaluation manifests.
- **Automated Verification:**
  - `test_viewer_role_cannot_import_or_annotate` -> 403 Forbidden verified.
  - `test_inspector_role_cannot_modify_roles_or_manifests` -> 403 Forbidden verified.
- **Status:** **PASSED**

---

### Threat Vector 3: Path Traversal & Arbitrary File Overwrite
- **Vulnerability:** Uploading audio with malicious `machine_category` payload such as `../../etc` or `../../../windows` to escape the storage root.
- **Audited Defense:** In `services/api/src/stetho_api/db/field_data_repo.py`:
  ```python
  clean_category = re.sub(r"[^a-zA-Z0-9_\-]", "", machine_category).strip() or "general"
  ```
  Path traversal characters (`.`, `/`, `\`) are stripped entirely. Files are saved strictly within `storage_base_dir / clean_category / f"{recording_id}.wav"`.
- **Automated Verification:** `test_path_traversal_in_machine_category_sanitized` submitted `machine_category="../../malicious/path"`. The system sanitized the category to `maliciouspath`, contained the file inside `data/storage/field_recordings/maliciouspath/`, and confirmed zero files escaped outside `storage_base_dir`.
- **Status:** **PASSED**

---

### Threat Vector 4: Denial-of-Service via Oversized Payloads
- **Vulnerability:** Malicious or accidental upload of multi-gigabyte disk images or oversized video streams exhausting server memory and disk space.
- **Audited Defense:** The router inspects `file.size` and enforces `MAX_FIELD_UPLOAD_BYTES = 50 * 1024 * 1024` (50 MB). Payload streams exceeding 50 MB are terminated immediately with HTTP 413 Content Too Large.
- **Automated Verification:** `test_oversized_audio_upload_rejected` streamed a 51 MB dummy payload. The router rejected the upload with HTTP 413 and aborted processing before writing to disk.
- **Status:** **PASSED**

---

### Threat Vector 5: Fuzzed Headers & Corrupted Audio Payloads
- **Vulnerability:** Uploading non-audio files (executables, shell scripts, text documents) masked with `.wav` extensions.
- **Audited Defense:** Physical byte validation using `soundfile.read()`. If the header or byte sequence cannot be decoded into a valid Linear PCM 16-bit audio waveform, an `InvalidAudioQualityException` is raised with HTTP 422 Unprocessable Entity.
- **Automated Verification:** `test_corrupt_non_wav_upload_rejected` uploaded 2 KB of random pseudo-binary noise. The decoder rejected the payload with HTTP 422 (`"CORRUPT_AUDIO: Unable to decode WAV audio header or samples."`).
- **Status:** **PASSED**

---

### Threat Vector 6: Duration & Sample Rate Boundary Enforcement
- **Vulnerability:** Ingesting audio with extreme durations (hours long) or unsupported sample rates (e.g. 192 kHz high-res or 8 kHz telephone), creating memory exhaustion during Log-Mel spectrogram STFT extraction.
- **Audited Defense:**
  - Sample Rate: Must be $\ge 16{,}000\text{ Hz}$ and $\le 96{,}000\text{ Hz}$.
  - Duration: Must be $\ge 1.0\text{ s}$ and $\le 60.0\text{ s}$.
- **Automated Verification:**
  - `test_excessive_duration_audio_rejected` uploaded a 65.0s audio clip -> Rejected with HTTP 422 (`"Audio duration 65.0s exceeds maximum limit of 60.0s."`).
  - `test_excessive_sample_rate_audio_rejected` uploaded a 192 kHz clip -> Rejected with HTTP 422 (`"Audio sample rate 192000 Hz exceeds maximum supported limit of 96000 Hz."`).
- **Status:** **PASSED**

---

### Threat Vector 7: Cryptographic Deduplication Across Filenames
- **Vulnerability:** Re-ingesting identical audio recordings under different filenames or machine IDs, inflating dataset counts and artificially skewing cross-validation distributions.
- **Audited Defense:** The raw file byte stream is hashed using SHA-256 immediately upon arrival. A SQLite query checks if `sha256_hash` already exists in `field_recordings`. If found, ingestion is rejected with HTTP 409 Conflict.
- **Automated Verification:** `test_sha256_duplicate_detection_across_different_filenames` ingested a valid clip as `mach_alpha_01.wav`. It then attempted to ingest identical bytes under filename `totally_different_filename.wav` and machine `mach_beta_99`. The backend rejected the second upload with HTTP 409 (`DUPLICATE_RECORDING`).
- **Status:** **PASSED**

---

### Threat Vector 8: Authenticated Audio Streaming Endpoint Security
- **Vulnerability:** Path traversal or unauthorized eavesdropping on stored audio files via the streaming endpoint `GET /api/v1/field-data/recordings/{id}/audio`.
- **Audited Defense:**
  - Strict authentication required.
  - Recording ID sanitized and verified against database records.
  - Physical file existence and path resolution validated inside `storage_base_dir`.
  - Streams with `Content-Type: audio/wav`.
- **Automated Verification:** `test_audio_streaming_endpoint_security_and_fidelity` verified that unauthenticated streaming requests return 401, while authenticated requests stream exact byte-for-byte identical audio with valid HTTP 200 responses.
- **Status:** **PASSED**

---

### Threat Vector 9: Dataset Separation & Calibration Enrollment Gating
- **Vulnerability:** Inserting abnormal acoustic recordings into the calibration enrollment split, poisoning the autoencoder's residual whitening baseline.
- **Audited Defense:** In `assign_dataset_role`:
  ```python
  if new_role == "calibration_enrollment" and (current_label != "normal" or rec["verification_status"] != "VERIFIED"):
      raise DatasetRoleConflictException(
          "Cannot assign role 'calibration_enrollment' to an abnormal recording or unverified recording. "
          "Enrollment audio must represent confirmed healthy normal baseline operation."
      )
  ```
- **Automated Verification:** `test_calibration_enrollment_strictly_gated_to_verified_normal` verified:
  - Unlabelled recording -> Assign calibration enrollment -> **Blocked with 422**
  - Abnormal recording -> Assign calibration enrollment -> **Blocked with 422**
  - Verified normal recording -> Assign calibration enrollment -> **Accepted with 200**
- **Status:** **PASSED**

---

### Threat Vector 10: Sealed Manifest Immutability & Disk-Level Tamper Detection
- **Vulnerability:** Silently replacing audio files on disk or reassigning recordings out of held-out benchmark splits.
- **Audited Defense:**
  - Manifest sealing computes a combined SHA-256 hash over the lexicographically sorted hashes of member recordings.
  - Once sealed (`is_sealed = 1`), database integrity constraints reject role transitions, label edits, or recording deletion with HTTP 409 Conflict (`MANIFEST_ALREADY_SEALED`).
  - Verification API (`GET /api/v1/field-data/manifests/{id}/verify`) reads physical disk files, computes their live SHA-256 hashes, and compares against manifest records.
- **Automated Verification:** `test_sealed_manifest_immutability_and_tamper_detection`:
  1. Created and sealed a manifest with a normal clip.
  2. Attempted to change the recording's role to `training` -> **Blocked with 409 Conflict**.
  3. Verified manifest with clean file -> **Valid, 0 tampered files**.
  4. Deliberately altered 1 byte in the physical WAV file on disk.
  5. Called verify endpoint -> **Immediate detection: `is_valid: false`, `tamper_detected: true`, identifying the corrupted recording ID**.
- **Status:** **PASSED**

---

## 3. OWASP Top 10 Security Mapping

| OWASP Risk Category | Stetho Vulnerability Risk | Implemented Architecture Defense | Audit Finding |
|---|---|---|---|
| **A01: Broken Access Control** | Viewers/Inspectors altering roles or manifests | RBAC middleware + service-level role verification | **MITIGATED** |
| **A02: Cryptographic Failures** | Storing unhashed or forgeable audio files | SHA-256 cryptographic hashing on all ingestions & manifests | **MITIGATED** |
| **A03: Injection** | SQL injection in metadata queries or Path Traversal | Parameterized SQLite queries + regex character stripping | **MITIGATED** |
| **A04: Insecure Design** | Contaminating calibration baselines with defects | Strict healthy-only gating on calibration enrollment | **MITIGATED** |
| **A05: Security Misconfiguration** | Insecure session cookies or open CORS | HTTP-only cookies, protocol-aware Secure flag, strict origins | **MITIGATED** |
| **A06: Vulnerable Components** | Outdated or insecure Python/Node dependencies | Clean `npm audit` and pinned `.venv` dependency lockfiles | **MITIGATED** |
| **A07: Identification Failures** | Session hijacking or brute force login | Bcrypt password hashing, timed JWT tokens | **MITIGATED** |
| **A08: Software & Data Integrity** | Out-of-band disk modification of evaluation data | Byte-level SHA-256 live re-verification on manifest endpoints | **MITIGATED** |
| **A09: Logging & Monitoring Failures** | Untracked diagnostic alterations | Multi-revision audit history with mandatory evidence tracking | **MITIGATED** |
| **A10: Server-Side Request Forgery** | Insecure upstream backend proxies | Next.js proxy strictly resolves internal `http://127.0.0.1:8000` | **MITIGATED** |

---

## 4. Final Security Certification

The Stetho Field Data Collection, Annotation, and Pilot Readiness subsystems demonstrate comprehensive defense-in-depth:
- Zero unauthenticated attack surfaces on state-mutating endpoints.
- Robust cryptographic immutability preventing both data poisoning and benchmark manipulation.
- Certified passing on all 12 automated security penetration test suites.

**Audit Status:** `GOVERNANCE & SECURITY AUDIT PASSED`

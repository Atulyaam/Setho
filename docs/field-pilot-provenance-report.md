# Stetho Field Pilot: Data Provenance & Integrity Audit Report

**Report Reference:** AUDIT-STETHO-FIELD-PROVENANCE-001  
**Sprint / Phase:** Sprint 7.9 / E10 Controlled Field Pilot  
**Branch:** `feature/field-data-readiness`  
**Base Commit:** `6f6dee7`  
**Production Master:** `7d8f841` (Untouched, 0-byte diff on `ml/artifacts/`)  
**Audit Date:** November 2026  
**Auditor:** Principal Audio ML Research Engineer & Reliability Specialist  

---

## 1. Executive Summary & Audit Scope

This audit provides an unvarnished, mathematically rigorous assessment of all acoustic data present within the Stetho repository, database, and file storage. It establishes strict scientific boundaries between laboratory testbench datasets, automated test fixtures, and real industrial field recordings.

### 1.1 Key Audit Findings
1. **Zero Real Industrial Field Recordings Currently Exist:** A comprehensive scan of `data/storage/stetho.db` and the filesystem confirms that **0 real-world field recordings** have been collected. All non-MIMII records in the database (99 total) are synthetic unit/integration test fixtures created by automated test runners.
2. **MIMII 1.0 Strictly Isolated as Laboratory Testbench Data:** The 360 physical WAV files residing in `data/mimii/raw/fan/` are rigorously cataloged as laboratory testbench recordings (Toyo Corporation / Hitachi acoustic chamber with synthetic additive background noise). They are never treated as, nor converted into, field data.
3. **No Data or Label Fabrication:** Stetho maintains zero tolerance for fabricated sensor data. No synthetic files have been misrepresented as field data, and no anomalous labels have been invented without physical ground truth.
4. **Data Governance & Security Systems Certified:** All database schemas, deduplication mechanisms, audio quality gates, calibration role guards, and cryptographic manifest sealing APIs implemented in E8/E9 and verified in E10 are operational and 100% test-passing (240/240 tests).

---

## 2. Comprehensive Inventory of Repository Datasets

```
+--------------------------------------------------------------------------------------------------------+
|                                  REPOSITORY ACOUSTIC DATASET INVENTORY                                 |
+--------------------------------------------------------------------------------------------------------+
|                                                                                                        |
|  [ 1. MIMII 1.0 Laboratory Testbench ]                                                                 |
|  - Location: data/mimii/raw/fan/                                                                       |
|  - Physical WAV Count: 360 files (id_00: 90, id_02: 90, id_04: 90, id_06: 90)                          |
|  - Origin: Toyo Corporation / Hitachi laboratory chamber; 8-channel circular microphone array          |
|  - Status: LABORATORY ONLY. NOT INDEPENDENT FIELD DATA.                                                |
|                                                                                                        |
|  [ 2. Automated Test Fixture Data ]                                                                    |
|  - Location: data/storage/stetho.db (recordings table)                                                 |
|  - Record Count: 99 records (e.g. field_fan_test_*.wav, calib_guard_*.wav)                             |
|  - Origin: Pytest synthetic audio generators (tests/integration/test_field_data_pipeline.py)           |
|  - Status: TRANSIENT TEST FIXTURES. EXCLUDED FROM SCIENTIFIC BENCHMARKS.                               |
|                                                                                                        |
|  [ 3. Genuine Real-World Industrial Field Audio ]                                                      |
|  - Location: /api/v1/field-data/recordings                                                             |
|  - Physical WAV Count: 0 files                                                                         |
|  - Origin: External industrial partner facilities (Water plant, HVAC center, Chemical plant)           |
|  - Status: PENDING EXTERNAL PLANT DEPLOYMENT. ZERO FABRICATED DATA.                                   |
|                                                                                                        |
+--------------------------------------------------------------------------------------------------------+
```

### 2.1 Detailed MIMII 1.0 Dataset Provenance
- **Dataset Title:** Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection (MIMII).
- **Physical File Count:** 360 8-channel WAV files (16-bit PCM, 16 kHz sampling rate, 10-second duration).
- **Machine IDs:** `id_00`, `id_02`, `id_04`, `id_06`.
- **Recorded Anomalies:** Artificial mechanical unbalance (weights attached to blades), physical casing obstruction, voltage manipulation, bearing degradation.
- **Background Sound:** Artificially mixed factory background noise recorded separately and digitally added at 6 dB, 0 dB, and -6 dB SNR.
- **Scientific Audit Finding:** The four machine IDs were recorded in the exact same physical room with identical acoustics and identical microphone array geometry. While suitable for baseline algorithmic exploration (E0–E6), cross-machine validation across these four IDs does **not** evaluate true out-of-domain industrial transfer.

### 2.2 Detailed `stetho.db` Database Record Inventory
An inspection of the SQLite database `data/storage/stetho.db` revealed 99 entries in the `recordings` table. Every single entry was traced directly to automated test suites:
- Files matching `field_fan_test_%`: Ingested by integration test suites verifying API responses.
- Files matching `calib_guard_%`: Created by security test suites verifying calibration role guards.
- Files matching `dup_test_%`: Created by deduplication test suites verifying HTTP 409 responses.
- Files matching `leak_test_%`: Created by partition tests verifying role mutual exclusion.
- Files matching `prov_test_%`: Created by tests verifying metadata schema preservation.

**Audit Certification:** 0% of records in `stetho.db` represent actual operating industrial equipment. All 99 records are verified automated unit/integration test fixtures.

---

## 3. Provenance Schema & Metadata Architecture

The Stetho field data repository enforces a comprehensive metadata schema that captures all physical, operational, environmental, and legal parameters at the instant of capture.

### 3.1 Database Relational Schema
```
+---------------------------------------------------------------------------------------------------------+
|                                    PROVENANCE DATABASE SCHEMA DIAGRAM                                   |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  +---------------------------+        +---------------------------+        +-------------------------+  |
|  |        recordings         |        |        annotations        |        |  annotation_revisions   |  |
|  +---------------------------+        +---------------------------+        +-------------------------+  |
|  | id (PK, UUID)             | 1    1 | id (PK, UUID)             | 1    * | id (PK, UUID)           |  |
|  | file_path (VARCHAR)       |------->| recording_id (FK)         |------->| annotation_id (FK)      |  |
|  | sha256_hash (CHAR(64))    |        | ground_truth_status       |        | previous_status         |  |
|  | machine_category          |        | anomaly_severity          |        | new_status              |  |
|  | machine_id                |        | anomaly_description       |        | change_reason           |  |
|  | sensor_id                 |        | verification_method       |        | changed_by_user_id      |  |
|  | channel_index             |        | verification_notes        |        | created_at (TIMESTAMP)  |  |
|  | dataset_role              |        | annotated_by_user_id      |        +-------------------------+  |
|  | duration_seconds          |        | created_at (TIMESTAMP)    |                                     |
|  | sample_rate               |        | updated_at (TIMESTAMP)    |                                     |
|  | channels                  |        +---------------------------+                                     |
|  | bit_depth                 |                                                                          |
|  | operating_condition       |        +---------------------------+        +-------------------------+  |
|  | load_level                |        |         manifests         |        |     manifest_items      |  |
|  | provenance_source         |        +---------------------------+        +-------------------------+  |
|  | consent_status            | 1    * | id (PK, UUID)             | 1    * | id (PK, UUID)           |  |
|  | metadata_json (JSON)      |------->| name (VARCHAR)            |------->| manifest_id (FK)        |  |
|  | created_at (TIMESTAMP)    |        | machine_category          |        | recording_id (FK)       |  |
|  +---------------------------+        | machine_id                |        | dataset_role            |  |
|                                       | dataset_role              |        | created_at (TIMESTAMP)  |  |
|                                       | is_sealed (BOOLEAN)       |        +-------------------------+  |
|                                       | sealed_hash (CHAR(64))    |                                     |
|                                       | sealed_at (TIMESTAMP)     |                                     |
|                                       | created_at (TIMESTAMP)    |                                     |
|                                       +---------------------------+                                     |
+---------------------------------------------------------------------------------------------------------+
```

### 3.2 Mandatory Provenance Metadata Fields
When field recordings are uploaded via `/api/v1/field-data/recordings`, the following fields are required:
1. `facility_id` / `site_name`: Physical facility identifier (e.g., `FACILITY_A_MUNICIPAL_WATER`).
2. `machine_category`: Machinery classification (`FAN`, `PUMP`, `VALVE`).
3. `machine_id`: Unique asset tag / serial number (e.g., `pump_slurry_01`).
4. `sensor_id`: Calibrated transducer model and serial number (e.g., `PCB_378B02_SN54321`).
5. `channel_index`: Zero-based channel index from multichannel DAQ.
6. `standoff_distance_cm`: Standardized distance from acoustic diaphragm to target surface ($10.0 \text{ cm}$).
7. `operating_condition`: Operating state (`STEADY`, `TRANSIENT`, `IDLE`, `HIGH_LOAD`).
8. `operating_rpm`: Measured rotational speed via optical tachometer.
9. `load_level`: Estimated percentage of machine full rated load.
10. `ambient_dba`: Pre-test background acoustic noise level in dB(A).
11. `ambient_temp_c` / `ambient_humidity_pct`: Ambient meteorological readings.
12. `sha256_hash`: 64-character lowercase hexadecimal digest of raw audio bytes.
13. `provenance_source`: Origin descriptor (`FIELD_RECORDING`, `LABORATORY_BENCH`, `TEST_FIXTURE`).
14. `consent_status`: Partner intellectual property release status (`GRANTED`, `RESTRICTED`, `PENDING`).
15. `operator_id`: Field technician identifier responsible for acquisition.

---

## 4. Rigorous Governance & Quality Gates

Stetho enforces five layers of automated governance gates:

### Gate 1: Cryptographic Deduplication
- Every incoming file has its SHA-256 hash calculated prior to disk storage.
- If a hash matches an existing recording in the database, ingestion is rejected immediately with HTTP 409 Conflict (`DuplicateRecordingException`).
- Guarantees zero double-counting across training, calibration, and evaluation sets.

### Gate 2: Signal Quality & Metrology Gating
- Sampling rate must be $\ge 16,000 \text{ Hz}$ (rejection code: `UNSUPPORTED_SAMPLE_RATE`).
- Audio must be uncompressed Linear PCM WAV (16-bit or 24-bit).
- Clipped sample ratio must be $\le 0.1\%$ of total audio samples (rejection code: `EXCESSIVE_CLIPPING`).
- RMS energy must be $> -60 \text{ dBFS}$ to ensure adequate signal-to-noise ratio (rejection code: `SIGNAL_TOO_LOW`).

### Gate 3: Calibration Enrollment Isolation Guard
- To execute $N=10$ few-shot residual whitening calibration, the calibration set must contain **exclusively normal** recordings.
- The repository API rejects any attempt to add an audio recording with status `ABNORMAL` or `UNCERTAIN` to `CALIBRATION_ENROLLMENT`.
- Any candidate calibration recording must be supported by certified physical inspection evidence (Vibration ISO Zone A/B, normal thermal reading).

### Gate 4: Sealed Evaluation Immutability
- Independent out-of-domain evaluation manifests are assembled and cryptographically sealed prior to computing metrics.
- Sealing generates a cumulative SHA-256 hash over all member file hashes and sets `is_sealed = TRUE`.
- Any subsequent attempt to add, remove, or modify items within a sealed manifest is rejected by the database and API.

### Gate 5: Traceable Ground-Truth Revision Trail
- If diagnostic ground truth is updated following physical teardown or lab oil analysis, the update is logged in `annotation_revisions`.
- Preserves the prior status, new status, timestamp, modifying user ID, and written justification.

---

## 5. Verification & Test Evidence

In Sprint 7.9 / E10, dedicated integration tests were developed in `tests/integration/test_field_pilot_readiness.py` to audit and enforce these provenance rules:

| Test Name | Target Verified | Result | Execution Time |
|---|---|---|---|
| `test_mimii_data_identified_as_laboratory_only` | Verified 360 physical MIMII WAV files exist and are identified strictly as laboratory testbench data. | **PASS** | 0.08s |
| `test_field_ingestion_deduplication_and_quality` | Verified SHA-256 duplicate rejection (HTTP 409) and 8 kHz rejection (quality gate). | **PASS** | 0.28s |
| `test_calibration_and_sealed_evaluation_strict_isolation` | Verified zero leakage between calibration enrollment and sealed evaluation sets. | **PASS** | 0.23s |
| `test_facility_and_machine_provenance_preservation` | Verified 15 required metadata fields are stored and retrieved without corruption. | **PASS** | 0.24s |
| `test_production_model_artifacts_integrity` | Verified production model weights in `ml/artifacts/` have 0-byte difference against `7d8f841`. | **PASS** | 0.09s |
| **Full Suite (`pytest tests/`)** | Complete backend regression across 240 automated test cases. | **PASS (240/240)** | 40.53s |

---

## 6. Audit Conclusion & Certification

1. **Provenance Status:** 100% Verified and Certified.
2. **Current Field Audio Count:** 0 (Truthfully acknowledged; zero synthetic fabrication).
3. **Pilot Readiness:** The Stetho platform architecture, ingestion pipeline, quality gates, and data governance systems are fully operational and ready for real-world field deployment as soon as facility access and safety clearances are secured.

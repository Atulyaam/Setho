# Stetho Field Pilot Readiness & Operational Evaluation Report

**Sprint Reference:** Sprint 7.7 / E8: Field Data Collection & Pilot Readiness  
**Base Commit:** `c5c4436` (Sprint 7.6 / E7 Independent Validation complete)  
**Branch:** `feature/field-data-readiness`  
**Production Master:** `7d8f841` (Untouched, 0-byte diff)  
**Evaluation Status:** Approved for Controlled Field Pilot  
**Final Verdict:** `FIELD DATA WORKFLOW READY`  

---

## 1. Executive Summary

Sprint 7.7 / E8 establishes the complete software, operational, and governance foundation required for deploying Stetho into a controlled industrial field pilot.

Prior research phases (E0 through E7) revealed that while Stetho's deep neural autoencoders achieve strong mathematical performance on synthetic test splits, **genuine cross-machine and out-of-domain industrial validation has not yet occurred**. Specifically:
1. Zero-shot transfer to unseen machinery without calibration exhibits degraded detection quality (AUC ~0.50–0.60).
2. Spatial-spectral residual whitening with $N=10$ enrollment recordings dramatically improves cross-machine performance (AUC 0.77–0.89 on MIMII), but has never been validated outside the original laboratory test bench.
3. The original MIMII 1.0 dataset shares identical room acoustics, microphone arrays, and machine housings across all IDs, precluding it from serving as a true independent benchmark.

To prepare Stetho for real industrial deployment without fabricating validation claims or premature claims of production readiness, Sprint 7.7 delivers:
- **Cryptographic Audio Ingestion Engine:** Automated SHA-256 deduplication and audio quality gating (rejecting sub-16 kHz, clipping $> 0.1\%$, and acoustic silence $< -60\text{ dBFS}$).
- **Multi-Revision Ground-Truth Annotation:** Operator diagnosis tracking (`normal`, `abnormal`, `uncertain`) with mandatory multi-modal physical evidence sources and complete audit history.
- **Dataset Isolation & Manifest Sealing:** Strict separation of dataset roles (`training`, `development`, `calibration_enrollment`, `sealed_evaluation`), preventing cross-contamination and tamper-sealing benchmark datasets with SHA-256 checksums.
- **Pilot Command Dashboard:** Next.js operator interface displaying ingestion KPIs, category/machine distributions, rejection logs, and an unmissable warning banner: `INSUFFICIENT INDEPENDENT EVIDENCE` until sealed evaluation criteria are satisfied.
- **Exportable Field Protocol:** Formal checklist and standard operating procedure for plant technicians.

---

## 2. Operational Architecture

```
  [ Field Technician / DAQ ]
              |
              | 1. Upload WAV Audio (16-bit PCM, ≥16 kHz, 10s) + Metadata
              v
  +---------------------------------------------------------------+
  | FASTAPI FIELD DATA ENGINE (/api/v1/field-data)                |
  |                                                               |
  |  +------------------------+  +-----------------------------+  |
  |  | SHA-256 Deduplication  |  | Audio Quality Gating        |  |
  |  | - Immediate hash calc  |  | - Sample rate ≥ 16 kHz      |  |
  |  | - Duplicate -> HTTP 409|  | - Peak amp ≤ 0.98           |  |
  |  | - No silent overwrite  |  | - Silence check (>-60 dBFS) |  |
  |  +------------------------+  +-----------------------------+  |
  |                                                               |
  |  +---------------------------------------------------------+  |
  |  | Dataset Governance & Role Separation Guardrails         |  |
  |  | - calibration_enrollment: Strictly Normal Clips         |  |
  |  | - sealed_evaluation: Locked against Reassignment      |  |
  |  +---------------------------------------------------------+  |
  +---------------------------------------------------------------+
              |                                     |
              v                                     v
  +------------------------+             +------------------------+
  | SQLite Storage         |             | Next.js Field Portal   |
  | - field_recordings     |             | - Dashboard Telemetry  |
  | - recording_annotations|             | - Warning Alert Banner |
  | - dataset_manifests    |             | - Operator Annotation  |
  | - field_import_history |             | - Protocol Download    |
  +------------------------+             +------------------------+
```

### 2.1 Storage & Schema Additions

Four dedicated relational tables manage field data lifecycle within `services/api/src/stetho_api/db/database.py`:

1. `field_recordings`: Stores physical file paths, SHA-256 cryptographic hashes, machine metadata, audio telemetry (sample rate, channels, RMS dBFS, peak amplitude, clipping ratio), dataset roles, and ground-truth verification statuses.
2. `recording_annotations`: Append-only, multi-revision audit table capturing every diagnostic diagnosis, annotator user ID, revision number, physical evidence source, and technical notes.
3. `dataset_manifests`: Frozen manifest records maintaining immutable JSON arrays of member recordings, sealed timestamps, and top-level SHA-256 checksums computed across member hashes.
4. `field_import_history`: Immutable log of every upload attempt (`SUCCESS`, `DUPLICATE_REJECTED`, `VALIDATION_FAILED`), preserving audit trails for compliance.

---

## 3. Automated Quality Gating & Data Hygiene

Every audio submission undergoes deterministic quality gating before ingestion:

1. **Format Validation:** Audio must be decodable as Linear PCM WAV via `soundfile`.
2. **Bandwidth Compliance:** Sample rate must be $\ge 16,000\text{ Hz}$ (rejecting legacy telephony audio that lacks high-frequency acoustic defect signatures).
3. **Clipping Guard:** Normalized peak amplitude must not exceed $0.98$ and clipping ratio must be $\le 0.001$ ($0.1\%$).
4. **Signal Energy Guard:** Signal RMS energy must be $\ge -60.0\text{ dBFS}$ to eliminate dead channels, broken cables, or muted transducers.
5. **Deduplication:** Hash matches against existing records are immediately rejected with HTTP 409 Conflict.

---

## 4. Dataset Governance & Ground-Truth Annotation

### 4.1 Strict Dataset Role Separation

To eliminate data leakage, Stetho enforces five distinct dataset roles:
- `unassigned`: Default state for newly imported recordings pending operator review.
- `training`: Audio utilized strictly for unsupervised model training (normal baseline only).
- `development`: Audio utilized for algorithm tuning, hyperparameter exploration, and pilot development.
- `calibration_enrollment`: Pristine healthy recordings ($N=10$ per machine) used exclusively to calibrate the spatial-spectral residual whitening transformation. **Abnormal recordings are mathematically barred from this role.**
- `sealed_evaluation`: Independent, held-out recordings for statistical benchmarking. **Once added to a sealed manifest, recordings cannot be reassigned or modified.**

### 4.2 Multi-Modal Ground-Truth Triangulation

To prevent circular reasoning, Stetho strictly prohibits using model anomaly scores or predictions as ground-truth labels. Technicians must provide independent physical evidence:
- `VISUAL_INSPECTION`: Borescope, stethoscope, or visual teardown.
- `VIBRATION_ORDER_ANALYSIS`: Accelerometer spectral analysis per ISO 10816-3.
- `TEARDOWN_INSPECTION`: Bearing or impeller physical overhaul.
- `THERMAL_SCAN`: Radiometric thermal imaging.
- `OIL_ANALYSIS`: Spectrometric or ferrographic lubricant assay.
- `OPERATIONAL_LOG`: Supervisory maintenance logs.

---

## 5. Independent Validation Status & Dashboard Warning

### 5.1 Verification Status

> [!WARNING]
> **Independent Validation Has NOT Yet Occurred.**
> 
> The Stetho platform has been evaluated extensively on the MIMII 1.0 benchmark dataset (achieving AUCs up to 0.89 under spatial-spectral calibration). However, this benchmark does not constitute independent validation. The current models must be treated as research prototypes until validated against genuine, independently collected field data.

### 5.2 Pilot Dashboard Gating

The Stetho web dashboard displays a prominent persistent amber alert banner whenever sealed independent evaluation data is insufficient:
- **Threshold Criterion:** $\ge 10$ verified normal and $\ge 10$ verified abnormal recordings permanently sealed in a `sealed_evaluation` manifest.
- **Current Status:** Gated (`INSUFFICIENT_INDEPENDENT_EVIDENCE`). Field data collection is actively underway.
- **Enforcement:** Production model promotion is blocked until independent evaluation passes all statistical acceptance criteria.

---

## 6. Pilot Readiness Verification Matrix

| Subsystem | Requirement | Implementation | Status |
| :--- | :--- | :--- | :--- |
| **API Ingestion** | WAV upload with metadata, SHA-256 deduplication, quality gating | `routers/field_data.py`, `POST /api/v1/field-data/import` | **PASS** |
| **Deduplication** | Reject duplicate file hashes immediately with HTTP 409 | `FieldDataRepo.import_recording()` | **PASS** |
| **Quality Filtering** | Reject $<16\text{ kHz}$, clipping $>0.1\%$, silence $<-60\text{ dBFS}$ | `FieldDataRepo._validate_audio_quality()` | **PASS** |
| **Annotation Audit** | Multi-revision ground-truth logging with evidence sources | `POST .../annotate`, `GET .../annotations` | **PASS** |
| **Calibration Purity**| Bar abnormal recordings from `calibration_enrollment` | `FieldDataRepo.assign_dataset_role()` | **PASS** |
| **Manifest Sealing** | Immutable manifests with sorted SHA-256 checksums | `POST .../manifests`, `FieldDataRepo.create_or_seal_manifest()` | **PASS** |
| **Operator UI** | Field data dashboard, KPI cards, warning banner, modals | `apps/web/src/app/field-data/page.tsx` | **PASS** |
| **Documentation** | Field collection checklist, readiness report, governance policy | `docs/` | **PASS** |
| **Unit & Integration** | Automated test suite covering all workflows and edge cases | 14 field tests, 223 total suite tests | **PASS (100%)** |

---

## 7. Official Final Decision

```
================================================================================
FINAL SPRINT 7.7 / E8 VERDICT: FIELD DATA WORKFLOW READY
================================================================================
The field data collection, quality gating, multi-revision annotation, and
cryptographic manifest governance workflows are fully operational and verified.

Stetho is officially cleared for controlled field pilot data collection.
Production model deployment remains conditioned upon completing sealed
independent evaluation using the newly established ingestion pipeline.
================================================================================
```

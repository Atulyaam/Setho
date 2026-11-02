# Stetho Acoustic Data Governance & Dataset Lifecycle Standard

**Policy Document ID:** POL-ML-008  
**Version:** 1.0.0  
**Effective Date:** 2026-11-02  
**Scope:** All Acoustic Datasets, Machine Calibration Profiles, Ingestion Workflows, and Model Evaluation Manifests  
**Applies To:** Data Scientists, ML Engineers, Reliability Technicians, QA Auditors  

---

## 1. Purpose & Principles

This standard establishes mandatory governance controls for all acoustic audio recordings acquired, stored, annotated, or utilized by the Stetho platform.

Industrial acoustic anomaly detection is susceptible to subtle data leakage, confirmation bias, and sensor overfitting. The fundamental principles of this governance policy are:

1. **Absolute Provenance:** Every acoustic recording must have an immutable SHA-256 cryptographic hash, known acquisition hardware, physical placement details, and verified operator attribution.
2. **Deterministic Role Separation:** Audio recordings must be explicitly classified into strict dataset roles. Cross-contamination between training, calibration, and evaluation sets is prohibited.
3. **Purity of Calibration:** Calibration enrollment recordings must be strictly verified as healthy (normal). Anomaly detection thresholds must never be calibrated using abnormal recordings.
4. **Independent Ground-Truth:** Model predictions must never be used to label training or validation data. Ground-truth classifications must be established through independent, multi-modal physical evidence.
5. **Immutable Sealing:** Evaluation benchmarks must be cryptographically sealed prior to model assessment to prevent data snooping, post-hoc threshold tuning, or silent dataset drift.
6. **Integrity of Claims:** The MIMII 1.0 dataset shall never be represented or claimed as an independent validation dataset.

---

## 2. Dataset Roles & Isolation Boundaries

Every recording in Stetho must be assigned to exactly one dataset role:

```
+------------------------------------------------------------------------------------+
|                                 STETHO AUDIO DATASET ROLES                         |
+---------------------+-------------------+---------------------+--------------------+
| ROLE                | PURPOSE           | ADMISSIBLE LABELS   | MUTABILITY         |
+---------------------+-------------------+---------------------+--------------------+
| unassigned          | Ingestion staging | unlabelled, normal, | Mutable            |
|                     | pending review    | abnormal, uncertain |                    |
+---------------------+-------------------+---------------------+--------------------+
| training            | Self-supervised & | Strictly NORMAL     | Mutable until      |
|                     | representation    |                     | training manifest  |
|                     | learning          |                     | sealed             |
+---------------------+-------------------+---------------------+--------------------+
| development         | Hyperparameter    | normal, abnormal,   | Mutable            |
|                     | exploration &     | uncertain           |                    |
|                     | architecture tests|                     |                    |
+---------------------+-------------------+---------------------+--------------------+
| calibration_        | Machine-specific  | Strictly NORMAL     | Locked upon asset  |
| enrollment          | spatial-spectral  | (Abnormal strictly  | calibration        |
|                     | residual whitening| prohibited)         |                    |
+---------------------+-------------------+---------------------+--------------------+
| sealed_evaluation   | Independent blind | Fully verified      | IMMUTABLE          |
|                     | validation        | normal & abnormal   | (Permanently       |
|                     | benchmarks        |                     | locked when sealed)|
+---------------------+-------------------+---------------------+--------------------+
```

### 2.1 Role-Specific Guardrails

1. **Calibration Enrollment Guardrail:**
   - Attempting to assign a recording with `current_label = 'abnormal'` to `calibration_enrollment` is rejected at the API level (HTTP 422 `DATASET_ROLE_CONFLICT`).
   - If a previously enrolled calibration clip is subsequently discovered to be abnormal, the machine's calibration status is invalidated, reverting to `uncalibrated`.

2. **Sealed Evaluation Guardrail:**
   - Once a recording is incorporated into a sealed manifest (`is_sealed = 1`), its dataset role is locked.
   - Any attempt to modify its role or remove it from the manifest is rejected with HTTP 409 Conflict (`MANIFEST_ALREADY_SEALED`).

---

## 3. Cryptographic Manifest Sealing

To eliminate dataset leakage and test-set tampering, all official evaluation benchmarks must be frozen into manifests:

### 3.1 Checksum Generation Algorithm

A manifest's cryptographic checksum is calculated deterministically:
1. Extract all member recordings $(R_1, R_2, \dots, R_k)$ in the manifest.
2. Retrieve the SHA-256 hash of each audio recording payload.
3. Sort the tuples by recording ID alphabetically: $\text{sort}((R_i, \text{hash}_i))$.
4. Format each entry as `"recording_id:sha256_hash\n"`.
5. Compute the top-level SHA-256 digest across the concatenated string:
   $$\text{Checksum}_{\text{manifest}} = \text{SHA256}\left(\bigoplus_{i=1}^k \left(R_i \mathbin{\Vert} \text{hash}_i \mathbin{\Vert} \text{"\n"}\right)\right)$$

This guarantees that:
- Any modification, replacement, or deletion of a recording invalidates the manifest checksum.
- Benchmark results are tied to a unique, immutable 64-character hexadecimal signature.

---

## 4. Ground-Truth Annotation & Multi-Revision Audit

### 4.1 Prohibition of Circular / Pseudo-Labeling

> [!CAUTION]
> **Prohibition of Self-Labeling:**
> 
> Under no circumstances may model anomaly scores, autoencoder reconstruction errors, or classifier outputs be used as ground-truth labels. Labeling data based on model outputs introduces circular self-reinforcement and masks blind spots.

### 4.2 Multi-Modal Defect Triangulation

All ground-truth labels must be backed by documented evidence from at least one approved source:

- **`VISUAL_INSPECTION`**: High-resolution optical, stethoscope, or borescope inspection.
- **`VIBRATION_ORDER_ANALYSIS`**: ISO 10816-3 accelerometer analysis confirming spectral velocity thresholds ($> 4.5\text{ mm/s RMS}$) or harmonic defect orders (1X, 2X, BPFO, BPFI).
- **`TEARDOWN_INSPECTION`**: Overhaul inspection confirming spalling, pitting, or galling.
- **`THERMAL_SCAN`**: Radiometric thermography showing bearing overheating ($\Delta T > 20^\circ\text{C}$).
- **`OIL_ANALYSIS`**: Lubricant spectrometry or ferrographic particle analysis.
- **`OPERATIONAL_LOG`**: Supervisory SCADA trip or maintenance work order.
- **`OTHER_GROUND_TRUTH`**: Documented third-party diagnostic assessment.

### 4.3 Multi-Revision Audit Trail

Annotations are stored in `recording_annotations` as an immutable append-only ledger:
- Every update generates a new revision number ($1, 2, 3, \dots$).
- The audit record retains the technician's user ID, timestamp, evidence source, diagnostic reason, and detailed notes.
- Historical revisions are never deleted or overwritten, ensuring complete regulatory traceability.

---

## 5. Benchmark Integrity: Status of MIMII 1.0

### 5.1 Clear Statement of Limitation

The ToyADMOS and MIMII 1.0 datasets were instrumental during initial algorithm development and architecture search (E0 through E6). However:

> [!WARNING]
> **MIMII 1.0 CANNOT SERVE AS AN INDEPENDENT BENCHMARK.**
> 
> All fan machine IDs in MIMII 1.0 (`id_00`, `id_02`, `id_04`, `id_06`) were recorded in the same acoustic chamber using identical microphone array fixtures and identical room reflections. Therefore, cross-machine evaluation within MIMII evaluates variation in machine geometry, but NOT acoustic domain generalization or factory environmental robustness.
> 
> True independent validation requires physical data collection in operating factories across diverse industrial environments.

---

## 6. Audio Data Quality & Hygiene Gates

All audio ingested into Stetho must satisfy mandatory quality thresholds:

| Quality Metric | Lower Bound | Upper Bound | Action on Violation |
| :--- | :--- | :--- | :--- |
| **Sample Rate** | $16,000\text{ Hz}$ | $96,000\text{ Hz}$ | Reject upload (HTTP 422). Log failure to audit history. |
| **Duration** | $1.0\text{ s}$ | $60.0\text{ s}$ | Reject upload (HTTP 422). |
| **Peak Amplitude** | — | $0.98$ (Normalized) | If clipped ($>0.1\%$), reject upload (HTTP 422). |
| **RMS Energy** | $-60.0\text{ dBFS}$ | $0.0\text{ dBFS}$ | If $<-60\text{ dBFS}$, reject as silent/disconnected (HTTP 422). |
| **Payload Hash** | Unique SHA-256 | Unique SHA-256 | If duplicate, reject with HTTP 409 Conflict. Never overwrite. |

---

## 7. Governance Compliance Auditing

1. **Automated Continuous Monitoring:** The Stetho dashboard continuously evaluates sealed evaluation counts against readiness criteria ($\ge 10$ normal, $\ge 10$ abnormal sealed clips).
2. **Audit Logging:** Every rejected import, whether due to duplicate SHA-256 or audio quality gating, is permanently logged to `field_import_history`.
3. **Model Promotion Gate:** The release pipeline blocks any model promotion to production until an evaluation manifest has been sealed and verified against independent field data.

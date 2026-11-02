# Stetho Field Data Manual Acceptance Test Results Log

**Sprint Reference:** Sprint 7.8 / E9: Field Workflow Acceptance & Security Audit  
**Execution Date:** November 2, 2026  
**Test Environment:** Windows 11, Node.js v20.x, Python 3.11.9, Google Chrome 131 (Headless Playwright Runner)  
**Base Commit:** `cf2a8a4` | **Active Branch:** `feature/field-data-readiness`  
**Test Harness:** `tests/e2e/field-data/test_field_data_browser.py`  
**Evidence Archive:** `docs/evidence/browser_e9/` (18 PNG Screenshots Captured)  
**Overall Result:** **18 / 18 SCENARIOS PASSED (100% SUCCESS)**

---

## 1. Scope & Test Methodology

To independently verify the Stetho Field Data Ingestion, Annotation, Governance, and Dashboard workflows prior to physical deployment in an industrial plant, 18 rigorous end-to-end scenarios were executed in real headless Google Chrome.

The test harness orchestrates:
- Live FastAPI backend bound to `127.0.0.1:8000` backed by SQLite.
- Next.js production build (`next start`) bound to `127.0.0.1:3000`.
- Headless Chromium with viewport resolution $1400 \times 950$ px.
- Synthetic and fuzzed acoustic waveforms generated on-the-fly (`soundfile` PCM 16-bit).
- Step-by-step high-resolution PNG screenshot capture into `docs/evidence/browser_e9/`.

---

## 2. Test Execution Matrix (18 Scenarios)

### Scenario 1: Open Field Data Dashboard & Telemetry Inspection
- **Objective:** Verify navigation to the Field Data portal and confirm initial KPI summary rendering.
- **Action:** Authenticate as administrator via `/login`, click the navigation link to `/field-data`.
- **Observed Behavior:** Page header `Field Data Collection & Pilot Readiness` rendered. Four top telemetry cards displayed: Total Ingested Audio, Verified Normal, Verified Abnormal, and Quality Reject & Audit Count.
- **HTTP Calls:** `GET /api/v1/field-data/dashboard` (200 OK), `GET /api/v1/field-data/recordings?limit=100` (200 OK), `GET /api/v1/field-data/manifests` (200 OK).
- **Result:** **PASSED**
- **Evidence:** `docs/evidence/browser_e9/01_dashboard_open.png`

---

### Scenario 2: Ingest Valid Field Acoustic WAV Recording
- **Objective:** Ingest an authorized Linear PCM WAV recording conforming to quality standards.
- **Input:** 3.0-second 16,000 Hz single-channel sine tone ($380\text{ Hz}$, peak amplitude $0.25$, RMS $-15.1\text{ dBFS}$) representing normal fan bearing audio. Machine ID: `fan_pilot_field_01`. Category: `fan`.
- **Action:** Click `Import Field Audio`, populate file picker with `field_fan_bearing_valid.wav`, fill machine ID, click `Verify & Ingest`.
- **Observed Behavior:** Form submitted via multipart form-data. Green notification banner displayed: `Field recording field_fan_bearing_valid.wav successfully imported and verified.` Modal automatically closed and table refreshed.
- **HTTP Calls:** `POST /api/v1/field-data/import` (201 Created).
- **Result:** **PASSED**
- **Evidence:** `docs/evidence/browser_e9/02_upload_valid_wav.png`

---

### Scenario 3: Verify Metadata & Quality Telemetry in Table Row
- **Objective:** Confirm ingested audio metadata displays with exact numerical fidelity.
- **Action:** Locate table row for `fan_pilot_field_01`.
- **Observed Behavior:** Table displayed Equipment Category `FAN`, Machine Instance `fan_pilot_field_01`, Transducer `mic_sensor_01`, Sample Rate `16000 Hz`, Duration `3.0s`, Quality Badge `PASSED`, Label `UNLABELLED`, Dataset Role `UNASSIGNED`.
- **Result:** **PASSED**
- **Evidence:** `docs/evidence/browser_e9/03_recording_metadata_confirmed.png`

---

### Scenario 4: Duplicate Ingestion Rejection (Cryptographic SHA-256)
- **Objective:** Verify that identical acoustic data cannot be re-uploaded under a different filename or machine ID.
- **Input:** Re-uploading the exact same 3.0s WAV file under filename `field_fan_bearing_valid.wav` but with Machine Instance ID `fan_pilot_duplicate`.
- **Action:** Click `Import Field Audio`, attach file, click `Verify & Ingest`.
- **Observed Behavior:** Upload immediately rejected. Amber error alert displayed: `Recording with hash ... already exists as fld_rec_...`. HTTP 409 Conflict returned by backend. No duplicate row created.
- **HTTP Calls:** `POST /api/v1/field-data/import` (409 Conflict).
- **Result:** **PASSED**
- **Evidence:** `docs/evidence/browser_e9/04_duplicate_rejection.png`

---

### Scenario 5: Sub-Nyquist Sample Rate Rejection (8 kHz Audio)
- **Objective:** Verify strict quality gate rejecting audio below industrial Nyquist threshold ($\ge 16\text{ kHz}$).
- **Input:** 2.0-second 8,000 Hz WAV recording (`subnyquist_8khz.wav`).
- **Action:** Select file in import dialog, enter Machine ID `fan_subnyquist_01`, click `Verify & Ingest`.
- **Observed Behavior:** Audio quality gate failed. Error banner rendered: `Field acoustic recording sample rate must be >= 16000 Hz, got 8000 Hz`. Ingestion rejected with HTTP 422.
- **HTTP Calls:** `POST /api/v1/field-data/import` (422 Unprocessable Entity).
- **Result:** **PASSED**
- **Evidence:** `docs/evidence/browser_e9/05_sample_rate_rejection.png`

---

### Scenario 6: Acoustic Silence / Disconnected Sensor Rejection
- **Objective:** Verify quality gate rejecting silent audio clips (dead channel or unplugged transducer).
- **Input:** 2.0-second 16,000 Hz all-zero amplitude audio clip (`silent_audio.wav`, RMS energy $< -90\text{ dBFS}$).
- **Action:** Select silent file, enter Machine ID `fan_silent_01`, click `Verify & Ingest`.
- **Observed Behavior:** Audio quality gate failed. Error banner displayed: `Clip is predominantly silent (RMS: -100.0 dBFS <= -60.0 dBFS)`. Upload blocked with HTTP 422.
- **HTTP Calls:** `POST /api/v1/field-data/import` (422 Unprocessable Entity).
- **Result:** **PASSED**
- **Evidence:** `docs/evidence/browser_e9/06_silence_quality_rejection.png`

---

### Scenario 7: Inspect Recording Details in Table View
- **Objective:** Inspect detailed acoustic quality metrics and audit history in the main recordings table.
- **Action:** Close import modal, inspect the active recordings table view.
- **Observed Behavior:** Table clearly formatted with equipment badge, machine instance, sensor mount (`primary_bearing`), sample rate, RMS dBFS level, label badge, dataset role badge, and action controls (`Annotate`, `Role`, `Play`).
- **Result:** **PASSED**
- **Evidence:** `docs/evidence/browser_e9/07_table_details_inspected.png`

---

### Scenario 8: Ground-Truth Annotation (Initial Revision: Normal)
- **Objective:** Set ground-truth diagnosis to Normal with verified visual inspection evidence.
- **Action:** Click `Annotate` on `fan_pilot_field_01`. Select Diagnosis `Normal (Confirmed healthy operation)`, select Evidence Source `Visual Stethoscope / Borescope Inspection`, enter Reason `"Technician confirmed normal acoustic and vibration baseline"`, click `Commit Revision`.
- **Observed Behavior:** Annotation committed. Notification displayed: `Recording fld_rec_... ground-truth updated to NORMAL.` Table row badge updated to green `NORMAL` (`VERIFIED`).
- **HTTP Calls:** `POST /api/v1/field-data/recordings/{id}/annotate` (201 Created).
- **Result:** **PASSED**
- **Evidence:** `docs/evidence/browser_e9/08_annotate_normal.png`

---

### Scenario 9: Ground-Truth Annotation (Second Revision: Uncertain)
- **Objective:** Revise diagnosis when supervisory maintenance report indicates inconclusive findings.
- **Action:** Click `Annotate` on `fan_pilot_field_01`. Change Diagnosis to `Uncertain (Inconclusive inspection)`, change Evidence Source to `Supervisory Maintenance Log`, enter Reason `"Minor acoustic hiss reported during peak load handover"`, click `Commit Revision`.
- **Observed Behavior:** Revision committed. Notification displayed: `Recording fld_rec_... ground-truth updated to UNCERTAIN.` Table badge updated to amber `UNCERTAIN`.
- **HTTP Calls:** `POST /api/v1/field-data/recordings/{id}/annotate` (201 Created).
- **Result:** **PASSED**
- **Evidence:** `docs/evidence/browser_e9/09_revised_annotation_added.png`

---

### Scenario 10: Verify Multi-Revision Audit Trail
- **Objective:** Confirm audit log immutability and visibility of previous revision history.
- **Action:** Click `Annotate` on `fan_pilot_field_01`. Inspect the `Audit Revision History` panel in modal.
- **Observed Behavior:** Both revisions displayed in reverse chronological order:
  - `Rev #2: UNCERTAIN` (Evidence: OPERATIONAL_LOG by admin, notes: `"Minor acoustic hiss reported during peak load handover"`).
  - `Rev #1: NORMAL` (Evidence: VISUAL_INSPECTION by admin, notes: `"Technician confirmed normal acoustic and vibration baseline"`).
- **Result:** **PASSED**
- **Evidence:** `docs/evidence/browser_e9/10_annotation_history_revisions_1_and_2.png`

---

### Scenario 11: Dataset Role Assignment (Development Split)
- **Objective:** Assign recording to internal research and development dataset split.
- **Action:** Click `Role` button on `fan_pilot_field_01`. In modal, select `Development`, click `Update Role`.
- **Observed Behavior:** Role successfully updated. Notification banner: `Dataset role for fld_rec_... updated to development.` Table row badge updated to `DEVELOPMENT`.
- **HTTP Calls:** `PUT /api/v1/field-data/recordings/{id}/role` (200 OK).
- **Result:** **PASSED**
- **Evidence:** `docs/evidence/browser_e9/11_role_assigned_development.png`

---

### Scenario 12: Calibration Gating Guardrail (Abnormal Recording Blocked)
- **Objective:** Verify security boundary preventing abnormal audio from polluting calibration enrollment baselines.
- **Action:** First annotate `fan_pilot_field_01` as `Abnormal (Defect or mechanical flaw verified)` with evidence `VIBRATION_ORDER_ANALYSIS` and notes `"Verified bearing cage fracture flaw"`. Then click `Role`, select `Calibration Enrollment`, click `Update Role`.
- **Observed Behavior:** Operation strictly blocked by backend governance rules. Red error banner displayed: `Cannot assign role 'calibration_enrollment' to an abnormal recording or unverified recording. Enrollment audio must represent confirmed healthy normal baseline operation.` HTTP 422 Unprocessable Entity. Recording role unchanged.
- **HTTP Calls:** `PUT /api/v1/field-data/recordings/{id}/role` (422 Unprocessable Entity).
- **Result:** **PASSED**
- **Evidence:** `docs/evidence/browser_e9/12_calibration_enrollment_rejected.png`

---

### Scenario 13: Dataset Manifest Generation & Cryptographic Sealing
- **Objective:** Create and seal an immutable evaluation manifest.
- **Action:** Re-annotate `fan_pilot_field_01` as verified `normal`. Assign dataset role `sealed_evaluation`. Click `Seal Manifest` button. In modal, enter Manifest Name `"Sprint 7.8 Pilot Fan Benchmark"`, select Role `Sealed Evaluation`, Category `Fan`, verify `Seal immediately` toggle is checked, click `Generate Manifest`.
- **Observed Behavior:** Manifest generated and cryptographically sealed. Notification displayed: `Dataset manifest 'Sprint 7.8 Pilot Fan Benchmark' created and sealed (SHA-256: ...).`
- **HTTP Calls:** `POST /api/v1/field-data/manifests` (201 Created).
- **Result:** **PASSED**
- **Evidence:** `docs/evidence/browser_e9/13_manifest_created_and_sealed.png`

---

### Scenario 14: Confirm Sealed Manifest in Dashboard UI
- **Objective:** Inspect manifest details in the `Sealed Manifests` tab.
- **Action:** Switch to `Sealed Manifests` navigation tab.
- **Observed Behavior:** Table displayed Manifest Name `Sprint 7.8 Pilot Fan Benchmark`, Role `SEALED_EVALUATION`, Category `FAN`, Member Count `1`, Status Badge `SEALED`, Full SHA-256 Checksum, Created By `admin`, and Verification Control.
- **Result:** **PASSED**
- **Evidence:** `docs/evidence/browser_e9/14_sealed_manifest_confirmed.png`

---

### Scenario 15: Sealed Role Immutability Guardrail
- **Objective:** Verify that once a recording is sealed in an evaluation manifest, its dataset role cannot be altered.
- **Action:** Switch back to `Field Audio Recordings` tab. Locate sealed recording `fan_pilot_field_01`. Click `Role` button, select `Training`, click `Update Role`.
- **Observed Behavior:** Modification blocked by database integrity trigger. Error alert displayed: `Recording 'fld_rec_...' belongs to sealed evaluation manifest 'Sprint 7.8 Pilot Fan Benchmark' (...) and cannot be reassigned.` HTTP 409 Conflict returned. Role remained strictly `sealed_evaluation`.
- **HTTP Calls:** `PUT /api/v1/field-data/recordings/{id}/role` (409 Conflict).
- **Result:** **PASSED**
- **Evidence:** `docs/evidence/browser_e9/15_sealed_recording_locked.png`

---

### Scenario 16: Browser Refresh Persistence Verification
- **Objective:** Confirm full state hydration and UI persistence following client-side page reload.
- **Action:** Execute `page.reload()` in Playwright. Wait for DOM hydration.
- **Observed Behavior:** Page reloaded smoothly. All recordings, labels (`NORMAL`), roles (`sealed_evaluation`), and manifest records re-fetched and rendered identically without session drop or data loss.
- **Result:** **PASSED**
- **Evidence:** `docs/evidence/browser_e9/16_browser_refresh_persistence.png`

---

### Scenario 17: Backend Server Restart Persistence Verification
- **Objective:** Verify physical database persistence across complete backend process shutdown and reboot.
- **Action:** Send `SIGTERM` to the Uvicorn FastAPI backend process, wait for termination, spawn a fresh Uvicorn process on `127.0.0.1:8000`, verify health check, and reload the browser page.
- **Observed Behavior:** FastAPI reconnected cleanly. Browser re-fetched `/api/v1/field-data/dashboard` and `/api/v1/field-data/recordings`. Every recording, annotation history revision, and sealed manifest persisted intact from disk SQLite storage.
- **Result:** **PASSED**
- **Evidence:** `docs/evidence/browser_e9/17_backend_restart_persistence.png`

---

### Scenario 18: Mandatory Governance Alert Banner Verification
- **Objective:** Verify prominent rendering of the unmissable amber alert banner warning operators of insufficient independent evidence.
- **Action:** Inspect the top viewport of `/field-data`.
- **Observed Behavior:** Prominent amber alert box with warning icon and badge `GOVERNANCE GATE ACTIVE` displayed:
  `INSUFFICIENT INDEPENDENT EVIDENCE: Controlled Field Pilot in Progress`
  Subtext correctly stated that laboratory model weights cannot be promoted to unmonitored production until at least 50 verified normal and 20 verified abnormal independent industrial recordings are collected and sealed.
- **Result:** **PASSED**
- **Evidence:** `docs/evidence/browser_e9/18_warning_banner_confirmed.png`

---

## 3. UI/UX & Responsive Evaluation

- **Desktop Layout ($1400 \times 950$ px):** Telemetry grid renders 4 balanced cards. Data tables utilize horizontal scrollbars for narrow viewports without clipping action buttons.
- **Modal Dialogs:** Modals render with `backdrop-blur-sm` and high $z$-index (`z-50`). Keyboard ESC and explicit `Cancel` / `Dismiss` buttons successfully release focus.
- **Notification Feedback:** Green emerald banners for successful mutations and rose/amber banners for rejected transactions render with clear contrast and dismiss buttons.
- **Zero Console Exceptions:** Headless Chrome console monitoring verified zero uncaught client-side JavaScript runtime exceptions throughout all 18 scenarios.

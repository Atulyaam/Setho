# Stetho Controlled Field Pilot: Dataset Collection Targets & Accounting Matrix

**Document Reference:** PLAN-STETHO-FIELD-TARGETS-001  
**Sprint / Phase:** Sprint 7.9 / E10 Controlled Field Pilot  
**Target Environment:** 3 Independent External Industrial Facilities  
**Governance Standard:** Cryptographic SHA-256 Provenance & Role Isolation  
**Current Real Field Data Collected:** **0 Recordings (Physical Plant Access Required)**  

---

## 1. Executive Summary & Design Principles

To validate Stetho's acoustic anomaly detection models beyond laboratory testbench limitations (such as Toyo/Hitachi MIMII 1.0), the field pilot establishes explicit quantitative recording targets across **three independent industrial facilities** and **three major industrial machinery categories**.

### 1.1 Critical Statistical Distinction: Workflow Gates vs. Statistical Sufficiency
A common flaw in industrial ML pilots is confusing an operational workflow test with genuine statistical validation. Stetho formally delineates these two thresholds:

```
+---------------------------------------------------------------------------------------------------------+
|                                TWO TIERS OF DATASET TARGET THRESHOLDS                                   |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  TIER 1: Operational Workflow Minimum                                                                   |
|  - Purpose: End-to-end pipeline verification (ingestion, quality gating, calibration, sealing).        |
|  - Requirement: N = 10 confirmed healthy clips for calibration enrollment                               |
|                 + 10 evaluation clips (5 healthy, 5 abnormal).                                          |
|  - Limitation: Statistically underpowered; wide AUC confidence intervals (e.g. +/- 0.22).               |
|                                                                                                         |
|  TIER 2: Statistical Validation Sufficiency                                                             |
|  - Purpose: Rigorous, publishable out-of-domain performance certification.                              |
|  - Requirement: >= 50 confirmed healthy clips across multiple operational cycles / shifts               |
|                 + >= 20 confirmed abnormal clips spanning multiple fault severities.                    |
|  - Power: Yields tight AUC 95% confidence intervals (+/- 0.04 to 0.06 via bootstrap resampling).        |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Target Facility Profiles & Machine Inventory

To guarantee true environmental, acoustic, and electrical diversity, recordings must be collected from three distinct industrial operational sites:

### Facility A: Municipal Water Treatment & Slurry Processing Plant
- **Site Environment:** Heavy industrial, wet processing, concrete floors, high acoustic reverberation, ambient pumps and motors operating continuously.
- **Ambient Noise Floor:** 78–84 dB(A) continuous background drone.
- **Machinery Focus:** Continuous-duty centrifugal slurry pumps and primary aeration blowers.

### Facility B: Commercial District Cooling & HVAC Energy Center
- **Site Environment:** Rooftop and mechanical plant rooms, variable weather exposure, metal decking, seasonal load fluctuations.
- **Ambient Noise Floor:** 72–78 dB(A) with periodic compressor cycling.
- **Machinery Focus:** Induced-draft cooling tower fans and chilled water circulation pumps.

### Facility C: Specialty Chemical & Bulk Powder Manufacturing Plant
- **Site Environment:** Class I Div 2 / ATEX Zone 2 hazardous environment, pneumatic transfer systems, enclosed pipe galleries.
- **Ambient Noise Floor:** 80–86 dB(A) with intermittent high-pressure venting.
- **Machinery Focus:** Pneumatic rotary process valves and heavy-duty ventilation extraction fans.

---

## 3. Machine Inventory & Target Collection Matrix

The target collection plan spans 9 individual industrial machines across 3 distinct equipment categories.

| Facility ID | Machine Category | Machine ID | Description & Operating Point | Target Healthy (Tier 1 / Tier 2) | Target Abnormal (Tier 1 / Tier 2) | Target Uncertain (Allowance) | Planned Evidence Modalities |
|---|---|---|---|---|---|---|---|
| **Facility A** | Centrifugal Pump | `pump_slurry_01` | 75 kW slurry pump, 1450 RPM, 80% load | 10 / 50 | 5 / 20 | $\le 5$ | ISO 10816 Vibration, Seal Leak Log, Motor Current |
| **Facility A** | Centrifugal Pump | `pump_slurry_02` | 75 kW slurry pump, 1450 RPM, 100% load | 10 / 50 | 5 / 20 | $\le 5$ | ISO 10816 Vibration, Seal Leak Log, Motor Current |
| **Facility A** | Industrial Fan | `fan_aeration_01` | 45 kW aeration blower, 2900 RPM direct drive | 10 / 50 | 5 / 20 | $\le 5$ | ISO 10816 Vibration, Radiometric Thermal, Filter DP |
| **Facility B** | Industrial Fan | `fan_cooling_01` | 30 kW induced-draft fan, 720 RPM gearbox | 10 / 50 | 5 / 20 | $\le 5$ | Accelerometer RMS, Gearbox Oil Analysis, Strobe |
| **Facility B** | Industrial Fan | `fan_cooling_02` | 30 kW induced-draft fan, 720 RPM gearbox | 10 / 50 | 5 / 20 | $\le 5$ | Accelerometer RMS, Gearbox Oil Analysis, Strobe |
| **Facility B** | Centrifugal Pump | `pump_chilled_01` | 55 kW chilled water pump, 1450 RPM | 10 / 50 | 5 / 20 | $\le 5$ | ISO 10816 Vibration, Differential Pressure, Thermal |
| **Facility C** | Process Valve | `valve_rotary_01` | 6-inch pneumatic rotary feed valve, 12 RPM | 10 / 50 | 5 / 20 | $\le 5$ | Ultrasonic Leak Detector, Actuator Cycle Telemetry |
| **Facility C** | Process Valve | `valve_rotary_02` | 6-inch pneumatic rotary feed valve, 12 RPM | 10 / 50 | 5 / 20 | $\le 5$ | Ultrasonic Leak Detector, Actuator Cycle Telemetry |
| **Facility C** | Industrial Fan | `fan_exhaust_01` | 37 kW chemical fume exhaust fan, 1750 RPM | 10 / 50 | 5 / 20 | $\le 5$ | ISO 10816 Vibration, Bearing Temp, Housing Flange |
| **TOTALS** | **3 Categories** | **9 Machines** | **3 Facilities** | **90 / 450** | **45 / 180** | **$\le 45$** | **Multi-Modal Physical Evidence** |

---

## 4. Rigorous Live Dataset Accounting Matrix

This table reflects the **exact live state of repository data**. Real field recordings require physical plant deployment and partner access agreements.

> [!IMPORTANT]
> To preserve scientific integrity, synthetic unit/integration test fixtures generated by test runners in `stetho.db` are strictly classified as automated test fixtures and **never** counted as field recordings.

| Facility ID | Machine ID | Category | Target Total (Tier 2) | Physical Field WAVs Ingested | Quality Gate Approved | Cryptographic Deduplicated | Multi-Modal Ground-Truthed | Calibration Eligible (Healthy Only) | Sealed Evaluation Reserved | Operational Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **Facility A** | `pump_slurry_01` | Pump | 70 | 0 | 0 | 0 | 0 | 0 | 0 | Pending External Site Access |
| **Facility A** | `pump_slurry_02` | Pump | 70 | 0 | 0 | 0 | 0 | 0 | 0 | Pending External Site Access |
| **Facility A** | `fan_aeration_01` | Fan | 70 | 0 | 0 | 0 | 0 | 0 | 0 | Pending External Site Access |
| **Facility B** | `fan_cooling_01` | Fan | 70 | 0 | 0 | 0 | 0 | 0 | 0 | Pending External Site Access |
| **Facility B** | `fan_cooling_02` | Fan | 70 | 0 | 0 | 0 | 0 | 0 | 0 | Pending External Site Access |
| **Facility B** | `pump_chilled_01` | Pump | 70 | 0 | 0 | 0 | 0 | 0 | 0 | Pending External Site Access |
| **Facility C** | `valve_rotary_01` | Valve | 70 | 0 | 0 | 0 | 0 | 0 | 0 | Pending External Site Access |
| **Facility C** | `valve_rotary_02` | Valve | 70 | 0 | 0 | 0 | 0 | 0 | 0 | Pending External Site Access |
| **Facility C** | `fan_exhaust_01` | Fan | 70 | 0 | 0 | 0 | 0 | 0 | 0 | Pending External Site Access |
| **PILOT TOTAL** | **9 Machines** | **3 Types** | **630** | **0** | **0** | **0** | **0** | **0** | **0** | **EXTERNAL ACCESS REQUIRED** |

---

## 5. Data Governance & Allocation Rules

Once external plant collection begins, the ingested audio must be allocated strictly following these four rules:

1. **Role Gating Rule 1 (Strict Calibration Purity):**
   - An audio recording may ONLY be assigned the `CALIBRATION_ENROLLMENT` role if its ground-truth status is confirmed `HEALTHY` backed by certified physical inspection evidence (Vibration Zone A/B, normal thermal reading).
   - Recordings with status `ABNORMAL` or `UNCERTAIN` are cryptographically blocked by the backend service from being assigned to calibration sets.

2. **Role Gating Rule 2 (Sealed Evaluation Immutability):**
   - When an evaluation benchmark is assembled for an unseen machine, it is assigned the `SEALED_EVALUATION` role.
   - Once sealed via `POST /api/v1/field-data/manifests/{id}/seal`, the manifest computes a SHA-256 manifest fingerprint across all member file hashes. No recordings can be added, modified, or removed.
   - Any modification attempt generates a tamper alert and invalidates downstream benchmark reports.

3. **Role Gating Rule 3 (Zero Contamination / Leakage):**
   - A single physical recording (identified by its SHA-256 content hash) can belong to AT MOST one role for a given machine:
     $$\text{Role}(\text{hash}) \in \{\text{CALIBRATION\_ENROLLMENT}\} \oplus \{\text{SEALED\_EVALUATION}\} \oplus \{\text{DEVELOPMENT}\}$$
   - Intersection between calibration and evaluation sets is strictly empty:
     $$\mathcal{S}_{\text{calib}} \cap \mathcal{S}_{\text{eval}} = \emptyset$$

4. **Role Gating Rule 4 (Traceable Revisions):**
   - Ground-truth annotations are versioned. If an annotation changes (e.g. from `UNCERTAIN` to `ABNORMAL` following a mechanical teardown report), the system archives the prior label, author ID, timestamp, and justification reason in `annotation_revisions`.

---

## 6. Pilot Execution Phases & Timeline

```
+---------------------------------------------------------------------------------------------------------+
|                                    PILOT TIMELINE & EXECUTION MILESTONES                                |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  [ Milestone M1: Governance & Protocol Acceptance ]  <-- CURRENT STATUS: COMPLETED IN SPRINT 7.9       |
|  - All SOPs, checklists, safety requirements, and targets codified.                                    |
|  - Ingestion APIs, deduplication, and sealing verified with 240/240 automated tests.                   |
|                                                                                                         |
|  [ Milestone M2: Legal, Safety & Partner Permitting ]  <-- NEXT GATE                                    |
|  - Sign data-sharing agreements with Facilities A, B, and C.                                            |
|  - Complete safety site inductions and obtain LOTO clearances.                                          |
|                                                                                                         |
|  [ Milestone M3: Field Deployment & Tier 1 Collection ]                                                 |
|  - Deploy calibrated Class 1 hardware to sites.                                                         |
|  - Capture N = 10 calibration clips + 10 evaluation clips per machine.                                  |
|  - Execute end-to-end operational pipeline check.                                                       |
|                                                                                                         |
|  [ Milestone M4: Full Tier 2 Dataset Collection & Sealing ]                                             |
|  - Expand collection to >= 50 healthy and >= 20 abnormal recordings per machine.                       |
|  - Complete multi-modal verification logs (vibration, thermal, teardown).                              |
|  - Cryptographically seal evaluation manifests for independent cross-facility benchmark.                |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

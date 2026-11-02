# Stetho Sprint 7.9 / E10: Controlled Field Pilot Master Plan

**Sprint Reference:** Sprint 7.9 / E10: Controlled Field Pilot Data Collection  
**Base Commit:** `6f6dee7` (Sprint 7.8 / E9 Field Workflow Acceptance & Security Audit complete)  
**Branch:** `feature/field-data-readiness`  
**Production Master:** `7d8f841` (Untouched, 0-byte diff on `ml/artifacts/`)  
**Pilot Status:** Workflow Accepted & Certified; External Plant Access Required  
**Governance Gate Status:** `ACTIVE — CONTROLLED FIELD PILOT PREPARATION`

---

## 1. Executive Summary & Strategic Rationale

Prior research sprints (E0 through E7) rigorously audited Stetho's deep autoencoder pipelines on the MIMII 1.0 dataset. While mathematical benchmarks achieved high detection quality on test splits (AUC 0.77–0.89 using $N=10$ spatial-spectral residual whitening), **genuine out-of-domain industrial validation has never occurred**:
1. **Identical Test Environment:** The four MIMII fan machine IDs (`id_00`, `id_02`, `id_04`, `id_06`) were recorded in the exact same laboratory room using identical microphone array fixtures and synthetic background noise mixing.
2. **Zero Cross-Facility Evidence:** Zero acoustic data has been collected from real-world manufacturing plants, commercial HVAC installations, or municipal pumping stations.
3. **No Autonomous Production Readiness:** Promoting laboratory checkpoints to unmonitored production without independent out-of-domain validation violates fundamental machine learning reproducibility and reliability engineering principles.

Sprint 7.9 / E10 establishes the operational architecture, safety standards, and governance protocols required to execute a controlled real-world field pilot across industrial partner facilities.

---

## 2. Pilot Objectives & Operational Scope

```
+--------------------------------------------------------------------------------------------------------+
|                                    CONTROLLED FIELD PILOT ARCHITECTURE                                 |
+--------------------------------------------------------------------------------------------------------+
|                                                                                                        |
|  [ Operational Target Machine ]                                                                        |
|  (Industrial Fan / Pump / Valve)                                                                       |
|        |                                                                                               |
|        | Multi-Modal Physical Inspection (ISO 10816 Vibration, Radiometric Thermal, Oil, LOTO Teardown)|
|        v                                                                                               |
|  [ Certified Field Technician ]                                                                        |
|        |                                                                                               |
|        | Calibrated Acoustic Acquisition (Class 1/2 Mic, Stand-off 10cm, PCM 16-bit, fs >= 16kHz)     |
|        v                                                                                               |
|  [ Stetho Field Ingestion Engine (/api/v1/field-data) ]                                                |
|        |                                                                                               |
|        +---> 1. Audio Quality Gating (fs >= 16kHz, clipping <= 0.1%, RMS > -60 dBFS)                   |
|        |                                                                                               |
|        +---> 2. Cryptographic Deduplication (Instant SHA-256 match -> HTTP 409)                       |
|        |                                                                                               |
|        +---> 3. Ground-Truth Annotation (Multi-modal evidence source + Immutable revision trail)       |
|        |                                                                                               |
|        +---> 4. Role Isolation & Gating (Calibration strictly normal; Sealed evaluation locked)       |
|        |                                                                                               |
|        v                                                                                               |
|  [ Cryptographic Manifest Sealing ]                                                                    |
|  (Immutable SHA-256 evaluation benchmark for cross-facility performance computation)                   |
+--------------------------------------------------------------------------------------------------------+
```

### 2.1 Core Objectives
1. **Acquire Diverse Industrial Field Audio:** Collect pristine Linear PCM WAV audio recordings across real operating machinery across at least three distinct industrial facilities.
2. **Prevent Calibration Contamination:** Guarantee that calibration enrollment sets contain exclusively confirmed healthy recordings ($N=10$) with multi-modal physical inspection proof.
3. **Establish Independent Sealed Benchmarks:** Seal held-out evaluation manifests with immutable cryptographic checksums prior to computing out-of-domain model metrics.
4. **Enforce Zero Data Fabrication:** Strictly forbid synthetic data generation, label guessing, or laboratory data substitution.

### 2.2 Target Equipment Categories
- **Industrial Fans:** Centrifugal ventilation blowers, HVAC induced-draft/forced-draft fans, and tunnel ventilation fans ($0.75\text{ kW}$ to $45\text{ kW}$).
- **Centrifugal Pumps:** Water intake pumps, cooling loop circulation pumps, and chemical metering slurry pumps.
- **Valves:** High-pressure steam valves, solenoid-actuated process valves, and pneumatic control valves.

---

## 3. Pilot Execution Phases

The field pilot is structured into five progressive execution phases:

### Phase A: Site Qualification & Instrumentation Calibration
- Partner site technical clearance, safety briefings, and area classification (ATEX / Class I Div 2 compliance if volatile gases present).
- Laboratory calibration of measurement microphones (Bruel & Kjaer / PCB Piezotronics / RODE NTG) with 1 kHz 94 dB calibrator.
- Network verification of field tablet or offline queue deployment.

### Phase B: Baseline Healthy Commissioning & Calibration
- Machine operating under steady nominal load ($80\%\text{--}100\%$ full load).
- Visual and vibration survey confirming normal condition ($< 2.8\text{ mm/s RMS}$ per ISO 10816-3).
- Acquisition of $N=10$ healthy commissioning clips ($10.0\text{s}$ duration each).
- Ingestion into `/field-data/import` and assignment to `calibration_enrollment`.

### Phase C: Operational Variation & Environmental Stress Tracking
- Acquisition during variable plant operating conditions: cold start, partial load ($50\%$), peak load ($100\%$), shut-down transient.
- Logging of ambient acoustic noise floor, ambient temperature ($-10^\circ\text{C}$ to $+45^\circ\text{C}$), and motor frequency inverter switching noise.
- Assignment to `development` split for spatial-spectral whitening stability research.

### Phase D: Anomaly Acquisition & Multi-Modal Triangulation
- Opportunistic and planned recording during confirmed mechanical degradation:
  - Bearing raceway spalling and cage fracture.
  - Impeller blade unbalance, erosion, or cavitation.
  - Belt misalignment or loose foundation bolts.
  - Valve internal leakage or seat wire-drawing.
- Mandatory ground-truth evidence logging: vibration order analysis, thermal imaging, oil ferrography, or maintenance teardown confirmation.
- Direct assignment to `sealed_evaluation` or `unassigned`.

### Phase E: Manifest Sealing & Independent Benchmark Evaluation
- Administrator review of all verified field recordings.
- Creation of sealed evaluation manifests with automated SHA-256 sealing.
- Evaluation of ResidualCNNAE, Stetho-MSAE, and residual whitening pipelines strictly against the sealed field manifest.

---

## 4. Personnel Safety & Equipment Hazard Rules

Industrial plant floors present mechanical, electrical, and thermal hazards. The following safety rules take absolute precedence over data collection:

1. **Non-Interference with Machinery:** Transducers must be mounted using non-invasive magnetic bases or rigid non-resonant tripods. Technicians must **never touch, lean against, or place hands near rotating shafts, belts, or unshielded couplings**.
2. **Personal Protective Equipment (PPE):** Hearing protection (ear defenders rated SNR $\ge 28\text{ dB}$), safety glasses with side shields, steel-toe boots, and high-visibility vests are mandatory at all times.
3. **No Unsafe Override:** Under no circumstances should field personnel bypass safety interlocks, open high-voltage junction boxes, or request plant operators to induce artificial mechanical failures that compromise plant safety.
4. **Adherence to Plant LOTO:** If mounting contact accelerometers or acoustic guides inside acoustic enclosures, complete Lockout/Tagout (LOTO) procedures must be verified with plant safety supervisors.

---

## 5. Required External Access & Dependencies

Real-world field execution cannot proceed solely within local developer workstations. The following external access clearances are formally required:

| Resource / Clearance | Description | Dependency Owner | Status |
|---|---|---|---|
| **Facility Access Clearance** | Plant entry authorization and safety orientation for at least two partner manufacturing sites. | Plant Reliability Director | **REQUIRED** |
| **Physical Equipment Authorization** | Formal sign-off permitting acoustic logging on operational fan and pump assets. | Operations Shift Manager | **REQUIRED** |
| **Calibrated Hardware DAQ** | Class 1 / Class 2 directional microphone, acoustic foam windscreen, 10 cm standoff guide. | Stetho Hardware Team | **REQUIRED** |
| **Multi-Modal Inspection Access** | Access to routine vibration inspection logs (ISO 10816) and thermal scans for ground-truth verification. | Maintenance Lead | **REQUIRED** |
| **Secure Data Transfer** | HTTPS access from plant tablets to Stetho API or offline synchronized storage queue. | IT / OT Security Admin | **READY** |

---

## 6. Pilot Governance Decision Matrix

To ensure absolute scientific integrity, the field pilot operates under binary milestone gates:

```
+---------------------------------------------------------------------------------------+
| Milestone Gate | Criteria Required                               | Current Status    |
+---------------------------------------------------------------------------------------+
| Gate 1: Ingestion Workflow   | 100% E2E browser and security test pass | CERTIFIED (E9)    |
| Gate 2: Plant Entry          | Facility access & safety sign-offs       | EXTERNAL REQUIRED |
| Gate 3: Baseline Enrollment  | >= 10 verified healthy clips per asset   | PENDING ACCESS    |
| Gate 4: Anomaly Benchmarking | >= 10 verified abnormal clips with proof | PENDING ACCESS    |
| Gate 5: Sealed Validation    | Cryptographically sealed manifest        | PENDING DATA      |
+---------------------------------------------------------------------------------------+
```

Until genuine field recordings are acquired and verified from live industrial equipment, Stetho strictly maintains its **Governance Gate Active** status and displays the mandatory warning banner: `INSUFFICIENT INDEPENDENT EVIDENCE: Controlled Field Pilot in Progress`.

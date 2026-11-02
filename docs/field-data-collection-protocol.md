# Stetho Industrial Acoustic Field Data Collection Protocol

**Document Version:** 1.0.0  
**Status:** Approved for Controlled Field Pilot  
**Target Systems:** Industrial Fans, Pumps, Valves, Compressors  
**Audience:** Field Technicians, Reliability Engineers, Acoustic Testing Specialists  

---

## 1. Scope & Objective

This document prescribes the mandatory Standard Operating Procedure (SOP) for acquiring, verifying, and logging industrial acoustic data for Stetho's anomaly detection platform.

Strict adherence to this protocol is necessary to:
1. Guarantee pristine signal quality across field environments.
2. Prevent transducer placement bias and acoustic reverberation artifacts.
3. Ensure $N=10$ healthy enrollment recordings meet the mathematical assumptions of spatial-spectral residual whitening.
4. Establish independent ground-truth through multi-modal triangulation (vibration, thermal, oil, physical teardown) rather than relying on model predictions.

---

## 2. Equipment Checklist

Field teams must verify that all instrumentation is inspected and calibrated before entering the machine hall.

| Item | Specification / Requirement | Verification |
| :--- | :--- | :--- |
| **Acoustic Transducer** | Class 1 or Class 2 measurement microphone capsule or industrial directional acoustic sensor with flat frequency response (20 Hz – 20 kHz ±1.0 dB). | Calibration certificate valid within 12 months. |
| **Acoustic Windscreen** | High-density polyurethane open-cell foam windscreen (mandatory for outdoor and fan intake/discharge areas). | Fitted securely; no tears or oil saturation. |
| **Stand-Off Positioning Guide** | Non-magnetic, non-resonant spacer fixture calibrated to $10\text{ cm} \pm 2\text{ cm}$. | Inspected for rigid mechanical mounting. |
| **Multi-Modal Triangulation: Vibration** | Handheld or tri-axial piezo accelerometer compliant with ISO 10816-3. | Battery charged; calibrated to $100\text{ Hz}$ reference. |
| **Multi-Modal Triangulation: Thermal** | Calibrated radiometric thermal camera or infrared pyrometer ($\pm 1.0^\circ\text{C}$). | Optics clean; emissivity set to machine surface ($0.95$ for cast iron). |
| **Data Acquisition Unit (DAQ)** | 24-bit linear PCM recorder supporting $\ge 16\text{ kHz}$ (or $48\text{ kHz}$) sampling rate, SNR $> 90\text{ dB}$. | Clean phantom power; storage formatted (exFAT). |
| **Field Terminal / Tablet** | Stetho Operator Portal or rugged field tablet running browser interface. | Synchronized UTC clock; API connectivity verified. |

---

## 3. Transducer Placement and Orientation

Acoustic anomaly detection depends on capturing structural mechanical emissions while minimizing diffuse airborne reverberation and turbulent air buffet.

```
       [ Machine Motor ] ==== [ Shaft / Coupling ] ==== [ Bearing Housing / Volute ]
                                                               |
                                                          Stand-off: 10 cm ± 2 cm
                                                               |
                                                          [ Microphone Capsule ]
                                                          (Acoustic Windscreen Fitted)
                                                          Orientation: 90° ± 15° to shaft
```

### 3.1 Placement Rules by Equipment Category

1. **Industrial Centrifugal & Axial Fans:**
   - **Primary Capsule:** Positioned $10\text{ cm} \pm 2\text{ cm}$ from the inboard bearing housing.
   - **Orientation:** $90^\circ \pm 15^\circ$ perpendicular to the rotational shaft axis.
   - **Airflow Precaution:** Never place the capsule directly in the blade exhaust or intake suction slipstream. If aerodynamic buffeting exceeds $-30\text{ dBFS}$ in the $0-100\text{ Hz}$ band, relocate radially behind the housing shield.

2. **Centrifugal Slurry & Water Pumps:**
   - **Primary Capsule:** Positioned $10\text{ cm}$ from the pump drive-end (DE) bearing bracket.
   - **Secondary Capsule (Multi-channel):** $10\text{ cm}$ from the suction volute cutwater to isolate cavitation acoustic spikes ($2\text{ kHz} - 8\text{ kHz}$).

3. **Industrial Valves (Control, Gate, Relief):**
   - **Primary Capsule:** Mounted $10\text{ cm}$ downstream of the valve seat on the pipe centerline.
   - **Precaution:** Avoid contact with high-temperature steam lines; maintain stand-off using insulated non-resonant fixture.

---

## 4. Healthy Baseline Commissioning Protocol ($N=10$ Enrollment)

To commission an unseen machine into Stetho using few-shot spatial-spectral calibration, exactly $N=10$ distinct healthy operational clips must be enrolled.

### 4.1 Step-by-Step Commissioning Procedure

1. **Verify Thermal Steady-State:**
   - Confirm machine has reached continuous operating temperature (run continuously for $\ge 30\text{ minutes}$ prior to capture).
   - Record bearing skin temperature via infrared pyrometer.

2. **Verify Steady Load Condition:**
   - The machine must operate at its nominal operating load ($100\% \pm 5\%$ speed or nominal pressure).
   - Fluctuating load recordings during start-up or shut-down transients are prohibited for enrollment.

3. **Capture $N=10$ Distinct Cycles:**
   - Ingest 10 sequential, non-overlapping $10.0\text{s}$ clips separated by at least 15 seconds.
   - Audio specifications: 16-bit or 24-bit Linear PCM, $16,000\text{ Hz}$ sample rate, 1 or more channels.

4. **Verify Quality Gating Telemetry:**
   - Automated quality gating must return `PASSED`:
     - Sample Rate $\ge 16,000\text{ Hz}$.
     - Signal duration $\ge 1.0\text{s}$ (nominal $10.0\text{s}$).
     - Peak amplitude $\le 0.98$ (Clipping ratio $< 0.1\%$).
     - Signal RMS $> -60.0\text{ dBFS}$ (rejects disconnected or muted capsules).

5. **Assign Dataset Role:**
   - Designate recordings as `calibration_enrollment`.
   - *Strict Rule:* No recording flagged as abnormal may ever be assigned to `calibration_enrollment`.

---

## 5. Multi-Modal Defect Triangulation & Ground-Truth

> [!IMPORTANT]
> **Prohibition of Circular Labeling:** Model anomaly scores or detector flags MUST NEVER be used as the ground-truth label. Every confirmed abnormal label must be corroborated by physical, thermodynamic, or vibration evidence.

### 5.1 Approved Evidence Categories

| Code | Evidence Source | Technical Requirement |
| :--- | :--- | :--- |
| `VISUAL_INSPECTION` | Borescope / Stethoscope / Visual | Direct visual confirmation of broken vanes, severe loose bolts, physical rubbing, or stethoscope acoustic confirmation. |
| `VIBRATION_ORDER_ANALYSIS` | ISO 10816-3 Spectral Vibration | Accelerometer confirms velocity RMS exceeding Zone C/D ($> 4.5\text{ mm/s}$), or 1X/2X shaft harmonics or bearing defect frequencies (BPFO, BPFI, BSF, FTF). |
| `TEARDOWN_INSPECTION` | Bearing / Component Overhaul | Physical inspection after teardown confirming fatigue spalling, cage fracture, or severe galling. |
| `THERMAL_SCAN` | Infrared Thermography | Bearing temperature rise $\Delta T > 20^\circ\text{C}$ above adjacent machinery at steady load. |
| `OIL_ANALYSIS` | Lubricant Ferrography | High particle count (ISO 4406 $> 22/19/16$) or metallic wear debris detected in laboratory assay. |
| `OPERATIONAL_LOG` | Supervisory SCADA Log | Formal maintenance log reporting machine trip, severe seal leakage, or emergency shutdown. |
| `OTHER_GROUND_TRUTH` | Documented Independent Source | Other certified independent engineering assessment. |

---

## 6. Data Provenance, Ingestion, and Manifest Sealing

1. **Immediate SHA-256 Checksumming:**
   - Every audio file imported through `/api/v1/field-data/import` is immediately hashed with SHA-256.
   - Duplicate hashes are rejected immediately with HTTP 409 Conflict and logged to `field_import_history`.
   - The platform will never silently overwrite or replace existing recordings.

2. **Sealed Evaluation Manifest Creation:**
   - Recordings intended for independent validation must be assigned to role `sealed_evaluation`.
   - Authorized QA personnel freeze the benchmark using `/api/v1/field-data/manifests`.
   - Once sealed, a cryptographic SHA-256 checksum is computed across sorted member hashes. Sealed recordings are permanently locked against role reclassification.

---

## 7. Field Operator Execution Checklist

Technicians must print or complete this checklist digitally for each collection session:

- [ ] **Pre-Trip:** DAQ charged, Class 1/2 microphone inspected, acoustic windscreen packed, vibration meter calibrated.
- [ ] **Safety Briefing:** Lock-Out/Tag-Out (LOTO) reviewed; safety glasses, hearing protection, steel-toe boots worn.
- [ ] **Machine Warm-Up:** Machine verified running at steady state for $\ge 30\text{ minutes}$.
- [ ] **Mounting:** Transducer secured at $10\text{ cm} \pm 2\text{ cm}$ stand-off, $90^\circ$ perpendicular to shaft. Windscreen fitted.
- [ ] **Level Verification:** DAQ gain adjusted so peak level remains between $-18\text{ dBFS}$ and $-6\text{ dBFS}$ (no red clip indicator).
- [ ] **Enrollment Capture:** 10 sequential 10-second clips recorded for healthy baseline commissioning.
- [ ] **Defect Triangulation (if abnormal):** Multi-modal inspection performed (vibration spectra, thermography, borescope) and uploaded with diagnostic notes.
- [ ] **Ingestion:** Files imported to Stetho with correct metadata (machine ID, sensor ID, load level, operator ID).
- [ ] **Audit Verification:** Confirm zero clipping errors and verify SHA-256 hashes registered in Stetho database.

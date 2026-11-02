# Stetho Field Pilot Recording Checklist & Standard Operating Procedure (SOP)

**Document Reference:** SOP-STETHO-FIELD-001  
**Sprint / Phase:** Sprint 7.9 / E10 Controlled Field Pilot  
**Target Environment:** Real-world Industrial Facilities (Pumps, Fans, Valves)  
**Safety Classification:** High — Rotating Machinery & Industrial Plant Operations  
**Compliance Standard:** ISO 10816 (Mechanical Vibration), IEC 61672 (Electroacoustics)  

---

## 1. Safety First — Industrial Environment Protocols

Industrial facilities present severe hazards including high-voltage power lines, rotating shafts, pressurized fluid lines, toxic/flammable atmospheres, and extreme acoustic energy. Field technicians must complete all site-specific safety orientations before approaching machinery.

### 1.1 Personal Protective Equipment (PPE) Mandatory Checklist
All personnel within active machine envelopes must wear:
- [ ] **Hearing Protection:** Dual protection (earplugs NRR >= 25 dB + earmuffs NRR >= 30 dB) when entering areas with ambient SPL >= 85 dB(A).
- [ ] **Eye & Face Protection:** ANSI Z87.1 approved safety glasses with side shields. Face shield required if inspecting pressurized fluid systems.
- [ ] **Footwear:** ASTM F2413 compliant steel-toe or composite-toe puncture-resistant boots.
- [ ] **Head Protection:** ANSI Z89.1 Type 1 Class E/G hard hat.
- [ ] **High-Visibility Clothing:** ANSI/ISEA 107 Class 2 or Class 3 high-visibility vest or jacket.
- [ ] **Gloves:** Cut-resistant gloves (ANSI Cut Level A3+) for setup; remove gloves when working immediately adjacent to rotating shafts or drive belts to prevent snagging.

### 1.2 Rotating Machinery & Non-Interference Safety Rules
> [!CAUTION]
> **STRICT PROHIBITION:** At no point during acoustic acquisition shall any sensor, cable, mounting arm, spacer, or technician body part contact a rotating shaft, pulley, impeller, coupling, or exposed electrical terminal.

1. **Clearance Envelope:** Maintain a minimum 300 mm (12 in) safety buffer from all moving machinery parts at all times during operation.
2. **Lockout / Tagout (LOTO):** If mounting magnetic brackets or installing standoff fixtures within a safety guard enclosure, the machine MUST be completely isolated, locked out, tagged out, and zero-energy verified by a licensed facility operator prior to sensor positioning.
3. **Emergency Stop (E-Stop) Awareness:** Identify the nearest two E-Stop buttons or pull-cords prior to beginning any recording or measurement.
4. **Intrinsically Safe / Explosive Atmosphere (ATEX / IECEx):** For Zone 1 / Zone 2 (Class I, Div 1 / Div 2) environments (e.g. chemical processing plants or fuel pumping bays), only ATEX-certified, intrinsically safe DAQ units and battery-powered microphones are permitted. Standard consumer electronics, smartphones, or non-certified laptops are strictly forbidden.

---

## 2. Equipment Specifications & Calibration Checklist

All hardware used for Stetho field acoustic data collection must adhere to strict scientific and industrial metrology requirements.

### 2.1 Acoustic Acquisition Hardware
- [ ] **Measurement Microphone:**
  - Standard: IEC 61672 Class 1 (preferred) or Class 2 free-field condenser measurement microphone (e.g., PCB Piezotronics 378B02, GRAS 46AE, or equivalent industrial omnidirectional measurement mic).
  - Dynamic Range: 20 dB(A) to 135 dB SPL.
  - Frequency Response: Flat ($\pm 1.0$ dB) from 20 Hz to 20,000 Hz.
  - Windscreen: Open-cell polyurethane acoustic foam windscreen installed at all times to eliminate false turbulent airflow transients.
- [ ] **Mechanical Fixture & Standoff Spacer:**
  - Standardized Standoff Distance: Exactly $10.0 \text{ cm} \pm 1.0 \text{ cm}$ from the machine bearing housing or casing.
  - Mounting: Heavy-duty magnetic base with articulated locking arm, or non-resonant carbon-fiber tripod isolated with silicone damping feet.
  - Orientation: Microphone capsule oriented normal ($90^\circ \pm 5^\circ$) to the target machine surface.
- [ ] **Data Acquisition (DAQ) / Audio Interface:**
  - Format: Uncompressed Linear PCM, 24-bit or 16-bit depth.
  - Sampling Rate: Minimum $16,000 \text{ Hz}$ ($48,000 \text{ Hz}$ preferred for full acoustic bandwidth).
  - Preamp / ADC SNR: $> 100 \text{ dB}$, Total Harmonic Distortion (THD+N) $< 0.005\%$.
  - Cable: Double-shielded, low-microphonics XLR/BNC cable secured with Velcro strain-relief wraps to prevent mechanical cable vibration transfer.
- [ ] **Acoustic Calibrator:**
  - Handheld acoustic calibrator (IEC 60942 Class 1, 94.0 dB / 114.0 dB at 1000 Hz) with valid calibration certificate within 12 months.

### 2.2 Multi-Modal Ground-Truth Verification Instruments
Acoustic labels must never be guessed or assigned without independent physical confirmation. The following instruments must be present:
- [ ] **Vibration Meter:** Triaxial or single-axis handheld accelerometer adhering to ISO 10816 standards (measurement of overall velocity RMS in mm/s from 10 Hz to 1000 Hz).
- [ ] **Radiometric Thermal Camera:** Calibrated thermal imager ($\pm 2^\circ\text{C}$ accuracy) with adjustable emissivity for bearing and casing thermal profile capture.
- [ ] **Digital Stroboscope / Optical Tachometer:** Contactless optical RPM measurement tool with reflective tape to measure true running speed.
- [ ] **Lubrication / Fluid Inspection Kit:** Dipstick, oil sample bottles, and grease gun pressure log for lubrication health assessment.

---

## 3. Step-by-Step Field Acquisition Procedure

```
+---------------------------------------------------------------------------------------------------------+
|                                  FIELD ACOUSTIC ACQUISITION WORKFLOW                                    |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  [ Step 1: Pre-Audit & Permitting ]                                                                     |
|  Obtain site permit -> Confirm machine ID -> Verify safety clearance -> LOTO if entering guard envelope|
|                                     |                                                                   |
|                                     v                                                                   |
|  [ Step 2: Environmental Survey ]                                                                       |
|  Record ambient sound pressure level (dB(A)) -> Log competing noise sources (compressors, traffic)     |
|                                     |                                                                   |
|                                     v                                                                   |
|  [ Step 3: Multi-Modal Inspection ]                                                                     |
|  Measure ISO 10816 vibration (RMS mm/s) -> Measure bearing temp (deg C) -> Log lubrication condition   |
|                                     |                                                                   |
|                                     v                                                                   |
|  [ Step 4: Sensor Positioning ]                                                                         |
|  Attach windscreen -> Position at exactly 10 cm standoff -> Secure strain-relief cables                 |
|                                     |                                                                   |
|                                     v                                                                   |
|  [ Step 5: Gain Staging & Calibration ]                                                                 |
|  Run 1000 Hz calibrator tone -> Set digital headroom (-18 to -6 dBFS peak) -> Verify RMS > -60 dBFS    |
|                                     |                                                                   |
|                                     v                                                                   |
|  [ Step 6: Audio Recording ]                                                                            |
|  Capture uncompressed WAV (fs >= 16kHz, >= 10s) -> Confirm zero clipping (<= 0.1% samples at ceiling)  |
|                                     |                                                                   |
|                                     v                                                                   |
|  [ Step 7: Quality Verification & Ingestion ]                                                           |
|  Headphone acoustic audit -> Calculate SHA-256 -> Upload to /api/v1/field-data/recordings               |
|                                     |                                                                   |
|                                     v                                                                   |
|  [ Step 8: Ground-Truth Annotation & Role Assignment ]                                                  |
|  Attach multi-modal evidence -> Assign HEALTHY / ABNORMAL / UNCERTAIN -> Assign Dataset Role            |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Step 1: Pre-Audit & Machine Identification
- [ ] Obtain plant work authorization and escort sign-off.
- [ ] Locate machine nameplate. Record:
  - Facility / Plant Site Name: ______________________________________
  - Machine Category (Fan, Pump, Valve): _____________________________
  - Machine Identifier (Serial / Asset Tag): ________________________
  - Manufacturer & Model Number: _____________________________________
  - Rated Power (kW or HP): __________________________________________
  - Rated Speed (RPM): _______________________________________________
  - Current Operating Mode (Continuous, Variable Load, Idle): _________
  - Operating Speed / Flow Rate during test: _________________________
  - Estimated Load Level (% of rated capacity): _____________________

### Step 2: Environmental Ambient Noise Survey
- [ ] Before recording the target machine, assess ambient acoustic interference:
  - If machine can be temporarily paused: Measure background noise level with target off: _________ dB(A).
  - If continuous operation: Measure ambient noise at 2 meters from machine envelope: _________ dB(A).
  - Identify dominant background sound sources (e.g. adjacent air compressor, overhead cooling tower, diesel forklift traffic, high-pressure steam venting).
  - Ambient Temperature: _________ $^\circ\text{C}$ | Ambient Relative Humidity: _________ $\%$.

### Step 3: Independent Multi-Modal Physical Health Inspection
> [!IMPORTANT]
> Audio must never be labeled as `HEALTHY` or `ABNORMAL` based on human listening alone. Ground-truth requires physical instrumentation evidence.

- [ ] **Vibration Measurement (ISO 10816):**
  - Drive-End (DE) Bearing Overall Velocity RMS: _________ mm/s.
  - Non-Drive-End (NDE) Bearing Overall Velocity RMS: _________ mm/s.
  - ISO 10816 Zone Assessment: [ ] Zone A (Good) | [ ] Zone B (Acceptable) | [ ] Zone C (Unsatisfactory) | [ ] Zone D (Unacceptable/Damage).
- [ ] **Thermal Inspection:**
  - DE Bearing Housing Temperature: _________ $^\circ\text{C}$.
  - NDE Bearing Housing Temperature: _________ $^\circ\text{C}$.
  - Ambient Differential ($\Delta T$ above ambient): _________ $^\circ\text{C}$.
  - Capture radiometric thermal JPEG image and record filename: ____________________.
- [ ] **Visual & Maintenance Log Audit:**
  - Check for mechanical looseness, cracked welds, damaged belts, pipe cavitation, or oil leakage.
  - Check plant CMMS (Computerized Maintenance Management System) work orders for last bearing replacement, alignment, or balancing date.

### Step 4: Transducer Positioning & Fixturing
- [ ] Verify windscreen is firmly fitted onto the microphone capsule.
- [ ] Mount microphone stand or magnetic base onto rigid machine frame or isolated ground tripod.
- [ ] Use calibrated 10.0 cm measurement gauge to position microphone diaphragm exactly 10 cm from target bearing housing or valve body.
- [ ] Ensure microphone axis is pointed directly at the acoustic emission point (e.g., DE bearing housing or pump volute).
- [ ] Dress all microphone cables with Velcro ties or gaffer tape to machine structure to prevent dangling or snagging.

### Step 5: Gain Staging, Calibration & Level Check
- [ ] Attach 94.0 dB 1000 Hz calibrator to microphone. Verify DAQ calibration tone registers cleanly without distortion.
- [ ] Power on target machine and allow it to reach thermal and mechanical steady-state (minimum 5 minutes runtime).
- [ ] Monitor real-time input meter on DAQ:
  - Target Peak Level: Between $-18 \text{ dBFS}$ and $-6 \text{ dBFS}$ (ensuring $> 6 \text{ dB}$ headroom above transient peaks).
  - Verify RMS signal energy $> -60 \text{ dBFS}$ to ensure high signal-to-noise ratio.
  - Verify NO red clipping indicators on preamps.

### Step 6: Audio Recording Execution
- [ ] Initiate recording in uncompressed WAV Linear PCM format.
- [ ] Capture duration: Standard $10.0 \text{ seconds}$ per clip (or continuous recording split into 10-second non-overlapping blocks).
- [ ] Capture minimum target sequence for session:
  - Healthy baseline enrollment: 10 to 30 clips under steady-state operating conditions.
  - Variable condition tests: 5 to 10 clips per discrete load step (e.g. 50%, 75%, 100% flow).
- [ ] During capture, technician must observe machine envelope:
  - [ ] No personnel spoke or shouted near microphone.
  - [ ] No mobile phone interference or radio transmissions occurred.
  - [ ] No dropped tools or transient impacts occurred during recording window.

### Step 7: Post-Capture Acoustic Verification & Ingestion
- [ ] Perform listening check using closed-back studio monitor headphones (e.g., Sennheiser HD 280 Pro or Sony MDR-7506):
  - [ ] Sound is clean, representative, and free from digital clicks, drops, or buffer underruns.
  - [ ] No wind rumble or handling rubbing noise detected.
- [ ] Run automated quality gate checks:
  - Sample Rate $\ge 16,000 \text{ Hz}$.
  - Bit Depth $\ge 16\text{-bit}$.
  - Clipped Sample Ratio $\le 0.1\%$.
  - RMS Level $> -60 \text{ dBFS}$.
- [ ] Ingest recording into Stetho platform via POST `/api/v1/field-data/recordings`:
  - Verify system computes SHA-256 cryptographic digest.
  - Confirm HTTP 201 Created (or HTTP 409 Conflict if duplicate file).

### Step 8: Ground-Truth Annotation & Role Assignment
- [ ] Open Stetho Field Data Portal (`/field-data`).
- [ ] Select ingested recording and click **Ground Truth Annotation**:
  - Ground Truth Status:
    - [ ] `HEALTHY` (Vibration ISO Zone A/B, normal bearing temp, no audible or mechanical distress).
    - [ ] `ABNORMAL` (Vibration ISO Zone C/D, elevated thermal signature, bearing cage rattle, cavitation, or unbalance confirmed).
    - [ ] `UNCERTAIN` (Ambiguous vibration, transient background interference, or unconfirmed anomaly — **strictly excluded from calibration**).
  - Anomaly Severity: `NONE`, `MILD`, `MODERATE`, or `SEVERE`.
  - Specific Anomaly Description (e.g., "Inner raceway bearing defect BPFI on DE bearing", "Severe suction cavitation at 60% valve closure").
  - Verification Method: Select `PHYSICAL_INSPECTION`, `VIBRATION_ISO10816`, `THERMAL_RADIOMETRIC`, `TEARDOWN_CONFIRMED`, or `CMMS_RECORD`.
- [ ] Assign Dataset Role:
  - [ ] `CALIBRATION_ENROLLMENT`: **STRICT CONDITION** — Status MUST be `HEALTHY` with verified multi-modal evidence. Never assign abnormal or uncertain audio to calibration.
  - [ ] `SEALED_EVALUATION`: Reserved exclusively for held-out benchmark manifests. Once manifest is cryptographically sealed, items become permanently read-only.
  - [ ] `DEVELOPMENT`: Training and tuning pool for exploratory research models.

---

## 4. Technician Sign-Off & Chain of Custody

| Role | Name | Organization / Facility | Signature | Date (UTC) |
|---|---|---|---|---|
| **Lead Field Acoustic Engineer** | _______________________ | _______________________ | _______________________ | ____________ |
| **Facility Operations Lead** | _______________________ | _______________________ | _______________________ | ____________ |
| **Certified Vibration Analyst** | _______________________ | _______________________ | _______________________ | ____________ |

*This checklist must be scanned, archived with the machine's cryptographic manifest, and stored under `docs/field-records/`.*

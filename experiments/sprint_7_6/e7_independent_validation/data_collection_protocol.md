# Stetho Industrial Acoustic Data Collection & Pilot Protocol

**Document ID:** `PROT-STETHO-PILOT-DATA-COLLECTION-V1`  
**Date:** November 2, 2026  
**Author:** Principal ML Systems Architect & Industrial Acoustic Engineer  
**Purpose:** Standard operating procedure for collecting, annotating, and freezing independent acoustic datasets for industrial plant pilots.  

---

## 1. Objectives & Scope

This protocol establishes the standardized procedure for acquiring genuinely independent acoustic data from operational customer facilities (manufacturing plants, mining ventilation, HVAC, chemical refineries) to validate the Stetho anomaly detection platform.

### Target Machine Categories:
1. **Centrifugal & Axial Industrial Fans:** Supply/exhaust blowers, cooling tower fans, tunnel ventilation.
2. **Positive Displacement & Centrifugal Pumps:** Slurry pumps, boiler feed pumps, chemical dosing pumps.
3. **Pneumatic & Solenoid Valves:** Fast-actuating process control valves, steam relief valves.
4. **Rotating Compressors & Gearboxes:** Industrial screw compressors, transmission gearboxes.

---

## 2. Hardware & Microphone Configuration

To avoid hardware filtering mismatch and directional acoustic degradation identified in E5 and E6, all recording hardware must comply with the following technical specifications:

### 2.1 Sensor Specifications
- **Transducer Type:** Industrial-grade pre-polarized electret or MEMS acoustic microphone with flat frequency response ($20\text{ Hz} - 20\text{ kHz} \pm 1.5\text{ dB}$).
- **Dynamic Range:** Minimum $94\text{ dB}$ SPL, maximum acoustic overload point (AOP) $\ge 120\text{ dB}$ SPL.
- **Enclosure:** IP65-rated dust- and moisture-resistant acoustic baffle with hydrophobic mesh.
- **Mounting:** Magnetic base or rigid mechanical clamp anchored to structural machine housing (not flexible sheet metal to avoid structural resonance).

### 2.2 Recording Hardware & Acquisition Settings
- **Sampling Rate:** Strict $16,000\text{ Hz}$ or $48,000\text{ Hz}$ (downsampled via Stetho polyphase filter to $16\text{ kHz}$). Minimum $16\text{ kHz}$ to prevent sub-Nyquist harmonic truncation.
- **Bit Depth:** $24\text{-bit}$ or $32\text{-bit}$ linear PCM (Float32 or Int24) to maintain $\ge 40\text{ dB}$ headroom above ambient noise floor.
- **Audio Channels:** Minimum 1 channel (designated primary), optional 2–4 channels for spatial beamforming and directional interference rejection.

### 2.3 Sensor Placement & Geometry Documentation
For every machine asset, record exact spatial coordinates:
- **Distance:** $10\text{ cm} \pm 2\text{ cm}$ from primary acoustic radiating surface (bearing housing, casing, or intake port).
- **Orientation:** $0^\circ$ direct line-of-sight toward moving assembly.
- **Photographic Record:** Three high-resolution photos: (1) wide-angle machine context, (2) close-up mounting clamp, (3) distance measurement against scale ruler.
- **Metadata Lock:** Every clip must be stamped with `(facility_id, machine_id, sensor_serial_no, channel_index)`.

---

## 3. Data Collection Regimes

### 3.1 Healthy Baseline Enrollment Phase ($N=10$ clips)
To calibrate spatial-spectral residual whitening:
- Capture exactly **10 non-overlapping recordings** of $10.0\text{ seconds}$ duration each during verified normal operating state.
- Space recordings across at least **2 operating hours** to capture minor thermal, load, and ambient shift variations.
- Document nominal machine operating parameters during enrollment:
  - RPM / rotational frequency.
  - Electrical load (kW or Amps).
  - Ambient plant temperature ($^\circ\text{C}$) and background noise level ($\text{dBA}$).
- **Freezing:** Lock and sign SHA-256 hashes of these 10 clips into `calibration_manifest.json`.

### 3.2 Evaluation Phase (Normal & Fault Data)
- **Normal Operational Range:** Collect normal audio across variable plant conditions:
  - Idle state / spin-up / nominal steady state / peak load state.
  - Night shift vs day shift (ambient noise floor swings).
- **Physical Fault Regimes:**
  Capture acoustic signatures of real, naturally occurring or physically seeded machinery defects:
  1. **Bearing Degradation:** Inner race spalling, ball defects, cage wear, lack of lubrication.
  2. **Rotor Unbalance:** Eccentric mass distribution, dust/material buildup on fan blades.
  3. **Shaft Misalignment:** Angular or parallel motor-to-shaft misalignment.
  4. **Acoustic Leaks / Blade Rub:** Airflow obstruction, loose casing rattles, valve seal blow-by.

---

## 4. Ground-Truth Verification & Labeling Protocol

To avoid synthetic label contamination or unverified assumptions:

### 4.1 Multi-Modal Ground Truth Triangulation
Every acoustic recording must be cross-verified against at least one independent physical diagnostic modality:
1. **Tri-Axial Vibration Accelerometry:** Overall RMS velocity ($\text{mm/s}$ per ISO 10816-3), peak acceleration ($g$), and FFT peak order tracking ($1\times, 2\times, \text{BPFO}, \text{BPFI}, \text{BSF}$).
2. **Thermal Imaging:** Infrared thermography logging bearing temperature ($\Delta T \ge 15^\circ\text{C}$ indicating friction).
3. **Physical Maintenance Teardown:** Maintenance log confirmation signed by a certified vibration analyst (ISO 18436-2 Level II or higher).

### 4.2 Label Hierarchy
- `CONFIRMED_NORMAL`: Asset verified healthy by vibration analysis ($< 2.8\text{ mm/s}$ RMS) and visual inspection.
- `CONFIRMED_ANOMALY`: Defect verified by vibration order harmonics ($> 4.5\text{ mm/s}$ RMS) or physical inspection.
- `UNCERTAIN_OPERATIONAL_TRANSIENT`: Process change, startup, or external tool noise. Excluded from primary benchmark.

---

## 5. Dataset Freezing & Quarantine Protocol

1. **Immutable Ingestion Directory:**
   Audio files transferred via SFTP/S3 into write-once read-many (WORM) storage.
2. **Cryptographic Sealing:**
   Compute SHA-256 for all WAV files immediately upon ingest.
3. **Manifest Creation:**
   Generate `pilot_evaluation_manifest.json` cataloging:
   - File path, SHA-256, duration, sample rate, peak amplitude, RMS energy.
   - Machine category, asset ID, confirmed label, fault type, severity level.
   - Triangulation evidence (vibration report ID, thermography log).
4. **Quarantine:**
   The evaluation set must remain strictly sealed. ML engineers and algorithms must NOT access evaluation audio during model training, tuning, or threshold selection.

---

## 6. Pilot Acceptance Criteria

Before authorizing production cutover, the Stetho complete pipeline must achieve the following milestones on the sealed pilot evaluation set:

1. **Detection Performance:**
   - Macro ROC-AUC $\ge 0.8500$ across all pilot machines.
   - False Positive Rate ($FPR$) $\le 5.0\%$ under verified normal operations.
   - False Negative Rate ($FNR$) $\le 5.0\%$ on critical mechanical faults (Severity Grade 3 or 4).
2. **Operational Stability:**
   - $0\%$ unhandled crashes or un-caught exceptions.
   - $100\%$ detection of sensor disconnection or hardware clipping by `AudioQualityGate`.
   - $100\%$ blocking of uncalibrated inference by `SensorGuard`.
   - P95 CPU inference latency $\le 100\text{ ms}$ on target edge runtime (Intel Core i5 / Jetson Orin Nano).

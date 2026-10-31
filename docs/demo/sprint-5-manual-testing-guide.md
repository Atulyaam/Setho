# Stetho — Sprint 5 Manual End-to-End Verification Guide

**Document:** Manual Verification Guide & Step-by-Step Operator Workflow  
**Target:** Stetho Web UI (`apps/web`) & FastAPI Backend (`services/api`)  
**Lead Roles:** Senior Full-Stack Engineer, Product Designer, QA Engineer  
**Date:** 2026-10-31  

---

## 1. Overview & Prerequisites

This guide provides the exact step-by-step verification procedure for an operator or auditor testing the complete integrated Stetho application workflow.

### Prerequisites
1. **Start Backend API Service:**
   ```powershell
   .\.venv\Scripts\uvicorn.exe stetho_api.main:app --host 0.0.0.0 --port 8000
   ```
   *Verify:* Navigate to `http://localhost:8000/ready` -> returns `"status": "ready"`.

2. **Start Next.js Frontend Application:**
   ```powershell
   npm run dev --prefix apps/web
   ```
   *Verify:* Navigate to `http://localhost:3000` -> page renders with green "Backend API: Connected" status badge.

---

## 2. Step-by-Step Verification Protocol

### Step 1: User Registration & Authentication
1. Navigate to `http://localhost:3000/register`.
2. Enter:
   - **Operator Username:** `technician_qa`
   - **Password:** `InspectorPass2026!`
   - **Confirm Password:** `InspectorPass2026!`
   - **Role Profile:** `Acoustic Inspector (Standard)`
3. Click **Complete Registration**.
4. **Verified Result:** The user account is created with PBKDF2-HMAC-SHA256 password hash. An `HttpOnly`, `SameSite=Lax` session cookie (`stetho_token`) is stored in the browser. The user is redirected to `/dashboard`.
5. The navigation header displays `technician_qa` and role badge `INSPECTOR`.

---

### Step 2: Register a New Machine Profile
1. On the dashboard, click **Register Machine** (or navigate to `/machines/new`).
2. Fill out the physical asset registration form:
   - **Machine Identifier:** `rooftop_fan_01`
   - **Display Name:** `Industrial Roof Ventilation Fan #1`
   - **Category:** `Industrial Electric Fan`
   - **Model Architecture Policy:** `Stetho-MSAE (Edge Champion, 560k params)`
   - **Baseline Operating Conditions:** `Normal speed, 1,450 RPM, ambient 65 dB`
   - **Description:** `Exhaust fan on roof of Building C, mounted on anti-vibration springs.`
3. Click **Save Machine Profile**.
4. **Verified Result:** Machine profile is saved in SQLite database with `owner_id = "usr_..."`. The browser redirects to `/machines/rooftop_fan_01`.
5. **Initial State:**
   - Enrolled Samples: `0`
   - Calibration Status: `uncalibrated`
   - Operating Thresholds: Not yet calculated.

---

### Step 3: Enroll Confirmed Normal Audio
1. On the machine detail page (`/machines/rooftop_fan_01`), click **Enroll Normal Audio** (navigates to `/machines/rooftop_fan_01/enroll`).
2. Select a confirmed normal WAV recording:
   - File: `data/mimii/raw/fan/id_00/normal/00000000.wav` (10.0s, 16,000 Hz, 16-bit PCM).
3. The interactive **Audio Player & Waveform Visualizer** decodes the audio buffer:
   - Time-Domain Waveform displays amplitude envelope $[-0.3, +0.3]$ across $10.0\text{ s}$.
   - Log-Frequency Spectrogram renders genuine STFT spectral energy distribution from $0\text{ Hz}$ to $8,000\text{ Hz}$.
   - Audio playback can be started, paused, and scrubbed.
4. Check the **Mandatory Normal Confirmation** checkbox:
   - *"I explicitly verify that this recording was captured when this physical machine was functioning under healthy, known-normal operating conditions..."*
5. Click **Submit Normal Enrollment**.

---

### Step 4: Verify Enrollment Count & Dynamic Recalibration
1. **Verified Result:** The backend executes `MachineCalibrationEngine.recalibrate_machine()`.
2. The UI renders the green success card:
   - **Enrolled Count:** `1 recordings`
   - **Calibration State:** `provisional`
   - **Threshold ($\theta_{\text{upper}}$):** Dynamically computed (e.g. `0.0428`)
   - **Calibration Version:** `1.1.0`
   - **Notice Displayed:** *"Provisional calibration (N=1 recordings). Minimum recommended is N >= 5 for initial variance estimates, and N >= 30 for stable false-positive rate control."*
3. Click **View Machine Dashboard** -> Enrolled count reflects `1`, status reflects `provisional`.

---

### Step 5: Upload a Normal Recording for Inspection
1. From the machine page, click **Perform Inspection** (navigates to `/machines/rooftop_fan_01/analyze`).
2. Model Policy: Keep `Stetho-MSAE (Edge Champion)`.
3. Select an operational recording:
   - File: `data/mimii/raw/fan/id_00/normal/00000001.wav`.
4. Click **Submit for Acoustic Analysis**.

---

### Step 6: Verify Asynchronous Job & Normal Inspection Result
1. **Verified Result:** Submits to `POST /api/v1/machines/rooftop_fan_01/analysis-jobs` (receives HTTP 202 Accepted).
2. The progress screen displays the progress bar advancing from $0\%$ to $30\%$ to $85\%$ to $100\%$.
3. When complete, the page redirects automatically to `/analyses/{analysis_id}`.
4. **Verified Findings:**
   - **Outcome:** **`NORMAL_LIKE`** (Emerald badge)
   - **Meaning Displayed:** *"The recording falls within the current machine-specific reference range. Reconstruction deviation (0.0125) is strictly below the operating threshold (0.0428)."*
   - **Visual Gauge:** The pin points inside the green `NORMAL_LIKE` zone ($\le \theta_{\text{lower}}$).
   - **AQG Status:** `ACCEPTED` (No clipping, good SNR).
   - **Provisional Notice:** Tagged `Provisional Calibration` because $N < 30$.
   - **Strict Engineering Disclaimer:** Explicitly states Stetho does not claim the machine is safe or diagnose mechanical faults.

---

### Step 7: Upload an Abnormal Recording for Inspection
1. Click **New Inspection** (navigates to `/machines/rooftop_fan_01/analyze`).
2. Select an abnormal recording:
   - File: `data/mimii/raw/fan/id_00/abnormal/00000000.wav`.
3. Click **Submit for Acoustic Analysis**.

---

### Step 8: Verify Abnormal Deviation Outcome
1. Job completes in $< 1.5\text{ s}$.
2. Redirects to `/analyses/{analysis_id}`.
3. **Verified Findings:**
   - **Outcome:** **`UNUSUAL`** (Rose badge)
   - **Meaning Displayed:** *"The recording shows an acoustic deviation from the current reference. Measured reconstruction error (0.0682) exceeds the calibrated normal threshold (0.0428)."*
   - **Visual Gauge:** The pin points inside the red `UNUSUAL DEVIATION` zone ($> \theta_{\text{upper}}$).
   - **Recommended Physical Action:** *"Perform hands-on physical inspection of the machine. Inspect rotating bearings, mounting brackets, and fan blades for mechanical wear, looseness, or imbalance..."*

---

### Step 9: Upload Silent / Rejected Audio
1. Click **New Inspection**.
2. Select a silent or corrupt WAV file (e.g., generated with RMS $0.0$).
3. Click **Submit for Acoustic Analysis**.
4. **Verified Result:** Rejection error banner displayed:
   - *"Enrollment audio rejected by Audio Quality Gate (REJECTED): Audio is silent or near-silent: RMS amplitude 0.00e+00 is below threshold 1.00e-04."*
5. No fake score or unverified diagnosis is recorded.

---

### Step 10: Open and Print Formal Inspection Report
1. On the abnormal analysis result page, click **Full Inspection Report** (navigates to `/analyses/{analysis_id}/report`).
2. **Verified Elements:**
   - Document Header with Report ID and ISO UTC Timestamp.
   - Target Machine Profile & Operational Baseline.
   - Technical Telemetry: Reconstruction Error (`0.0682`), Threshold (`0.0428`), AQG Status (`ACCEPTED`), Model Policy (`stetho_msae v1.0.0`).
   - Detailed Technical Findings bullet points.
   - Recommended Physical Inspection Actions.
   - Certified Engineering Disclaimer.
3. Click **Print / Save as PDF** -> Invokes clean browser print dialog with navigation and action buttons hidden via `@media print`.

---

### Step 11: Multi-Tenant Cross-User Isolation
1. Click the **Sign Out** button in the navigation header.
2. Register a new second user: `operator_two` / `OperatorPass2026!`.
3. Navigate to `/dashboard` -> Registered Machines count is `0` (clean empty state).
4. Attempt to navigate directly to User A's machine: `http://localhost:3000/machines/rooftop_fan_01`.
5. **Verified Result:** Red error card displayed:
   - *"Machine Inaccessible: FORBIDDEN: You do not have permission to access or modify this machine."*
6. User B cannot view User A's audio, enrollments, analyses, or inspection reports.

---

### Step 12: Session Logout & Protected Route Enforcement
1. Click **Sign Out**.
2. **Verified Result:** The `stetho_token` cookie is cleared with `maxAge: 0`.
3. Attempt to access `/dashboard` or `/machines` directly in the URL bar.
4. **Verified Result:** Immediately intercepted by `ProtectedRoute` and redirected to `/login`.


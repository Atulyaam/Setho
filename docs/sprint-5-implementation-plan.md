# Stetho — Sprint 5 Implementation Plan: Frontend Integration & Complete User Workflow

**Target:** Sprint 5 Full Frontend-to-Backend Integration  
**Lead Roles:** Senior Full-Stack Engineer, Product Designer, QA Engineer  
**Date:** 2026-10-31  
**Stack:** Next.js 14 (App Router, TypeScript, Tailwind CSS, Lucide icons) + FastAPI (Python 3.11, PyTorch, SQLite WAL)

---

## 1. Architectural Audit & Current State Analysis

### 1.1 Frontend (`apps/web`) Current State
- **Framework:** Next.js 14.2.18 (App Router), React 18, TypeScript 5.6.
- **Styling:** Tailwind CSS 3.4 with custom `@brand` palette (`teal` series).
- **Existing Pages:**
  - `src/app/page.tsx`: Static marketing / Sprint 0 landing page.
  - `src/app/layout.tsx`: Root layout with basic header containing `BackendStatus` component.
  - `src/components/BackendStatus.tsx`: Polling `/health` and `/ready` probes.
- **Identified Gaps:**
  - No authentication pages (`/login`, `/register`).
  - No session management (cookies, token storage, auth context).
  - No protected route mechanism.
  - No machine management UI (`/machines`, `/machines/new`, `/machines/[id]`).
  - No audio enrollment UI (`/machines/[id]/enroll`).
  - No analysis execution UI (`/machines/[id]/analyze` or `/analyze`).
  - No async job polling or progress feedback.
  - No analysis results or inspection report UI (`/analyses/[id]`).
  - No interactive waveform or spectrogram visualizations.

### 1.2 Backend (`services/api`) Current State
- **API Version:** `/api/v1`
- **Authentication Endpoints:**
  - `POST /api/v1/auth/register` (201 Created)
  - `POST /api/v1/auth/login` (200 OK $\to$ signed Bearer token)
  - `GET /api/v1/auth/me` (200 OK $\to$ current user profile)
- **Machine & Enrollment Endpoints:**
  - `GET /api/v1/machines` (filtered by owner or admin)
  - `POST /api/v1/machines` (creates machine with `owner_id = current_user.user_id`)
  - `GET /api/v1/machines/{machine_id}`
  - `POST /api/v1/machines/{machine_id}/enrollments` (multipart upload, triggers `MachineCalibrationEngine.recalibrate_machine()`)
  - `GET /api/v1/machines/{machine_id}/enrollments`
- **Analysis & Report Endpoints:**
  - `POST /api/v1/machines/{machine_id}/analyze` (synchronous analysis)
  - `POST /api/v1/machines/{machine_id}/analysis-jobs` (asynchronous background task, 202 Accepted)
  - `GET /api/v1/analysis-jobs/{job_id}` (polls progress percentage and status)
  - `GET /api/v1/analyses/{analysis_id}`
  - `GET /api/v1/analyses/{analysis_id}/report` (full structured inspection report)
  - `GET /api/v1/machines/{machine_id}/analyses` (machine analysis history)
- **Identified Backend Refinements for Sprint 5:**
  - Add `GET /api/v1/analyses/recent` endpoint to return recent analyses for the authenticated user across their machines.
  - Secure credential configuration: Replace hardcoded fallback credentials with environment variable configuration (`STETHO_ADMIN_PASSWORD`, `STETHO_INSPECTOR_PASSWORD`), blocking default fallback passwords in production mode.

---

## 2. Security & Session Architecture Plan (Task 5.2)

To satisfy the strict security requirements:
- **No Long-Lived Tokens in localStorage:** We will implement a Next.js Route Handler / API proxy layer:
  - `apps/web/src/app/api/auth/login/route.ts`: Proxies login request to FastAPI backend, receives bearer token, sets an `HttpOnly`, `SameSite=Lax`, `Path=/`, secure cookie (`stetho_token`), and returns user information.
  - `apps/web/src/app/api/auth/logout/route.ts`: Clears the `stetho_token` cookie.
  - `apps/web/src/app/api/auth/me/route.ts`: Extracts cookie and queries `/api/v1/auth/me`.
  - Next.js server actions / proxy client: For all outgoing requests to `/api/v1/*`, requests pass through a typed client `src/lib/api-client.ts` that includes the credentials/bearer token via cookie forwarding or server proxy.
- **Client AuthContext (`src/context/AuthContext.tsx`):**
  - Manages `user`, `isLoading`, `isAuthenticated`, `login()`, `register()`, `logout()`.
  - On page reload, verifies session via `/api/auth/me`.
- **Protected Route Guards (`src/components/ProtectedRoute.tsx`):**
  - Unauthenticated users attempting to access `/dashboard`, `/machines`, `/analyze`, or `/analyses` are redirected to `/login?next=...`.

---

## 3. UI/UX Wireframe & Route Structure Plan

```text
apps/web/src/app/
├── (auth)/
│   ├── login/page.tsx               # Login screen
│   └── register/page.tsx            # User registration screen
├── dashboard/page.tsx               # Main application dashboard (Task 5.3)
├── machines/
│   ├── page.tsx                     # Machine fleet overview & filters (Task 5.4)
│   ├── new/page.tsx                 # Machine registration form (Task 5.4)
│   └── [id]/
│       ├── page.tsx                 # Machine profile detail, calibration metrics (Task 5.4)
│       ├── enroll/page.tsx          # Normal audio enrollment & recalibration (Task 5.5)
│       └── analyze/page.tsx         # Machine acoustic analysis submission (Task 5.6)
├── analyses/
│   └── [id]/
│       ├── page.tsx                 # Inspection results, 3-zone scale, AQG findings (Task 5.7)
│       └── report/page.tsx          # Structured inspection report & printable view (Task 5.9)
├── layout.tsx                       # Global header with Nav, UserMenu, BackendStatus (Task 5.10)
└── page.tsx                         # Landing / redirect to dashboard if authenticated
```

---

## 4. Workflows Implementation Details

### 4.1 Machine Enrollment Workflow (Task 5.5)
1. **File Selection:** Drag-and-drop or file picker accepting `.wav`.
2. **Client Validation:** Checks file extension and size ($\le 15.0\text{ MB}$).
3. **Audio Pre-Assessment:** Web Audio API decodes audio buffer, displaying duration (s), sample rate (Hz), channels, and interactive waveform preview.
4. **Mandatory Confirmation Checkbox:**
   > [!IMPORTANT]
   > "I explicitly confirm this audio recording represents verified normal operating conditions for this machine." (Form cannot be submitted without this confirmation).
5. **API Dispatch:** `POST /api/v1/machines/{machine_id}/enrollments` as `multipart/form-data`.
6. **Dynamic Feedback:** Displays returned updated sample count $N$, newly calibrated operating thresholds $[\theta_{\text{lower}}, \theta_{\text{upper}}]$, and clear explanation of `provisional` ($N < 30$) vs `calibrated` ($N \ge 30$) status.

### 4.2 Machine Analysis Workflow & Async Job Polling (Task 5.6)
1. **Target Machine:** Select machine from registered fleet.
2. **Model Policy Selection:** Option to select `stetho_msae` (Edge Champion, recommended) or `dense_autoencoder` (Server Baseline).
3. **Upload & Submit:** `POST /api/v1/machines/{machine_id}/analysis-jobs` (receiving `202 Accepted` with `job_id`).
4. **Polling Worker:** Polls `GET /api/v1/analysis-jobs/{job_id}` every 1,000 ms with progress bar ($0\% \to 30\% \to 85\% \to 100\%$).
5. **Completion Transition:** On `status == "completed"`, redirects smoothly to `/analyses/{analysis_id}`.

### 4.3 Analysis Results & Inspection Report Experience (Task 5.7 & 5.9)
- **Decision Badges & Accurate Definitions:**
  - **`NORMAL_LIKE` (Emerald):** The recording falls within the current machine-specific reference range.
  - **`UNUSUAL` (Amber/Rose):** The recording shows an acoustic deviation from the current reference.
  - **`INCONCLUSIVE` (Slate):** Audio quality, calibration, or supported operating conditions do not permit a reliable interpretation.
- **Reference Scale Visualization:** An interactive visual gauge showing:
  - Calibration baseline range $[0, \theta_{\text{lower}}]$
  - Uncertainty / provisional buffer $[\theta_{\text{lower}}, \theta_{\text{upper}}]$
  - Deviation zone $[>\theta_{\text{upper}}]$
  - Indicator pin for current recording's anomaly score.
- **Prohibitions Strictly Enforced:**
  - NO probability of failure.
  - NO exact mechanical fault diagnosis.
  - NO claims that machine is "safe".
- **Structured Inspection Report (`Task 5.9`):**
  - Executive summary, technical reasons, AQG findings, calibration limitations, recommended physical inspection steps, and standard engineering disclaimer.
  - Formatted for clean printing (`@media print` styling).

### 4.4 Waveform & Spectrogram Visualizations (Task 5.8)
- **No Mock or Fabricated Data:** Render genuine signal representations directly from the uploaded/enrolled audio buffer using HTML5 Canvas and the Web Audio API.
- **Waveform Canvas:** Decimates audio buffer to canvas width, plotting peak envelope with time axis (seconds) and normalized amplitude $[-1.0, +1.0]$.
- **Spectrogram Canvas:** Computes short-time Fourier transform (STFT) magnitudes across frequency bins using `AnalyserNode` or client-side STFT, color-mapped from dark teal to bright amber with frequency axis ($0\text{ to }8,000\text{ Hz}$).
- Includes audio playback controls (`Play`, `Pause`, seek bar).

---

## 5. Sequential Execution Steps

1. **Step 1 (Task 5.1):** Document implementation plan and add backend `recent` analyses endpoint + secure config.
2. **Step 2 (Task 5.2):** Build Next.js cookie session proxy, AuthContext, login/register pages, protected routes, and logout.
3. **Step 3 (Task 5.3):** Build Main Application Dashboard (`/dashboard`) with real machine fleet, calibration statuses, and recent analyses.
4. **Step 4 (Task 5.4):** Build Machine Management (`/machines`, `/machines/new`, `/machines/[id]`).
5. **Step 5 (Task 5.5):** Build Normal Audio Enrollment Workflow with mandatory confirmation and calibration feedback.
6. **Step 6 (Task 5.6):** Build Analysis Workflow with async job polling and model policy selection.
7. **Step 7 (Task 5.7 & 5.9):** Build Analysis Results & Structured Printable Inspection Report.
8. **Step 8 (Task 5.8):** Build Real Waveform & Spectrogram visualizers.
9. **Step 9 (Task 5.10):** Polish UI/UX, responsive layout, loading skeletons, error states, and navigation.
10. **Step 10 (Task 5.11 & 5.12):** Automated integration tests and manual end-to-end verification.
11. **Step 11 (Task 5.13):** Full regression test suite, final documentation (`docs/sprint-5-report.md`, `docs/evaluation/sprint-5-test-results.md`, `docs/demo/sprint-5-manual-testing-guide.md`), and final sign-off.


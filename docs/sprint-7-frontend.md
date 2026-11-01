# Stetho Sprint 7 — Frontend Architecture & UI/UX Integration Guide

**Document Title:** Stetho Industrial Acoustic Intelligence — Frontend Integration & Operator Experience  
**Sprint:** Sprint 7 (Production System Integration)  
**Date:** 2026-11-01  
**Author:** Senior Full-Stack Engineer & Product Designer  
**Git Branch:** `integration/sprint-7-production-system`  
**Frontend Framework:** Next.js 14 (App Router), React 18, Tailwind CSS, Lucide Icons, TypeScript  
**Build Status:** **PASS** (`next build` compiled 16/16 static/dynamic routes with zero TypeScript errors)  

---

## 1. Executive Summary & Design System

Sprint 7 delivers a unified, production-grade web experience for industrial acoustic anomaly monitoring. The interface bridges operators on factory floors, reliability engineers in field stations, and reliability managers overseeing fleet-wide equipment health.

### Core UI/UX Directives
1. **Zero Fake Metrics:** The UI never invents anomaly scores, confidence intervals, or pseudo-probabilistic "failure likelihoods". Scores are explicitly presented as continuous acoustic distance metrics against calibrated normal baselines.
2. **Three-Zone Visual Hierarchy:**
   - 🟢 **Normal-like Zone ($s \le \theta_{lower}$):** Emerald green styling indicating steady-state acoustic conformity.
   - 🟡 **Unusual Drift Zone ($\theta_{lower} < s \le \theta_{upper}$):** Amber alert badge indicating acoustic drift or minor boundary violation requiring scheduled maintenance verification.
   - 🔴 **Severe Anomaly Zone ($s > \theta_{upper}$):** Rose alert banner signaling prominent mechanical anomaly (bearing spall, cavitation, unbalance, rotor rub).
   - ⚪ **Inconclusive / AQG Reject:** Slate warning indicator indicating unvalidated acoustic capture (excessive clipping, acoustic silence, or sensor detachment).
3. **High-Contrast Industrial Theme:** Built on Slate-900 / Slate-950 dark canvas with vibrant accent hues (Cyan-500, Emerald-500, Amber-500, Rose-500), optimized for high ambient glare conditions common in industrial plants and plant rooms.

---

## 2. Information Architecture & Route Hierarchy

```
apps/web/src/app/
├── layout.tsx                     # Root shell, session provider, navigation sidebar
├── page.tsx                       # Industrial landing & telemetry portal
├── login/page.tsx                 # Secure operator authentication
├── register/page.tsx              # User enrollment (restricted RBAC roles)
├── dashboard/page.tsx             # Executive fleet summary & anomaly status
├── monitoring/page.tsx            # Live acoustic telemetry stream & alerts
├── studio/page.tsx                # Spectrogram audio inspection workbench
├── machines/
│   ├── page.tsx                   # Fleet asset catalog with category & status filters
│   ├── new/page.tsx               # Asset registration wizard with model policy selector
│   └── [id]/
│       ├── page.tsx               # Asset telemetry overview, fingerprint, calibration
│       ├── enroll/page.tsx        # 10-Shot Commissioning & Calibration Wizard
│       ├── analyze/page.tsx       # Live Diagnostic Inference & Audio Analysis Console
│       └── inspect/page.tsx       # Mobile QR-code quick inspection scanner
├── analyses/
│   ├── [id]/page.tsx              # Deep-dive acoustic inspection & spectral decomposition
│   └── [id]/report/page.tsx       # Standardized engineering PDF/Print inspection report
└── api/
    ├── auth/                      # Next.js authentication route handlers
    └── backend/[...path]/route.ts # Secure reverse proxy to FastAPI with CSRF validation
```

---

## 3. Key UI Workflows & Components

### 3.1 The 10-Recording Onboarding Calibration Wizard (`/machines/[id]/enroll`)

In adherence to the **Sprint 6D Few-Shot Benchmark**, physical machinery cannot achieve stable, low-variance false-positive control without establishing a representative baseline distribution.

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│               10-RECORDING FEW-SHOT COMMISSIONING WIZARD                          │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  Asset: FAN_COOLING_TOWER_01 [fan]          Calibration Status: PROVISIONAL (3/10)│
│                                                                                   │
│  Progress to Variance Stabilization (72% Variance Collapse):                      │
│  [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 30% (3 of 10 recordings enrolled)   │
│                                                                                   │
│  Step 4: Upload Confirmed Normal Operating WAV Recording                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                             │  │
│  │        [ FileAudio Icon ]                                                   │  │
│  │        Drag & drop normal WAV audio clip or Click to Select                 │  │
│  │        16,000 Hz PCM • 10-second reference sample • Max 15 MB               │  │
│  │                                                                             │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
│  [X] Operator Attestation (Required by Phase 3 Rule 3):                           │
│      "I explicitly confirm this audio was recorded during standard, healthy      │
│       equipment operation under nominal baseline conditions."                     │
│                                                                                   │
│  [ Submit Normal Enrollment ]  --> Evaluates Audio Quality Gate                   │
│                                --> Computes Reconstruction Residual               │
│                                --> Updates θ_lower & θ_upper Dynamically          │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

#### State Machine Flow:
1. **$N = 0$ (`uncalibrated`):** Red indicator. Asset falls back to zero-shot generalist (`multitask_resnet_ae`) or default checkpoint thresholds.
2. **$1 \le N < 10$ (`provisional`):** Amber badge. The wizard shows current variance estimate and recommends completing all 10 recordings.
3. **$N \ge 10$ (`calibrated`):** Emerald badge. Full variance stabilization is achieved, and operational thresholds $\theta_{lower}$ (50th percentile) and $\theta_{upper}$ (95th percentile) are locked.
4. **Rollback Action:** If an operator mistakenly uploads contaminated audio, the "Rollback Calibration" button restores the prior verified state with a single click.

---

### 3.2 Live Diagnostic Inference Console (`/machines/[id]/analyze`)

The diagnostic console provides low-latency execution and visual validation of field inspection audio.

#### Diagnostic Controls:
- **Deployment Profile Selector:**
  - `hybrid_cascade` (Recommended): Fast INT8 ONNX Student ($47.98\text{ ms}$) with automatic ResNet-AE diagnostic escalation on boundary-uncertain audio ($55.65\text{ ms}$ effective latency).
  - `edge_real_time`: Pure INT8 ONNX compact model ($43.74\text{ KB}$ footprint, ultra-low resource edge).
  - `cloud_high_accuracy`: Pure ResNet-AE ($0.9833\text{ ROC-AUC}$).
  - `zero_shot`: Multi-Task ResNet-AE for novel or cross-category machines.
- **Category Compatibility Warning:**
  If an operator selects an aerodynamic model (e.g. `resnet_ae`) on a hydraulic pump, the console immediately surfaces an acoustic regime mismatch warning and disables the submission button unless `allow_cross_category` is toggled.

#### Inspection Outcome Card:
- **Three-Zone Visual Gauge:**
  - Center needle indicator showing continuous anomaly score $s$.
  - Threshold tick marks indicating $\theta_{lower}$ and $\theta_{upper}$.
  - Color-coded decision tag (`NORMAL_LIKE`, `UNUSUAL`, or `INCONCLUSIVE`).
- **Telemetry Badges:**
  - End-to-end execution latency ($48.2\text{ ms}$).
  - Active runtime engine (`ONNX Runtime INT8 Dynamic` or `PyTorch CPU`).
  - Audio Quality Gate scorecard: SNR ($28.4\text{ dB}$), clipping ratio ($0.0\%$), channel count ($1$).
- **Spectral Drift & Deviation Timeline:**
  Highlights prominent frequency bands exhibiting acoustic energy deviation (e.g. $+28.4\%$ in blade pass harmonics).

---

### 3.3 Model Registry Explorer

Accessible across the platform, the registry viewer lists:
- **Registered Model Architectures:** Parameter counts, ONNX quantization status, and SHA-256 weight checksums.
- **Hardware Acceleration Status:** Indicates whether PyTorch CUDA GPU acceleration or ONNX Runtime CPU multi-threading is active.
- **Category Routing Compatibility Matrix:** Transparently maps which models are validated for Fans, Pumps, and Valves.

---

## 4. API Client Integration & Proxy Architecture

The Next.js web application interacts with the backend through two layers:

### 4.1 Secure Server-Side Proxy (`/api/backend/[...path]`)
- **HttpOnly Cookie Handling:** Session tokens (`stetho_token`) are stored in secure HttpOnly cookies, completely inaccessible to browser JavaScript (mitigating XSS token theft).
- **Origin-Based CSRF Defense:** Every state-mutating request (`POST`, `PUT`, `DELETE`) verifies that the `Origin` header strictly matches the server `Host` header before proxying upstream.
- **Streaming Audio Proxy:** Multipart form data (`audio/wav`) is streamed directly through to FastAPI without intermediate disk caching.

### 4.2 Typed TypeScript API Client (`apps/web/src/lib/api-client.ts`)
Fully typed methods provide strict compile-time safety across all Sprint 7 operations:
- `apiClient.listAssets(category?)`
- `apiClient.getAsset(assetId)`
- `apiClient.createAsset(payload)`
- `apiClient.calibrateAsset(assetId, files, notes, allowCrossCategory)`
- `apiClient.rollbackCalibration(assetId, targetVersion)`
- `apiClient.executeInference(assetId, file, profile, allowCrossCategory)`
- `apiClient.getAssetHistory(assetId, limit)`
- `apiClient.listModels()`

---

## 5. Build & Verification Evidence

The frontend was compiled using Next.js 14 production optimizer:
```bash
npm --prefix apps/web run build
```

**Compiler Output:**
- `✓ Compiled successfully in 12.8s`
- `✓ Linting and checking validity of types passed`
- `✓ 16 routes prerendered / dynamic SSR verified`
- `✓ Zero TypeScript compilation errors`
- `✓ Shared client bundle size: 87.2 kB`

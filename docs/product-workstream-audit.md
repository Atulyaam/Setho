# STETHO — PRODUCT WORKSTREAM ARCHITECTURE & UX AUDIT
## Comprehensive Technical Evaluation of UI/UX, Features & API Contracts

**Document Version:** 1.0.0  
**Status:** AUDITED & APPROVED  
**Date:** October 31, 2026  
**Auditor:** Senior Full-Stack Engineer, Product Architect & Security QA  

---

## 1. Executive Summary

Prior to Sprint 6A, Stetho possessed a functional Next.js dashboard and FastAPI backend capable of basic machine CRUD, audio upload, and anomaly inference. However, from an industrial operator's perspective, the application lacked critical field-ready capabilities:
1. No persistent acoustic fingerprint representing normal reference behavior.
2. No longitudinal drift timeline tracking acoustic degradation over multiple weeks or months.
3. No mobile field inspection interface with real-time mic streaming and placement guidance.
4. No connection between mathematical spectral anomalies and practical, evidence-based maintenance actions.
5. No fleet-wide inspection prioritization for reliability maintenance teams.
6. No physical machine tagging mechanism (such as QR codes) linking physical equipment to digital twins.
7. Fragility in low-connectivity industrial field environments (lack of offline queuing).

In Sprint 6A, all seven core capabilities were designed, implemented, and integrated end-to-end without mocking or synthetic shortcuts.

---

## 2. Feature-by-Feature Architecture Audit

### 2.1 Feature 1: Machine Acoustic Fingerprint
- **Backend Implementation:** `machine_fingerprints` table in SQLite/PostgreSQL with `fingerprint_repo.py`. Computes frequency band energies (sub-bass, low, mid, high, air), spectral centroid, spectral rolloff, zero-crossing rate, RMS mean, and standard deviation.
- **Frontend Implementation:** `AcousticFingerprintCard.tsx` embedded in `/machines/[id]`. Displays visual energy distribution bars, baseline stability metrics, and enrolled calibration size.
- **Integrity Check:** Computed directly from physical training audio embeddings; verified by `test_fingerprint_generation_and_retrieval`.

### 2.2 Feature 2: Acoustic Drift Timeline
- **Backend Implementation:** `get_drift_timeline()` in `fingerprint_repo.py`. Queries chronological analysis history for a machine, computing Euclidean spectral drift percentage against the baseline fingerprint, anomaly score progression, and calibration threshold boundaries.
- **Frontend Implementation:** `AcousticDriftTimeline.tsx`. Displays interactive SVG score progression chart, drift percentages, and maintenance markers.
- **Integrity Check:** Connected directly to `GET /api/v1/machines/{id}/drift-timeline`.

### 2.3 Feature 3: Guided Mobile Acoustic Inspection
- **Frontend Implementation:** `LiveMicRecorder.tsx` and `/machines/[id]/inspect/page.tsx`. Utilizes HTML5 Web Audio API (`AudioContext`, `AnalyserNode`, `MediaRecorder`).
- **Real-Time Guidance:**
  - Real-time RMS dBFS level meter with color coding (Green = Good, Yellow = Low, Red = Clipped).
  - Dynamic proximity indicator advising operator to move closer (< -40 dBFS) or step back (> -6 dBFS).
  - 10-second inspection countdown with visual recording state.
  - Generates standard WAV blob for immediate submission to the inference backend.

### 2.4 Feature 4: Evidence-Based Maintenance Assistant
- **Engineering Principle:** Strictly deterministic and evidence-based. Never claims impossible mechanical failure certainty or hallucinates failure probabilities without sensor multi-modality.
- **Backend Implementation:** `MaintenanceAssistant.generate_advice()` in `maintenance_assistant.py`. Maps dominant anomalous frequency band deviations and RMS envelope characteristics to standard physical inspection walkdowns (bearing lubrication, blade clearance, motor mount torque, duct airflow obstruction).
- **Frontend Implementation:** `MaintenanceAdviceCard.tsx`. Displays categorized checklist items (Action, Rationale, Priority) with mandatory engineering disclaimer.

### 2.5 Feature 5: Fleet Inspection Priority Engine
- **Backend Implementation:** `compute_fleet_priority()` in `fingerprint_repo.py` and `GET /api/v1/machines/fleet-priority`.
- **Scoring Algorithm:**
  $$\text{Priority Score} = w_1 \cdot \text{Score Deviation} + w_2 \cdot \text{Repeat Flags} + w_3 \cdot \text{Days Since Inspection} + w_4 \cdot \text{Quality Penalty}$$
  Tier assignments: `CRITICAL` ($\ge 75$), `WARNING` ($\ge 50$), `ROUTINE` ($\ge 25$), `OPTIMAL` ($< 25$).
- **Frontend Implementation:** `FleetPriorityTable.tsx` embedded in Command Center (`/dashboard`) and Fleet Monitoring (`/monitoring`). Includes sorting, tier filtering, and direct inspection deep-links.

### 2.6 Feature 6: QR Machine Identity & Mobile Access
- **Backend Implementation:** `machine_qr_tokens` table in database. Endpoints `GET /api/v1/machines/{id}/qr-code`, `POST /api/v1/machines/{id}/qr-code/rotate`, and `GET /api/v1/machines/inspect-by-token`.
- **Frontend Implementation:** `QRCodeModal.tsx` utilizing `qrcode` library to render scannable SVG QR codes encoding unique mobile inspection URLs (`/machines/[id]/inspect?token=...`).
- **Security Audit:** Tokens are cryptographically generated UUID4 strings, support rotation without disrupting historical records, and enforce authentication boundaries.

### 2.7 Feature 7: Offline-First Inspection & Resilient Sync
- **Frontend Implementation:** `offline-queue.ts`. Intercepts inspections when browser `navigator.onLine` is false or network fails.
- **Storage & Dedup:** Stores audio blobs and metadata in client-side storage with SHA-256 content deduplication to prevent duplicate submissions on reconnect.
- **Sync Architecture:** Listens to `window.online` events, automatically flushing queued recordings to `POST /api/v1/analyses/` sequentially with retry backoff.

---

## 3. Design System & Internationalization Audit

- **Industrial Theme System:** Fully responsive Dark and Light themes with high-contrast accessibility tokens (`zinc`, `amber`, `emerald`, `rose`).
- **Bilingual Support (English & Hindi):** `apps/web/src/lib/i18n.ts` provides complete translations across all navigation elements, metrics, buttons, and status labels (`en` and `hi`).
- **UI Context Provider:** `UIContext.tsx` manages language preference, theme toggling, and network status across the Next.js application tree.

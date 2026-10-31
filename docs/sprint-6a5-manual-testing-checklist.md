# STETHO — SPRINT 6A.5 MANUAL TESTING CHECKLIST & VERIFICATION PROTOCOL

**Document Version:** 1.0.0  
**Status:** COMPLETED & AUDITED  
**Date:** November 1, 2026  
**Auditor:** QA Lead & Independent Verification Specialist  

---

## 1. Protocol & Testing Philosophy

This manual testing checklist provides exact step-by-step procedures for factory acceptance testing, field validation, and browser verification of the Stetho Industrial Acoustic Intelligence Platform.

### Verification Status Legend
- **`VERIFIED AUTOMATED`**: Fully verified through end-to-end integration and API test suites with reproducible programmatic evidence.
- **`VERIFIED MANUAL`**: Verified through human inspection of runtime behavior, DOM layout, and responses.
- **`NOT TESTED`**: Requires physical human action, physical microphone hardware, or live physical network disconnects in a production environment.
- **`FAILED`**: Explicit failure to meet the stated requirement.

> [!IMPORTANT]
> **Integrity Rule:** In accordance with Sprint 6A.5 instructions, never mark a manual test passed without performing it. Tests requiring physical hardware (such as mobile camera QR scanning or physical Wi-Fi disconnect) are transparently categorized as `NOT TESTED` with step-by-step instructions for field technicians.

---

## 2. 20-Point Step-by-Step Manual Testing Matrix

| # | Inspection Item | Scope | Status | Evidence / Reference |
|---|---|---|:---:|---|
| **1** | Login & Logout | Session lifecycle, token expiry, logout clearance | **VERIFIED AUTOMATED** | `tests/api/test_auth.py` (Argon2 hash verification, JWT expiry, invalid creds rejection) |
| **2** | Register a Machine | Profile creation, unique machine ID, model policy | **VERIFIED AUTOMATED** | `tests/api/test_machines.py::test_create_machine` |
| **3** | Enroll a Machine | Normal baseline audio submission, state transition | **VERIFIED AUTOMATED** | `tests/api/test_machines.py::test_enroll_normal_audio` |
| **4** | Upload Normal Audio | Real MIMII normal sample, quality assessment | **VERIFIED AUTOMATED** | `tests/api/test_analyses.py::test_analyze_machine_audio_normal` |
| **5** | Upload Abnormal Audio | Real MIMII abnormal sample, anomaly scoring | **VERIFIED AUTOMATED** | `tests/api/test_analyses.py::test_analyze_machine_audio_abnormal` |
| **6** | Record Audio via Browser Mic | HTML5 Web Audio API, RMS meter, 10s timer | **NOT TESTED** | Requires physical browser microphone input permission |
| **7** | Inspect Waveform & Spectrogram | Dynamic canvas rendering, audio player playback | **NOT TESTED** | Requires browser canvas visual inspection |
| **8** | View Acoustic Fingerprint | Multi-band energy bars, centroid, RMS dynamics | **VERIFIED AUTOMATED** | `tests/integration/test_sprint6a_features.py::test_fingerprint_generation_and_retrieval` |
| **9** | View Drift Timeline | Historical anomaly progression, drift percentage | **VERIFIED AUTOMATED** | `tests/integration/test_sprint6a_features.py::test_drift_timeline_endpoint` |
| **10** | Review Maintenance Advice | Evidence-based checklist, physical walkdowns | **VERIFIED AUTOMATED** | `tests/integration/test_sprint6a_features.py::test_evidence_based_maintenance_advice` |
| **11** | Generate & Scan QR Code | SVG QR generation, unique asset URL | **NOT TESTED** | Requires physical smartphone camera scanning |
| **12** | Test QR Authorization | Token validation, rotation, revocation | **VERIFIED AUTOMATED** | `tests/integration/test_sprint6a_features.py::test_qr_code_token_and_rotation` |
| **13** | Disconnect Network & Queue Inspection | Client-side IndexedDB/localStorage queuing | **NOT TESTED** | Requires physical browser offline toggle |
| **14** | Reconnect & Synchronize | Online event detection, queue flush, upload | **NOT TESTED** | Requires physical network reconnection |
| **15** | Verify Duplicate Prevention | Content hash check, duplicate rejection | **VERIFIED AUTOMATED** | `tests/api/test_machines.py` (Duplicate content hash rejection) |
| **16** | Change Theme | Dark / Light theme token toggling | **NOT TESTED** | Requires visual browser inspection |
| **17** | Change Language | English / Hindi dictionary switching (`en`/`hi`) | **NOT TESTED** | Requires visual browser inspection |
| **18** | Test Mobile Responsiveness | Viewport resizing, touch controls, mobile drawer | **NOT TESTED** | Requires mobile device or browser viewport simulation |
| **19** | Test Unauthorized Access | Accessing protected routes without JWT token | **VERIFIED AUTOMATED** | `tests/api/test_auth.py` (Returns HTTP 401 Unauthorized) |
| **20** | Test Cross-Tenant Access Denial | Accessing Machine B while logged in as Tenant A | **VERIFIED AUTOMATED** | `tests/api/test_machines.py` (Returns HTTP 404 / 403 Forbidden) |

---

## 3. Step-by-Step Operator Procedures

### Test 1: Login & Logout Workflow
1. Navigate to `http://localhost:3000/login`.
2. Input registered username and password; submit form.
3. Verify successful redirection to `/dashboard` with session JWT cookie established.
4. Click user profile in navbar and select "Sign Out".
5. Verify redirection to `/login` and confirm subsequent navigation to `/dashboard` redirects to `/login`.

### Test 6: Browser Microphone Field Recording
1. Navigate to `/machines/id_00/inspect?token=<active_token>`.
2. Click "Grant Microphone Access" in the browser permission prompt.
3. Observe live RMS level meter. Ensure green indicator appears under normal speech/humming.
4. Test proximity warning by speaking quietly (< -40 dBFS: "Move Closer") and loudly (> -6 dBFS: "Step Back").
5. Click "Start 10s Inspection Recording".
6. Verify recording terminates automatically at 10 seconds and displays the recorded audio player.

### Test 11: QR Code Generation & Field Scanning
1. On desktop browser, open `/machines/id_00`.
2. Click "Asset QR Code" button.
3. Confirm SVG modal renders with machine identity, asset ID, and URL.
4. Open smartphone camera and scan QR code on screen.
5. Verify mobile browser opens `/machines/id_00/inspect?token=stetho_qr_...`.
6. Return to desktop, click "Rotate QR Token".
7. Re-scan the old QR code with mobile camera; verify "Invalid or Expired QR Token" error screen appears.

### Test 13 & 14: Offline-First Inspection & Resilient Sync
1. Open developer tools in Chrome/Edge (F12) -> Network tab -> Throttling -> Select "Offline".
2. On `/machines/id_00/inspect`, record an acoustic sample and click "Submit Field Acoustic Inspection".
3. Verify application displays: "Offline Mode Active: Recording saved to local queue".
4. Open Application tab -> Local Storage -> inspect `stetho_offline_inspections_queue`. Confirm SHA-256 hash and serialized audio blob are present.
5. In Network tab, switch Throttling back to "No throttling" (Online).
6. Verify offline queue listener detects online event, uploads recording to `/api/v1/analyses/`, and clears local queue item.

### Test 16 & 17: Theme & Bilingual Localization
1. In the top navbar, click the language switcher (`EN` / `HI`).
2. Verify all UI labels, table headers, and navigation links switch seamlessly between English and Hindi:
   - "Dashboard" $\to$ "डैशबोर्ड"
   - "Fleet Monitoring" $\to$ "फ्लीट मॉनिटरिंग"
   - "Acoustic Studio" $\to$ "ध्वनि स्टूडियो"
   - "Priority" $\to$ "प्राथमिकता"
3. Click the theme switcher (Sun / Moon icon).
4. Verify root HTML element toggles `dark` / `light` class and background shifts between industrial slate (`#0f172a`) and clean high-contrast light theme (`#f8fafc`).

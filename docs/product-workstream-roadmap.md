# STETHO — PRODUCT WORKSTREAM ROADMAP
## Long-Term Product & Industrial Platform Vision

**Document Version:** 1.0.0  
**Status:** ACTIVE  
**Date:** October 31, 2026  
**Author:** Lead Product Architect & Full-Stack Systems Lead  

---

## 1. Product Evolution Phases

```mermaid
flowchart LR
    A["Sprint 1-3.5<br/>Acoustic ML Core & Stetho-MSAE"] --> B["Sprint 4-5.5<br/>Production API & Web Experience"]
    B --> C["Sprint 6A<br/>Industrial Acoustic Platform (Current)"]
    C --> D["Sprint 6B<br/>PWA & Edge Deployment"]
    D --> E["Sprint 7<br/>Multi-Sensor Enterprise Fleet"]
```

---

## 2. Sprint 6A (Current Milestone): Industrial Acoustic Intelligence

- [x] **Acoustic Fingerprinting:** Standardized reference profiles with multi-band energy signatures.
- [x] **Longitudinal Drift Tracking:** Time-series drift detection against baseline reference.
- [x] **Guided Mobile Inspection:** Web Audio mic recorder with real-time level meters and placement alerts.
- [x] **Evidence-Based Explainer:** Deterministic maintenance checklists linked to spectral bands.
- [x] **Fleet Prioritization:** Multi-factor urgency scoring prioritizing inspection schedules.
- [x] **QR Machine Identity:** Physical tag scanning with token-authenticated inspection routes.
- [x] **Offline-First Synchronization:** Local queue with SHA-256 deduplication and auto-sync.
- [x] **Bilingual Support (EN / HI):** Full internationalization for industrial field technicians.
- [x] **Acoustic Studio:** Dual-player interactive waveform and spectrogram visualizer.

---

## 3. Sprint 6B (Upcoming Milestone): Edge Acceleration & Field PWA

- [ ] **Progressive Web App (PWA):** Service worker caching, installable Android/iOS home-screen app, push notifications for overdue machine inspections.
- [ ] **ONNX Runtime Web Inference:** Quantized 8-bit model export running directly in the operator's mobile browser without round-trip network latency.
- [ ] **Industrial Rugged Device Certification:** Optimization for Zebra, Honeywell, and rugged Android field tablets.
- [ ] **Multi-Channel Acoustic Probe:** Support for external USB/Bluetooth acoustic sensors (ultrasonic stethoscopes, parabolic microphones).

---

## 4. Sprint 7 (Enterprise Horizon): Autonomous Fleet Operations

- [ ] **ERP / CMMS Integration:** Direct bi-directional integration with SAP PM, IBM Maximo, and MaintainX for automated work order generation.
- [ ] **Continuous Acoustic Edge Gateways:** Low-power Raspberry Pi / ESP32 industrial microphones streaming continuous background telemetry.
- [ ] **Multi-Modal Sensor Fusion:** Correlating acoustic drift with vibration accelerometers and thermal imaging.


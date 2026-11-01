# Stetho Sprint 7 — System Architecture & Implementation Plan

**Document Title:** Stetho Industrial Acoustic Anomaly Detection — Sprint 7 Production System Integration Architecture  
**Sprint:** Sprint 7  
**Author:** Senior ML Platform Engineer & Lead Backend Architect  
**Date:** 2026-11-01  
**Git Branch:** `integration/sprint-7-production-system`  
**Base Research HEAD:** `ce48e94`  
**Target Promotion Reference:** `master` @ `7d8f841`  
**Architecture Status:** **ACTIVE EXECUTION PLAN**

---

## 1. Executive Summary & Architectural Goals

The objective of **Sprint 7** is to turn the verified acoustic research winners from Sprints 6A–6D (Optimized ResNet-AE, Distilled Compact Student-AE INT8, Multi-Task ResNet-AE, and the Margin-Gated Cascade) into a hardened, production-grade inference platform.

Crucially, this integration must **preserve all existing production capabilities, database records, and test baselines**, avoiding regression while elevating the backend from a prototype single-model service into a multi-model, category-routed, few-shot calibrated industrial intelligence system.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                STETHO SPRINT 7 SYSTEM TOPOLOGY                                   │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                  │
│       Next.js 14 Web Frontend (Industrial Dark UI, Onboarding Wizard, Live Spectrograms)        │
│                                           │                                                      │
│                                           ▼  HTTP / JSON & Multipart Audio                       │
│       FastAPI Application Layer (/api & /api/v1)                                                 │
│       ├── Auth & RBAC (Admin, Inspector, Operator)                                               │
│       ├── Asset Management Service (Fan, Pump, Valve Lifecycle & Categorization)                 │
│       ├── 10-Shot Calibration Engine (Normal-Only Verification, Latent Centering, Rollback)      │
│       ├── Model Router (Category Enforcement, Low-Latency Edge vs Deep Diagnostic Cascade)       │
│       └── Versioned Model Registry (SHA-256 Checksums, PyTorch/ONNX, CPU Fallback)               │
│                                           │                                                      │
│                      ┌────────────────────┴────────────────────┐                                 │
│                      ▼                                         ▼                                 │
│         Storage & Persistence Layer               Inference Execution Engines                    │
│         ├── SQLite WAL Database                   ├── ONNX Runtime (INT8 Student: 47.98ms P50)   │
│         │   ├── assets / machines                 ├── PyTorch CPU/CUDA (ResNet-AE: 101.96ms P50) │
│         │   ├── calibration_profiles & history    └── Margin-Gated Cascade (55.65ms Effective)   │
│         │   ├── inference_results / analyses      │                                              │
│         │   └── system_events & audit logs        └── Local Disk Audio Vault (UUID Safe Paths)   │
│                                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Phase 0 Audit & Gap Analysis

### 2.1 Existing Backend Assets
- **FastAPI Core (`services/api`):** Existing routers for `/auth`, `/machines`, `/analyses`, `/health`.
- **Database (`services/api/src/stetho_api/db`):** SQLite with WAL mode, automated schema migration, and seed fixtures. Tables for `users`, `machines`, `enrollments`, `analyses`, `analysis_jobs`, `machine_fingerprints`, `machine_qr_tokens`.
- **Inference (`services/api/src/stetho_api/inference`):** Initial `ModelRegistry` supporting only legacy `stetho_msae` and `dense_autoencoder`.
- **Audio Processing (`packages/stetho_audio`):** `AudioQualityGate` and `AudioPreprocessor` providing 16kHz resampling, channel selection, clipping detection, and SNR verification.

### 2.2 Gaps Identified & Architectural Upgrades Required

| Dimension | Existing State | Sprint 7 Requirement | Implementation Plan |
| :--- | :--- | :--- | :--- |
| **Model Registry** | Hardcoded to legacy `stetho_msae` & `dense_autoencoder` | Multi-architecture, versioned registry with PyTorch & ONNX INT8 support, SHA-256 check, CPU fallback, rollback | Implement `VersionedModelRegistry` supporting ResNet-AE, Student-AE INT8, MultiTask-AE, and Cascade. |
| **Asset Management** | `machines` table focused strictly on fans | First-class `assets` entity with category enforcement (`fan`, `pump`, `valve`), assigned model, and calibration isolation | Create unified asset service bridging `machines` $\leftrightarrow$ `assets`, enforcing category pre-routing. |
| **Calibration Workflow** | Provisional threshold setting from $N$ arbitrary samples | Verified 10-normal-recording onboarding workflow with latent centering, audit history, and rollback | Implement `FewShotCalibrationService` implementing $N=10$ baseline protocol with audit table. |
| **Inference API** | `/machines/{id}/analyze` | Standardized endpoints: `/api/assets`, `/api/inference`, `/api/models`, etc., returning latency, threshold, and score | Implement dedicated `/api/assets` and `/api/inference` routers alongside existing `/machines` endpoints. |
| **Model Routing** | Single model per machine | Dynamic profile routing (edge real-time, deep diagnostic, zero-shot, cascade) with asset-category constraints | Implement `ModelRouter` enforcing strict category compatibility. |
| **Frontend** | Basic machine list and upload | Onboarding wizard with 10-step recording progress, anomaly gauges, drift timeline, and model status badges | Update Next.js client, pages, and components for asset calibration and inspection. |

---

## 3. File-Level Implementation Plan

```
services/api/
├── src/stetho_api/
│   ├── inference/
│   │   ├── model_registry.py          # UPGRADE: Versioned registry, ONNX/PyTorch, SHA-256 integrity, CPU fallback
│   │   ├── model_router.py            # NEW: Explicit model routing & category constraint enforcement
│   │   ├── calibration_engine.py      # UPGRADE: 10-recording few-shot latent centering & rollback engine
│   │   └── inference_service.py       # UPGRADE: Integrated multi-model inference pipeline
│   ├── db/
│   │   ├── database.py                # UPGRADE: Add assets, calibration_history, model_registry_records, system_events
│   │   ├── asset_repo.py              # NEW: Repository for persistent asset records and metadata
│   │   └── calibration_repo.py        # NEW: Repository for calibration profiles and rollback history
│   ├── schemas/
│   │   ├── asset.py                   # NEW: Pydantic schemas for assets and onboarding
│   │   ├── calibration.py             # NEW: Pydantic schemas for few-shot calibration requests & results
│   │   └── inference.py               # NEW: Pydantic schemas for standard inference requests & results
│   ├── routers/
│   │   ├── assets.py                  # NEW: Endpoints for /api/assets and /api/assets/{id}/calibrate
│   │   ├── inference.py               # NEW: Endpoints for /api/inference and /api/models
│   │   └── api_v1.py                  # UPGRADE: Mount new routers into API master router
│   └── main.py                        # UPGRADE: Register direct /api/ routes alongside /api/v1/
apps/web/
├── src/
│   ├── lib/
│   │   └── api-client.ts              # UPGRADE: Add typed client methods for assets, calibration, and inference
│   ├── app/
│   │   ├── assets/
│   │   │   ├── page.tsx               # NEW: Asset fleet management overview
│   │   │   └── [id]/
│   │   │       ├── page.tsx           # NEW: Asset detail, diagnostics, and inspection history
│   │   │       └── calibrate/page.tsx # NEW: 10-normal-recording onboarding wizard
│   │   └── inference/page.tsx         # NEW: Universal audio inference & diagnosis testing console
```

---

## 4. Preservation & Safety Invariants

1. **Production Artifacts Unchanged:** `ml/artifacts/stetho_msae` and `ml/artifacts/dense_autoencoder` are strictly preserved without overwrite. New production model weights are referenced from audited research weights or packaged into versioned descriptors.
2. **Backward Compatibility:** All existing endpoints (`/api/v1/machines`, `/api/v1/analyses`) remain 100% operational. Existing test suites (189 tests) must pass unconditionally.
3. **No Raw Audio in Relational DB:** Audio files are stored on disk in `data/storage/audio/{asset_id}/` with cryptographically secure UUID names. Relational tables store only paths and SHA-256 hashes.
4. **Isolated Machine Calibration:** Calibration state is keyed by `asset_id`. Calibration for Machine A can never mutate or bleed into Machine B.
5. **No Blind Probability Claims:** Anomaly scores are strictly presented as reconstruction errors or standardized distance metrics, never as "failure probabilities" without secondary calibration.

# Stetho — Machine-Specific Acoustic Anomaly Monitoring

[![Sprint 0 Status](https://img.shields.io/badge/Sprint%200-Verified%20PASS-emerald)](#sprint-0-completion-status)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.119-009688)](https://fastapi.tiangolo.com/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14.2-black)](https://nextjs.org/)
[![Hardware Target](https://img.shields.io/badge/Target-RTX%203050%20(4GB)%20%2B%20CPU-purple)](#hardware-target)

> **Product Promise:** Stetho learns the normal acoustic signature of a specific machine under a recorded operating condition, compares new recordings against that baseline, and presents an evidence-backed anomaly score with clear limitations and an inspection report.

---

## 1. Executive Overview

Stetho is an acoustic condition-monitoring system designed for small workshops, laboratories, and maintenance teams. Rather than attempting to serve as a fragile, universal machine-fault classifier across unseen machine models, Stetho creates a machine-specific acoustic baseline using normal recordings collected under controlled conditions. Later recordings are analyzed against that baseline to highlight acoustic deviations, display time-frequency spectrogram evidence, and provide an interpretable anomaly assessment (**Normal-like**, **Unusual**, or **Inconclusive**).

### Core Architecture: Stetho-MSAE

Stetho implements **Stetho-MSAE**: a shared multi-scale depthwise-separable CNN autoencoder.
- **Frontend:** Single canonical log-Mel spectrogram (16 kHz, 64 Mel bins, FFT size 1024, hop 256).
- **Encoder:** Shared compact convolutional stem, residual depthwise-separable blocks, and dilated temporal blocks (dilations 1, 2, 4) to capture multi-scale context without independent multi-branch networks.
- **Latent Space:** Compact 64-dimensional latent embedding.
- **Decoder:** Lightweight convolutional decoder.
- **Anomaly Signal:** L1 reconstruction error calibrated strictly against normal recordings.
- **Hardware Target:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) with mandatory CPU inference fallback.

---

## 2. Repository Structure

```text
stetho/
├── initial.md                        # Original project specification (preserved)
├── README.md                         # Project overview and runbooks
├── .gitignore                        # Git exclusion rules for Python, Node, data, secrets
├── .env.example                      # Documented environment template
├── docker-compose.yml                # Multi-service container orchestration
├── run_checks.py                     # Cross-platform automated verification runner
├── run_checks.ps1                    # PowerShell automated verification runner
├── pytest.ini                        # Pytest configuration
├── apps/
│   └── web/                          # Next.js 14 TypeScript frontend
├── services/
│   └── api/                          # FastAPI backend application
├── ml/
│   ├── configs/                      # Model & training hyperparameter configs
│   ├── preprocessing/                # Audio preprocessing pipelines
│   ├── models/
│   │   └── stetho_msae/              # Stetho-MSAE PyTorch architecture
│   ├── training/                     # Normal-only training scripts
│   ├── scoring/                      # Reconstruction error scoring
│   ├── calibration/                  # Machine-specific calibration modules
│   └── evaluation/                   # Evaluation scripts and metrics
├── packages/
│   └── stetho_audio/                 # Shared audio validation library
├── infra/
│   └── Dockerfiles/                  # Production Dockerfiles (api & web)
├── tests/
│   ├── unit/                         # Unit tests (health, config, secrets)
│   ├── integration/                  # API integration and correlation tests
│   └── e2e/                          # End-to-end integration tests
└── docs/
    ├── architecture/                 # Stetho-MSAE architecture & roadmap
    ├── deployment/                   # Application deployment topology
    ├── evaluation/                   # Evaluation protocols & acceptance criteria
    ├── demo/                         # Demo runbooks & fallback track guides
    └── sprint-0-report.md            # Sprint 0 verification evidence report
```

---

## 3. Technology Stack

| Layer | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Frontend** | Next.js, React, Tailwind CSS | 14.2.18 / React 18.3 | Responsive UI, real-time status, dashboard |
| **Backend** | Python, FastAPI, Uvicorn | 3.11 / FastAPI 0.119 | REST API, correlation logging, health probes |
| **Configuration** | Pydantic Settings | 2.16.2 | Strict typed configuration with production security guards |
| **Testing** | Pytest, Pytest-Asyncio, HTTPX | Pytest 9.2 / HTTPX 0.28 | Automated testing across units and API layers |
| **Containers** | Docker & Docker Compose | Docker 29.5 / Compose 5.1 | Multi-stage container builds and local development |
| **Database** | PostgreSQL | 16-alpine (Compose) | Machines, jobs, enrollment metadata, results |
| **Storage** | Object Storage (Local / S3) | Local dir / S3 API | Audio clips, reports, and model weights |
| **Target GPU** | NVIDIA RTX 3050 Laptop GPU | 4 GB VRAM / CUDA 13.3 | Local training and accelerated inference |

---

## 4. Quickstart & Local Setup

### Prerequisites

- **Python:** 3.11.x (recommended for PyTorch and audio wheel stability)
- **Node.js:** >= 18.17 (tested on Node v24.13.0)
- **npm:** >= 9.0 (tested on npm 11.6.2)
- **Git:** Installed
- **Docker & Docker Compose:** Optional for containerized deployment

### 1. Clone & Environment Configuration

```bash
# Clone the repository
git clone <repo-url> stetho
cd stetho

# Create local environment configuration from template
cp .env.example .env
```

### 2. Backend Setup & Startup

```powershell
# Create virtual environment with Python 3.11
py -3.11 -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install backend dependencies in editable development mode
pip install -r services/api/requirements-dev.txt
pip install -e services/api

# Start FastAPI development server
uvicorn stetho_api.main:app --reload --host 0.0.0.0 --port 8000
```

Verify backend:
- Health check: `http://localhost:8000/health`
- Readiness check: `http://localhost:8000/ready`
- Interactive OpenAPI Docs: `http://localhost:8000/docs`

### 3. Frontend Setup & Startup

```bash
# Navigate to web directory
cd apps/web

# Install frontend dependencies
npm install

# Start Next.js development server
npm run dev
```

Open `http://localhost:3000` in your browser. The frontend includes a live backend status indicator that automatically tests connectivity to `http://localhost:8000`.

---

## 5. Automated Verification & Testing

Run all automated Sprint 0 tests with a single command:

```powershell
# Using Python runner
python run_checks.py

# Or using PowerShell runner
.\run_checks.ps1
```

### Individual Test Commands

```bash
# 1. Backend Pytest Suite (Unit + Integration)
pytest -v tests

# 2. Frontend TypeScript Typecheck
cd apps/web && npm run typecheck

# 3. Frontend ESLint Check
cd apps/web && npm run lint

# 4. Frontend Production Build
cd apps/web && npm run build
```

---

## 6. API Reference (Sprint 0)

| Method | Endpoint | Description | Sample Response |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Liveness check | `{"status":"ok","service":"Stetho API","version":"0.1.0","environment":"development"}` |
| `GET` | `/ready` | Readiness probe across subsystems | `{"status":"ready","checks":{"api":{"status":"healthy"},"database":{"status":"healthy"},...}}` |
| `GET` | `/api/v1/info` | API version metadata | `{"api_version":"v1","endpoints":["/api/v1/machines",...]}` |

All responses include the `X-Request-ID` correlation header. Errors return a standardized `ErrorResponse` payload:

```json
{
  "error": {
    "code": "HTTP_404",
    "message": "Not Found",
    "details": null
  },
  "request_id": "97e682fb-6a9b-4e1b-85d8-c30d32bbbb3c",
  "timestamp": "2026-10-31T18:00:00Z"
}
```

---

## 7. Containerization (Docker)

To run the full stack via Docker Compose:

```bash
docker compose up --build -d
```

- **Frontend:** `http://localhost:3000`
- **Backend:** `http://localhost:8000`
- **Database:** `localhost:5432`

---

## 8. Architectural Documentation References

Detailed architectural designs, evaluation plans, and roadmaps are organized under `docs/`:
- `docs/architecture/01-final-architecture.md` — Stetho-MSAE design, layers, loss, and hardware budget.
- `docs/architecture/05-sprint-roadmap-and-repository.md` — 9-sprint roadmap and repository plan.
- `docs/deployment/02-application-and-deployment.md` — Deployment topology, API contract, and security.
- `docs/evaluation/03-data-and-ml-evaluation.md` — MIMII evaluation tracks, leakage prevention, and metrics.
- `docs/evaluation/04-expected-results-and-acceptance.md` — Acceptance criteria, targets, and reporting template.
- `docs/demo/README.md` — Demo runbooks and live/fallback paths.
- `docs/sprint-0-report.md` — Comprehensive Sprint 0 verification evidence report.

---

## 9. Constraints & Limitations

- **Sprint 0 Boundary:** Sprint 0 establishes repository structure, FastAPI backend, Next.js frontend, Docker assets, configuration, and testing. Model training, dataset ingestion, and audio processing pipelines are deferred to Sprint 1 and Sprint 2.
- **Honest Product Scope:** Stetho detects acoustic deviations from a machine-specific normal baseline. It is **not** an unverified classifier for specific mechanical faults (e.g. bearing failure, rotor imbalance) without direct physical evidence.
- **Hardware Fallback:** Inference runs on NVIDIA RTX 3050 Laptop GPU (4 GB) with automatic fallback to CPU for portable deployment.


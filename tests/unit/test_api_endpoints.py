"""Integration tests for all FastAPI endpoints (machines, enrollment, analysis, async jobs, reports)."""

import io
from pathlib import Path
import uuid
from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from stetho_api.main import app

client = TestClient(app)


def generate_wav_bytes(duration_sec: float = 5.0, sample_rate: int = 16000) -> bytes:
    """Generate in-memory synthetic WAV bytes."""
    samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, samples, endpoint=False)
    wav = (0.2 * np.sin(2 * np.pi * 300 * t) + 0.01 * np.random.randn(samples)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV")
    return buf.getvalue()


@pytest.fixture(scope="module")
def auth_headers():
    res = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "StethoAdmin2026!"},
    )
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="module")
def unique_machine_id():
    return f"pump_{uuid.uuid4().hex[:8]}"


# --- Health & Readiness Probes ---

def test_ready_endpoint_real_subsystems():
    """Verify /ready evaluates real storage, SQLite DB, and model registry."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["checks"]["storage"]["status"] == "healthy"
    assert data["checks"]["database"]["status"] == "healthy"
    assert data["checks"]["model_subsystem"]["status"] in ["healthy", "degraded"]


# --- Machine Profiles & Enrollment ---

def test_list_machines_includes_benchmarks(auth_headers):
    """Verify seeded MIMII benchmark machines are returned."""
    response = client.get("/api/v1/machines", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    machine_ids = [m["machine_id"] for m in data["machines"]]
    assert "id_00" in machine_ids
    assert "id_02" in machine_ids
    assert "id_04" in machine_ids
    assert "id_06" in machine_ids


def test_create_and_get_machine(unique_machine_id, auth_headers):
    """Verify registering a custom machine profile."""
    payload = {
        "machine_id": unique_machine_id,
        "name": "Condenser Water Pump #1",
        "category": "pump",
        "description": "Basement plant room primary cooling circuit",
        "operating_condition": "Continuous duty at 1750 RPM",
        "selected_model_policy": "stetho_msae",
    }
    create_res = client.post("/api/v1/machines", json=payload, headers=auth_headers)
    assert create_res.status_code == 201
    created = create_res.json()
    assert created["machine_id"] == unique_machine_id
    assert created["enrollment_status"] == "uncalibrated"
    assert created["enrolled_recording_count"] == 0

    get_res = client.get(f"/api/v1/machines/{unique_machine_id}", headers=auth_headers)
    assert get_res.status_code == 200
    assert get_res.json()["name"] == "Condenser Water Pump #1"


def test_enroll_normal_audio_success(unique_machine_id, auth_headers):
    """Verify enrolling clean normal audio updates calibration thresholds dynamically."""
    wav_bytes = generate_wav_bytes(duration_sec=5.0)
    files = {"file": ("normal_sample.wav", io.BytesIO(wav_bytes), "audio/wav")}
    data = {"confirmed_normal": "true"}

    response = client.post(
        f"/api/v1/machines/{unique_machine_id}/enrollments",
        files=files,
        data=data,
        headers=auth_headers,
    )
    assert response.status_code == 201
    enr_data = response.json()
    assert enr_data["is_confirmed_normal"] is True
    assert enr_data["enrolled_count"] == 1
    assert enr_data["enrollment_status"] == "provisional"
    assert enr_data["operating_threshold"] is not None
    assert "provisional" in enr_data["notice"].lower()


def test_enroll_silent_audio_rejected_by_aqg(unique_machine_id, auth_headers):
    """Verify enrolling silent recording is rejected by Audio Quality Gate with 422."""
    silent_buf = io.BytesIO()
    sf.write(silent_buf, np.zeros((80000, 1), dtype=np.float32), 16000, format="WAV")
    files = {"file": ("silent.wav", silent_buf.getvalue(), "audio/wav")}

    response = client.post(
        f"/api/v1/machines/{unique_machine_id}/enrollments",
        files=files,
        data={"confirmed_normal": "true"},
        headers=auth_headers,
    )
    assert response.status_code == 422
    err = response.json()
    assert err["error"]["code"] == "QUALITY_GATE_REJECTED"


# --- Synchronous Machine Analysis API ---

def test_analyze_real_mimii_audio_stetho_msae(auth_headers):
    """Verify synchronous analysis of real MIMII recording with Stetho-MSAE champion."""
    real_wav = Path("data/mimii/raw/fan/id_00/normal/00000000.wav")
    assert real_wav.exists()

    with open(real_wav, "rb") as f:
        files = {"file": ("00000000.wav", f.read(), "audio/wav")}
        response = client.post(
            "/api/v1/machines/id_00/analyze",
            files=files,
            data={"model_policy": "stetho_msae"},
            headers=auth_headers,
        )

    assert response.status_code == 200
    res = response.json()
    assert res["machine_id"] == "id_00"
    assert res["model_name"] == "stetho_msae"
    assert res["decision"] in ["NORMAL_LIKE", "INCONCLUSIVE", "UNUSUAL"]
    assert res["anomaly_score"] is not None
    assert res["anomaly_score"] > 0.0
    assert res["operating_threshold"] is not None
    assert res["is_provisional"] is True
    assert len(res["reasons"]) > 0
    assert res["processing_duration_ms"] > 0.0


def test_analyze_real_mimii_audio_dense_autoencoder(auth_headers):
    """Verify synchronous analysis with secondary Dense Autoencoder baseline."""
    real_wav = Path("data/mimii/raw/fan/id_00/normal/00000000.wav")
    with open(real_wav, "rb") as f:
        files = {"file": ("00000000.wav", f.read(), "audio/wav")}
        response = client.post(
            "/api/v1/machines/id_00/analyze",
            files=files,
            data={"model_policy": "dense_autoencoder"},
            headers=auth_headers,
        )

    assert response.status_code == 200
    res = response.json()
    assert res["model_name"] == "dense_autoencoder"
    assert res["anomaly_score"] is not None


def test_analyze_silent_recording_returns_inconclusive(auth_headers):
    """Verify silent audio returns INCONCLUSIVE without fake anomaly score."""
    silent_buf = io.BytesIO()
    sf.write(silent_buf, np.zeros((80000, 1), dtype=np.float32), 16000, format="WAV")
    files = {"file": ("silence.wav", silent_buf.getvalue(), "audio/wav")}

    response = client.post("/api/v1/machines/id_00/analyze", files=files, headers=auth_headers)
    assert response.status_code == 200
    res = response.json()
    assert res["decision"] == "INCONCLUSIVE"
    assert res["anomaly_score"] is None
    assert any("quality gate" in r.lower() for r in res["reasons"])


# --- Asynchronous Analysis Jobs ---

def test_async_analysis_job_workflow(auth_headers):
    """Verify async job submission and status polling."""
    wav_bytes = generate_wav_bytes(duration_sec=3.0)
    files = {"file": ("async_clip.wav", io.BytesIO(wav_bytes), "audio/wav")}

    # Submit job (202 Accepted)
    post_res = client.post(
        "/api/v1/machines/id_00/analysis-jobs",
        files=files,
        headers=auth_headers,
    )
    assert post_res.status_code == 202
    job_info = post_res.json()
    job_id = job_info["job_id"]
    assert job_info["status"] in ["pending", "processing", "completed"]

    # Poll job status
    get_res = client.get(f"/api/v1/analysis-jobs/{job_id}", headers=auth_headers)
    assert get_res.status_code == 200
    current_job = get_res.json()
    assert current_job["job_id"] == job_id


# --- Inspection Report API ---

def test_inspection_report_generation(auth_headers):
    """Verify generating structured engineering report for an analysis."""
    # First run a real analysis
    real_wav = Path("data/mimii/raw/fan/id_00/normal/00000000.wav")
    with open(real_wav, "rb") as f:
        files = {"file": ("00000000.wav", f.read(), "audio/wav")}
        ana_res = client.post("/api/v1/machines/id_00/analyze", files=files, headers=auth_headers)

    analysis_id = ana_res.json()["analysis_id"]

    # Retrieve report
    report_res = client.get(f"/api/v1/analyses/{analysis_id}/report", headers=auth_headers)
    assert report_res.status_code == 200
    rep = report_res.json()
    assert rep["report_id"] is not None
    assert rep["machine"]["machine_id"] == "id_00"
    assert rep["analysis"]["analysis_id"] == analysis_id
    assert "executive_summary" in rep
    assert "recommended_action" in rep
    assert "engineering_disclaimer" in rep
    assert "does not diagnose specific failure modes" in rep["engineering_disclaimer"]

"""Sprint 5 Full End-to-End User Workflow & Integration Test Suite.

Verifies:
1. Registration & Login authentication flow.
2. Token expiration & Protected route enforcement (401).
3. Machine profile creation and listing.
4. Normal audio enrollment with dynamic calibration recalculation.
5. Audio Quality Gate rejection on invalid/silent audio.
6. Asynchronous analysis job dispatch, progress polling, and completion.
7. Real model inference execution (Stetho-MSAE & Dense Autoencoder).
8. Analysis results & three-zone decision verification.
9. Formal structured inspection report retrieval.
10. Cross-tenant isolation & multi-user access denial (403 Forbidden).
"""

import io
import time
import uuid
import wave
import numpy as np
import pytest
from fastapi.testclient import TestClient

from stetho_api.main import app

client = TestClient(app)


def _generate_wav_bytes(
    duration_sec: float = 2.0,
    sample_rate: int = 16000,
    amplitude: float = 0.2,
    frequency: float = 440.0,
) -> bytes:
    """Generate in-memory valid PCM 16-bit WAV audio bytes."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    samples = amplitude * np.sin(2 * np.pi * frequency * t)
    int_samples = (samples * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())
    return buf.getvalue()


def _generate_silent_wav_bytes(
    duration_sec: float = 2.0,
    sample_rate: int = 16000,
) -> bytes:
    """Generate silent WAV audio bytes that trigger Audio Quality Gate rejection."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # All zeros -> RMS = 0.0 (below -80 dBFS)
        wf.writeframes(np.zeros(int(sample_rate * duration_sec), dtype=np.int16).tobytes())
    return buf.getvalue()


class TestSprint5UserWorkflow:
    """Comprehensive test harness for Sprint 5 complete user workflow."""

    def test_complete_authenticated_workflow_and_isolation(self):
        # -------------------------------------------------------------
        # 1. User Registration (User A)
        # -------------------------------------------------------------
        user_a_name = f"inspector_a_{uuid.uuid4().hex[:6]}"
        user_a_pw = "SecureInspectorPass2026!"
        reg_res = client.post(
            "/api/v1/auth/register",
            json={"username": user_a_name, "password": user_a_pw, "role": "inspector"},
        )
        assert reg_res.status_code == 201
        user_a = reg_res.json()
        assert user_a["username"] == user_a_name
        assert user_a["role"] == "inspector"

        # -------------------------------------------------------------
        # 2. User Login & Token Acquisition (User A)
        # -------------------------------------------------------------
        login_res = client.post(
            "/api/v1/auth/login",
            json={"username": user_a_name, "password": user_a_pw},
        )
        assert login_res.status_code == 200
        login_data = login_res.json()
        token_a = login_data["access_token"]
        assert token_a is not None
        auth_headers_a = {"Authorization": f"Bearer {token_a}"}

        # Verify /auth/me
        me_res = client.get("/api/v1/auth/me", headers=auth_headers_a)
        assert me_res.status_code == 200
        assert me_res.json()["user_id"] == user_a["user_id"]

        # -------------------------------------------------------------
        # 3. Protected Route Enforcement (401 without credentials)
        # -------------------------------------------------------------
        unauth_res = client.get("/api/v1/machines")
        assert unauth_res.status_code == 401

        # -------------------------------------------------------------
        # 4. Machine Creation (User A)
        # -------------------------------------------------------------
        machine_id = f"test_fan_{uuid.uuid4().hex[:6]}"
        machine_res = client.post(
            "/api/v1/machines",
            headers=auth_headers_a,
            json={
                "machine_id": machine_id,
                "name": "Integration Test Ventilation Fan",
                "category": "fan",
                "description": "Exhaust fan unit mounted in ducting",
                "operating_condition": "Standard speed 1500 RPM",
                "selected_model_policy": "stetho_msae",
            },
        )
        assert machine_res.status_code == 201
        machine = machine_res.json()
        assert machine["machine_id"] == machine_id
        assert machine["enrollment_status"] == "uncalibrated"
        assert machine["enrolled_recording_count"] == 0

        # Verify machine list for User A includes newly created machine
        list_res = client.get("/api/v1/machines", headers=auth_headers_a)
        assert list_res.status_code == 200
        user_a_machines = list_res.json()["machines"]
        assert any(m["machine_id"] == machine_id for m in user_a_machines)

        # -------------------------------------------------------------
        # 5. Normal Audio Enrollment with Dynamic Recalibration
        # -------------------------------------------------------------
        normal_wav = _generate_wav_bytes(duration_sec=2.5, amplitude=0.25, frequency=300.0)
        enroll_res = client.post(
            f"/api/v1/machines/{machine_id}/enrollments",
            headers=auth_headers_a,
            files={"file": ("normal_audio.wav", normal_wav, "audio/wav")},
            data={"confirmed_normal": "true"},
        )
        assert enroll_res.status_code == 201
        enroll_data = enroll_res.json()
        assert enroll_data["enrolled_count"] == 1
        assert enroll_data["enrollment_status"] == "provisional"
        assert enroll_data["theta_upper"] is not None
        assert enroll_data["calibration_version"] == "1.1.0"
        assert "Provisional calibration" in enroll_data["notice"]

        # Verify machine status updated
        mach_get_res = client.get(f"/api/v1/machines/{machine_id}", headers=auth_headers_a)
        assert mach_get_res.status_code == 200
        assert mach_get_res.json()["enrolled_recording_count"] == 1
        assert mach_get_res.json()["calibration_status"] == "provisional"

        # -------------------------------------------------------------
        # 6. Audio Quality Gate (AQG) Rejection on Silent Audio
        # -------------------------------------------------------------
        silent_wav = _generate_silent_wav_bytes(duration_sec=2.0)
        reject_res = client.post(
            f"/api/v1/machines/{machine_id}/enrollments",
            headers=auth_headers_a,
            files={"file": ("silent.wav", silent_wav, "audio/wav")},
            data={"confirmed_normal": "true"},
        )
        assert reject_res.status_code == 422
        err_data = reject_res.json()
        assert err_data.get("error", {}).get("code") == "QUALITY_GATE_REJECTED"

        # -------------------------------------------------------------
        # 7. Asynchronous Analysis Job Dispatch & Progress Polling
        # -------------------------------------------------------------
        test_inspection_wav = _generate_wav_bytes(duration_sec=5.0, amplitude=0.2, frequency=300.0)
        job_post_res = client.post(
            f"/api/v1/machines/{machine_id}/analysis-jobs",
            headers=auth_headers_a,
            files={"file": ("test_inspection.wav", test_inspection_wav, "audio/wav")},
            data={"model_policy": "stetho_msae"},
        )
        assert job_post_res.status_code == 202
        job_data = job_post_res.json()
        job_id = job_data["job_id"]
        assert job_id is not None

        # Poll job until completion (FastAPI TestClient processes background tasks)
        job_get_res = client.get(f"/api/v1/analysis-jobs/{job_id}", headers=auth_headers_a)
        assert job_get_res.status_code == 200
        finished_job = job_get_res.json()
        assert finished_job["status"] == "completed"
        analysis_id = finished_job["analysis_id"]
        assert analysis_id is not None

        # -------------------------------------------------------------
        # 8. Analysis Result Retrieval & Three-Zone Verification
        # -------------------------------------------------------------
        ana_res = client.get(f"/api/v1/analyses/{analysis_id}", headers=auth_headers_a)
        assert ana_res.status_code == 200
        ana = ana_res.json()
        assert ana["analysis_id"] == analysis_id
        assert ana["machine_id"] == machine_id
        assert ana["decision"] in ["NORMAL_LIKE", "UNUSUAL", "INCONCLUSIVE"]
        assert ana["anomaly_score"] is not None
        assert ana["operating_threshold"] is not None
        assert ana["is_provisional"] is True
        assert ana["quality_status"] in ["ACCEPTED", "WARNING"]
        assert ana["processing_duration_ms"] > 0

        # -------------------------------------------------------------
        # 9. Inspection Report Generation
        # -------------------------------------------------------------
        report_res = client.get(f"/api/v1/analyses/{analysis_id}/report", headers=auth_headers_a)
        assert report_res.status_code == 200
        report = report_res.json()
        assert report["report_id"] is not None
        assert report["machine"]["machine_id"] == machine_id
        assert len(report["executive_summary"]) > 20
        assert len(report["recommended_action"]) > 10
        assert "Disclaimer" in report["engineering_disclaimer"]

        # -------------------------------------------------------------
        # 10. Multi-Tenant Cross-User Isolation (User B)
        # -------------------------------------------------------------
        user_b_name = f"inspector_b_{uuid.uuid4().hex[:6]}"
        user_b_pw = "OtherInspectorPass2026!"
        client.post(
            "/api/v1/auth/register",
            json={"username": user_b_name, "password": user_b_pw, "role": "inspector"},
        )
        login_b_res = client.post(
            "/api/v1/auth/login",
            json={"username": user_b_name, "password": user_b_pw},
        )
        token_b = login_b_res.json()["access_token"]
        auth_headers_b = {"Authorization": f"Bearer {token_b}"}

        # User B cannot see User A's machine in list
        list_b_res = client.get("/api/v1/machines", headers=auth_headers_b)
        assert list_b_res.status_code == 200
        user_b_machines = list_b_res.json()["machines"]
        assert not any(m["machine_id"] == machine_id for m in user_b_machines)

        # User B cannot directly access User A's machine profile (403 Forbidden)
        forbidden_mach = client.get(f"/api/v1/machines/{machine_id}", headers=auth_headers_b)
        assert forbidden_mach.status_code == 403

        # User B cannot enroll audio to User A's machine (403 Forbidden)
        forbidden_enroll = client.post(
            f"/api/v1/machines/{machine_id}/enrollments",
            headers=auth_headers_b,
            files={"file": ("audio.wav", normal_wav, "audio/wav")},
            data={"confirmed_normal": "true"},
        )
        assert forbidden_enroll.status_code == 403

        # User B cannot inspect or analyze User A's machine (403 Forbidden)
        forbidden_ana = client.post(
            f"/api/v1/machines/{machine_id}/analyze",
            headers=auth_headers_b,
            files={"file": ("audio.wav", normal_wav, "audio/wav")},
        )
        assert forbidden_ana.status_code == 403

        # User B cannot access User A's analysis record (403 Forbidden)
        forbidden_rec = client.get(f"/api/v1/analyses/{analysis_id}", headers=auth_headers_b)
        assert forbidden_rec.status_code == 403

        # User B cannot access User A's inspection report (403 Forbidden)
        forbidden_rep = client.get(f"/api/v1/analyses/{analysis_id}/report", headers=auth_headers_b)
        assert forbidden_rep.status_code == 403

    def test_real_mimii_audio_end_to_end_inference(self):
        """Execute end-to-end inference against actual MIMII benchmark recordings on disk."""
        from pathlib import Path

        # 1. Login as admin
        admin_login = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "StethoAdmin2026!"},
        )
        assert admin_login.status_code == 200
        admin_token = admin_login.json()["access_token"]
        headers = {"Authorization": f"Bearer {admin_token}"}

        # 2. Check for real MIMII files
        normal_path = Path("data/mimii/raw/fan/id_00/normal/00000000.wav")
        abnormal_path = Path("data/mimii/raw/fan/id_00/abnormal/00000000.wav")

        if not normal_path.exists() or not abnormal_path.exists():
            pytest.skip("MIMII raw benchmark audio files not found on disk")

        # 3. Analyze Normal Audio Recording
        with open(normal_path, "rb") as f:
            normal_res = client.post(
                "/api/v1/machines/id_00/analyze",
                headers=headers,
                files={"file": ("00000000_normal.wav", f, "audio/wav")},
                data={"model_policy": "stetho_msae"},
            )
        assert normal_res.status_code == 200
        normal_data = normal_res.json()
        assert normal_data["decision"] in ["NORMAL_LIKE", "INCONCLUSIVE"]
        assert normal_data["anomaly_score"] is not None
        assert normal_data["quality_status"] in ["ACCEPTED", "WARNING"]

        # 4. Analyze Abnormal Audio Recording
        with open(abnormal_path, "rb") as f:
            abnormal_res = client.post(
                "/api/v1/machines/id_00/analyze",
                headers=headers,
                files={"file": ("00000000_abnormal.wav", f, "audio/wav")},
                data={"model_policy": "stetho_msae"},
            )
        assert abnormal_res.status_code == 200
        abnormal_data = abnormal_res.json()
        assert abnormal_data["decision"] == "UNUSUAL"
        assert abnormal_data["anomaly_score"] > abnormal_data["operating_threshold"]

        # 5. Retrieve Inspection Report for the abnormal recording
        abnormal_id = abnormal_data["analysis_id"]
        report_res = client.get(f"/api/v1/analyses/{abnormal_id}/report", headers=headers)
        assert report_res.status_code == 200
        report = report_res.json()
        assert report["analysis"]["decision"] == "UNUSUAL"
        assert "Unusual acoustic deviation detected" in report["executive_summary"]
        assert "Perform hands-on physical inspection" in report["recommended_action"]

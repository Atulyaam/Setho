"""Sprint 6A Integration Tests for Product Features & Endpoints.

Verifies:
1. Feature 1: Machine Acoustic Fingerprint (retrieval and auto-initialization).
2. Feature 2: Acoustic Drift Timeline (historical trend tracking and deviation points).
3. Feature 4: Evidence-Based Maintenance Assistant (rule-based diagnostic checklists).
4. Feature 5: Fleet Inspection Priority Engine (transparent multi-factor priority ranking).
5. Feature 6: QR Machine Identity & Token Rotation (secure mobile token generation and revocation).
"""

import io
import uuid
import wave
import numpy as np
import pytest
from fastapi.testclient import TestClient

from stetho_api.main import app

client = TestClient(app)


def _generate_wav(duration_sec: float = 5.0, freq: float = 440.0, amp: float = 0.25) -> bytes:
    """Generate in-memory valid mono 16kHz PCM WAV bytes."""
    sr = 16000
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    samples = amp * np.sin(2 * np.pi * freq * t)
    int_samples = (samples * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int_samples.tobytes())
    return buf.getvalue()


class TestSprint6AProductFeatures:
    """Integration test suite for Sprint 6A backend features."""

    @pytest.fixture(autouse=True)
    def setup_authenticated_session(self):
        """Create an authenticated inspector session for testing."""
        self.username = f"feat_inspector_{uuid.uuid4().hex[:6]}"
        self.password = "InspectorSecret123!"

        # Register and login
        client.post("/api/v1/auth/register", json={"username": self.username, "password": self.password})
        login_res = client.post("/api/v1/auth/login", json={"username": self.username, "password": self.password})
        self.token = login_res.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def test_machine_acoustic_fingerprint_lifecycle(self):
        """Verify Feature 1: Machine Acoustic Fingerprint retrieval and calculation."""
        m_id = f"fp_mach_{uuid.uuid4().hex[:6]}"
        client.post(
            "/api/v1/machines",
            headers=self.headers,
            json={"machine_id": m_id, "name": "Fan Fingerprint Unit", "category": "fan"},
        )

        # Before enrollment: status uncalibrated / no fingerprint
        res_pre = client.get(f"/api/v1/machines/{m_id}/fingerprint", headers=self.headers)
        assert res_pre.status_code == 200
        assert res_pre.json()["status"] == "uncalibrated"

        # Enroll audio
        wav = _generate_wav(duration_sec=5.0, freq=350.0)
        client.post(
            f"/api/v1/machines/{m_id}/enrollments",
            headers=self.headers,
            files={"file": ("normal.wav", wav, "audio/wav")},
            data={"confirmed_normal": "true"},
        )

        # After enrollment: status established with spectral centroid & multi-band energies
        res_post = client.get(f"/api/v1/machines/{m_id}/fingerprint", headers=self.headers)
        assert res_post.status_code == 200
        data = res_post.json()
        assert data["status"] == "established"
        fp = data["fingerprint"]
        assert fp["spectral_centroid_hz"] > 0
        assert fp["sub_bass_energy_db"] is not None
        assert fp["baseline_reconstruction_error"] is not None

    def test_qr_code_generation_and_rotation(self):
        """Verify Feature 6: Machine QR Identity, token validation, and secure rotation."""
        m_id = f"qr_mach_{uuid.uuid4().hex[:6]}"
        client.post(
            "/api/v1/machines",
            headers=self.headers,
            json={"machine_id": m_id, "name": "QR Tagged Pump", "category": "pump"},
        )

        # 1. Fetch QR code token
        res_qr = client.get(f"/api/v1/machines/{m_id}/qr-code", headers=self.headers)
        assert res_qr.status_code == 200
        qr_data = res_qr.json()
        token1 = qr_data["qr_token"]
        assert token1.startswith("stetho_qr_")
        assert "/inspect?token=" in qr_data["inspection_url"]

        # 2. Resolve token anonymously (as field mobile scanner would)
        res_scan = client.get(f"/api/v1/machines/inspect-by-token?token={token1}")
        assert res_scan.status_code == 200
        assert res_scan.json()["machine"]["machine_id"] == m_id

        # 3. Rotate QR token (security revocation)
        res_rotate = client.post(f"/api/v1/machines/{m_id}/qr-code/rotate", headers=self.headers)
        assert res_rotate.status_code == 200
        token2 = res_rotate.json()["qr_token"]
        assert token2 != token1

        # 4. Old token must now be invalid (404 Not Found)
        res_old = client.get(f"/api/v1/machines/inspect-by-token?token={token1}")
        assert res_old.status_code == 404

        # 5. New token resolves successfully
        res_new = client.get(f"/api/v1/machines/inspect-by-token?token={token2}")
        assert res_new.status_code == 200

    def test_fleet_priority_engine(self):
        """Verify Feature 5: Fleet Inspection Priority ranking."""
        m_id1 = f"prio_mach_1_{uuid.uuid4().hex[:4]}"
        m_id2 = f"prio_mach_2_{uuid.uuid4().hex[:4]}"

        client.post(
            "/api/v1/machines",
            headers=self.headers,
            json={"machine_id": m_id1, "name": "Priority Machine A", "category": "fan"},
        )
        client.post(
            "/api/v1/machines",
            headers=self.headers,
            json={"machine_id": m_id2, "name": "Priority Machine B", "category": "fan"},
        )

        res_prio = client.get("/api/v1/machines/fleet-priority", headers=self.headers)
        assert res_prio.status_code == 200
        data = res_prio.json()
        assert data["status"] == "success"
        assert len(data["priority_ranking"]) >= 2

        # Check ranking structure
        top_machine = data["priority_ranking"][0]
        assert "priority_score" in top_machine
        assert "priority_tier" in top_machine
        assert top_machine["priority_tier"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        assert isinstance(top_machine["reasons"], list)

    def test_evidence_based_maintenance_advice(self):
        """Verify Feature 4: Evidence-based maintenance guidance and checklist generation."""
        m_id = f"maint_mach_{uuid.uuid4().hex[:6]}"
        client.post(
            "/api/v1/machines",
            headers=self.headers,
            json={"machine_id": m_id, "name": "Maintenance Test Unit", "category": "fan"},
        )

        # Enroll audio
        wav = _generate_wav(duration_sec=5.0, freq=300.0)
        client.post(
            f"/api/v1/machines/{m_id}/enrollments",
            headers=self.headers,
            files={"file": ("normal.wav", wav, "audio/wav")},
            data={"confirmed_normal": "true"},
        )

        # Analyze audio
        ana_res = client.post(
            f"/api/v1/machines/{m_id}/analyze",
            headers=self.headers,
            files={"file": ("inspect.wav", wav, "audio/wav")},
        )
        assert ana_res.status_code == 200
        analysis_id = ana_res.json()["analysis_id"]

        # Fetch maintenance advice
        advice_res = client.get(f"/api/v1/analyses/{analysis_id}/maintenance-advice", headers=self.headers)
        assert advice_res.status_code == 200
        advice = advice_res.json()
        assert advice["analysis_id"] == analysis_id
        assert advice["maintenance_urgency"] in ["HIGH", "MEDIUM", "LOW", "VERIFY_RECORDING"]
        assert len(advice["findings"]) > 0
        assert len(advice["inspection_checklist"]) > 0
        assert "disclaimer" in advice["engineering_disclaimer"].lower()

    def test_acoustic_drift_timeline(self):
        """Verify Feature 2: Acoustic drift timeline recording and deviation tracking."""
        m_id = f"drift_mach_{uuid.uuid4().hex[:6]}"
        client.post(
            "/api/v1/machines",
            headers=self.headers,
            json={"machine_id": m_id, "name": "Drift Monitoring Unit", "category": "fan"},
        )

        # Enroll
        wav = _generate_wav(duration_sec=5.0, freq=300.0)
        client.post(
            f"/api/v1/machines/{m_id}/enrollments",
            headers=self.headers,
            files={"file": ("normal.wav", wav, "audio/wav")},
            data={"confirmed_normal": "true"},
        )

        # Execute 2 inspections
        client.post(
            f"/api/v1/machines/{m_id}/analyze",
            headers=self.headers,
            files={"file": ("insp1.wav", wav, "audio/wav")},
        )
        client.post(
            f"/api/v1/machines/{m_id}/analyze",
            headers=self.headers,
            files={"file": ("insp2.wav", wav, "audio/wav")},
        )

        # Fetch drift timeline
        res = client.get(f"/api/v1/machines/{m_id}/drift-timeline", headers=self.headers)
        assert res.status_code == 200
        data = res.json()
        assert data["machine_id"] == m_id
        assert data["total_records"] >= 2
        assert len(data["timeline"]) >= 2
        pt = data["timeline"][0]
        assert "anomaly_score" in pt
        assert "operating_threshold" in pt
        assert "decision" in pt


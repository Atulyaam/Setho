"""Integration tests for Stetho Field Data Collection, Annotation, Governance, and Dashboard APIs."""

import io
import uuid
from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from stetho_api.main import create_app


def create_synthetic_wav(
    duration_sec: float = 2.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    amplitude: float = 0.2,
    channels: int = 1,
) -> bytes:
    """Generate in-memory WAV byte stream containing clean synthetic audio with unique entropy."""
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False, dtype=np.float32)
    sig = amplitude * np.sin(2 * np.pi * frequency * t)
    sig += np.random.uniform(-0.002, 0.002, size=num_samples).astype(np.float32)
    if channels > 1:
        sig = np.column_stack([sig] * channels)
    buf = io.BytesIO()
    sf.write(buf, sig, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


@pytest.fixture
def auth_headers(client):
    res = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "StethoAdmin2026!"},
    )
    assert res.status_code == 200, res.text
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_field_data_import_and_duplicate_prevention(client, auth_headers):
    """Verify WAV upload, quality gating, and SHA-256 deduplication."""
    wav_bytes = create_synthetic_wav(duration_sec=3.0, sample_rate=16000, frequency=250.0)
    filename = f"field_fan_test_{uuid.uuid4().hex[:6]}.wav"

    # 1. Successful upload
    response = client.post(
        "/api/v1/field-data/import",
        headers=auth_headers,
        data={
            "machine_category": "fan",
            "machine_id": "fan_alpha_01",
            "sensor_id": "mic_pos_1",
            "operating_condition": "steady_full_load",
            "load_level": "100%",
            "consent_status": "authorized",
            "dataset_role": "unassigned",
        },
        files={"file": (filename, wav_bytes, "audio/wav")},
    )
    assert response.status_code == 201, response.text
    rec = response.json()
    assert rec["machine_category"] == "fan"
    assert rec["quality_status"].upper() == "PASSED"
    recording_id = rec["recording_id"]
    assert recording_id.startswith("fld_rec_")

    # 2. Duplicate upload of identical file bytes -> must be rejected with 409
    dup_response = client.post(
        "/api/v1/field-data/import",
        headers=auth_headers,
        data={
            "machine_category": "fan",
            "machine_id": "fan_alpha_01",
            "sensor_id": "mic_pos_1",
        },
        files={"file": ("duplicate_upload.wav", wav_bytes, "audio/wav")},
    )
    assert dup_response.status_code == 409, dup_response.text
    dup_data = dup_response.json()
    assert dup_data["error"]["code"] == "DUPLICATE_RECORDING"
    assert "already exists" in dup_data["error"]["message"]


def test_field_data_quality_rejection(client, auth_headers):
    """Verify rejection of low sample-rate audio (< 16 kHz)."""
    low_sr_wav = create_synthetic_wav(duration_sec=2.0, sample_rate=8000)
    filename = f"field_low_sr_{uuid.uuid4().hex[:6]}.wav"

    response = client.post(
        "/api/v1/field-data/import",
        headers=auth_headers,
        data={
            "machine_category": "pump",
            "machine_id": "pump_p100",
            "sensor_id": "mic_pos_2",
        },
        files={"file": (filename, low_sr_wav, "audio/wav")},
    )
    assert response.status_code == 422, response.text
    err_body = response.json()
    assert err_body["error"]["code"] == "INVALID_AUDIO_QUALITY"
    assert "Sample rate" in err_body["error"]["message"]


def test_annotation_lifecycle_and_multi_revision_history(client, auth_headers):
    """Verify operator annotation workflow and multi-revision audit trail."""
    wav_bytes = create_synthetic_wav(duration_sec=2.5, sample_rate=16000, frequency=330.0)
    filename = f"annot_test_{uuid.uuid4().hex[:6]}.wav"

    # Ingest
    res = client.post(
        "/api/v1/field-data/import",
        headers=auth_headers,
        data={
            "machine_category": "valve",
            "machine_id": "valve_v20",
            "sensor_id": "mic_valve_inlet",
        },
        files={"file": (filename, wav_bytes, "audio/wav")},
    )
    assert res.status_code == 201
    recording_id = res.json()["recording_id"]

    # Revision 1: Annotate as normal
    rev1_res = client.post(
        f"/api/v1/field-data/recordings/{recording_id}/annotate",
        headers=auth_headers,
        json={
            "label": "normal",
            "reason": "Baseline acoustic signature matches ISO limits",
            "evidence_source": "VISUAL_INSPECTION",
            "verification_status": "verified",
            "inspection_notes": "Visual and stethoscope inspection confirmed smooth seating",
        },
    )
    assert rev1_res.status_code == 201, rev1_res.text
    data1 = rev1_res.json()
    assert data1["label"] == "normal"
    assert data1["revision_number"] == 1

    # Check updated recording
    rec_res = client.get(f"/api/v1/field-data/recordings/{recording_id}", headers=auth_headers)
    assert rec_res.status_code == 200
    assert rec_res.json()["current_label"] == "normal"

    # Revision 2: Subsequent discovery of subtle cavitation
    rev2_res = client.post(
        f"/api/v1/field-data/recordings/{recording_id}/annotate",
        headers=auth_headers,
        json={
            "label": "abnormal",
            "reason": "Teardown confirmed internal cavitation pitting",
            "evidence_source": "TEARDOWN_INSPECTION",
            "verification_status": "verified",
            "inspection_notes": "Ultrasonic thickness gauge and teardown reveal localized erosion",
        },
    )
    assert rev2_res.status_code == 201, rev2_res.text
    data2 = rev2_res.json()
    assert data2["label"] == "abnormal"
    assert data2["revision_number"] == 2

    # Query audit history
    hist_res = client.get(
        f"/api/v1/field-data/recordings/{recording_id}/annotations",
        headers=auth_headers,
    )
    assert hist_res.status_code == 200
    history = hist_res.json()
    assert len(history) == 2
    assert history[0]["revision_number"] == 1
    assert history[0]["label"] == "normal"
    assert history[1]["revision_number"] == 2
    assert history[1]["label"] == "abnormal"


def test_dataset_role_and_calibration_guardrails(client, auth_headers):
    """Verify that abnormal recordings cannot be assigned to calibration_enrollment."""
    wav_bytes = create_synthetic_wav(duration_sec=2.0, sample_rate=16000, frequency=400.0)
    filename = f"calib_guard_{uuid.uuid4().hex[:6]}.wav"

    res = client.post(
        "/api/v1/field-data/import",
        headers=auth_headers,
        data={
            "machine_category": "fan",
            "machine_id": "fan_c1",
            "sensor_id": "mic_c1",
        },
        files={"file": (filename, wav_bytes, "audio/wav")},
    )
    recording_id = res.json()["recording_id"]

    # Mark as abnormal
    ann_res = client.post(
        f"/api/v1/field-data/recordings/{recording_id}/annotate",
        headers=auth_headers,
        json={
            "label": "abnormal",
            "reason": "Known mechanical unbalance",
            "evidence_source": "VIBRATION_ORDER_ANALYSIS",
            "verification_status": "verified",
        },
    )
    assert ann_res.status_code == 201

    # Attempt to assign abnormal recording to calibration_enrollment
    role_res = client.put(
        f"/api/v1/field-data/recordings/{recording_id}/role",
        headers=auth_headers,
        json={"dataset_role": "calibration_enrollment"},
    )
    assert role_res.status_code == 422
    assert "abnormal" in role_res.json()["error"]["message"].lower()


def test_manifest_creation_and_sealing(client, auth_headers):
    """Verify manifest creation, cryptographic sealing, and tamper-resistance."""
    rec_ids = []
    for freq in [500.0, 600.0]:
        wav_bytes = create_synthetic_wav(duration_sec=2.0, sample_rate=16000, frequency=freq)
        filename = f"manifest_test_{uuid.uuid4().hex[:6]}.wav"
        res = client.post(
            "/api/v1/field-data/import",
            headers=auth_headers,
            data={
                "machine_category": "pump",
                "machine_id": "pump_m1",
                "sensor_id": "mic_m1",
                "dataset_role": "sealed_evaluation",
            },
            files={"file": (filename, wav_bytes, "audio/wav")},
        )
        assert res.status_code == 201
        rec_ids.append(res.json()["recording_id"])

    # Create sealed evaluation manifest
    manifest_res = client.post(
        "/api/v1/field-data/manifests",
        headers=auth_headers,
        json={
            "name": f"Field Pilot Sealed Eval {uuid.uuid4().hex[:4]}",
            "dataset_role": "sealed_evaluation",
            "machine_category": "pump",
            "recording_ids": rec_ids,
            "seal_now": True,
        },
    )
    assert manifest_res.status_code == 201, manifest_res.text
    manifest = manifest_res.json()
    assert manifest["is_sealed"] is True
    assert len(manifest["sha256_checksum"]) == 64
    assert manifest["recording_count"] == 2

    # Attempt to change role of sealed recording must fail
    locked_role_res = client.put(
        f"/api/v1/field-data/recordings/{rec_ids[0]}/role",
        headers=auth_headers,
        json={"dataset_role": "training"},
    )
    assert locked_role_res.status_code == 409
    assert "sealed" in locked_role_res.json()["error"]["message"].lower()


def test_dashboard_metrics_and_readiness_warning(client, auth_headers):
    """Verify dashboard endpoint returns aggregate metrics and independent readiness warning."""
    dash_res = client.get("/api/v1/field-data/dashboard", headers=auth_headers)
    assert dash_res.status_code == 200, dash_res.text
    dash = dash_res.json()

    assert "total_imported" in dash
    assert "by_dataset_role" in dash
    assert "by_category" in dash
    assert "duplicates_detected" in dash
    assert "independent_evaluation_readiness" in dash

    readiness = dash["independent_evaluation_readiness"]
    assert "is_ready" in readiness
    assert readiness["verdict"] in ["READY_FOR_SEALED_EVALUATION", "INSUFFICIENT_INDEPENDENT_EVIDENCE"]
    if not readiness["is_ready"]:
        assert readiness["verdict"] == "INSUFFICIENT_INDEPENDENT_EVIDENCE"
        assert readiness["warning"] is not None


def test_protocol_checklist_export(client, auth_headers):
    """Verify field data collection protocol checklist API in JSON and Markdown formats."""
    # JSON export
    json_res = client.get("/api/v1/field-data/protocol-checklist?export_format=json", headers=auth_headers)
    assert json_res.status_code == 200
    checklist_data = json_res.json()
    assert checklist_data["title"] == "Stetho Field Acoustic Data Collection Checklist"
    assert len(checklist_data["steps"]) >= 5

    # Markdown export
    md_res = client.get("/api/v1/field-data/protocol-checklist?export_format=markdown", headers=auth_headers)
    assert md_res.status_code == 200
    assert "text/markdown" in md_res.headers["content-type"]
    assert "# Stetho Field Acoustic Data Collection Checklist" in md_res.text

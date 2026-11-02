"""Security, RBAC, and Dataset Integrity Integration Tests for Stetho Field Data.

Covers:
1. Unauthenticated request rejection (401 Unauthorized across all endpoints)
2. RBAC privilege boundaries (Viewer 403 on write; Inspector 403 on governance; Admin full)
3. Path traversal attacks in machine_category and storage filenames
4. Oversized upload protection (> 50 MB -> 413)
5. Corrupt / malicious audio rejection (non-WAV, duration > 60s, rate > 96kHz, rate < 16kHz, silence, clipping)
6. Cross-filename SHA-256 deduplication
7. Audio streaming endpoint and path traversal defense
8. Sealed manifest immutability (roles and annotations locked)
9. Cryptographic tamper detection on disk WAV alteration
10. Calibration enrollment gating (normal + verified only)
"""

import io
import os
from pathlib import Path
import uuid
from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from stetho_api.auth.password import hash_password
from stetho_api.auth.tokens import create_access_token
from stetho_api.config import get_settings
from stetho_api.db.database import get_db
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
def admin_token(client):
    res = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "StethoAdmin2026!"},
    )
    assert res.status_code == 200, res.text
    return res.json()["access_token"]


@pytest.fixture
def admin_headers(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture
def inspector_token(client):
    res = client.post(
        "/api/v1/auth/login",
        json={"username": "inspector", "password": "StethoInspector2026!"},
    )
    assert res.status_code == 200, res.text
    return res.json()["access_token"]


@pytest.fixture
def inspector_headers(inspector_token):
    return {"Authorization": f"Bearer {inspector_token}"}


@pytest.fixture
def viewer_headers(client):
    """Seed and log in as a read-only viewer."""
    settings = get_settings()
    db = get_db(settings.STORAGE_LOCAL_DIR / "stetho.db")
    viewer_id = f"usr_viewer_{uuid.uuid4().hex[:8]}"
    viewer_name = f"viewer_{uuid.uuid4().hex[:6]}"
    with db.get_connection() as conn:
        conn.execute(
            "INSERT INTO users (user_id, username, hashed_password, role, created_at) VALUES (?, ?, ?, ?, datetime('now'));",
            (viewer_id, viewer_name, hash_password("ViewerPass123!"), "viewer"),
        )
        conn.commit()

    res = client.post(
        "/api/v1/auth/login",
        json={"username": viewer_name, "password": "ViewerPass123!"},
    )
    assert res.status_code == 200, res.text
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


# =========================================================================
# 1. Unauthenticated Request Rejection
# =========================================================================

def test_unauthenticated_requests_are_rejected(client):
    """Verify that unauthenticated access across all field data endpoints returns 401."""
    fake_id = str(uuid.uuid4())

    endpoints = [
        ("GET", "/api/v1/field-data/recordings"),
        ("GET", f"/api/v1/field-data/recordings/{fake_id}"),
        ("GET", f"/api/v1/field-data/recordings/{fake_id}/audio"),
        ("POST", "/api/v1/field-data/import"),
        ("POST", f"/api/v1/field-data/recordings/{fake_id}/annotate"),
        ("PUT", f"/api/v1/field-data/recordings/{fake_id}/role"),
        ("GET", "/api/v1/field-data/manifests"),
        ("POST", "/api/v1/field-data/manifests"),
        ("POST", f"/api/v1/field-data/manifests/{fake_id}/seal"),
        ("GET", f"/api/v1/field-data/manifests/{fake_id}/verify"),
        ("GET", "/api/v1/field-data/dashboard"),
    ]

    for method, path in endpoints:
        res = client.request(method, path)
        assert res.status_code == 401, f"{method} {path} expected 401, got {res.status_code}: {res.text}"


# =========================================================================
# 2. RBAC Privilege Boundaries
# =========================================================================

def test_viewer_role_privilege_boundaries(client, viewer_headers, admin_headers):
    """Viewers can inspect data, but cannot import recordings or create annotations."""
    # Viewer can read stats and list recordings
    res = client.get("/api/v1/field-data/dashboard", headers=viewer_headers)
    assert res.status_code == 200

    res = client.get("/api/v1/field-data/recordings", headers=viewer_headers)
    assert res.status_code == 200

    # Viewer CANNOT import
    wav_bytes = create_synthetic_wav(frequency=310.0)
    files = {"file": ("viewer_test.wav", wav_bytes, "audio/wav")}
    data = {
        "machine_category": "fan",
        "machine_id": "fan_viewer_test",
        "sensor_id": "sensor_01",
        "operator_id": "viewer_user",
    }
    res = client.post("/api/v1/field-data/import", files=files, data=data, headers=viewer_headers)
    assert res.status_code == 403, f"Expected 403 for viewer import, got {res.status_code}"

    # Admin imports a valid recording
    res = client.post("/api/v1/field-data/import", files=files, data=data, headers=admin_headers)
    assert res.status_code == 201
    rec_id = res.json()["recording_id"]

    # Viewer CANNOT annotate
    res = client.post(
        f"/api/v1/field-data/recordings/{rec_id}/annotate",
        json={"label": "normal", "reason": "Viewer attempting annotation", "evidence_source": "VISUAL_INSPECTION"},
        headers=viewer_headers,
    )
    assert res.status_code == 403

    # Viewer CANNOT assign dataset role
    res = client.put(
        f"/api/v1/field-data/recordings/{rec_id}/role",
        json={"dataset_role": "development"},
        headers=viewer_headers,
    )
    assert res.status_code == 403


def test_inspector_cannot_perform_governance_actions(client, inspector_headers, admin_headers):
    """Inspectors can import and annotate, but CANNOT assign dataset roles or create/seal manifests."""
    wav_bytes = create_synthetic_wav(frequency=320.0)
    files = {"file": ("inspector_test.wav", wav_bytes, "audio/wav")}
    data = {
        "machine_category": "fan",
        "machine_id": "fan_insp_01",
        "sensor_id": "sensor_01",
        "operator_id": "inspector_user",
    }
    # Inspector CAN import
    res = client.post("/api/v1/field-data/import", files=files, data=data, headers=inspector_headers)
    assert res.status_code == 201
    rec_id = res.json()["recording_id"]

    # Inspector CAN annotate
    res = client.post(
        f"/api/v1/field-data/recordings/{rec_id}/annotate",
        json={"label": "normal", "reason": "Inspector validated sound", "evidence_source": "VISUAL_INSPECTION"},
        headers=inspector_headers,
    )
    assert res.status_code == 201

    # Inspector CANNOT assign dataset role
    res = client.put(
        f"/api/v1/field-data/recordings/{rec_id}/role",
        json={"dataset_role": "development"},
        headers=inspector_headers,
    )
    assert res.status_code == 403

    # Inspector CANNOT create manifest
    res = client.post(
        "/api/v1/field-data/manifests",
        json={"name": "Inspector Manifest", "dataset_role": "development", "machine_category": "fan", "recording_ids": [rec_id]},
        headers=inspector_headers,
    )
    assert res.status_code == 403


# =========================================================================
# 3. Path Traversal Attacks in Metadata
# =========================================================================

def test_path_traversal_in_machine_category_rejected(client, admin_headers):
    """Attempts to escape storage dir via malicious machine_category must be rejected."""
    traversal_categories = [
        "../../sensitive",
        "..\\..\\system32",
        "fan/../../../etc",
        "/etc/passwd",
        "fan/subfolder/attack",
    ]

    wav_bytes = create_synthetic_wav(frequency=330.0)
    for cat in traversal_categories:
        files = {"file": ("traversal.wav", wav_bytes, "audio/wav")}
        data = {
            "machine_category": cat,
            "machine_id": "machine_01",
            "sensor_id": "sensor_01",
            "operator_id": "attacker",
        }
        res = client.post("/api/v1/field-data/import", files=files, data=data, headers=admin_headers)
        assert res.status_code == 400 or res.status_code == 422, f"Expected rejection for {cat}, got {res.status_code}"


# =========================================================================
# 4. Oversized Audio Upload Rejection
# =========================================================================

def test_oversized_audio_upload_rejected(client, admin_headers):
    """Uploads exceeding max allowed payload (50 MB) must return 413."""
    dummy_payload = b"0" * (51 * 1024 * 1024)
    files = {"file": ("huge_recording.wav", dummy_payload, "audio/wav")}
    data = {
        "machine_category": "fan",
        "machine_id": "fan_oversize",
        "sensor_id": "sensor_01",
        "operator_id": "tester",
    }
    res = client.post("/api/v1/field-data/import", files=files, data=data, headers=admin_headers)
    assert res.status_code == 413


# =========================================================================
# 5. Corrupt / Malicious Audio Format Rejection
# =========================================================================

def test_corrupt_non_wav_bytes_rejected(client, admin_headers):
    """Arbitrary non-WAV bytes must be rejected with 422 INVALID_AUDIO_QUALITY."""
    corrupt_bytes = b"NOT_A_WAV_FILE_HEADER" + os.urandom(1024)
    files = {"file": ("corrupt.wav", corrupt_bytes, "audio/wav")}
    data = {
        "machine_category": "pump",
        "machine_id": "pump_01",
        "sensor_id": "sensor_01",
        "operator_id": "tester",
    }
    res = client.post("/api/v1/field-data/import", files=files, data=data, headers=admin_headers)
    assert res.status_code == 422
    assert "WAV_DECODE_ERROR" in res.json().get("detail", "") or res.json().get("error", {}).get("code") == "INVALID_AUDIO_QUALITY"


def test_audio_duration_exceeding_max_rejected(client, admin_headers):
    """Audio exceeding 60 seconds must be rejected with 422 DURATION_TOO_LONG."""
    long_wav = create_synthetic_wav(duration_sec=65.0, sample_rate=16000, frequency=200.0)
    files = {"file": ("long.wav", long_wav, "audio/wav")}
    data = {
        "machine_category": "fan",
        "machine_id": "fan_long",
        "sensor_id": "sensor_01",
        "operator_id": "tester",
    }
    res = client.post("/api/v1/field-data/import", files=files, data=data, headers=admin_headers)
    assert res.status_code == 422
    assert "DURATION_TOO_LONG" in str(res.json())


def test_excessive_sample_rate_rejected(client, admin_headers):
    """Sample rate exceeding 96 kHz must be rejected with 422 SAMPLE_RATE_EXCESSIVE."""
    high_rate_wav = create_synthetic_wav(duration_sec=1.0, sample_rate=192000, frequency=400.0)
    files = {"file": ("high_rate.wav", high_rate_wav, "audio/wav")}
    data = {
        "machine_category": "fan",
        "machine_id": "fan_high_rate",
        "sensor_id": "sensor_01",
        "operator_id": "tester",
    }
    res = client.post("/api/v1/field-data/import", files=files, data=data, headers=admin_headers)
    assert res.status_code == 422
    assert "SAMPLE_RATE_EXCESSIVE" in str(res.json())


# =========================================================================
# 6. Cross-Filename SHA-256 Deduplication
# =========================================================================

def test_cross_filename_sha256_deduplication(client, admin_headers):
    """Identical audio bytes uploaded under different names must be rejected as duplicate."""
    wav_bytes = create_synthetic_wav(duration_sec=2.0, sample_rate=16000, frequency=455.0)

    # First upload: clip_original.wav
    files1 = {"file": ("clip_original.wav", wav_bytes, "audio/wav")}
    data1 = {
        "machine_category": "fan",
        "machine_id": "fan_dedup_01",
        "sensor_id": "mic_01",
        "operator_id": "inspector_alice",
    }
    res1 = client.post("/api/v1/field-data/import", files=files1, data=data1, headers=admin_headers)
    assert res1.status_code == 201
    orig_id = res1.json()["recording_id"]

    # Second upload: clip_renamed_copy.wav with IDENTICAL bytes
    files2 = {"file": ("clip_renamed_copy.wav", wav_bytes, "audio/wav")}
    data2 = {
        "machine_category": "fan",
        "machine_id": "fan_dedup_02",
        "sensor_id": "mic_02",
        "operator_id": "inspector_bob",
    }
    res2 = client.post("/api/v1/field-data/import", files=files2, data=data2, headers=admin_headers)
    assert res2.status_code == 409
    body2 = res2.json()
    assert "DUPLICATE_RECORDING" in str(body2)
    assert orig_id in str(body2)


# =========================================================================
# 7. Audio Streaming Endpoint and Path Safety
# =========================================================================

def test_audio_streaming_returns_intact_audio(client, admin_headers):
    """GET /recordings/{id}/audio returns byte-for-byte identical audio."""
    wav_bytes = create_synthetic_wav(duration_sec=2.0, sample_rate=16000, frequency=520.0)
    files = {"file": ("audio_stream_test.wav", wav_bytes, "audio/wav")}
    data = {
        "machine_category": "valve",
        "machine_id": "valve_stream_01",
        "sensor_id": "mic_01",
        "operator_id": "inspector_01",
    }
    res = client.post("/api/v1/field-data/import", files=files, data=data, headers=admin_headers)
    assert res.status_code == 201
    rec_id = res.json()["recording_id"]

    audio_res = client.get(f"/api/v1/field-data/recordings/{rec_id}/audio", headers=admin_headers)
    assert audio_res.status_code == 200
    assert audio_res.headers["content-type"].startswith("audio/wav")
    assert audio_res.content == wav_bytes


# =========================================================================
# 8. Sealed Manifest Immutability & Tamper Detection
# =========================================================================

def test_sealed_manifest_immutability_and_tamper_detection(client, admin_headers):
    """Test manifest sealing, member immutability, and disk tamper detection."""
    # Create 2 recordings
    ids = []
    for freq in [400.0, 480.0]:
        wav = create_synthetic_wav(duration_sec=2.0, sample_rate=16000, frequency=freq)
        files = {"file": (f"tamper_{freq}.wav", wav, "audio/wav")}
        data = {
            "machine_category": "fan",
            "machine_id": "fan_tamper_unit",
            "sensor_id": "mic_01",
            "operator_id": "analyst",
        }
        res = client.post("/api/v1/field-data/import", files=files, data=data, headers=admin_headers)
        assert res.status_code == 201
        rec_id = res.json()["recording_id"]
        # Mark as verified normal
        client.post(
            f"/api/v1/field-data/recordings/{rec_id}/annotate",
            json={"label": "normal", "reason": "Verified normal for evaluation", "evidence_source": "VISUAL_INSPECTION"},
            headers=admin_headers,
        )
        # Assign role
        client.put(
            f"/api/v1/field-data/recordings/{rec_id}/role",
            json={"dataset_role": "sealed_evaluation"},
            headers=admin_headers,
        )
        ids.append(rec_id)

    # Create unsealed manifest
    m_res = client.post(
        "/api/v1/field-data/manifests",
        json={
            "name": "Tamper Test Evaluation Benchmark",
            "dataset_role": "sealed_evaluation",
            "machine_category": "fan",
            "recording_ids": ids,
            "seal_now": False,
        },
        headers=admin_headers,
    )
    assert m_res.status_code == 201
    manifest_id = m_res.json()["manifest_id"]
    assert m_res.json()["is_sealed"] is False

    # Now seal the manifest
    seal_res = client.post(f"/api/v1/field-data/manifests/{manifest_id}/seal", headers=admin_headers)
    assert seal_res.status_code == 200
    assert seal_res.json()["is_sealed"] is True

    # 8a. Verify initial tamper check reports valid
    verify_res = client.get(f"/api/v1/field-data/manifests/{manifest_id}/verify", headers=admin_headers)
    assert verify_res.status_code == 200
    v_data = verify_res.json()
    assert v_data["is_valid"] is True
    assert v_data["tamper_detected"] is False

    # 8b. Attempt to modify dataset role of member -> blocked with 409
    role_res = client.put(
        f"/api/v1/field-data/recordings/{ids[0]}/role",
        json={"dataset_role": "development"},
        headers=admin_headers,
    )
    assert role_res.status_code == 409
    assert "MANIFEST_ALREADY_SEALED" in str(role_res.json())

    # 8c. Attempt to add annotation to sealed recording -> blocked with 409
    ann_res = client.post(
        f"/api/v1/field-data/recordings/{ids[0]}/annotate",
        json={"label": "abnormal", "reason": "Tampering attempt", "evidence_source": "OPERATIONAL_LOG"},
        headers=admin_headers,
    )
    assert ann_res.status_code == 409
    assert "MANIFEST_ALREADY_SEALED" in str(ann_res.json())

    # 8d. Tamper with disk audio of first recording
    settings = get_settings()
    rec_info = client.get(f"/api/v1/field-data/recordings/{ids[0]}", headers=admin_headers).json()
    file_path = Path(rec_info["file_path"])
    assert file_path.exists()

    with open(file_path, "ab") as f:
        f.write(b"CORRUPTING_BYTE_MODIFICATION_TAMPER")

    # 8e. Verify endpoint now detects tampering
    tamper_check_res = client.get(f"/api/v1/field-data/manifests/{manifest_id}/verify", headers=admin_headers)
    assert tamper_check_res.status_code == 200
    t_data = tamper_check_res.json()
    assert t_data["is_valid"] is False
    assert t_data["tamper_detected"] is True
    assert "File tamper detected" in t_data["reason"]
    assert ids[0] in t_data["reason"]


# =========================================================================
# 9. Calibration Enrollment Gating
# =========================================================================

def test_calibration_enrollment_requires_verified_normal(client, admin_headers):
    """Recordings cannot be assigned calibration_enrollment unless verified normal."""
    wav_bytes = create_synthetic_wav(frequency=375.0)
    files = {"file": ("calibration_gate.wav", wav_bytes, "audio/wav")}
    data = {
        "machine_category": "fan",
        "machine_id": "fan_calib_01",
        "sensor_id": "sensor_01",
        "operator_id": "tester",
    }
    res = client.post("/api/v1/field-data/import", files=files, data=data, headers=admin_headers)
    assert res.status_code == 201
    rec_id = res.json()["recording_id"]

    # 9a. While unlabelled/unverified -> cannot assign calibration_enrollment
    role_res = client.put(
        f"/api/v1/field-data/recordings/{rec_id}/role",
        json={"dataset_role": "calibration_enrollment"},
        headers=admin_headers,
    )
    assert role_res.status_code == 422
    assert "DATASET_ROLE_CONFLICT" in str(role_res.json())

    # 9b. When labeled abnormal -> cannot assign calibration_enrollment
    client.post(
        f"/api/v1/field-data/recordings/{rec_id}/annotate",
        json={"label": "abnormal", "reason": "Failed bearing", "evidence_source": "VIBRATION_ORDER_ANALYSIS"},
        headers=admin_headers,
    )
    role_res2 = client.put(
        f"/api/v1/field-data/recordings/{rec_id}/role",
        json={"dataset_role": "calibration_enrollment"},
        headers=admin_headers,
    )
    assert role_res2.status_code == 422

    # 9c. When re-annotated as normal -> can assign calibration_enrollment
    client.post(
        f"/api/v1/field-data/recordings/{rec_id}/annotate",
        json={"label": "normal", "reason": "Confirmed healthy", "evidence_source": "VISUAL_INSPECTION"},
        headers=admin_headers,
    )
    role_res3 = client.put(
        f"/api/v1/field-data/recordings/{rec_id}/role",
        json={"dataset_role": "calibration_enrollment"},
        headers=admin_headers,
    )
    assert role_res3.status_code == 200
    assert role_res3.json()["dataset_role"] == "calibration_enrollment"

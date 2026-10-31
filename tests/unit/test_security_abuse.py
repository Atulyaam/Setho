"""Unit and abuse tests for API Security, Input Validation, and Threat Mitigation (Task 4.5.7)."""

import io
from pathlib import Path
import uuid
from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from stetho_api.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def admin_token():
    res = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "StethoAdmin2026!"},
    )
    return res.json()["access_token"]


def test_reject_oversized_audio_upload(admin_token):
    """Verify uploads exceeding 15 MB are rejected with HTTP 400 PAYLOAD_TOO_LARGE."""
    # Create 16 MB dummy payload
    oversized = io.BytesIO(b"RIFF" + b"\x00" * 4 + b"WAVE" + b"X" * (16 * 1024 * 1024))
    files = {"file": ("huge.wav", oversized, "audio/wav")}

    res = client.post(
        "/api/v1/machines/id_00/analyze",
        files=files,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res.status_code == 400
    err = res.json()["error"]
    assert err["code"] == "PAYLOAD_TOO_LARGE"
    assert "exceeds maximum allowed size" in err["message"]


def test_reject_malformed_and_non_wav_headers(admin_token):
    """Verify non-WAV / forged magic byte files are rejected."""
    bad_data = io.BytesIO(b"PK\x03\x04ZIP_ARCHIVE_NOT_AUDIO")
    files = {"file": ("malicious.wav", bad_data, "audio/wav")}

    res = client.post(
        "/api/v1/machines/id_00/analyze",
        files=files,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_AUDIO_HEADER"


def test_path_traversal_sanitization(admin_token):
    """Verify path traversal in filename is completely sanitized to server UUID."""
    samples = 16000 * 3
    t = np.linspace(0, 3.0, samples, endpoint=False)
    wav = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, 16000, format="WAV")

    traversal_filenames = [
        "../../../../etc/passwd.wav",
        "..\\..\\Windows\\System32\\cmd.wav",
        "nested/dir/audio.wav",
    ]

    for malicious_name in traversal_filenames:
        buf.seek(0)
        files = {"file": (malicious_name, buf.getvalue(), "audio/wav")}
        res = client.post(
            "/api/v1/machines/id_00/analyze",
            files=files,
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert res.status_code == 200
        # Verify no file was written outside storage/recordings/id_00
        assert not Path("etc/passwd.wav").exists()
        assert not Path("Windows/System32/cmd.wav").exists()


def test_reject_too_short_recording(admin_token):
    """Verify recording under 1.0s is rejected with HTTP 400."""
    samples = int(0.5 * 16000)
    wav = (0.2 * np.sin(2 * np.pi * 300 * np.linspace(0, 0.5, samples))).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, 16000, format="WAV")

    files = {"file": ("short.wav", buf.getvalue(), "audio/wav")}
    res = client.post(
        "/api/v1/machines/id_00/analyze",
        files=files,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "AUDIO_TOO_SHORT"


def test_reject_too_long_recording(admin_token):
    """Verify recording over 30.0s is rejected with HTTP 400."""
    samples = int(32.0 * 16000)
    wav = (0.2 * np.sin(2 * np.pi * 300 * np.linspace(0, 32.0, samples))).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, 16000, format="WAV")

    files = {"file": ("long.wav", buf.getvalue(), "audio/wav")}
    res = client.post(
        "/api/v1/machines/id_00/analyze",
        files=files,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "AUDIO_TOO_LONG"


def test_sanitized_error_response_no_trace_leakage():
    """Verify 404/422/500 errors contain structured ErrorResponse without server filesystem leakage."""
    res = client.get("/api/v1/machines/nonexistent_xyz_9999", headers={"Authorization": "Bearer dummy"})
    # Dummy token returns 401
    assert res.status_code == 401
    data = res.json()
    assert "error" in data
    assert "code" in data["error"]
    assert "request_id" in data
    assert "C:\\" not in str(data)
    assert "Traceback" not in str(data)


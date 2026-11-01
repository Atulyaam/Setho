"""Integration tests for Sprint 7 Assets, Calibration, Inference, and Routing APIs."""

import io
import uuid
from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from stetho_api.auth.tokens import create_access_token
from stetho_api.main import create_app


def create_synthetic_wav(duration_sec: float = 10.0, sample_rate: int = 16000) -> bytes:
    """Generate in-memory WAV byte stream containing clean synthetic audio."""
    num_samples = int(duration_sec * sample_rate)
    # 440 Hz pure tone with low noise
    t = np.linspace(0, duration_sec, num_samples, endpoint=False, dtype=np.float32)
    tone = 0.1 * np.sin(2 * np.pi * 440 * t) + 0.01 * np.random.randn(num_samples).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, tone, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


@pytest.fixture
def auth_headers():
    from stetho_api.config import get_settings
    settings = get_settings()
    payload = {"sub": "usr_admin", "username": "admin", "role": "admin"}
    token = create_access_token(payload, secret_key=settings.SECRET_KEY, expires_in_seconds=3600)
    return {"Authorization": f"Bearer {token}"}


def test_models_discovery_endpoint(client, auth_headers):
    """Verify GET /api/models returns all registered production models with benchmarks."""
    response = client.get("/api/models", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] >= 6
    model_types = [m["type"] for m in data["models"]]
    assert "compact_student_int8" in model_types
    assert "resnet_ae" in model_types
    assert "cascade_ensemble" in model_types
    assert "multitask_resnet_ae" in model_types


def test_asset_lifecycle_crud(client, auth_headers):
    """Verify POST /api/assets, GET /api/assets, and GET /api/assets/{id}."""
    asset_id = f"test_fan_s7_{uuid.uuid4().hex[:6]}"

    # 1. Create asset
    payload = {
        "asset_id": asset_id,
        "name": "Sprint 7 Exhaust Fan Unit 1",
        "category": "fan",
        "description": "Roof-mounted ventilation exhaust unit",
        "operating_condition": "1750 RPM, Continuous Duty",
        "assigned_model": "cascade_ensemble",
    }
    create_res = client.post("/api/assets", json=payload, headers=auth_headers)
    assert create_res.status_code == 201
    created_data = create_res.json()
    assert created_data["asset_id"] == asset_id
    assert created_data["calibration_status"] == "uncalibrated"

    # 2. Duplicate ID raises 409 Conflict
    dup_res = client.post("/api/assets", json=payload, headers=auth_headers)
    assert dup_res.status_code == 409

    # 3. Retrieve asset
    get_res = client.get(f"/api/assets/{asset_id}", headers=auth_headers)
    assert get_res.status_code == 200
    assert get_res.json()["name"] == "Sprint 7 Exhaust Fan Unit 1"

    # 4. List assets with filter
    list_res = client.get("/api/assets?category=fan", headers=auth_headers)
    assert list_res.status_code == 200
    assert list_res.json()["total_count"] >= 1


def test_asset_category_validation(client, auth_headers):
    """Verify asset registration rejects invalid categories."""
    payload = {
        "asset_id": "invalid_cat_asset",
        "name": "Invalid Machine",
        "category": "compressor",  # Not in fan, pump, valve
    }
    res = client.post("/api/assets", json=payload, headers=auth_headers)
    assert res.status_code == 422


def test_few_shot_calibration_workflow(client, auth_headers):
    """Verify few-shot calibration workflow: N=2 (provisional), N=10 (calibrated), and rollback."""
    asset_id = f"test_cal_asset_{uuid.uuid4().hex[:6]}"
    # Create asset
    client.post(
        "/api/assets",
        json={"asset_id": asset_id, "name": "Calibrating Fan", "category": "fan"},
        headers=auth_headers,
    )

    wav_bytes = create_synthetic_wav(10.0, 16000)

    # 1. Unconfirmed normal recording must be rejected (Phase 3 Rule 3)
    files = [("files", ("baseline_01.wav", wav_bytes, "audio/wav"))]
    rejected_res = client.post(
        f"/api/assets/{asset_id}/calibrate",
        files=files,
        data={"is_confirmed_normal": False},
        headers=auth_headers,
    )
    assert rejected_res.status_code == 400
    assert "strictly requires operator confirmation" in rejected_res.text

    # 2. Calibration with 2 recordings -> status='provisional'
    files_2 = [
        ("files", (f"norm_{i}.wav", wav_bytes, "audio/wav")) for i in range(2)
    ]
    cal_res_2 = client.post(
        f"/api/assets/{asset_id}/calibrate",
        files=files_2,
        data={"is_confirmed_normal": True, "notes": "Initial provisional 2-shot baseline"},
        headers=auth_headers,
    )
    assert cal_res_2.status_code == 200
    res_data_2 = cal_res_2.json()
    assert res_data_2["calibration_status"] == "provisional"
    assert res_data_2["enrolled_sample_count"] == 2
    assert res_data_2["operating_threshold"] > 0.0

    # 3. Calibration with 10 recordings -> status='calibrated' (Sprint 6D 10-shot standard)
    files_10 = [
        ("files", (f"norm_10_{i}.wav", wav_bytes, "audio/wav")) for i in range(10)
    ]
    cal_res_10 = client.post(
        f"/api/assets/{asset_id}/calibrate",
        files=files_10,
        data={"is_confirmed_normal": True, "notes": "Sprint 6D verified 10-shot commissioning"},
        headers=auth_headers,
    )
    assert cal_res_10.status_code == 200
    res_data_10 = cal_res_10.json()
    assert res_data_10["calibration_status"] == "calibrated"
    assert res_data_10["is_provisional"] is False
    assert res_data_10["enrolled_sample_count"] >= 10

    # 4. Rollback to previous version
    rollback_res = client.post(
        f"/api/assets/{asset_id}/calibration/rollback",
        headers=auth_headers,
    )
    assert rollback_res.status_code == 200
    rb_data = rollback_res.json()
    assert rb_data["calibration_version"] == "1.2.0"


def test_inference_and_category_routing_enforcement(client, auth_headers):
    """Verify POST /api/inference, category routing rules, and cross-category bypass."""
    # 1. Create pump asset
    pump_id = f"test_pump_s7_{uuid.uuid4().hex[:6]}"
    client.post(
        "/api/assets",
        json={"asset_id": pump_id, "name": "Primary Water Pump", "category": "pump"},
        headers=auth_headers,
    )

    wav_bytes = create_synthetic_wav(10.0, 16000)

    # 2. Attempting to force ResNet-AE (fan-only) on Pump WITHOUT cross-category flag must fail (Phase 5)
    files = {"file": ("pump_sample.wav", wav_bytes, "audio/wav")}
    bad_route_res = client.post(
        "/api/inference",
        files=files,
        data={"asset_id": pump_id, "profile": "resnet_ae", "allow_cross_category": False},
        headers=auth_headers,
    )
    assert bad_route_res.status_code == 400
    assert "Category routing error" in bad_route_res.text

    # 3. Running with allowable universal model (compact_student_int8) succeeds
    files_ok = {"file": ("pump_sample.wav", wav_bytes, "audio/wav")}
    ok_res = client.post(
        "/api/inference",
        files=files_ok,
        data={"asset_id": pump_id, "profile": "compact_student_int8"},
        headers=auth_headers,
    )
    assert ok_res.status_code == 200
    data = ok_res.json()
    assert data["asset_id"] == pump_id
    assert data["model_name"] == "compact_student_int8"
    assert data["processing_latency_ms"] > 0.0
    assert data["anomaly_decision"] in ["NORMAL_LIKE", "UNUSUAL", "INCONCLUSIVE"]
    assert data["result_id"] is not None

    # 4. Running with allow_cross_category=True permits override
    files_bypass = {"file": ("pump_sample.wav", wav_bytes, "audio/wav")}
    bypass_res = client.post(
        "/api/inference",
        files=files_bypass,
        data={"asset_id": pump_id, "profile": "resnet_ae", "allow_cross_category": True},
        headers=auth_headers,
    )
    assert bypass_res.status_code == 200
    assert bypass_res.json()["model_name"] == "resnet_ae"

    # 5. Retrieve stored inference result
    result_id = ok_res.json()["result_id"]
    fetch_res = client.get(f"/api/inference/{result_id}", headers=auth_headers)
    assert fetch_res.status_code == 200
    assert fetch_res.json()["result_id"] == result_id

    # 6. Check asset inspection history
    hist_res = client.get(f"/api/assets/{pump_id}/history", headers=auth_headers)
    assert hist_res.status_code == 200
    assert hist_res.json()["total_count"] >= 1

"""End-to-end system integration verification test."""

import pytest
from fastapi.testclient import TestClient
from stetho_api.main import app

client = TestClient(app)


def test_full_system_health_and_cors_flow():
    """Verify frontend-to-backend communication prerequisites."""
    # 1. Backend liveness
    health_res = client.get("/health")
    assert health_res.status_code == 200
    health_data = health_res.json()
    assert health_data["status"] == "ok"
    assert health_data["service"] == "Stetho API"

    # 2. Backend readiness
    ready_res = client.get("/ready")
    assert ready_res.status_code == 200
    ready_data = ready_res.json()
    assert ready_data["status"] == "ready"
    assert ready_data["checks"]["api"]["status"] == "healthy"
    assert ready_data["checks"]["database"]["status"] == "healthy"
    assert ready_data["checks"]["storage"]["status"] == "healthy"
    assert ready_data["checks"]["model_subsystem"]["status"] == "healthy"

    # 3. Frontend origin preflight CORS
    cors_res = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert cors_res.status_code == 200
    assert cors_res.headers.get("access-control-allow-origin") == "http://localhost:3000"

    # 4. Frontend origin GET request
    get_res = client.get(
        "/health",
        headers={"Origin": "http://localhost:3000"},
    )
    assert get_res.status_code == 200
    assert get_res.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert "X-Request-ID" in get_res.headers

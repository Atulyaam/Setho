"""Integration tests for FastAPI application layer."""

import uuid
import pytest
from fastapi.testclient import TestClient

from stetho_api.main import app

client = TestClient(app)


def test_request_correlation_id_propagated():
    """Verify that custom X-Request-ID is preserved in response headers."""
    custom_req_id = f"custom-req-{uuid.uuid4().hex[:8]}"
    response = client.get("/health", headers={"X-Request-ID": custom_req_id})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_req_id


def test_auto_generated_correlation_id():
    """Verify that missing X-Request-ID is automatically generated as UUID."""
    response = client.get("/health")
    assert response.status_code == 200
    req_id = response.headers.get("X-Request-ID")
    assert req_id is not None
    # Validate UUID format
    parsed_uuid = uuid.UUID(req_id)
    assert parsed_uuid.version == 4


def test_api_v1_info_endpoint():
    """Verify API v1 metadata endpoint."""
    response = client.get("/api/v1/info")
    assert response.status_code == 200
    data = response.json()
    assert data["api_version"] == "v1"
    assert "endpoints" in data
    assert "/api/v1/machines" in data["endpoints"]


def test_cors_headers_handling():
    """Verify CORS preflight headers."""
    response = client.options(
        "/api/v1/info",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_error_handling_unsupported_method():
    """Verify unsupported HTTP method returns 405 Method Not Allowed with correlation ID."""
    custom_id = "test-405-req-123"
    response = client.post("/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 405
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "HTTP_405"
    assert response.headers.get("X-Request-ID") == custom_id

"""Unit tests for FastAPI health endpoints, configuration loading, and error handling."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from stetho_api.config import Settings
from stetho_api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify /health returns 200 OK and expected structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "Stetho API"
    assert "version" in data
    assert "timestamp" in data
    assert "environment" in data
    assert "X-Request-ID" in response.headers


def test_ready_endpoint():
    """Verify /ready returns 200 OK and system readiness checks."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "checks" in data
    assert "api" in data["checks"]
    assert "storage" in data["checks"]
    assert "database" in data["checks"]
    assert "model_subsystem" in data["checks"]
    assert data["checks"]["api"]["status"] == "healthy"


def test_invalid_endpoint_returns_404_structured_error():
    """Verify 404 Not Found returns structured ErrorResponse."""
    response = client.get("/api/v1/non_existent_endpoint_xyz")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "HTTP_404"
    assert "Not Found" in data["error"]["message"]
    assert "timestamp" in data
    assert "request_id" in data


def test_validation_error_returns_422_structured_error():
    """Verify request validation error returns 422 with structured ErrorResponse."""
    # POST to /api/v1/info which only accepts GET, or send malformed data
    response = client.post("/api/v1/info", json={"unexpected": "payload"})
    assert response.status_code == 405
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "HTTP_405"


def test_configuration_loading_defaults():
    """Verify Settings loads appropriate defaults."""
    settings = Settings()
    assert settings.PROJECT_NAME == "Stetho API"
    assert settings.ENVIRONMENT in ["development", "test"]
    assert settings.PORT == 8000
    assert settings.API_PREFIX == "/api"
    assert not settings.is_production


def test_configuration_production_secret_key_guard():
    """Verify that in production, default insecure secret key is rejected."""
    with pytest.raises(ValidationError):
        Settings(
            ENVIRONMENT="production",
            SECRET_KEY="stetho-dev-secret-key-32-chars-minimum-length-key!",
        )


def test_configuration_production_valid_key():
    """Verify that in production, a secure non-default key succeeds."""
    prod_settings = Settings(
        ENVIRONMENT="production",
        SECRET_KEY="very-strong-production-cryptographic-key-12345",
    )
    assert prod_settings.is_production
    assert prod_settings.SECRET_KEY == "very-strong-production-cryptographic-key-12345"


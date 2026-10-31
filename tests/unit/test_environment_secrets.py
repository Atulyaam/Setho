"""Unit tests for environment variable loading, secrets management, and production guards."""

import os
import pytest
from pydantic import ValidationError

from stetho_api.config import Settings


def test_env_file_loaded(monkeypatch):
    """Verify that settings can load from local environment variables or .env."""
    monkeypatch.setenv("PROJECT_NAME", "Stetho Test Suite")
    monkeypatch.setenv("PORT", "8099")
    monkeypatch.setenv("ENVIRONMENT", "test")

    settings = Settings()
    assert settings.PROJECT_NAME == "Stetho Test Suite"
    assert settings.PORT == 8099
    assert settings.ENVIRONMENT == "test"
    assert settings.is_test


def test_production_environment_missing_secure_secret_fails():
    """Verify that setting ENVIRONMENT=production with default secret raises ValidationError."""
    with pytest.raises(ValidationError) as excinfo:
        Settings(
            ENVIRONMENT="production",
            SECRET_KEY="stetho-dev-secret-key-32-chars-minimum-length-key!",
        )
    assert "Production environment requires a secure, non-default SECRET_KEY." in str(excinfo.value)


def test_production_environment_with_strong_secret_succeeds():
    """Verify that setting ENVIRONMENT=production with a strong unique secret succeeds."""
    settings = Settings(
        ENVIRONMENT="production",
        SECRET_KEY="e4d909c290d0fb1ca068ffaddf22cbd0",
    )
    assert settings.is_production
    assert settings.SECRET_KEY == "e4d909c290d0fb1ca068ffaddf22cbd0"


def test_storage_and_database_settings_placeholders():
    """Verify storage type and database URL default parsing."""
    settings = Settings()
    assert "postgresql://" in settings.DATABASE_URL
    assert settings.STORAGE_TYPE in ["local", "s3", "minio"]
    assert settings.MODEL_ARTIFACTS_DIR.as_posix() == "ml/artifacts"


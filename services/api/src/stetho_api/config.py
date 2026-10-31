"""Environment configuration for the Stetho backend service."""

from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    PROJECT_NAME: str = "Stetho API"
    VERSION: str = "0.1.0"
    ENVIRONMENT: Literal["development", "test", "staging", "production"] = "development"
    DEBUG: bool = False

    # Server Network
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_PREFIX: str = "/api"
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "console"] = "console"

    # Database
    DATABASE_URL: str = Field(
        default="postgresql://stetho_user:stetho_password@localhost:5432/stetho_db",
        description="PostgreSQL connection string.",
    )

    # Audio & Object Storage
    STORAGE_TYPE: Literal["local", "s3", "minio"] = "local"
    STORAGE_LOCAL_DIR: Path = Field(
        default=Path("./data/storage"),
        description="Directory for local storage of audio and reports.",
    )
    S3_ENDPOINT_URL: Optional[str] = None
    S3_BUCKET_NAME: str = "stetho-audio"
    S3_ACCESS_KEY_ID: Optional[str] = None
    S3_SECRET_ACCESS_KEY: Optional[str] = None
    S3_REGION: str = "us-east-1"

    # ML & Model Artifacts
    MODEL_ARTIFACTS_DIR: Path = Field(
        default=Path("./ml/artifacts"),
        description="Path to machine baseline model artifacts.",
    )
    DEVICE_FALLBACK_TO_CPU: bool = True

    # Security
    SECRET_KEY: str = Field(
        default="stetho-dev-secret-key-32-chars-minimum-length-key!",
        description="JWT and encryption secret key.",
    )

    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, v: str, info) -> str:
        # In production, prevent the default insecure secret key
        env = info.data.get("ENVIRONMENT", "development")
        if env == "production" and "stetho-dev-secret" in v:
            raise ValueError("Production environment requires a secure, non-default SECRET_KEY.")
        return v

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_test(self) -> bool:
        return self.ENVIRONMENT == "test"


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()


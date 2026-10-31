"""Common schemas and response models for Stetho API."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


def get_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class HealthResponse(BaseModel):
    """Liveness check response."""
    status: str = Field(default="ok", json_schema_extra={"example": "ok"})
    service: str = Field(default="stetho-api")
    version: str = Field(default="0.1.0")
    environment: str = Field(default="development")
    timestamp: str = Field(default_factory=get_utc_now)


class ComponentStatus(BaseModel):
    """Component-specific readiness indicator."""
    status: str = Field(description="'healthy', 'degraded', or 'unhealthy'")
    details: Optional[str] = None


class ReadyResponse(BaseModel):
    """Readiness check response including subsystem statuses."""
    status: str = Field(description="'ready' or 'not_ready'")
    service: str = Field(default="stetho-api")
    version: str = Field(default="0.1.0")
    environment: str = Field(default="development")
    timestamp: str = Field(default_factory=get_utc_now)
    checks: Dict[str, ComponentStatus] = Field(default_factory=dict)


class ErrorDetail(BaseModel):
    """Structured error information."""
    code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error description")
    details: Optional[Any] = Field(default=None, description="Detailed validation or trace payload")


class ErrorResponse(BaseModel):
    """Standardized API error response format."""
    error: ErrorDetail
    request_id: Optional[str] = None
    timestamp: str = Field(default_factory=get_utc_now)

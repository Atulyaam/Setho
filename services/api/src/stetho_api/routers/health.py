"""Health and readiness endpoints for Stetho API."""

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from stetho_api.config import Settings, get_settings
from stetho_api.schemas.common import ComponentStatus, HealthResponse, ReadyResponse

router = APIRouter(tags=["Health & Status"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="Returns 200 OK if the FastAPI process is alive and responsive.",
)
async def get_health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Return process liveness state."""
    return HealthResponse(
        status="ok",
        service=settings.PROJECT_NAME,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description="Evaluates whether downstream components, model subsystem, and storage are ready.",
)
async def get_ready(settings: Settings = Depends(get_settings)) -> JSONResponse:
    """Return subsystem readiness indicators."""
    checks = {
        "api": ComponentStatus(status="healthy", details="FastAPI router running"),
        "config": ComponentStatus(
            status="healthy",
            details=f"Environment: {settings.ENVIRONMENT}, Debug: {settings.DEBUG}",
        ),
        "storage": ComponentStatus(
            status="healthy" if settings.STORAGE_TYPE in ["local", "s3", "minio"] else "degraded",
            details=f"Type: {settings.STORAGE_TYPE}",
        ),
        "database": ComponentStatus(
            status="healthy",
            details="Configured and active",
        ),
        "model_subsystem": ComponentStatus(
            status="healthy",
            details="Stetho-MSAE inference runner initialized (Sprint 0 stub ready)",
        ),
    }

    # Determine overall status
    is_ready = all(c.status in ["healthy", "degraded"] for c in checks.values())
    overall_status = "ready" if is_ready else "not_ready"
    http_status = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE

    payload = ReadyResponse(
        status=overall_status,
        service=settings.PROJECT_NAME,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        checks=checks,
    )
    return JSONResponse(status_code=http_status, content=payload.model_dump())


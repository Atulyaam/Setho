"""Health and readiness endpoints for Stetho API."""

from pathlib import Path
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from stetho_api.config import Settings, get_settings
from stetho_api.db.database import get_db
from stetho_api.inference.model_registry import get_model_registry
from stetho_api.schemas.common import ComponentStatus, HealthResponse, ReadyResponse

router = APIRouter(tags=["Health & Status"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="Returns 200 OK if the FastAPI process is alive and responsive.",
)
@router.get(
    "/health/live",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
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
@router.get(
    "/health/ready",
    response_model=ReadyResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
async def get_ready(settings: Settings = Depends(get_settings)) -> JSONResponse:
    """Return subsystem readiness indicators."""
    checks = {}

    # 1. API Core
    checks["api"] = ComponentStatus(status="healthy", details="FastAPI router running")

    # 2. Config
    checks["config"] = ComponentStatus(
        status="healthy",
        details=f"Environment: {settings.ENVIRONMENT}, Debug: {settings.DEBUG}",
    )

    # 3. Storage
    try:
        storage_path = Path(settings.STORAGE_LOCAL_DIR)
        storage_path.mkdir(parents=True, exist_ok=True)
        # Test writeability
        test_file = storage_path / ".readiness_probe_test"
        test_file.write_text("probe")
        test_file.unlink(missing_ok=True)
        checks["storage"] = ComponentStatus(
            status="healthy",
            details=f"Local storage active at {storage_path}",
        )
    except Exception as e:
        checks["storage"] = ComponentStatus(
            status="unhealthy",
            details=f"Storage inaccessible: {str(e)}",
        )

    # 4. Database
    try:
        db = get_db(settings.STORAGE_LOCAL_DIR / "stetho.db")
        conn = db.get_connection()
        conn.execute("SELECT 1;").fetchone()
        checks["database"] = ComponentStatus(
            status="healthy",
            details=f"SQLite ACID persistence verified at {db.db_path}",
        )
    except Exception as e:
        checks["database"] = ComponentStatus(
            status="unhealthy",
            details=f"Database error: {str(e)}",
        )

    # 5. Model Subsystem
    try:
        registry = get_model_registry()
        reg_status = registry.get_status()
        details_str = (
            f"Device: {reg_status['target_device']} "
            f"(loaded: {reg_status['loaded_count']}, failed: {reg_status['failed_count']})"
        )
        checks["model_subsystem"] = ComponentStatus(
            status=reg_status["status"],
            details=details_str,
        )
    except Exception as e:
        checks["model_subsystem"] = ComponentStatus(
            status="unhealthy",
            details=f"Model registry failure: {str(e)}",
        )

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

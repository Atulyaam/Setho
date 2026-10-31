"""API v1 master router aggregating machines and analyses."""

from fastapi import APIRouter

from stetho_api.routers.analyses import router as analyses_router
from stetho_api.routers.machines import router as machines_router

router = APIRouter(prefix="/v1")

# Include domain routers
router.include_router(machines_router)
router.include_router(analyses_router)


@router.get("/info", tags=["API v1"])
async def get_api_info():
    """Return API metadata and route outline."""
    return {
        "api_version": "v1",
        "description": "Stetho Machine-Specific Acoustic Inspection API",
        "endpoints": [
            "/api/v1/machines",
            "/api/v1/machines/{machine_id}",
            "/api/v1/machines/{machine_id}/enrollments",
            "/api/v1/machines/{machine_id}/analyze",
            "/api/v1/machines/{machine_id}/analysis-jobs",
            "/api/v1/analysis-jobs/{job_id}",
            "/api/v1/analyses/{analysis_id}",
            "/api/v1/analyses/{analysis_id}/report",
        ],
    }

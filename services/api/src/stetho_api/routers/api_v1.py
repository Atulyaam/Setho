"""API v1 placeholder router defining future machine inspection routes."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/v1", tags=["API v1"])


class MachineCreateRequest(BaseModel):
    name: str
    machine_type: str
    operating_mode: str


@router.get("/info")
async def get_api_info():
    """Return API metadata."""
    return {
        "api_version": "v1",
        "description": "Stetho Machine-Specific Acoustic Inspection API",
        "endpoints": [
            "/api/v1/machines",
            "/api/v1/analyses",
        ],
    }


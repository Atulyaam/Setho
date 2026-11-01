"""FastAPI Router for Asset Lifecycle Management and Few-Shot Calibration (Sprint 7)."""

import io
import logging
from typing import List, Optional, Union
import uuid
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
import numpy as np
import soundfile as sf

from stetho_api.audio.uploader import AudioIngestionService
from stetho_api.auth.dependencies import get_current_user
from stetho_api.config import Settings, get_settings
from stetho_api.db import (
    Asset,
    AssetRepository,
    CalibrationHistoryRecord,
    CalibrationProfile,
    CalibrationRepository,
    DatabaseManager,
    InferenceRepository,
    get_db,
)
from stetho_api.db.user_repo import User
from stetho_api.exceptions import ValidationException
from stetho_api.inference.calibration_engine import MachineCalibrationEngine
from stetho_api.schemas.asset import (
    AssetCalibrationResponse,
    AssetCreateRequest,
    AssetListResponse,
    AssetResponse,
)
from stetho_api.schemas.inference import InferenceHistoryListResponse, InferenceResponse

logger = logging.getLogger("stetho.assets_router")
router = APIRouter(tags=["Asset Management & Calibration"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB limit


def get_services(settings: Settings = Depends(get_settings)):
    db = get_db(settings.STORAGE_LOCAL_DIR / "stetho.db")
    return {
        "db": db,
        "asset_repo": AssetRepository(db),
        "calibration_repo": CalibrationRepository(db),
        "inference_repo": InferenceRepository(db),
        "calibration_engine": MachineCalibrationEngine(db=db),
        "ingestion": AudioIngestionService(settings.STORAGE_LOCAL_DIR),
    }


@router.post(
    "/assets",
    response_model=AssetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new industrial asset",
    description="Registers a physical machine (fan, pump, or valve) for acoustic monitoring.",
)
async def create_asset(
    request: AssetCreateRequest,
    current_user: User = Depends(get_current_user),
    services=Depends(get_services),
) -> AssetResponse:
    repo: AssetRepository = services["asset_repo"]
    asset = repo.create(
        asset_id=request.asset_id,
        name=request.name,
        category=request.category,
        description=request.description or "",
        operating_condition=request.operating_condition or "",
        assigned_model=request.assigned_model or "cascade_ensemble",
        assigned_model_version="1.0.0",
        metadata=request.metadata or {},
        owner_id=current_user.user_id,
    )
    return AssetResponse(**asset.to_dict())


@router.get(
    "/assets",
    response_model=AssetListResponse,
    summary="List industrial assets",
    description="Lists registered machinery assets accessible to the authenticated user.",
)
async def list_assets(
    category: Optional[str] = Query(None, description="Optional category filter: fan, pump, or valve"),
    current_user: User = Depends(get_current_user),
    services=Depends(get_services),
) -> AssetListResponse:
    repo: AssetRepository = services["asset_repo"]
    is_admin = current_user.role == "admin"
    assets = repo.list_all(
        user_id=current_user.user_id,
        is_admin=is_admin,
        category=category,
    )
    return AssetListResponse(
        assets=[AssetResponse(**a.to_dict()) for a in assets],
        total_count=len(assets),
    )


@router.get(
    "/assets/{asset_id}",
    response_model=AssetResponse,
    summary="Get asset details",
    description="Retrieves asset metadata, assigned model policy, and calibration state.",
)
async def get_asset(
    asset_id: str,
    current_user: User = Depends(get_current_user),
    services=Depends(get_services),
) -> AssetResponse:
    repo: AssetRepository = services["asset_repo"]
    asset = repo.get_or_raise(asset_id)
    return AssetResponse(**asset.to_dict())


@router.post(
    "/assets/{asset_id}/calibrate",
    response_model=AssetCalibrationResponse,
    summary="Execute few-shot onboarding calibration",
    description=(
        "Executes 10-normal-recording onboarding calibration for a machine. "
        "Strictly requires operator confirmation that uploaded audio represents normal baseline operation."
    ),
)
async def calibrate_asset(
    asset_id: str,
    files: List[UploadFile] = File(..., description="1 to 10 confirmed normal WAV audio recordings."),
    is_confirmed_normal: Union[bool, str] = Form(
        ...,
        description="Mandatory operator attestation that all submitted recordings represent normal operational acoustic behavior.",
    ),
    notes: Optional[str] = Form("", description="Optional calibration notes or operator comments."),
    current_user: User = Depends(get_current_user),
    services=Depends(get_services),
) -> AssetCalibrationResponse:
    confirmed = (
        is_confirmed_normal
        if isinstance(is_confirmed_normal, bool)
        else str(is_confirmed_normal).lower() in ("true", "1", "yes")
    )
    if not confirmed:
        raise ValidationException(
            "Calibration strictly requires operator confirmation (is_confirmed_normal=true). "
            "Unverified or anomalous audio recordings cannot be used to establish normal baselines."
        )

    if not files:
        raise ValidationException("At least one audio file must be uploaded for calibration.")

    ingestion: AudioIngestionService = services["ingestion"]
    cal_engine: MachineCalibrationEngine = services["calibration_engine"]

    decoded_recordings = []

    for f in files:
        contents = await f.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            raise ValidationException(f"File '{f.filename}' exceeds 25 MB limit.")

        try:
            audio_data, sr = sf.read(io.BytesIO(contents), dtype="float32", always_2d=True)
            if audio_data.ndim > 1:
                wave_1d = audio_data[:, 0]
            else:
                wave_1d = audio_data
            decoded_recordings.append((wave_1d, sr))

            # Store file safely in storage vault
            ingestion.save_recording(
                machine_id=asset_id,
                file_bytes=contents,
                original_filename=f.filename or "normal_baseline.wav",
            )
        except Exception as e:
            raise ValidationException(f"Invalid audio format in '{f.filename}': {e}")

    result = cal_engine.calibrate_asset(
        asset_id=asset_id,
        audio_recordings=decoded_recordings,
        operator_id=current_user.user_id,
        is_confirmed_normal=True,
        notes=notes or "",
    )

    return AssetCalibrationResponse(
        asset_id=result.asset_id,
        model_type=result.model_type,
        calibration_version=result.calibration_version,
        calibration_status=result.calibration_status,
        operating_threshold=result.operating_threshold,
        theta_lower=result.theta_lower,
        theta_upper=result.theta_upper,
        enrolled_sample_count=result.enrolled_sample_count,
        enrolled_mean=result.enrolled_mean,
        enrolled_std=result.enrolled_std,
        is_provisional=result.is_provisional,
        notice=result.notice,
        recalculated_at_utc=result.recalculated_at_utc,
        enrolled_scores=result.enrolled_scores,
    )


@router.post(
    "/assets/{asset_id}/calibration/rollback",
    response_model=AssetCalibrationResponse,
    summary="Rollback calibration profile",
    description="Restores the previous verified calibration profile for this asset.",
)
async def rollback_calibration(
    asset_id: str,
    target_version: Optional[str] = Form(None, description="Optional target version to restore."),
    current_user: User = Depends(get_current_user),
    services=Depends(get_services),
) -> AssetCalibrationResponse:
    cal_engine: MachineCalibrationEngine = services["calibration_engine"]
    result = cal_engine.rollback_calibration(
        asset_id=asset_id,
        target_version=target_version,
        operator_id=current_user.user_id,
    )
    return AssetCalibrationResponse(
        asset_id=result.asset_id,
        model_type=result.model_type,
        calibration_version=result.calibration_version,
        calibration_status=result.calibration_status,
        operating_threshold=result.operating_threshold,
        theta_lower=result.theta_lower,
        theta_upper=result.theta_upper,
        enrolled_sample_count=result.enrolled_sample_count,
        enrolled_mean=result.enrolled_mean,
        enrolled_std=result.enrolled_std,
        is_provisional=result.is_provisional,
        notice=result.notice,
        recalculated_at_utc=result.recalculated_at_utc,
        enrolled_scores=result.enrolled_scores,
    )


@router.get(
    "/assets/{asset_id}/history",
    response_model=InferenceHistoryListResponse,
    summary="Get inspection history",
    description="Returns chronological inspection results and anomaly trends for an asset.",
)
async def get_asset_history(
    asset_id: str,
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    services=Depends(get_services),
) -> InferenceHistoryListResponse:
    repo: InferenceRepository = services["inference_repo"]
    records = repo.list_by_asset(asset_id, limit=limit)
    return InferenceHistoryListResponse(
        results=[InferenceResponse(**r.to_dict()) for r in records],
        total_count=len(records),
    )

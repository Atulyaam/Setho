"""FastAPI Router for Acoustic Inference Execution and Model Discovery (Sprint 7)."""

import hashlib
import io
import logging
from typing import Optional, Union
import uuid
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
import numpy as np
import soundfile as sf

from stetho_api.audio.uploader import AudioIngestionService
from stetho_api.auth.dependencies import get_current_user
from stetho_api.config import Settings, get_settings
from stetho_api.db import DatabaseManager, InferenceRecord, InferenceRepository, get_db
from stetho_api.db.user_repo import User
from stetho_api.exceptions import ValidationException
from stetho_api.inference import (
    AcousticInferenceService,
    ModelRegistry,
    get_inference_service,
    get_model_registry,
)
from stetho_api.schemas.inference import (
    InferenceHistoryListResponse,
    InferenceResponse,
    ModelInfoResponse,
    ModelListResponse,
)

logger = logging.getLogger("stetho.inference_router")
router = APIRouter(tags=["Acoustic Inference & Models"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB max limit


def get_dependencies(settings: Settings = Depends(get_settings)):
    db = get_db(settings.STORAGE_LOCAL_DIR / "stetho.db")
    return {
        "db": db,
        "inference_repo": InferenceRepository(db),
        "inference_service": get_inference_service(db=db),
        "model_registry": get_model_registry(),
        "ingestion": AudioIngestionService(settings.STORAGE_LOCAL_DIR),
    }


@router.post(
    "/inference",
    response_model=InferenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute acoustic anomaly inspection",
    description=(
        "Executes end-to-end acoustic inference on an uploaded WAV recording for a machine asset. "
        "Enforces asset-category routing restrictions and returns three-zone anomaly decisions."
    ),
)
async def execute_inference(
    asset_id: str = Form(..., description="Target machine asset ID."),
    file: UploadFile = File(..., description="10-second industrial WAV audio recording."),
    profile: Optional[str] = Form(
        None,
        description="Optional deployment profile override ('edge_real_time', 'cloud_high_accuracy', 'hybrid_cascade', 'zero_shot').",
    ),
    allow_cross_category: Union[bool, str] = Form(
        False,
        description="Flag to bypass physical asset-category restrictions (strictly logged for audits).",
    ),
    current_user: User = Depends(get_current_user),
    deps=Depends(get_dependencies),
) -> InferenceResponse:
    cross_cat = (
        allow_cross_category
        if isinstance(allow_cross_category, bool)
        else str(allow_cross_category).lower() in ("true", "1", "yes")
    )
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise ValidationException(f"Uploaded audio exceeds maximum limit of 25 MB ({len(contents)} bytes).")

    if len(contents) < 44:
        raise ValidationException("Uploaded audio is corrupt or too small to contain a valid WAV header.")

    # Compute SHA-256 hash of raw uploaded audio
    audio_hash = hashlib.sha256(contents).hexdigest()

    # Audio format validation
    try:
        audio_data, sr = sf.read(io.BytesIO(contents), dtype="float32", always_2d=True)
    except Exception as e:
        raise ValidationException(f"Invalid audio format in '{file.filename}': {e}")

    # Save audio safely to storage vault
    ingestion: AudioIngestionService = deps["ingestion"]
    safe_path = ingestion.save_recording(
        machine_id=asset_id,
        file_bytes=contents,
        original_filename=file.filename or "inspection.wav",
    )

    inference_service: AcousticInferenceService = deps["inference_service"]
    result = inference_service.analyze_audio(
        audio=audio_data,
        sample_rate=sr,
        machine_id=asset_id,
        model_override=profile,
        allow_cross_category=cross_cat,
        operator_id=current_user.user_id,
        audio_storage_path=str(safe_path),
        audio_hash=audio_hash,
    )

    return InferenceResponse(**result.to_dict())


@router.get(
    "/inference/history/{asset_id}",
    response_model=InferenceHistoryListResponse,
    summary="Retrieve asset inspection history",
    description="Returns chronological inspection results and anomaly trends for an asset.",
)
async def get_asset_inference_history(
    asset_id: str,
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    deps=Depends(get_dependencies),
) -> InferenceHistoryListResponse:
    repo: InferenceRepository = deps["inference_repo"]
    records = repo.list_by_asset(asset_id, limit=limit)
    return InferenceHistoryListResponse(
        results=[InferenceResponse(**r.to_dict()) for r in records],
        total_count=len(records),
    )


@router.get(
    "/inference/{result_id}",
    response_model=InferenceResponse,
    summary="Retrieve inference record",
    description="Retrieves a historical acoustic inspection record by ID.",
)
async def get_inference_result(
    result_id: str,
    current_user: User = Depends(get_current_user),
    deps=Depends(get_dependencies),
) -> InferenceResponse:
    repo: InferenceRepository = deps["inference_repo"]
    record = repo.get_or_raise(result_id)
    return InferenceResponse(**record.to_dict())


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List registered models",
    description="Returns metadata, supported categories, and latency benchmarks for all registered ML models.",
)
async def list_models(
    current_user: User = Depends(get_current_user),
    deps=Depends(get_dependencies),
) -> ModelListResponse:
    registry: ModelRegistry = deps["model_registry"]
    raw_models = registry.list_available_models()
    models = [ModelInfoResponse(**m) for m in raw_models]
    return ModelListResponse(models=models, total_count=len(models))

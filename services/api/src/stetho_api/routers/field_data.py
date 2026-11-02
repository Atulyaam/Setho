"""FastAPI Router for Field Data Collection, Annotation, and Pilot Readiness (Sprint 7.7 / E8)."""

import json
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import FileResponse

from stetho_api.auth.dependencies import get_current_user
from stetho_api.config import Settings, get_settings
from stetho_api.db import DatabaseManager, FieldDataRepo, get_db
from stetho_api.db.user_repo import User
from stetho_api.schemas.field_data import (
    DatasetRoleUpdateRequest,
    FieldAnnotationCreateRequest,
    FieldAnnotationResponse,
    FieldDashboardStatsResponse,
    FieldRecordingListResponse,
    FieldRecordingResponse,
    ManifestCreateRequest,
    ManifestResponse,
    ProtocolChecklistResponse,
)

logger = logging.getLogger("stetho.field_data_router")
router = APIRouter(prefix="/field-data", tags=["Field Data & Pilot Governance"])

MAX_FIELD_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB max per field clip


def get_field_repo(settings: Settings = Depends(get_settings)) -> FieldDataRepo:
    db = get_db(settings.STORAGE_LOCAL_DIR / "stetho.db")
    storage_dir = settings.STORAGE_LOCAL_DIR / "field_recordings"
    return FieldDataRepo(db=db, storage_base_dir=storage_dir)


@router.post(
    "/import",
    response_model=FieldRecordingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Import field acoustic recording",
    description="Ingests real industrial WAV audio with cryptographic hash verification, metadata logging, and quality gating.",
)
async def import_field_recording(
    file: UploadFile = File(..., description="Raw field WAV audio file"),
    machine_category: str = Form(..., description="'fan', 'pump', 'valve', 'compressor', etc."),
    machine_id: str = Form(..., description="Unique machine instance ID"),
    sensor_id: str = Form(..., description="Microphone/transducer serial number"),
    channel_index: int = Form(0, description="Active channel index"),
    microphone_position: str = Form("primary_bearing", description="Physical mounting location"),
    operating_condition: str = Form("nominal", description="e.g. nominal, idle, full_load"),
    load_level: str = Form("full_load", description="Machine mechanical load"),
    captured_at: Optional[str] = Form(None, description="ISO capture timestamp"),
    asset_id: Optional[str] = Form(None, description="Optional asset linkage"),
    provenance_source: str = Form("field_pilot", description="Origin of recording"),
    consent_status: str = Form("authorized", description="Authorization status"),
    dataset_role: str = Form("unassigned", description="Dataset role ('unassigned', 'training', 'development', 'calibration_enrollment', 'sealed_evaluation')"),
    metadata_json: Optional[str] = Form("{}", description="Additional operational metadata as JSON"),
    current_user: User = Depends(get_current_user),
    repo: FieldDataRepo = Depends(get_field_repo),
):
    if current_user.role == "viewer":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Viewers are not permitted to import recordings.",
        )

    # 1. Read file bytes with limit check
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )
    if len(file_bytes) > MAX_FIELD_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds maximum limit of {MAX_FIELD_UPLOAD_BYTES // (1024*1024)} MB.",
        )

    # 2. Parse optional metadata JSON
    try:
        parsed_metadata = json.loads(metadata_json or "{}")
    except json.JSONDecodeError:
        parsed_metadata = {}

    # 3. Import through repository
    recording = repo.import_recording(
        file_bytes=file_bytes,
        original_filename=file.filename or "recording.wav",
        machine_category=machine_category,
        machine_id=machine_id,
        sensor_id=sensor_id,
        operator_id=current_user.username,
        channel_index=channel_index,
        microphone_position=microphone_position,
        operating_condition=operating_condition,
        load_level=load_level,
        captured_at=captured_at,
        asset_id=asset_id,
        provenance_source=provenance_source,
        consent_status=consent_status,
        dataset_role=dataset_role,
        metadata=parsed_metadata,
    )

    return recording


@router.get(
    "/recordings",
    response_model=FieldRecordingListResponse,
    summary="List field recordings",
    description="Retrieve paginated field recordings with optional filtering by category, machine, role, label, and status.",
)
async def list_recordings(
    machine_category: Optional[str] = Query(None, description="Filter by equipment category"),
    machine_id: Optional[str] = Query(None, description="Filter by machine ID"),
    dataset_role: Optional[str] = Query(None, description="Filter by role (training, sealed_evaluation, etc.)"),
    current_label: Optional[str] = Query(None, description="Filter by label (normal, abnormal, etc.)"),
    verification_status: Optional[str] = Query(None, description="Filter by status (VERIFIED, PENDING)"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    repo: FieldDataRepo = Depends(get_field_repo),
):
    items, total = repo.list_recordings(
        machine_category=machine_category,
        machine_id=machine_id,
        dataset_role=dataset_role,
        current_label=current_label,
        verification_status=verification_status,
        limit=limit,
        offset=offset,
    )
    return {
        "items": items,
        "total_count": total,
        "limit": limit,
        "offset": offset,
    }


@router.get(
    "/recordings/{recording_id}",
    response_model=FieldRecordingResponse,
    summary="Get field recording details",
)
async def get_recording(
    recording_id: str,
    current_user: User = Depends(get_current_user),
    repo: FieldDataRepo = Depends(get_field_repo),
):
    rec = repo.get_recording(recording_id)
    if not rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recording '{recording_id}' not found.",
        )
    return rec


@router.get(
    "/recordings/{recording_id}/audio",
    summary="Stream or download field recording audio",
    description="Streams raw WAV audio with authentication and path safety verification.",
)
async def get_recording_audio(
    recording_id: str,
    current_user: User = Depends(get_current_user),
    repo: FieldDataRepo = Depends(get_field_repo),
):
    path = repo.get_recording_audio_path(recording_id)
    return FileResponse(
        path=str(path),
        media_type="audio/wav",
        filename=f"{recording_id}.wav",
    )


@router.post(
    "/recordings/{recording_id}/annotate",
    response_model=FieldAnnotationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit ground-truth annotation",
    description="Submit an independently confirmed diagnosis (normal/abnormal/uncertain) with evidence source.",
)
async def annotate_recording(
    recording_id: str,
    payload: FieldAnnotationCreateRequest,
    current_user: User = Depends(get_current_user),
    repo: FieldDataRepo = Depends(get_field_repo),
):
    if current_user.role == "viewer":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Viewers are not permitted to annotate recordings.",
        )
    return repo.add_annotation(
        recording_id=recording_id,
        label=payload.label,
        reason=payload.reason,
        evidence_source=payload.evidence_source,
        annotator_id=current_user.username,
        inspection_notes=payload.inspection_notes or "",
    )


@router.get(
    "/recordings/{recording_id}/annotations",
    response_model=List[FieldAnnotationResponse],
    summary="Get annotation history",
    description="Retrieve the chronological revision history of human annotations for a recording.",
)
async def get_annotation_history(
    recording_id: str,
    current_user: User = Depends(get_current_user),
    repo: FieldDataRepo = Depends(get_field_repo),
):
    return repo.get_annotation_history(recording_id)


@router.put(
    "/recordings/{recording_id}/role",
    response_model=FieldRecordingResponse,
    summary="Assign dataset role",
    description="Assign a dataset role ('training', 'development', 'calibration_enrollment', 'sealed_evaluation').",
)
async def update_dataset_role(
    recording_id: str,
    payload: DatasetRoleUpdateRequest,
    current_user: User = Depends(get_current_user),
    repo: FieldDataRepo = Depends(get_field_repo),
):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Only administrators can modify dataset separation roles.",
        )
    return repo.assign_dataset_role(
        recording_id=recording_id,
        new_role=payload.dataset_role,
        operator_id=current_user.username,
    )


@router.post(
    "/manifests",
    response_model=ManifestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create or seal dataset manifest",
    description="Freezes and checksums an immutable manifest for training or sealed evaluation.",
)
async def create_manifest(
    payload: ManifestCreateRequest,
    current_user: User = Depends(get_current_user),
    repo: FieldDataRepo = Depends(get_field_repo),
):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Only administrators can create dataset manifests.",
        )
    return repo.create_or_seal_manifest(
        name=payload.name,
        dataset_role=payload.dataset_role,
        machine_category=payload.machine_category,
        recording_ids=payload.recording_ids,
        created_by=current_user.username,
        seal_now=payload.seal_now,
        metadata=payload.metadata,
    )


@router.post(
    "/manifests/{manifest_id}/seal",
    response_model=ManifestResponse,
    summary="Seal existing dataset manifest",
    description="Freezes and checksums an existing open manifest.",
)
async def seal_manifest(
    manifest_id: str,
    current_user: User = Depends(get_current_user),
    repo: FieldDataRepo = Depends(get_field_repo),
):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Only administrators can seal dataset manifests.",
        )
    return repo.seal_manifest(manifest_id=manifest_id, operator_id=current_user.username)


@router.get(
    "/manifests/{manifest_id}/verify",
    summary="Cryptographically verify manifest integrity",
    description="Verifies all member recordings against disk and recalculates manifest SHA-256.",
)
async def verify_manifest(
    manifest_id: str,
    current_user: User = Depends(get_current_user),
    repo: FieldDataRepo = Depends(get_field_repo),
):
    return repo.verify_manifest(manifest_id)


@router.get(
    "/manifests",
    response_model=List[ManifestResponse],
    summary="List dataset manifests",
)
async def list_manifests(
    dataset_role: Optional[str] = Query(None, description="Filter by role"),
    current_user: User = Depends(get_current_user),
    repo: FieldDataRepo = Depends(get_field_repo),
):
    return repo.list_manifests(dataset_role=dataset_role)


@router.get(
    "/dashboard",
    response_model=FieldDashboardStatsResponse,
    summary="Field data pilot telemetry and readiness",
    description="Returns aggregate counts, label distributions, and independent evaluation readiness with warning indicators.",
)
async def get_dashboard_stats(
    current_user: User = Depends(get_current_user),
    repo: FieldDataRepo = Depends(get_field_repo),
):
    return repo.get_dashboard_stats()


@router.get(
    "/protocol-checklist",
    summary="Download field data collection protocol",
    description="Returns the standard operating procedure and checklist for field technicians (JSON or Markdown).",
)
async def get_protocol_checklist(
    export_format: str = Query("json", description="'json' or 'markdown'"),
    current_user: User = Depends(get_current_user),
    repo: FieldDataRepo = Depends(get_field_repo),
):
    data = repo.get_collection_protocol_checklist()
    if export_format.lower() == "markdown":
        md = [
            f"# {data['title']} (v{data['version']})\n",
            "## Technical Specifications",
            f"- **Audio Format:** {data['specifications']['format']}",
            f"- **Sample Rate:** {data['specifications']['sample_rate']}",
            f"- **Clip Duration:** {data['specifications']['duration']}",
            f"- **Channels:** {data['specifications']['channels']}\n",
            "## Field Collection Procedure Steps\n",
        ]
        for s in data["steps"]:
            md.append(f"### Step {s['step']}: {s['title']}")
            md.append(f"{s['action']}\n")
        md.append("## Approved Multi-Modal Evidence Sources")
        for ev in data["evidence_sources"]:
            md.append(f"- `{ev}`")
        md_text = "\n".join(md)
        return Response(
            content=md_text,
            media_type="text/markdown",
            headers={"Content-Disposition": 'attachment; filename="stetho_field_collection_protocol.md"'},
        )

    return data

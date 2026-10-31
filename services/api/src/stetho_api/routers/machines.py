"""FastAPI Router for Machine Profiles and Audio Enrollment."""

from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from stetho_api.audio import (
    AudioIngestionService,
    AudioQualityGate,
    QualityStatus,
)
from stetho_api.auth.dependencies import get_current_user
from stetho_api.config import Settings, get_settings
from stetho_api.db import (
    DatabaseManager,
    EnrollmentRepository,
    MachineProfile,
    MachineRepository,
    get_db,
)
from stetho_api.db.user_repo import User
from stetho_api.exceptions import StethoAPIException
from stetho_api.inference.calibration_engine import MachineCalibrationEngine
from stetho_api.inference.model_registry import get_model_registry
from stetho_api.schemas.machine import (
    EnrollmentResponse,
    MachineCreateRequest,
    MachineListResponse,
    MachineResponse,
)

router = APIRouter(tags=["Machine Profiles & Enrollment"])


def get_repos(settings: Settings = Depends(get_settings)):
    db = get_db(settings.STORAGE_LOCAL_DIR / "stetho.db")
    registry = get_model_registry()
    return {
        "db": db,
        "machine_repo": MachineRepository(db),
        "enrollment_repo": EnrollmentRepository(db),
        "ingestion": AudioIngestionService(settings.STORAGE_LOCAL_DIR),
        "quality_gate": AudioQualityGate(),
        "calibration_engine": MachineCalibrationEngine(db=db, registry=registry),
    }


@router.post(
    "/machines",
    response_model=MachineResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create machine profile",
    description="Registers a new physical machine for acoustic anomaly monitoring with authenticated user ownership.",
)
async def create_machine(
    request: MachineCreateRequest,
    current_user: User = Depends(get_current_user),
    repos=Depends(get_repos),
) -> MachineResponse:
    repo: MachineRepository = repos["machine_repo"]
    profile = repo.create(
        machine_id=request.machine_id,
        name=request.name,
        category=request.category,
        description=request.description or "",
        operating_condition=request.operating_condition or "",
        selected_model_policy=request.selected_model_policy or "stetho_msae",
        owner_id=current_user.user_id,
    )
    return MachineResponse(**profile.to_dict())


@router.get(
    "/machines",
    response_model=MachineListResponse,
    summary="List machine profiles",
    description="Returns registered machines visible to the authenticated user.",
)
async def list_machines(
    current_user: User = Depends(get_current_user),
    repos=Depends(get_repos),
) -> MachineListResponse:
    repo: MachineRepository = repos["machine_repo"]
    is_admin = current_user.role == "admin"
    profiles = repo.list_all(user_id=current_user.user_id, is_admin=is_admin)
    return MachineListResponse(
        machines=[MachineResponse(**p.to_dict()) for p in profiles],
        total_count=len(profiles),
    )


@router.get(
    "/machines/{machine_id}",
    response_model=MachineResponse,
    summary="Get machine profile",
    description="Retrieves a machine profile, verifying ownership authorization.",
)
async def get_machine(
    machine_id: str,
    current_user: User = Depends(get_current_user),
    repos=Depends(get_repos),
) -> MachineResponse:
    repo: MachineRepository = repos["machine_repo"]
    is_admin = current_user.role == "admin"
    profile = repo.get_authorized(machine_id, user_id=current_user.user_id, is_admin=is_admin)
    return MachineResponse(**profile.to_dict())


@router.post(
    "/machines/{machine_id}/enrollments",
    response_model=EnrollmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Enroll normal machine recording and recalculate calibration",
    description="Registers an explicitly confirmed normal baseline audio recording and dynamically updates calibrated decision thresholds.",
)
async def enroll_normal_audio(
    machine_id: str,
    file: UploadFile = File(..., description="Normal reference WAV audio recording"),
    confirmed_normal: bool = Form(True, description="Must be explicitly confirmed normal by inspector"),
    current_user: User = Depends(get_current_user),
    repos=Depends(get_repos),
) -> EnrollmentResponse:
    machine_repo: MachineRepository = repos["machine_repo"]
    enrollment_repo: EnrollmentRepository = repos["enrollment_repo"]
    ingestion: AudioIngestionService = repos["ingestion"]
    quality_gate: AudioQualityGate = repos["quality_gate"]
    calibration_engine: MachineCalibrationEngine = repos["calibration_engine"]

    # 1. Verify machine profile exists and user is authorized
    is_admin = current_user.role == "admin"
    machine = machine_repo.get_authorized(machine_id, user_id=current_user.user_id, is_admin=is_admin)

    if not confirmed_normal:
        raise StethoAPIException(
            "Enrollment audio must be explicitly confirmed as normal baseline operating acoustic sound.",
            code="UNCONFIRMED_ENROLLMENT",
            status_code=400,
        )

    # 2. Ingest and validate file container
    meta, raw_audio = ingestion.ingest_audio(
        file_obj=file.file,
        filename=file.filename,
        content_type=file.content_type,
        machine_id=machine_id,
    )

    # 3. Audio Quality Gate assessment
    mono_audio = raw_audio[:, 0] if raw_audio.ndim > 1 else raw_audio
    q_report = quality_gate.assess_waveform(mono_audio, sample_rate=meta.sample_rate)

    if q_report.status in [QualityStatus.REJECTED, QualityStatus.INCONCLUSIVE]:
        raise StethoAPIException(
            f"Enrollment audio rejected by Audio Quality Gate ({q_report.status.value}): {'; '.join(q_report.reasons)}",
            code="QUALITY_GATE_REJECTED",
            status_code=422,
            details=q_report.to_dict(),
        )

    # 4. Record enrollment in DB (with deduplication hash check)
    rec = enrollment_repo.record_enrollment(
        machine_id=machine_id,
        upload_id=meta.upload_id,
        stored_path=meta.stored_path,
        duration_sec=meta.duration_sec,
        sample_rate=meta.sample_rate,
        channels=meta.channels,
        quality_status=q_report.status.value,
        is_confirmed_normal=True,
        content_hash=meta.content_hash,
    )

    # 5. DYNAMIC CALIBRATION RECALCULATION (Task 4.5.2)
    # Genuinely score enrolled audio, recompute distribution and operating thresholds
    cal_result = calibration_engine.recalibrate_machine(
        machine_id=machine_id,
        newest_waveform=mono_audio,
        sample_rate=meta.sample_rate,
    )

    return EnrollmentResponse(
        enrollment_id=rec.enrollment_id,
        machine_id=machine_id,
        upload_id=meta.upload_id,
        duration_sec=meta.duration_sec,
        sample_rate=meta.sample_rate,
        channels=meta.channels,
        quality_status=q_report.status.value,
        is_confirmed_normal=True,
        created_at=rec.created_at,
        enrolled_count=cal_result.enrolled_sample_count,
        enrollment_status=cal_result.calibration_status,
        operating_threshold=cal_result.operating_threshold,
        theta_lower=cal_result.theta_lower,
        theta_upper=cal_result.theta_upper,
        calibration_version=cal_result.calibration_version,
        notice=cal_result.notice,
    )


@router.get(
    "/machines/{machine_id}/enrollments",
    summary="List machine enrollments",
    description="Returns previous normal reference enrollment recordings for a machine.",
)
async def list_enrollments(
    machine_id: str,
    current_user: User = Depends(get_current_user),
    repos=Depends(get_repos),
):
    machine_repo: MachineRepository = repos["machine_repo"]
    enrollment_repo: EnrollmentRepository = repos["enrollment_repo"]
    is_admin = current_user.role == "admin"
    machine_repo.get_authorized(machine_id, user_id=current_user.user_id, is_admin=is_admin)

    records = enrollment_repo.list_by_machine(machine_id)
    return {
        "machine_id": machine_id,
        "enrollments": [r.to_dict() for r in records],
        "total_count": len(records),
    }

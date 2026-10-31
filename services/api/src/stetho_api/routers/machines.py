"""FastAPI Router for Machine Profiles and Audio Enrollment."""

from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from stetho_api.audio import (
    AudioIngestionService,
    AudioQualityGate,
    QualityStatus,
)
from stetho_api.config import Settings, get_settings
from stetho_api.db import (
    EnrollmentRepository,
    MachineProfile,
    MachineRepository,
    get_db,
)
from stetho_api.exceptions import StethoAPIException
from stetho_api.schemas.machine import (
    EnrollmentResponse,
    MachineCreateRequest,
    MachineListResponse,
    MachineResponse,
)

router = APIRouter(tags=["Machine Profiles & Enrollment"])


def get_repos(settings: Settings = Depends(get_settings)):
    db = get_db(settings.STORAGE_LOCAL_DIR / "stetho.db")
    return {
        "machine_repo": MachineRepository(db),
        "enrollment_repo": EnrollmentRepository(db),
        "ingestion": AudioIngestionService(settings.STORAGE_LOCAL_DIR),
        "quality_gate": AudioQualityGate(),
    }


@router.post(
    "/machines",
    response_model=MachineResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create machine profile",
    description="Registers a new physical machine for acoustic anomaly monitoring.",
)
async def create_machine(
    request: MachineCreateRequest,
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
    )
    return MachineResponse(**profile.to_dict())


@router.get(
    "/machines",
    response_model=MachineListResponse,
    summary="List machine profiles",
    description="Returns all registered machines and their enrollment/calibration status.",
)
async def list_machines(repos=Depends(get_repos)) -> MachineListResponse:
    repo: MachineRepository = repos["machine_repo"]
    profiles = repo.list_all()
    return MachineListResponse(
        machines=[MachineResponse(**p.to_dict()) for p in profiles],
        total_count=len(profiles),
    )


@router.get(
    "/machines/{machine_id}",
    response_model=MachineResponse,
    summary="Get machine profile",
    description="Retrieves a machine profile, including enrolled recording counts and model policy.",
)
async def get_machine(machine_id: str, repos=Depends(get_repos)) -> MachineResponse:
    repo: MachineRepository = repos["machine_repo"]
    profile = repo.get_or_raise(machine_id)
    return MachineResponse(**profile.to_dict())


@router.post(
    "/machines/{machine_id}/enrollments",
    response_model=EnrollmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Enroll normal machine recording",
    description="Registers an explicitly confirmed normal baseline audio recording for machine calibration.",
)
async def enroll_normal_audio(
    machine_id: str,
    file: UploadFile = File(..., description="Normal reference WAV audio recording"),
    confirmed_normal: bool = Form(True, description="Must be explicitly confirmed normal by inspector"),
    repos=Depends(get_repos),
) -> EnrollmentResponse:
    machine_repo: MachineRepository = repos["machine_repo"]
    enrollment_repo: EnrollmentRepository = repos["enrollment_repo"]
    ingestion: AudioIngestionService = repos["ingestion"]
    quality_gate: AudioQualityGate = repos["quality_gate"]

    # 1. Verify machine profile exists
    machine = machine_repo.get_or_raise(machine_id)

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

    # 4. Record enrollment in DB
    rec = enrollment_repo.record_enrollment(
        machine_id=machine_id,
        upload_id=meta.upload_id,
        stored_path=meta.stored_path,
        duration_sec=meta.duration_sec,
        sample_rate=meta.sample_rate,
        channels=meta.channels,
        quality_status=q_report.status.value,
        is_confirmed_normal=True,
    )

    # 5. Update machine enrollment count & status
    updated_machine = machine_repo.increment_enrollment(machine_id)

    # Calibration limitation notice
    notice = (
        "Calibration is provisional (N < 30 reference recordings). "
        "Acoustic threshold carries statistical variance and provisional decision flags."
        if updated_machine.enrolled_recording_count < 30
        else "Machine baseline is fully calibrated (N >= 30 normal reference recordings)."
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
        enrolled_count=updated_machine.enrolled_recording_count,
        enrollment_status=updated_machine.enrollment_status,
        notice=notice,
    )


@router.get(
    "/machines/{machine_id}/enrollments",
    summary="List machine enrollments",
    description="Returns previous normal reference enrollment recordings for a machine.",
)
async def list_enrollments(machine_id: str, repos=Depends(get_repos)):
    machine_repo: MachineRepository = repos["machine_repo"]
    enrollment_repo: EnrollmentRepository = repos["enrollment_repo"]
    machine_repo.get_or_raise(machine_id)
    records = enrollment_repo.list_by_machine(machine_id)
    return {
        "machine_id": machine_id,
        "enrollments": [r.to_dict() for r in records],
        "total_count": len(records),
    }


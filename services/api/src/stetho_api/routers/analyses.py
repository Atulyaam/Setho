"""FastAPI Router for Acoustic Analysis, Background Jobs, and Inspection Reports."""

import asyncio
import logging
from typing import Optional
import uuid
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile, status

from stetho_api.audio import AudioIngestionService
from stetho_api.auth.dependencies import get_current_user
from stetho_api.config import Settings, get_settings
from stetho_api.db import (
    AnalysisRecord,
    AnalysisRepository,
    JobRepository,
    MachineRepository,
    get_db,
    get_utc_now_iso,
)
from stetho_api.db.user_repo import User
from stetho_api.exceptions import StethoAPIException
from stetho_api.inference import (
    AcousticInferenceService,
    InferenceResult,
    ModelType,
    get_inference_service,
)
from stetho_api.schemas.analysis import (
    AnalysisListResponse,
    AnalysisResponse,
    InspectionReportResponse,
    JobResponse,
)
from stetho_api.schemas.machine import MachineResponse

logger = logging.getLogger("stetho.analysis_router")

router = APIRouter(tags=["Acoustic Analysis & Inspection"])


def get_dependencies(settings: Settings = Depends(get_settings)):
    db = get_db(settings.STORAGE_LOCAL_DIR / "stetho.db")
    return {
        "machine_repo": MachineRepository(db),
        "analysis_repo": AnalysisRepository(db),
        "job_repo": JobRepository(db),
        "ingestion": AudioIngestionService(settings.STORAGE_LOCAL_DIR),
        "inference": get_inference_service(),
    }


def generate_recommendation_and_summary(
    machine_name: str,
    decision: str,
    anomaly_score: Optional[float],
    threshold: Optional[float],
    is_provisional: bool,
) -> tuple[str, str]:
    """Produce conservative, evidence-based engineering recommendations."""
    if decision == "NORMAL_LIKE":
        summary = (
            f"Acoustic analysis of {machine_name} confirms normal operating behavior. "
            f"Reconstruction error ({anomaly_score:.4f}) is strictly within the calibrated baseline."
        )
        recommendation = (
            "No maintenance action required. Continue routine scheduled acoustic inspections."
        )
    elif decision == "UNUSUAL":
        summary = (
            f"Unusual acoustic deviation detected on {machine_name}. "
            f"Measured reconstruction error ({anomaly_score:.4f}) exceeds the calibrated normal threshold ({threshold:.4f})."
        )
        recommendation = (
            "Perform hands-on physical inspection of the machine. "
            "Inspect rotating bearings, mounting brackets, and fan blades for mechanical wear, looseness, or imbalance. "
            "Compare current sound profile with the baseline enrollment recording."
        )
    else:  # INCONCLUSIVE
        summary = (
            f"Acoustic inspection for {machine_name} returned an INCONCLUSIVE outcome. "
            "The recording exhibited borderline acoustic scoring, high ambient background noise, or provisional calibration limits."
        )
        recommendation = (
            "Re-record acoustic sample in a low-noise ambient environment. "
            "Ensure the acoustic probe or microphone is firmly coupled to the machine casing."
        )

    if is_provisional:
        summary += " [PROVISIONAL CALIBRATION NOTICE: Enrollment count N < 30]."

    return summary, recommendation


@router.post(
    "/machines/{machine_id}/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Synchronous machine acoustic analysis",
    description="Submits audio recording, validates audio quality, executes model inference, and returns calibrated three-zone anomaly decision.",
)
async def analyze_machine_audio(
    machine_id: str,
    file: UploadFile = File(..., description="Target machine audio recording (WAV)"),
    model_policy: Optional[str] = Form(
        None,
        description="Optional model override: 'stetho_msae' (edge champion) or 'dense_autoencoder' (server baseline)",
    ),
    current_user: User = Depends(get_current_user),
    deps=Depends(get_dependencies),
) -> AnalysisResponse:
    machine_repo: MachineRepository = deps["machine_repo"]
    analysis_repo: AnalysisRepository = deps["analysis_repo"]
    ingestion: AudioIngestionService = deps["ingestion"]
    inference: AcousticInferenceService = deps["inference"]

    # 1. Verify machine profile exists and user is authorized
    is_admin = current_user.role == "admin"
    machine = machine_repo.get_authorized(machine_id, user_id=current_user.user_id, is_admin=is_admin)
    selected_model = model_policy or machine.selected_model_policy

    # 2. Ingest audio
    meta, raw_audio = ingestion.ingest_audio(
        file_obj=file.file,
        filename=file.filename,
        content_type=file.content_type,
        machine_id=machine_id,
    )

    # 3. Execute inference pipeline
    result: InferenceResult = inference.analyze_audio(
        audio=raw_audio,
        sample_rate=meta.sample_rate,
        machine_id=machine_id,
        model_override=selected_model,
    )

    # 4. Persist analysis record
    rec = analysis_repo.create(
        machine_id=machine_id,
        model_name=result.model_name,
        model_version=result.model_version,
        analysis_status="completed",
        quality_status=result.quality_assessment.status.value,
        decision=result.decision,
        anomaly_score=result.anomaly_score,
        operating_threshold=result.operating_threshold,
        theta_lower=result.theta_lower,
        theta_upper=result.theta_upper,
        is_provisional=result.is_provisional,
        processing_duration_ms=result.processing_duration_ms,
        reasons=result.reasons,
        metrics=result.metrics,
    )

    return AnalysisResponse(**rec.to_dict())


def _execute_background_analysis_task(
    job_id: str,
    machine_id: str,
    raw_audio,
    sample_rate: int,
    selected_model: str,
    deps: dict,
):
    """Background worker function executing acoustic inference asynchronously."""
    analysis_repo: AnalysisRepository = deps["analysis_repo"]
    job_repo: JobRepository = deps["job_repo"]
    inference: AcousticInferenceService = deps["inference"]

    try:
        job_repo.update_status(job_id, status="processing", progress_pct=30)
        result: InferenceResult = inference.analyze_audio(
            audio=raw_audio,
            sample_rate=sample_rate,
            machine_id=machine_id,
            model_override=selected_model,
        )

        job_repo.update_status(job_id, status="processing", progress_pct=85)
        rec = analysis_repo.create(
            machine_id=machine_id,
            model_name=result.model_name,
            model_version=result.model_version,
            analysis_status="completed",
            quality_status=result.quality_assessment.status.value,
            decision=result.decision,
            anomaly_score=result.anomaly_score,
            operating_threshold=result.operating_threshold,
            theta_lower=result.theta_lower,
            theta_upper=result.theta_upper,
            is_provisional=result.is_provisional,
            processing_duration_ms=result.processing_duration_ms,
            reasons=result.reasons,
            metrics=result.metrics,
        )

        job_repo.update_status(
            job_id,
            status="completed",
            progress_pct=100,
            analysis_id=rec.analysis_id,
        )
        logger.info(f"Background analysis job {job_id} completed -> analysis {rec.analysis_id}")
    except Exception as e:
        logger.error(f"Background analysis job {job_id} failed: {e}", exc_info=True)
        job_repo.update_status(
            job_id,
            status="failed",
            error_message=str(e),
        )


@router.post(
    "/machines/{machine_id}/analysis-jobs",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit asynchronous analysis job",
    description="Submits an audio recording for background analysis without keeping the HTTP connection open.",
)
async def create_analysis_job(
    machine_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio recording for analysis"),
    model_policy: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    deps=Depends(get_dependencies),
) -> JobResponse:
    machine_repo: MachineRepository = deps["machine_repo"]
    job_repo: JobRepository = deps["job_repo"]
    ingestion: AudioIngestionService = deps["ingestion"]

    is_admin = current_user.role == "admin"
    machine = machine_repo.get_authorized(machine_id, user_id=current_user.user_id, is_admin=is_admin)
    selected_model = model_policy or machine.selected_model_policy

    meta, raw_audio = ingestion.ingest_audio(
        file_obj=file.file,
        filename=file.filename,
        content_type=file.content_type,
        machine_id=machine_id,
    )

    job = job_repo.create_job(machine_id)

    # Dispatch to background task
    background_tasks.add_task(
        _execute_background_analysis_task,
        job_id=job.job_id,
        machine_id=machine_id,
        raw_audio=raw_audio,
        sample_rate=meta.sample_rate,
        selected_model=selected_model,
        deps=deps,
    )

    return JobResponse(**job.to_dict())


@router.get(
    "/analysis-jobs/{job_id}",
    response_model=JobResponse,
    summary="Get analysis job status",
    description="Polls the execution state and progress of an asynchronous analysis job.",
)
async def get_analysis_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    deps=Depends(get_dependencies),
) -> JobResponse:
    job_repo: JobRepository = deps["job_repo"]
    machine_repo: MachineRepository = deps["machine_repo"]

    job = job_repo.get_or_raise(job_id)
    is_admin = current_user.role == "admin"
    machine_repo.get_authorized(job.machine_id, user_id=current_user.user_id, is_admin=is_admin)

    return JobResponse(**job.to_dict())


@router.get(
    "/analyses/recent",
    summary="List recent inspection analyses",
    description="Retrieves recent acoustic analyses visible to the authenticated user.",
)
async def list_recent_analyses(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    deps=Depends(get_dependencies),
):
    analysis_repo: AnalysisRepository = deps["analysis_repo"]
    is_admin = current_user.role == "admin"
    records = analysis_repo.list_recent(user_id=current_user.user_id, is_admin=is_admin, limit=limit)
    return {
        "analyses": [r.to_dict() for r in records],
        "total_count": len(records),
    }


@router.get(
    "/analyses/{analysis_id}",
    response_model=AnalysisResponse,
    summary="Get analysis result",
    description="Retrieves a completed acoustic inspection record by ID.",
)
async def get_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    deps=Depends(get_dependencies),
) -> AnalysisResponse:
    analysis_repo: AnalysisRepository = deps["analysis_repo"]
    machine_repo: MachineRepository = deps["machine_repo"]

    rec = analysis_repo.get_or_raise(analysis_id)
    is_admin = current_user.role == "admin"
    machine_repo.get_authorized(rec.machine_id, user_id=current_user.user_id, is_admin=is_admin)

    return AnalysisResponse(**rec.to_dict())


@router.get(
    "/analyses/{analysis_id}/report",
    response_model=InspectionReportResponse,
    summary="Get structured inspection report",
    description="Generates an evidence-based acoustic inspection report with engineering recommendations.",
)
async def get_inspection_report(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    deps=Depends(get_dependencies),
) -> InspectionReportResponse:
    analysis_repo: AnalysisRepository = deps["analysis_repo"]
    machine_repo: MachineRepository = deps["machine_repo"]

    analysis = analysis_repo.get_or_raise(analysis_id)
    is_admin = current_user.role == "admin"
    machine = machine_repo.get_authorized(analysis.machine_id, user_id=current_user.user_id, is_admin=is_admin)

    summary, recommendation = generate_recommendation_and_summary(
        machine_name=machine.name,
        decision=analysis.decision,
        anomaly_score=analysis.anomaly_score or 0.0,
        threshold=analysis.operating_threshold or 0.0,
        is_provisional=analysis.is_provisional,
    )

    disclaimer = (
        "Stetho Acoustic Assistant Disclaimer: Stetho evaluates acoustic deviations against an "
        "individual machine's learned baseline. It does not diagnose specific failure modes, "
        "estimate exact time to failure, or replace certified mechanical inspection."
    )

    return InspectionReportResponse(
        report_id=str(uuid.uuid4()),
        generated_at=get_utc_now_iso(),
        machine=MachineResponse(**machine.to_dict()),
        analysis=AnalysisResponse(**analysis.to_dict()),
        executive_summary=summary,
        recommended_action=recommendation,
        technical_findings=analysis.reasons,
        engineering_disclaimer=disclaimer,
    )


@router.get(
    "/machines/{machine_id}/analyses",
    response_model=AnalysisListResponse,
    summary="List machine analysis history",
    description="Returns previous acoustic analysis inspections for a machine.",
)
async def list_machine_analyses(
    machine_id: str,
    current_user: User = Depends(get_current_user),
    deps=Depends(get_dependencies),
) -> AnalysisListResponse:
    machine_repo: MachineRepository = deps["machine_repo"]
    analysis_repo: AnalysisRepository = deps["analysis_repo"]

    is_admin = current_user.role == "admin"
    machine_repo.get_authorized(machine_id, user_id=current_user.user_id, is_admin=is_admin)
    records = analysis_repo.list_by_machine(machine_id)

    return AnalysisListResponse(
        machine_id=machine_id,
        analyses=[AnalysisResponse(**r.to_dict()) for r in records],
        total_count=len(records),
    )

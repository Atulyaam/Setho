"""Repositories for Enrollments, Analysis Records, and Asynchronous Jobs."""

from dataclasses import dataclass
import json
import logging
from typing import Any, Dict, List, Optional
import uuid

from stetho_api.db.database import DatabaseManager, get_utc_now_iso
from stetho_api.exceptions import ResourceNotFoundException

logger = logging.getLogger("stetho.analysis_repo")


@dataclass
class EnrollmentRecord:
    enrollment_id: str
    machine_id: str
    upload_id: str
    stored_path: str
    duration_sec: float
    sample_rate: int
    channels: int
    quality_status: str
    is_confirmed_normal: bool
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enrollment_id": self.enrollment_id,
            "machine_id": self.machine_id,
            "upload_id": self.upload_id,
            "duration_sec": self.duration_sec,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "quality_status": self.quality_status,
            "is_confirmed_normal": self.is_confirmed_normal,
            "created_at": self.created_at,
        }


@dataclass
class AnalysisRecord:
    analysis_id: str
    machine_id: str
    model_name: str
    model_version: str
    analysis_status: str
    quality_status: str
    decision: str  # "NORMAL_LIKE", "UNUSUAL", "INCONCLUSIVE"
    anomaly_score: Optional[float]
    operating_threshold: Optional[float]
    theta_lower: Optional[float]
    theta_upper: Optional[float]
    is_provisional: bool
    processing_duration_ms: float
    reasons: List[str]
    metrics: Dict[str, Any]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "machine_id": self.machine_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "analysis_status": self.analysis_status,
            "quality_status": self.quality_status,
            "decision": self.decision,
            "anomaly_score": round(self.anomaly_score, 6) if self.anomaly_score is not None else None,
            "operating_threshold": round(self.operating_threshold, 6) if self.operating_threshold is not None else None,
            "theta_lower": round(self.theta_lower, 6) if self.theta_lower is not None else None,
            "theta_upper": round(self.theta_upper, 6) if self.theta_upper is not None else None,
            "is_provisional": self.is_provisional,
            "processing_duration_ms": round(self.processing_duration_ms, 2),
            "reasons": self.reasons,
            "metrics": self.metrics,
            "created_at": self.created_at,
        }


@dataclass
class JobRecord:
    job_id: str
    machine_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress_pct: int
    analysis_id: Optional[str]
    error_message: Optional[str]
    retry_count: int
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "machine_id": self.machine_id,
            "status": self.status,
            "progress_pct": self.progress_pct,
            "analysis_id": self.analysis_id,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class EnrollmentRepository:
    """Data access repository for audio enrollments."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def record_enrollment(
        self,
        machine_id: str,
        upload_id: str,
        stored_path: str,
        duration_sec: float,
        sample_rate: int,
        channels: int,
        quality_status: str,
        is_confirmed_normal: bool = True,
    ) -> EnrollmentRecord:
        enrollment_id = str(uuid.uuid4())
        now = get_utc_now_iso()
        with self.db.transaction() as cursor:
            cursor.execute("""
            INSERT INTO enrollments (
                enrollment_id, machine_id, upload_id, stored_path, duration_sec,
                sample_rate, channels, quality_status, is_confirmed_normal, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                enrollment_id,
                machine_id,
                upload_id,
                stored_path,
                duration_sec,
                sample_rate,
                channels,
                quality_status,
                1 if is_confirmed_normal else 0,
                now,
            ))

        return EnrollmentRecord(
            enrollment_id=enrollment_id,
            machine_id=machine_id,
            upload_id=upload_id,
            stored_path=stored_path,
            duration_sec=duration_sec,
            sample_rate=sample_rate,
            channels=channels,
            quality_status=quality_status,
            is_confirmed_normal=is_confirmed_normal,
            created_at=now,
        )

    def list_by_machine(self, machine_id: str) -> List[EnrollmentRecord]:
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM enrollments WHERE machine_id = ? ORDER BY created_at DESC;",
            (machine_id,),
        )
        rows = cursor.fetchall()
        return [
            EnrollmentRecord(
                enrollment_id=r["enrollment_id"],
                machine_id=r["machine_id"],
                upload_id=r["upload_id"],
                stored_path=r["stored_path"],
                duration_sec=r["duration_sec"],
                sample_rate=r["sample_rate"],
                channels=r["channels"],
                quality_status=r["quality_status"],
                is_confirmed_normal=bool(r["is_confirmed_normal"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]


class AnalysisRepository:
    """Data access repository for machine inspection analysis records."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def create(
        self,
        machine_id: str,
        model_name: str,
        model_version: str,
        analysis_status: str,
        quality_status: str,
        decision: str,
        anomaly_score: Optional[float],
        operating_threshold: Optional[float],
        theta_lower: Optional[float],
        theta_upper: Optional[float],
        is_provisional: bool,
        processing_duration_ms: float,
        reasons: List[str],
        metrics: Dict[str, Any],
    ) -> AnalysisRecord:
        analysis_id = str(uuid.uuid4())
        now = get_utc_now_iso()
        reasons_json = json.dumps(reasons)
        metrics_json = json.dumps(metrics)

        with self.db.transaction() as cursor:
            cursor.execute("""
            INSERT INTO analyses (
                analysis_id, machine_id, model_name, model_version, analysis_status,
                quality_status, decision, anomaly_score, operating_threshold,
                theta_lower, theta_upper, is_provisional, processing_duration_ms,
                reasons, metrics, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                analysis_id,
                machine_id,
                model_name,
                model_version,
                analysis_status,
                quality_status,
                decision,
                anomaly_score,
                operating_threshold,
                theta_lower,
                theta_upper,
                1 if is_provisional else 0,
                processing_duration_ms,
                reasons_json,
                metrics_json,
                now,
            ))

        return AnalysisRecord(
            analysis_id=analysis_id,
            machine_id=machine_id,
            model_name=model_name,
            model_version=model_version,
            analysis_status=analysis_status,
            quality_status=quality_status,
            decision=decision,
            anomaly_score=anomaly_score,
            operating_threshold=operating_threshold,
            theta_lower=theta_lower,
            theta_upper=theta_upper,
            is_provisional=is_provisional,
            processing_duration_ms=processing_duration_ms,
            reasons=reasons,
            metrics=metrics,
            created_at=now,
        )

    def get_by_id(self, analysis_id: str) -> Optional[AnalysisRecord]:
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analyses WHERE analysis_id = ?;", (analysis_id,))
        r = cursor.fetchone()
        if not r:
            return None

        return AnalysisRecord(
            analysis_id=r["analysis_id"],
            machine_id=r["machine_id"],
            model_name=r["model_name"],
            model_version=r["model_version"],
            analysis_status=r["analysis_status"],
            quality_status=r["quality_status"],
            decision=r["decision"],
            anomaly_score=r["anomaly_score"],
            operating_threshold=r["operating_threshold"],
            theta_lower=r["theta_lower"],
            theta_upper=r["theta_upper"],
            is_provisional=bool(r["is_provisional"]),
            processing_duration_ms=r["processing_duration_ms"],
            reasons=json.loads(r["reasons"] or "[]"),
            metrics=json.loads(r["metrics"] or "{}"),
            created_at=r["created_at"],
        )

    def get_or_raise(self, analysis_id: str) -> AnalysisRecord:
        record = self.get_by_id(analysis_id)
        if not record:
            raise ResourceNotFoundException("Analysis", analysis_id)
        return record

    def list_by_machine(self, machine_id: str) -> List[AnalysisRecord]:
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM analyses WHERE machine_id = ? ORDER BY created_at DESC;",
            (machine_id,),
        )
        rows = cursor.fetchall()
        return [
            AnalysisRecord(
                analysis_id=r["analysis_id"],
                machine_id=r["machine_id"],
                model_name=r["model_name"],
                model_version=r["model_version"],
                analysis_status=r["analysis_status"],
                quality_status=r["quality_status"],
                decision=r["decision"],
                anomaly_score=r["anomaly_score"],
                operating_threshold=r["operating_threshold"],
                theta_lower=r["theta_lower"],
                theta_upper=r["theta_upper"],
                is_provisional=bool(r["is_provisional"]),
                processing_duration_ms=r["processing_duration_ms"],
                reasons=json.loads(r["reasons"] or "[]"),
                metrics=json.loads(r["metrics"] or "{}"),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def list_recent(self, user_id: str, is_admin: bool = False, limit: int = 20) -> List[AnalysisRecord]:
        conn = self.db.get_connection()
        cursor = conn.cursor()
        if is_admin:
            cursor.execute(
                "SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?;",
                (limit,),
            )
        else:
            cursor.execute(
                """
                SELECT a.* FROM analyses a
                JOIN machines m ON a.machine_id = m.machine_id
                WHERE m.owner_id = ?
                ORDER BY a.created_at DESC LIMIT ?;
                """,
                (user_id, limit),
            )
        rows = cursor.fetchall()
        return [
            AnalysisRecord(
                analysis_id=r["analysis_id"],
                machine_id=r["machine_id"],
                model_name=r["model_name"],
                model_version=r["model_version"],
                analysis_status=r["analysis_status"],
                quality_status=r["quality_status"],
                decision=r["decision"],
                anomaly_score=r["anomaly_score"],
                operating_threshold=r["operating_threshold"],
                theta_lower=r["theta_lower"],
                theta_upper=r["theta_upper"],
                is_provisional=bool(r["is_provisional"]),
                processing_duration_ms=r["processing_duration_ms"],
                reasons=json.loads(r["reasons"] or "[]"),
                metrics=json.loads(r["metrics"] or "{}"),
                created_at=r["created_at"],
            )
            for r in rows
        ]


class JobRepository:
    """Data access repository for asynchronous analysis jobs."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def create_job(self, machine_id: str) -> JobRecord:
        job_id = str(uuid.uuid4())
        now = get_utc_now_iso()
        with self.db.transaction() as cursor:
            cursor.execute("""
            INSERT INTO analysis_jobs (
                job_id, machine_id, status, progress_pct, retry_count, created_at, updated_at
            ) VALUES (?, ?, 'pending', 0, 0, ?, ?);
            """, (job_id, machine_id, now, now))

        return JobRecord(
            job_id=job_id,
            machine_id=machine_id,
            status="pending",
            progress_pct=0,
            analysis_id=None,
            error_message=None,
            retry_count=0,
            created_at=now,
            updated_at=now,
        )

    def get_by_id(self, job_id: str) -> Optional[JobRecord]:
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analysis_jobs WHERE job_id = ?;", (job_id,))
        r = cursor.fetchone()
        if not r:
            return None

        return JobRecord(
            job_id=r["job_id"],
            machine_id=r["machine_id"],
            status=r["status"],
            progress_pct=r["progress_pct"],
            analysis_id=r["analysis_id"],
            error_message=r["error_message"],
            retry_count=r["retry_count"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )

    def get_or_raise(self, job_id: str) -> JobRecord:
        job = self.get_by_id(job_id)
        if not job:
            raise ResourceNotFoundException("AnalysisJob", job_id)
        return job

    def update_status(
        self,
        job_id: str,
        status: str,
        progress_pct: Optional[int] = None,
        analysis_id: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> JobRecord:
        now = get_utc_now_iso()
        current = self.get_or_raise(job_id)
        progress = progress_pct if progress_pct is not None else current.progress_pct

        with self.db.transaction() as cursor:
            cursor.execute("""
            UPDATE analysis_jobs
            SET status = ?,
                progress_pct = ?,
                analysis_id = COALESCE(?, analysis_id),
                error_message = COALESCE(?, error_message),
                updated_at = ?
            WHERE job_id = ?;
            """, (status, progress, analysis_id, error_message, now, job_id))

        return self.get_or_raise(job_id)


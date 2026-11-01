"""Inference Repository for Stetho API (Sprint 7).

Persists standardized inference results, latency metrics, anomaly decisions,
and diagnostic metadata for long-term historical tracking and auditing.
"""

from dataclasses import dataclass, field
import json
import logging
from typing import Any, Dict, List, Optional
import uuid

from stetho_api.db.database import DatabaseManager, get_utc_now_iso
from stetho_api.exceptions import ResourceNotFoundException

logger = logging.getLogger("stetho.inference_repo")


@dataclass
class InferenceRecord:
    """Persistent acoustic inference execution record."""
    result_id: str
    asset_id: str
    model_name: str
    model_version: str
    anomaly_score: Optional[float]
    operating_threshold: Optional[float]
    anomaly_decision: str  # "NORMAL_LIKE", "UNUSUAL", "INCONCLUSIVE"
    processing_latency_ms: float
    calibration_status: str
    quality_status: str
    is_provisional: bool = True
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    audio_storage_path: Optional[str] = None
    audio_hash: Optional[str] = None
    operator_id: Optional[str] = "usr_admin"
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_id": self.result_id,
            "asset_id": self.asset_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "anomaly_score": round(self.anomaly_score, 6) if self.anomaly_score is not None else None,
            "operating_threshold": round(self.operating_threshold, 6) if self.operating_threshold is not None else None,
            "anomaly_decision": self.anomaly_decision,
            "processing_latency_ms": round(self.processing_latency_ms, 2),
            "calibration_status": self.calibration_status,
            "quality_status": self.quality_status,
            "is_provisional": self.is_provisional,
            "reasons": self.reasons,
            "metrics": self.metrics,
            "audio_storage_path": self.audio_storage_path,
            "created_at": self.created_at,
        }


class InferenceRepository:
    """Repository handling persistence and retrieval of inference results."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def save(self, record: InferenceRecord) -> InferenceRecord:
        """Save an inference result to the database."""
        now = get_utc_now_iso()
        if not record.result_id:
            record.result_id = f"inf_{uuid.uuid4().hex[:12]}"
        if not record.created_at:
            record.created_at = now

        reasons_json = json.dumps(record.reasons)
        metrics_json = json.dumps(record.metrics)

        with self.db.transaction() as cur:
            cur.execute("""
            INSERT INTO inference_results (
                result_id, asset_id, model_name, model_version,
                anomaly_score, operating_threshold, anomaly_decision,
                processing_latency_ms, calibration_status, quality_status,
                is_provisional, reasons_json, metrics_json, audio_storage_path,
                audio_hash, operator_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                record.result_id, record.asset_id, record.model_name,
                record.model_version, record.anomaly_score,
                record.operating_threshold, record.anomaly_decision,
                record.processing_latency_ms, record.calibration_status,
                record.quality_status, 1 if record.is_provisional else 0,
                reasons_json, metrics_json, record.audio_storage_path,
                record.audio_hash, record.operator_id, record.created_at
            ))

            # Backward compatibility with analyses table
            cur.execute("""
            INSERT OR IGNORE INTO analyses (
                analysis_id, machine_id, model_name, model_version,
                analysis_status, quality_status, decision, anomaly_score,
                operating_threshold, theta_lower, theta_upper, is_provisional,
                processing_duration_ms, reasons, metrics, created_at
            ) VALUES (?, ?, ?, ?, 'COMPLETED', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                record.result_id, record.asset_id, record.model_name,
                record.model_version, record.quality_status,
                record.anomaly_decision, record.anomaly_score,
                record.operating_threshold,
                record.operating_threshold * 0.8 if record.operating_threshold else None,
                record.operating_threshold * 1.2 if record.operating_threshold else None,
                1 if record.is_provisional else 0, record.processing_latency_ms,
                reasons_json, metrics_json, record.created_at
            ))

        logger.info(
            f"Saved inference result '{record.result_id}' for asset '{record.asset_id}': "
            f"decision={record.anomaly_decision}, score={record.anomaly_score}, "
            f"latency={record.processing_latency_ms:.1f}ms"
        )
        return record

    def get(self, result_id: str) -> Optional[InferenceRecord]:
        """Retrieve an inference result by ID."""
        conn = self.db.get_connection()
        row = conn.execute("SELECT * FROM inference_results WHERE result_id = ?;", (result_id,)).fetchone()
        if not row:
            # Check legacy analyses table
            a_row = conn.execute("SELECT * FROM analyses WHERE analysis_id = ?;", (result_id,)).fetchone()
            if a_row:
                try:
                    reasons = json.loads(a_row["reasons"]) if a_row["reasons"] else []
                except Exception:
                    reasons = []
                try:
                    metrics = json.loads(a_row["metrics"]) if a_row["metrics"] else {}
                except Exception:
                    metrics = {}
                return InferenceRecord(
                    result_id=a_row["analysis_id"],
                    asset_id=a_row["machine_id"],
                    model_name=a_row["model_name"],
                    model_version=a_row["model_version"],
                    anomaly_score=a_row["anomaly_score"],
                    operating_threshold=a_row["operating_threshold"],
                    anomaly_decision=a_row["decision"],
                    processing_latency_ms=a_row["processing_duration_ms"] or 0.0,
                    calibration_status="calibrated" if not a_row["is_provisional"] else "provisional",
                    quality_status=a_row["quality_status"],
                    is_provisional=bool(a_row["is_provisional"]),
                    reasons=reasons,
                    metrics=metrics,
                    created_at=a_row["created_at"],
                )
            return None

        try:
            reasons = json.loads(row["reasons_json"]) if row["reasons_json"] else []
        except Exception:
            reasons = []
        try:
            metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else {}
        except Exception:
            metrics = {}

        return InferenceRecord(
            result_id=row["result_id"],
            asset_id=row["asset_id"],
            model_name=row["model_name"],
            model_version=row["model_version"],
            anomaly_score=row["anomaly_score"],
            operating_threshold=row["operating_threshold"],
            anomaly_decision=row["anomaly_decision"],
            processing_latency_ms=row["processing_latency_ms"],
            calibration_status=row["calibration_status"],
            quality_status=row["quality_status"],
            is_provisional=bool(row["is_provisional"]),
            reasons=reasons,
            metrics=metrics,
            audio_storage_path=row["audio_storage_path"],
            audio_hash=row["audio_hash"],
            operator_id=row["operator_id"],
            created_at=row["created_at"],
        )

    def get_or_raise(self, result_id: str) -> InferenceRecord:
        record = self.get(result_id)
        if not record:
            raise ResourceNotFoundException("InferenceResult", result_id)
        return record

    def list_by_asset(self, asset_id: str, limit: int = 50) -> List[InferenceRecord]:
        """List historical inference records for an asset."""
        conn = self.db.get_connection()
        rows = conn.execute("""
        SELECT * FROM inference_results
        WHERE asset_id = ?
        ORDER BY created_at DESC LIMIT ?;
        """, (asset_id, limit)).fetchall()

        results = []
        for r in rows:
            try:
                reasons = json.loads(r["reasons_json"]) if r["reasons_json"] else []
            except Exception:
                reasons = []
            try:
                metrics = json.loads(r["metrics_json"]) if r["metrics_json"] else {}
            except Exception:
                metrics = {}
            results.append(
                InferenceRecord(
                    result_id=r["result_id"],
                    asset_id=r["asset_id"],
                    model_name=r["model_name"],
                    model_version=r["model_version"],
                    anomaly_score=r["anomaly_score"],
                    operating_threshold=r["operating_threshold"],
                    anomaly_decision=r["anomaly_decision"],
                    processing_latency_ms=r["processing_latency_ms"],
                    calibration_status=r["calibration_status"],
                    quality_status=r["quality_status"],
                    is_provisional=bool(r["is_provisional"]),
                    reasons=reasons,
                    metrics=metrics,
                    audio_storage_path=r["audio_storage_path"],
                    audio_hash=r["audio_hash"],
                    operator_id=r["operator_id"],
                    created_at=r["created_at"],
                )
            )
        return results

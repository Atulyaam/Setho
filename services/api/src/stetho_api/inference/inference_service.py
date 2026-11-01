"""Inference Service for Stetho API (Sprint 7).

Stateless, decoupled domain service orchestrating the complete acoustic inference pipeline:
Audio Preprocessing -> Audio Quality Gate -> Model Routing & Category Verification ->
Model Registry -> Reconstruction Anomaly Scoring -> Three-Zone Calibration -> Result Persistence.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from stetho_api.audio.quality_gate import AudioQualityGate, QualityAssessment, QualityStatus
from stetho_api.db.asset_repo import AssetRepository
from stetho_api.db.calibration_repo import CalibrationProfile, CalibrationRepository
from stetho_api.db.database import DatabaseManager, get_db, get_utc_now_iso
from stetho_api.db.inference_repo import InferenceRecord, InferenceRepository
from stetho_api.exceptions import StethoAPIException, ValidationException
from stetho_api.inference.model_registry import (
    LoadedModel,
    ModelRegistry,
    ModelType,
    get_model_registry,
)
from stetho_api.inference.model_router import ModelRouter
from stetho_audio.channel_policy import ChannelHandler, ChannelPolicy
from stetho_audio.preprocessing import AudioPreprocessor, PreprocessConfig

logger = logging.getLogger("stetho.inference_service")


class DecisionZone(str):
    NORMAL_LIKE = "NORMAL_LIKE"
    UNUSUAL = "UNUSUAL"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class InferenceResult:
    """Structured, interpretable acoustic inspection inference result."""
    asset_id: str
    model_name: str
    model_version: str
    decision: str  # "NORMAL_LIKE", "UNUSUAL", "INCONCLUSIVE"
    anomaly_score: Optional[float]
    operating_threshold: Optional[float]
    theta_lower: Optional[float]
    theta_upper: Optional[float]
    is_provisional: bool
    quality_assessment: QualityAssessment
    reasons: List[str]
    processing_duration_ms: float
    calibration_status: str = "calibrated"
    metrics: Dict[str, Any] = field(default_factory=dict)
    result_id: Optional[str] = None

    @property
    def machine_id(self) -> str:
        """Backward-compatible alias for machine_id."""
        return self.asset_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_id": self.result_id,
            "asset_id": self.asset_id,
            "machine_id": self.asset_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "decision": self.decision,
            "anomaly_decision": self.decision,
            "anomaly_score": round(self.anomaly_score, 6) if self.anomaly_score is not None else None,
            "operating_threshold": round(self.operating_threshold, 6) if self.operating_threshold is not None else None,
            "threshold_used": round(self.operating_threshold, 6) if self.operating_threshold is not None else None,
            "theta_lower": round(self.theta_lower, 6) if self.theta_lower is not None else None,
            "theta_upper": round(self.theta_upper, 6) if self.theta_upper is not None else None,
            "is_provisional": self.is_provisional,
            "calibration_status": self.calibration_status,
            "quality_status": self.quality_assessment.status.value,
            "quality_reasons": self.quality_assessment.reasons,
            "reasons": self.reasons,
            "processing_duration_ms": round(self.processing_duration_ms, 2),
            "processing_latency_ms": round(self.processing_duration_ms, 2),
            "metrics": self.metrics,
        }


class AcousticInferenceService:
    """Orchestrates acoustic validation, category routing, model scoring, and decision analysis."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        quality_gate: Optional[AudioQualityGate] = None,
        db: Optional[DatabaseManager] = None,
    ):
        self.registry = registry or get_model_registry()
        self.quality_gate = quality_gate or AudioQualityGate()
        self.db = db
        self.asset_repo = AssetRepository(db) if db else None
        self.calibration_repo = CalibrationRepository(db) if db else None
        self.inference_repo = InferenceRepository(db) if db else None

        self.preprocessor = AudioPreprocessor(
            PreprocessConfig(
                target_sample_rate=16000,
                channel_policy=ChannelPolicy.CHANNEL_0,
                normalization="peak",
                validate_input=False,
            )
        )

    def analyze_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        machine_id: str,
        model_override: Optional[Union[ModelType, str]] = None,
        allow_cross_category: bool = False,
        operator_id: str = "usr_admin",
        audio_storage_path: Optional[str] = None,
        audio_hash: Optional[str] = None,
    ) -> InferenceResult:
        """Execute full inference pipeline on raw decoded audio waveform."""
        t0 = time.perf_counter()
        asset_id = machine_id

        # 1. Resolve Asset and Category Routing
        target_model_type: Optional[ModelType] = None
        target_version: str = "1.0.0"
        cal_status = "provisional"

        if self.asset_repo:
            asset = self.asset_repo.get(asset_id)
            if asset:
                cal_status = asset.calibration_status
                target_model_type, target_version = ModelRouter.resolve_model(
                    asset=asset,
                    profile_override=str(model_override) if model_override else None,
                    allow_cross_category=allow_cross_category,
                )

        if target_model_type is None:
            if model_override:
                target_model_type = (
                    model_override
                    if isinstance(model_override, ModelType)
                    else ModelType(str(model_override).lower())
                )
            else:
                target_model_type = ModelType.CASCADE_ENSEMBLE

        # 2. Multi-channel to mono conversion (Channel 0 default)
        mono_raw = ChannelHandler.apply(audio, policy=ChannelPolicy.CHANNEL_0)

        # 3. Audio Quality Gate evaluation
        quality_report = self.quality_gate.assess_waveform(mono_raw, sample_rate=sample_rate)

        # Fatal audio defects: Return immediately without running model
        if quality_report.status == QualityStatus.REJECTED:
            duration_ms = (time.perf_counter() - t0) * 1000
            logger.warning(
                f"Audio rejected by quality gate for asset={asset_id}: {'; '.join(quality_report.reasons)}"
            )
            return InferenceResult(
                asset_id=asset_id,
                model_name=target_model_type.value,
                model_version=target_version,
                decision=DecisionZone.INCONCLUSIVE,
                anomaly_score=None,
                operating_threshold=None,
                theta_lower=None,
                theta_upper=None,
                is_provisional=True,
                calibration_status=cal_status,
                quality_assessment=quality_report,
                reasons=[
                    "Recording rejected by Audio Quality Gate. Cannot perform acoustic anomaly analysis.",
                    *quality_report.reasons,
                ],
                processing_duration_ms=duration_ms,
                metrics=quality_report.metrics,
            )

        if quality_report.status == QualityStatus.INCONCLUSIVE:
            duration_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                f"Audio flagged inconclusive by quality gate for asset={asset_id}: {'; '.join(quality_report.reasons)}"
            )
            return InferenceResult(
                asset_id=asset_id,
                model_name=target_model_type.value,
                model_version=target_version,
                decision=DecisionZone.INCONCLUSIVE,
                anomaly_score=None,
                operating_threshold=None,
                theta_lower=None,
                theta_upper=None,
                is_provisional=True,
                calibration_status=cal_status,
                quality_assessment=quality_report,
                reasons=[
                    "Recording quality is inconclusive (excessive ambient noise or uncharacteristic acoustic environment).",
                    *quality_report.reasons,
                ],
                processing_duration_ms=duration_ms,
                metrics=quality_report.metrics,
            )

        # 4. Preprocess audio to standard 16 kHz peak-normalized array
        mono_processed, _ = self.preprocessor.process_array(mono_raw, sample_rate=sample_rate)

        # 5. Load model from registry
        spec_dir = self.registry.artifacts_root / target_model_type.value / asset_id
        effective_id = asset_id if spec_dir.exists() else "id_00"

        loaded_model = self.registry.get_model(
            machine_id=effective_id,
            model_type=target_model_type,
            version=target_version,
        )

        # 6. Execute model scoring
        raw_score, diag = loaded_model.score_waveform(mono_processed, sample_rate=16000)

        # 7. Apply Three-Zone Decision Analysis
        active_cal = None
        if self.calibration_repo:
            try:
                active_cal = self.calibration_repo.get_active_profile(asset_id)
            except Exception as e:
                logger.warning(f"Could not load active calibration profile for {asset_id}: {e}")

        if active_cal:
            theta_upper = float(active_cal.operating_threshold or active_cal.theta_upper)
            theta_lower = float(active_cal.theta_lower)
            is_provisional = bool(active_cal.enrolled_count < 5)
            cal_status = "provisional" if is_provisional else "calibrated"
        else:
            cal = loaded_model.calibration
            theta_upper = float(cal.get("operating_threshold", getattr(loaded_model.detector, "threshold", 0.35208) or 0.35208))
            theta_lower = float(cal.get("theta_lower", theta_upper * 0.85))
            is_provisional = bool(cal.get("is_provisional", cal_status != "calibrated"))

        reasons: List[str] = []
        if raw_score <= theta_lower:
            decision = DecisionZone.NORMAL_LIKE
            reasons.append(
                f"Acoustic reconstruction error ({raw_score:.5f}) is strictly within the normal reference baseline (<= {theta_lower:.5f})."
            )
        elif raw_score >= theta_upper:
            decision = DecisionZone.UNUSUAL
            reasons.append(
                f"Acoustic reconstruction error ({raw_score:.5f}) exceeds abnormal decision boundary (>= {theta_upper:.5f}). Unusual acoustic deviation detected."
            )
        else:
            decision = DecisionZone.INCONCLUSIVE
            reasons.append(
                f"Acoustic reconstruction error ({raw_score:.5f}) lies in the calibrated transition zone [{theta_lower:.5f}, {theta_upper:.5f}]. Anomaly status is inconclusive."
            )

        if quality_report.status == QualityStatus.WARNING:
            reasons.extend(quality_report.reasons)

        if is_provisional:
            reasons.append(
                "Provisional calibration notice: Operating boundary carries statistical uncertainty (N < 10 normal baseline recordings)."
            )

        duration_ms = (time.perf_counter() - t0) * 1000

        metrics = {
            **quality_report.metrics,
            "raw_anomaly_score": round(raw_score, 6),
            "runtime_engine": loaded_model.metadata.runtime_engine,
            "model_parameters": loaded_model.metadata.total_parameters,
            "inference_device": loaded_model.metadata.device,
            "diagnostic_details": diag,
        }

        # 8. Persist result in database if repository is available
        result_id = None
        if self.inference_repo:
            try:
                rec = InferenceRecord(
                    result_id="",
                    asset_id=asset_id,
                    model_name=loaded_model.model_type.value,
                    model_version=target_version,
                    anomaly_score=raw_score,
                    operating_threshold=theta_upper,
                    anomaly_decision=decision,
                    processing_latency_ms=duration_ms,
                    calibration_status=cal_status,
                    quality_status=quality_report.status.value,
                    is_provisional=is_provisional,
                    reasons=reasons,
                    metrics=metrics,
                    audio_storage_path=audio_storage_path,
                    audio_hash=audio_hash,
                    operator_id=operator_id,
                    created_at=get_utc_now_iso(),
                )
                saved = self.inference_repo.save(rec)
                result_id = saved.result_id
            except Exception as e:
                logger.warning(f"Could not persist inference record: {e}")

        logger.info(
            f"Inference completed for {asset_id} [{loaded_model.model_type.value}]: "
            f"score={raw_score:.5f}, decision={decision} in {duration_ms:.2f}ms"
        )

        return InferenceResult(
            result_id=result_id,
            asset_id=asset_id,
            model_name=loaded_model.model_type.value,
            model_version=target_version,
            decision=decision,
            anomaly_score=raw_score,
            operating_threshold=theta_upper,
            theta_lower=theta_lower,
            theta_upper=theta_upper,
            is_provisional=is_provisional,
            calibration_status=cal_status,
            quality_assessment=quality_report,
            reasons=reasons,
            processing_duration_ms=duration_ms,
            metrics=metrics,
        )


# Singleton
_inference_service: Optional[AcousticInferenceService] = None


def get_inference_service(db: Optional[DatabaseManager] = None) -> AcousticInferenceService:
    global _inference_service
    if _inference_service is None or (_inference_service.db is None and db is not None):
        database = db or get_db()
        _inference_service = AcousticInferenceService(db=database)
    return _inference_service

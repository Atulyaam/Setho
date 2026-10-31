"""Inference Service for Stetho API.

Stateless, decoupled domain service orchestrating the complete acoustic inference pipeline:
Audio Preprocessing -> Audio Quality Gate -> Model Registry -> Reconstruction Anomaly Scoring
-> Three-Zone Machine Calibration -> Uncertainty Evaluation.
"""

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from stetho_api.audio.quality_gate import AudioQualityGate, QualityAssessment, QualityStatus
from stetho_api.db.database import get_utc_now_iso
from stetho_api.exceptions import StethoAPIException
from stetho_api.inference.model_registry import (
    LoadedModel,
    ModelRegistry,
    ModelType,
    get_model_registry,
)
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
    machine_id: str
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
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "machine_id": self.machine_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "decision": self.decision,
            "anomaly_score": round(self.anomaly_score, 6) if self.anomaly_score is not None else None,
            "operating_threshold": round(self.operating_threshold, 6) if self.operating_threshold is not None else None,
            "theta_lower": round(self.theta_lower, 6) if self.theta_lower is not None else None,
            "theta_upper": round(self.theta_upper, 6) if self.theta_upper is not None else None,
            "is_provisional": self.is_provisional,
            "quality_status": self.quality_assessment.status.value,
            "quality_reasons": self.quality_assessment.reasons,
            "reasons": self.reasons,
            "processing_duration_ms": round(self.processing_duration_ms, 2),
            "metrics": self.metrics,
        }


class AcousticInferenceService:
    """Orchestrates acoustic validation, model scoring, and calibrated decision analysis."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        quality_gate: Optional[AudioQualityGate] = None,
    ):
        self.registry = registry or get_model_registry()
        self.quality_gate = quality_gate or AudioQualityGate()
        self.preprocessor = AudioPreprocessor(
            PreprocessConfig(
                target_sample_rate=16000,
                channel_policy=ChannelPolicy.CHANNEL_0,
                normalization="peak",
                validate_input=False,  # Input already validated by AQG
            )
        )

    def analyze_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        machine_id: str,
        model_override: Optional[Union[ModelType, str]] = None,
    ) -> InferenceResult:
        """Execute full inference pipeline on raw decoded audio waveform."""
        t0 = time.perf_counter()

        # 1. Multi-channel to mono conversion via standard ChannelPolicy (Channel 0 default)
        mono_raw = ChannelHandler.apply(audio, policy=ChannelPolicy.CHANNEL_0)

        # 2. Audio Quality Gate evaluation
        quality_report = self.quality_gate.assess_waveform(mono_raw, sample_rate=sample_rate)

        # If audio is fatally flawed, stop immediately without running ML model
        if quality_report.status == QualityStatus.REJECTED:
            duration_ms = (time.perf_counter() - t0) * 1000
            logger.warning(
                f"Audio rejected by quality gate for machine={machine_id}: {'; '.join(quality_report.reasons)}"
            )
            return InferenceResult(
                machine_id=machine_id,
                model_name=str(model_override or "none"),
                model_version="1.0.0",
                decision=DecisionZone.INCONCLUSIVE,
                anomaly_score=None,
                operating_threshold=None,
                theta_lower=None,
                theta_upper=None,
                is_provisional=True,
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
                f"Audio flagged inconclusive by quality gate for machine={machine_id}: {'; '.join(quality_report.reasons)}"
            )
            return InferenceResult(
                machine_id=machine_id,
                model_name=str(model_override or "none"),
                model_version="1.0.0",
                decision=DecisionZone.INCONCLUSIVE,
                anomaly_score=None,
                operating_threshold=None,
                theta_lower=None,
                theta_upper=None,
                is_provisional=True,
                quality_assessment=quality_report,
                reasons=[
                    "Recording quality is inconclusive (excessive ambient noise or uncharacteristic acoustic environment).",
                    *quality_report.reasons,
                ],
                processing_duration_ms=duration_ms,
                metrics=quality_report.metrics,
            )

        # 3. Preprocess audio to model input standard (16 kHz, float32, peak normalized)
        mono_processed, _ = self.preprocessor.process_array(
            mono_raw, sample_rate=sample_rate
        )

        # 4. Load requested or champion model from registry
        loaded_model = self.registry.get_model(
            machine_id=machine_id,
            model_type=model_override,
        )

        # 5. Execute model scoring
        raw_score, diag = loaded_model.score_waveform(mono_processed)

        # 6. Apply three-zone calibration
        cal = loaded_model.calibration
        theta_lower = float(cal.get("theta_lower", cal.get("operating_threshold", 0.0)))
        theta_upper = float(cal.get("theta_upper", cal.get("operating_threshold", 0.0)))
        is_provisional = bool(cal.get("is_provisional", True))

        reasons: List[str] = []

        if raw_score <= theta_lower:
            decision = DecisionZone.NORMAL_LIKE
            reasons.append(
                f"Acoustic reconstruction error ({raw_score:.5f}) is within the normal reference baseline (<= {theta_lower:.5f})."
            )
        elif raw_score > theta_upper:
            decision = DecisionZone.UNUSUAL
            reasons.append(
                f"Acoustic reconstruction error ({raw_score:.5f}) exceeds abnormal decision boundary (> {theta_upper:.5f}). Unusual acoustic deviation detected."
            )
        else:
            decision = DecisionZone.INCONCLUSIVE
            reasons.append(
                f"Acoustic reconstruction error ({raw_score:.5f}) lies in the calibrated transition zone between [{theta_lower:.5f}, {theta_upper:.5f}]. Anomaly status is inconclusive."
            )

        # Quality warnings note
        if quality_report.status == QualityStatus.WARNING:
            reasons.extend(quality_report.reasons)

        # Provisional notice
        if is_provisional:
            reasons.append(
                "Provisional calibration notice: Reference baseline derived from limited development samples (N < 30). Operating boundary carries statistical uncertainty."
            )

        duration_ms = (time.perf_counter() - t0) * 1000

        metrics = {
            **quality_report.metrics,
            "raw_anomaly_score": round(raw_score, 6),
            "model_parameters": loaded_model.metadata.total_parameters,
            "inference_device": loaded_model.metadata.device,
        }

        logger.info(
            f"Inference completed for {machine_id} [{loaded_model.model_type.value}]: "
            f"score={raw_score:.5f}, decision={decision} in {duration_ms:.2f}ms"
        )

        return InferenceResult(
            machine_id=machine_id,
            model_name=loaded_model.model_type.value,
            model_version=loaded_model.metadata.version,
            decision=decision,
            anomaly_score=raw_score,
            operating_threshold=theta_upper,
            theta_lower=theta_lower,
            theta_upper=theta_upper,
            is_provisional=is_provisional,
            quality_assessment=quality_report,
            reasons=reasons,
            processing_duration_ms=duration_ms,
            metrics=metrics,
        )


# Singleton
_inference_service: Optional[AcousticInferenceService] = None


def get_inference_service() -> AcousticInferenceService:
    global _inference_service
    if _inference_service is None:
        _inference_service = AcousticInferenceService()
    return _inference_service


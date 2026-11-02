"""Unified Adaptive Inference Engine for Stetho Acoustic Anomaly Detection.

Orchestrates the 7-stage unified inference pipeline:
1. Shared audio validation & quality gating
2. Standardized Log-Mel feature extraction
3. Fast Student INT8/FP32 first-pass inference
4. Category-aware specialist model selection
5. Machine-specific residual whitening
6. Uncertainty-aware escalation & decision
7. Unified diagnostic response
"""

from dataclasses import dataclass, field
import hashlib
import io
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

from ml.inference.adapters import (
    BaseModelAdapter,
    ModelOutput,
    MultiTaskAdapter,
    ResNetAEAdapter,
    ResidualWhiteningAdapter,
    StudentONNXAdapter,
)
from ml.inference.routing import (
    AdaptiveRouter,
    DecisionConfidence,
    DecisionStatus,
    RoutingPath,
    RoutingPolicyConfig,
    RoutingResult,
)
from ml.preprocessing.audio_quality_gate import AudioQualityGate, QualityGateConfig, QualityGateResult
from ml.preprocessing.feature_suite import UnifiedFeatureSuite

logger = logging.getLogger("stetho_unified_engine")


@dataclass
class DiagnosticResponse:
    """Comprehensive, structured diagnostic result for acoustic inspection."""
    asset_id: str
    machine_category: str
    decision: str  # "NORMAL", "ANOMALY", "UNCERTAIN", "REJECTED"
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    anomaly_score: Optional[float]
    operating_threshold: Optional[float]
    execution_path: str  # "FAST_STUDENT_EXIT", "SPECIALIST_ESCALATED", "RESIDUAL_WHITENED", "QUALITY_REJECTED"
    models_executed: int
    models_used: List[str]
    processing_duration_ms: float
    student_latency_ms: Optional[float] = None
    specialist_latency_ms: Optional[float] = None
    quality_passed: bool = True
    quality_status: str = "PASSED"
    quality_telemetry: Dict[str, Any] = field(default_factory=dict)
    rejection_reasons: List[str] = field(default_factory=list)
    uncertainty_reason: Optional[str] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "machine_id": self.asset_id,
            "machine_category": self.machine_category,
            "decision": self.decision,
            "anomaly_decision": self.decision,
            "confidence": self.confidence,
            "anomaly_score": round(self.anomaly_score, 5) if self.anomaly_score is not None else None,
            "operating_threshold": round(self.operating_threshold, 5) if self.operating_threshold is not None else None,
            "execution_path": self.execution_path,
            "models_executed": self.models_executed,
            "models_used": self.models_used,
            "processing_duration_ms": round(self.processing_duration_ms, 2),
            "student_latency_ms": round(self.student_latency_ms, 2) if self.student_latency_ms else None,
            "specialist_latency_ms": round(self.specialist_latency_ms, 2) if self.specialist_latency_ms else None,
            "quality_passed": self.quality_passed,
            "quality_status": self.quality_status,
            "quality_telemetry": self.quality_telemetry,
            "rejection_reasons": self.rejection_reasons,
            "uncertainty_reason": self.uncertainty_reason,
            "telemetry": self.telemetry,
        }


class UnifiedAdaptiveInferenceEngine:
    """Thread-safe, lazy-loading unified adaptive inference engine."""

    _instance: Optional["UnifiedAdaptiveInferenceEngine"] = None
    _init_lock = threading.Lock()

    def __init__(
        self,
        routing_config: Optional[RoutingPolicyConfig] = None,
        quality_config: Optional[QualityGateConfig] = None,
        checkpoints_dir: Union[str, Path] = "ml/experiments/research_artifacts",
    ):
        self.routing_config = routing_config or RoutingPolicyConfig()
        self.quality_config = quality_config or QualityGateConfig()
        self.checkpoints_dir = Path(checkpoints_dir)

        # Core components
        self.quality_gate = AudioQualityGate(self.quality_config)
        self.feature_suite = UnifiedFeatureSuite(preset_name="standard_64mel")
        self.router = AdaptiveRouter(self.routing_config)

        # Model cache: key -> BaseModelAdapter
        self._models_lock = threading.Lock()
        self._cached_adapters: Dict[str, BaseModelAdapter] = {}
        self._calibration_profiles: Dict[str, ResidualWhiteningAdapter] = {}

    @classmethod
    def get_instance(cls) -> "UnifiedAdaptiveInferenceEngine":
        """Return singleton instance of UnifiedAdaptiveInferenceEngine."""
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _get_student_adapter(self) -> StudentONNXAdapter:
        with self._models_lock:
            if "student_onnx" not in self._cached_adapters:
                onnx_path = self.checkpoints_dir / "phase8/exported_models/student_ae_int8.onnx"
                self._cached_adapters["student_onnx"] = StudentONNXAdapter(
                    onnx_path=onnx_path,
                    operating_threshold=self.routing_config.student_threshold,
                    prefer_fp32_for_latency=True,
                )
            return self._cached_adapters["student_onnx"]

    def _get_fan_specialist_adapter(self, machine_id: str) -> ResNetAEAdapter:
        cache_key = f"fan_resnet_{machine_id}"
        with self._models_lock:
            if cache_key not in self._cached_adapters:
                ckpt = self.checkpoints_dir / f"phase4/residual_cnn_ae/{machine_id}/model.pt"
                if not ckpt.exists():
                    ckpt = self.checkpoints_dir / "phase4/residual_cnn_ae/id_00/model.pt"
                self._cached_adapters[cache_key] = ResNetAEAdapter(
                    checkpoint_path=ckpt,
                    operating_threshold=self.routing_config.fan_specialist_threshold,
                    use_smooth_l1=True,
                    smooth_l1_beta=0.1,
                )
            return self._cached_adapters[cache_key]

    def _get_multitask_adapter(self) -> MultiTaskAdapter:
        with self._models_lock:
            if "multitask_resnet" not in self._cached_adapters:
                ckpt = self.checkpoints_dir / "phase4/residual_cnn_ae/id_00/model.pt"
                self._cached_adapters["multitask_resnet"] = MultiTaskAdapter(
                    checkpoint_path=ckpt,
                    operating_threshold=self.routing_config.multitask_threshold,
                )
            return self._cached_adapters["multitask_resnet"]

    def register_calibration_profile(
        self,
        machine_id: str,
        baseline_mean: np.ndarray,
        baseline_std: np.ndarray,
        calibrated_threshold: float,
        machine_category: str = "fan",
        n_shots: int = 10,
        two_sided: bool = True,
    ) -> None:
        """Register machine-specific few-shot residual whitening profile."""
        key = f"{machine_category.lower()}_{machine_id}"
        with self._models_lock:
            adapter = ResidualWhiteningAdapter(
                baseline_mean=baseline_mean,
                baseline_std=baseline_std,
                calibrated_threshold=calibrated_threshold,
                n_shots=n_shots,
                two_sided=two_sided,
            )
            self._calibration_profiles[key] = adapter
            self._calibration_profiles[machine_id] = adapter
            logger.info(f"Registered residual whitening profile for {key} (threshold={calibrated_threshold:.4f})")

    def inspect_audio(
        self,
        audio_input: Union[str, Path, bytes, np.ndarray],
        sample_rate: Optional[int] = None,
        machine_id: str = "asset_unknown",
        machine_category: str = "fan",
        channel_index: int = 0,
        force_specialist: bool = False,
        direct_calibrated: Optional[bool] = None,
    ) -> DiagnosticResponse:
        """Execute unified, adaptive inference on an incoming audio recording."""
        t_start = time.perf_counter()

        # Step 1: Decode Audio Payload
        if isinstance(audio_input, (str, Path)):
            raw_data, sr = sf.read(str(audio_input))
        elif isinstance(audio_input, bytes):
            raw_data, sr = sf.read(io.BytesIO(audio_input))
        elif isinstance(audio_input, np.ndarray):
            raw_data = audio_input
            sr = sample_rate or 16000
        else:
            raise ValueError(f"Unsupported audio_input type: {type(audio_input)}")

        # Step 2: Shared Audio Validation Gate
        prep_wav, q_result = self.quality_gate.validate_and_prepare(
            waveform=raw_data, sample_rate=sr, channel=channel_index
        )

        if not q_result.is_valid or prep_wav is None:
            total_duration_ms = (time.perf_counter() - t_start) * 1000.0
            return DiagnosticResponse(
                asset_id=machine_id,
                machine_category=machine_category,
                decision="REJECTED",
                confidence="HIGH",
                anomaly_score=None,
                operating_threshold=None,
                execution_path=RoutingPath.QUALITY_REJECTED.value,
                models_executed=0,
                models_used=[],
                processing_duration_ms=total_duration_ms,
                quality_passed=False,
                quality_status=q_result.status,
                quality_telemetry=q_result.telemetry,
                rejection_reasons=q_result.rejection_reasons,
                uncertainty_reason="AUDIO_QUALITY_GATE_FAILURE",
            )

        # Step 3: Feature Extraction (64-bin Log-Mel)
        if len(prep_wav) < 1024:
            total_duration_ms = (time.perf_counter() - t_start) * 1000.0
            return DiagnosticResponse(
                asset_id=machine_id,
                machine_category=machine_category,
                decision="REJECTED",
                confidence="HIGH",
                anomaly_score=None,
                operating_threshold=None,
                execution_path=RoutingPath.QUALITY_REJECTED.value,
                models_executed=0,
                models_used=[],
                processing_duration_ms=total_duration_ms,
                quality_passed=False,
                quality_status="EMPTY_SPECTROGRAM",
                rejection_reasons=["AUDIO_TOO_SHORT: Sample count is insufficient for spectral windowing."],
                uncertainty_reason="INSUFFICIENT_AUDIO_LENGTH",
            )

        # Standardize peak amplitude to match training representation
        peak_amp = float(np.max(np.abs(prep_wav)))
        if peak_amp > 1e-6:
            norm_wav = prep_wav / peak_amp
        else:
            norm_wav = prep_wav

        patches, _ = self.feature_suite.extract_patches_from_waveform(
            norm_wav, patch_frames=64, hop_frames=32
        )
        if len(patches) == 0:
            total_duration_ms = (time.perf_counter() - t_start) * 1000.0
            return DiagnosticResponse(
                asset_id=machine_id,
                machine_category=machine_category,
                decision="REJECTED",
                confidence="HIGH",
                anomaly_score=None,
                operating_threshold=None,
                execution_path=RoutingPath.QUALITY_REJECTED.value,
                models_executed=0,
                models_used=[],
                processing_duration_ms=total_duration_ms,
                quality_passed=False,
                quality_status="EMPTY_SPECTROGRAM",
                rejection_reasons=["No spectrogram patches could be extracted from waveform."],
            )

        # Step 4: Routing Check & Fast Calibration Bypass
        cal_key = f"{machine_category.lower()}_{machine_id}"
        has_calib = cal_key in self._calibration_profiles
        whitening_adapter = self._calibration_profiles.get(cal_key)
        
        use_direct = (
            self.routing_config.enable_direct_calibrated_routing
            if direct_calibrated is None
            else direct_calibrated
        )

        # Fast direct specialist path for commissioned/calibrated machines
        if has_calib and whitening_adapter is not None and use_direct:
            specialist_type = self.router.select_specialist_model_type(machine_category)
            if specialist_type == "resnet_ae_specialist":
                specialist = self._get_fan_specialist_adapter(machine_id)
            else:
                specialist = self._get_multitask_adapter()

            specialist_out: ModelOutput = specialist.score_patches(patches)
            spec_lat_ms = specialist_out.latency_ms

            whitened_patch_scores, whitened_score = whitening_adapter.whiten_reconstruction(
                input_patches=patches, reconstruction=specialist_out.reconstruction
            )
            route_path = RoutingPath.RESIDUAL_WHITENED
            status, conf, unc_reason = self.router.determine_decision_and_confidence(
                score=whitened_score,
                threshold=whitening_adapter.calibrated_threshold,
                patch_scores=whitened_patch_scores,
                route_path=route_path,
                quality_warnings=q_result.warnings,
            )

            total_duration_ms = (time.perf_counter() - t_start) * 1000.0
            return DiagnosticResponse(
                asset_id=machine_id,
                machine_category=machine_category,
                decision=status.value,
                confidence=conf.value,
                anomaly_score=whitened_score,
                operating_threshold=whitening_adapter.calibrated_threshold,
                execution_path=route_path.value,
                models_executed=1,
                models_used=[specialist.get_metadata()["model_name"]],
                processing_duration_ms=total_duration_ms,
                student_latency_ms=None,
                specialist_latency_ms=spec_lat_ms,
                quality_passed=True,
                quality_status=q_result.status,
                quality_telemetry=q_result.telemetry,
                uncertainty_reason=unc_reason,
                telemetry={
                    "specialist_raw_score": round(specialist_out.mean_score, 5),
                    "specialist_format": specialist.get_metadata()["model_format"],
                    "residual_whitening_applied": True,
                    "direct_calibrated_path": True,
                    "num_patches": len(patches),
                },
            )

        # Standard Cascade: Fast Student INT8/FP32 First-Pass
        student_adapter = self._get_student_adapter()
        student_out: ModelOutput = student_adapter.score_patches(patches)
        student_score = student_out.mean_score
        student_lat_ms = student_out.latency_ms

        can_exit, exit_reason = self.router.evaluate_student_stage(
            student_score=student_score, has_calibration_profile=has_calib
        )

        # Check for Early Exit
        if can_exit and not force_specialist:
            status, conf, unc_reason = self.router.determine_decision_and_confidence(
                score=student_score,
                threshold=self.routing_config.student_threshold,
                patch_scores=student_out.patch_scores,
                route_path=RoutingPath.FAST_STUDENT_EXIT,
                quality_warnings=q_result.warnings,
            )
            total_duration_ms = (time.perf_counter() - t_start) * 1000.0
            return DiagnosticResponse(
                asset_id=machine_id,
                machine_category=machine_category,
                decision=status.value,
                confidence=conf.value,
                anomaly_score=student_score,
                operating_threshold=self.routing_config.student_threshold,
                execution_path=RoutingPath.FAST_STUDENT_EXIT.value,
                models_executed=1,
                models_used=[student_adapter.model_path.name],
                processing_duration_ms=total_duration_ms,
                student_latency_ms=student_lat_ms,
                specialist_latency_ms=None,
                quality_passed=True,
                quality_status=q_result.status,
                quality_telemetry=q_result.telemetry,
                uncertainty_reason=unc_reason,
                telemetry={
                    "stage1_decision_margin": round(abs(student_score - self.routing_config.student_threshold), 4),
                    "exit_reason": exit_reason,
                    "num_patches": len(patches),
                },
            )

        # Step 5: Specialist Escalation
        specialist_type = self.router.select_specialist_model_type(machine_category)
        if specialist_type == "resnet_ae_specialist":
            specialist = self._get_fan_specialist_adapter(machine_id)
        else:
            specialist = self._get_multitask_adapter()

        specialist_out: ModelOutput = specialist.score_patches(patches)
        spec_lat_ms = specialist_out.latency_ms
        models_used = [student_adapter.model_path.name, specialist.get_metadata()["model_name"]]

        # Step 6: Machine-Specific Residual Whitening (if calibrated)
        if has_calib and whitening_adapter is not None:
            whitened_patch_scores, whitened_score = whitening_adapter.whiten_reconstruction(
                input_patches=patches, reconstruction=specialist_out.reconstruction
            )
            effective_score = whitened_score
            effective_threshold = whitening_adapter.calibrated_threshold
            effective_patch_scores = whitened_patch_scores
            route_path = RoutingPath.RESIDUAL_WHITENED
        else:
            effective_score = specialist_out.mean_score
            effective_threshold = specialist.operating_threshold
            effective_patch_scores = specialist_out.patch_scores
            route_path = RoutingPath.SPECIALIST_ESCALATED

        # Step 7: Uncertainty-Aware Decision Zone
        status, conf, unc_reason = self.router.determine_decision_and_confidence(
            score=effective_score,
            threshold=effective_threshold,
            patch_scores=effective_patch_scores,
            route_path=route_path,
            quality_warnings=q_result.warnings,
        )

        total_duration_ms = (time.perf_counter() - t_start) * 1000.0
        return DiagnosticResponse(
            asset_id=machine_id,
            machine_category=machine_category,
            decision=status.value,
            confidence=conf.value,
            anomaly_score=effective_score,
            operating_threshold=effective_threshold,
            execution_path=route_path.value,
            models_executed=2,
            models_used=models_used,
            processing_duration_ms=total_duration_ms,
            student_latency_ms=student_lat_ms,
            specialist_latency_ms=spec_lat_ms,
            quality_passed=True,
            quality_status=q_result.status,
            quality_telemetry=q_result.telemetry,
            uncertainty_reason=unc_reason,
            telemetry={
                "student_score": round(student_score, 5),
                "specialist_raw_score": round(specialist_out.mean_score, 5),
                "specialist_format": specialist.get_metadata()["model_format"],
                "residual_whitening_applied": has_calib,
                "num_patches": len(patches),
            },
        )

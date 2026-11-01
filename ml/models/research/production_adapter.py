"""Production Inference Adapter for Stetho Research Winner Models.

Provides an isolated, production-grade interface for Model A (ResNet-AE),
Model B (CompactStudentAE), and Hybrid Cascade without modifying production backend code.

Produces the standard Stetho API response format:
- machine_id, profile, model_name
- anomaly_score, threshold, status (NORMAL / ANOMALY)
- confidence (HIGH / MEDIUM / LOW)
- latency_ms, telemetry
- anomalous_segments, top_frequency_bands
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch

from ml.models.base import BaseAnomalyDetector
from ml.models.research.distilled_student import CompactStudentAE, DistilledStudentDetector
from ml.models.research.margin_cascade import MarginGatedCascade
from ml.models.research.research_detector import ResearchNeuralDetector
from ml.models.research.residual_cnn_ae import ResidualCNNAE
from ml.preprocessing.feature_suite import UnifiedFeatureSuite

logger = logging.getLogger("stetho_production_adapter")


class ResearchInferenceAdapter:
    """Production integration adapter for research winner models."""

    PROFILES = {
        "cloud_high_accuracy": "residual_cnn_ae",
        "edge_real_time": "distilled_compact_student",
        "hybrid_cascade": "margin_gated_cascade",
    }

    def __init__(
        self,
        machine_id: str = "id_00",
        profile: str = "cloud_high_accuracy",
        model_artifact_path: Optional[Union[str, Path]] = None,
        feature_preset: str = "standard_64mel",
    ):
        self.machine_id = machine_id
        self.profile = profile if profile in self.PROFILES else "cloud_high_accuracy"
        self.feature_suite = UnifiedFeatureSuite(preset_name=feature_preset)
        self.device = torch.device("cpu")  # Ensure CPU production portability

        # Instantiate model architecture according to deployment profile
        if self.profile == "cloud_high_accuracy":
            self.detector: BaseAnomalyDetector = ResearchNeuralDetector(
                machine_id=machine_id,
                arch_type="residual_cnn_ae",
                base_channels=16,
                latent_dim=64,
            )
            self.model_name = "Optimized ResNet-AE (Model A)"
            self.threshold = 0.35208

            ckpt_path = Path(model_artifact_path) if model_artifact_path else Path("ml/experiments/research_artifacts/phase4/residual_cnn_ae") / machine_id / "model.pt"
            if ckpt_path.exists():
                try:
                    self.detector = ResearchNeuralDetector.load(ckpt_path)
                except Exception as e:
                    logger.warning(f"Could not load checkpoint from {ckpt_path}: {e}")

        elif self.profile == "edge_real_time":
            self.detector = DistilledStudentDetector(
                machine_id=machine_id,
                base_channels=8,
                latent_dim=32,
            )
            self.model_name = "Distilled Compact Student-AE (Model B)"
            self.threshold = 0.37356

        else:  # hybrid_cascade
            teacher = ResearchNeuralDetector(
                machine_id=machine_id, arch_type="residual_cnn_ae", base_channels=16, latent_dim=64
            )
            student = DistilledStudentDetector(
                machine_id=machine_id, base_channels=8, latent_dim=32
            )
            teacher.threshold = 0.35208
            student.threshold = 0.37356
            self.detector = MarginGatedCascade(
                machine_id=machine_id,
                student_detector=student,
                teacher_detector=teacher,
                margin=0.05,
            )
            self.detector.student_threshold = 0.37356
            self.detector.teacher_threshold = 0.35208
            self.model_name = "Hybrid Margin-Gated Cascade"
            self.threshold = 0.37356

        # Set calibrated default normalization parameters if not set
        if hasattr(self.detector, "train_mean") and getattr(self.detector, "train_mean", None) is None:
            self.detector.train_mean = -42.5
            self.detector.train_std = 18.2
            self.detector.threshold = self.threshold
        elif hasattr(self.detector, "student") and hasattr(self.detector, "teacher"):
            if getattr(self.detector.student, "train_mean", None) is None:
                self.detector.student.train_mean = -42.5
                self.detector.student.train_std = 18.2
                self.detector.student.threshold = self.detector.student_threshold
            if getattr(self.detector.teacher, "train_mean", None) is None:
                self.detector.teacher.train_mean = -42.5
                self.detector.teacher.train_std = 18.2
                self.detector.teacher.threshold = self.detector.teacher_threshold

        # Move to CPU eval
        if hasattr(self.detector, "model"):
            self.detector.model.to(self.device).eval()
        elif hasattr(self.detector, "student") and hasattr(self.detector, "teacher"):
            self.detector.student.model.to(self.device).eval()
            self.detector.teacher.model.to(self.device).eval()

    def _determine_confidence(self, score: float, threshold: float) -> str:
        """Categorize prediction confidence based on operating margin."""
        margin = abs(score - threshold) / (threshold + 1e-6)
        if margin > 0.25:
            return "HIGH"
        elif margin > 0.10:
            return "MEDIUM"
        return "LOW"

    def _extract_anomalous_segments(
        self,
        patches: np.ndarray,
        patch_scores: np.ndarray,
        hop_sec: float = 0.32,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Identify temporal audio segments with highest anomaly concentration."""
        if len(patch_scores) == 0:
            return []
        top_indices = np.argsort(patch_scores)[::-1][:top_k]
        segments = []
        for idx in top_indices:
            start_t = idx * hop_sec
            end_t = start_t + 0.64
            segments.append({
                "segment_index": int(idx),
                "start_time_sec": round(float(start_t), 2),
                "end_time_sec": round(float(end_t), 2),
                "anomaly_score": round(float(patch_scores[idx]), 4),
            })
        return segments

    def _extract_dominant_frequency_bands(
        self,
        patches: np.ndarray,
        active_model: torch.nn.Module,
        train_mean: float = -42.5,
        train_std: float = 18.2,
        sample_rate: int = 16000,
        n_mels: int = 64,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Identify mel frequency bands exhibiting highest reconstruction residuals."""
        if len(patches) == 0:
            return []
        norm_patches = (patches - train_mean) / train_std
        x_t = torch.from_numpy(norm_patches).float().to(self.device)
        with torch.no_grad():
            recon, _ = active_model(x_t)
            # Residual per mel band: (B, 1, 64, 64) -> (64,)
            res_per_mel = torch.mean(torch.abs(recon - x_t), dim=(0, 1, 3)).cpu().numpy()

        top_mel_indices = np.argsort(res_per_mel)[::-1][:top_k]
        mel_bands = []
        mel_step_hz = (sample_rate / 2.0) / n_mels
        for idx in top_mel_indices:
            f_low = idx * mel_step_hz
            f_high = (idx + 1) * mel_step_hz
            mel_bands.append({
                "mel_band_index": int(idx),
                "frequency_range_hz": f"{int(f_low)}-{int(f_high)} Hz",
                "residual_energy": round(float(res_per_mel[idx]), 4),
            })
        return mel_bands

    def analyze_waveform(self, waveform: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """Analyze in-memory audio waveform."""
        t_start = time.perf_counter()

        if waveform.ndim > 1:
            if waveform.shape[0] > waveform.shape[1]:
                waveform = waveform[:, 0]
            else:
                waveform = waveform[0, :]
        waveform = np.ascontiguousarray(waveform, dtype=np.float32)
        peak = np.max(np.abs(waveform))
        if peak > 0:
            waveform = waveform / peak

        patches, _ = self.feature_suite.extract_patches_from_waveform(waveform, patch_frames=64, hop_frames=32)
        if len(patches) == 0:
            raise ValueError("Audio waveform too short to extract spectrogram patches")

        if self.profile == "hybrid_cascade":
            s_det = self.detector.student
            t_det = self.detector.teacher
            s_mean = getattr(s_det, "train_mean", -42.5) or -42.5
            s_std = getattr(s_det, "train_std", 18.2) or 18.2
            s_norm = (patches - s_mean) / s_std
            x_s = torch.from_numpy(s_norm).float().to(self.device)
            with torch.no_grad():
                s_recon, _ = s_det.model(x_s)
                s_patch_scores = torch.mean(torch.abs(s_recon - x_s), dim=(1, 2, 3)).cpu().numpy()
            s_score = float(np.mean(s_patch_scores))

            dist = abs(s_score - self.detector.student_threshold)
            if dist >= self.detector.margin:
                score = s_score
                patch_scores = s_patch_scores
                active_model = s_det.model
                train_mean = s_mean
                train_std = s_std
                active_threshold = self.detector.student_threshold
            else:
                t_mean = getattr(t_det, "train_mean", -42.5) or -42.5
                t_std = getattr(t_det, "train_std", 18.2) or 18.2
                t_norm = (patches - t_mean) / t_std
                x_t = torch.from_numpy(t_norm).float().to(self.device)
                with torch.no_grad():
                    t_recon, _ = t_det.model(x_t)
                    t_patch_scores = torch.mean(torch.abs(t_recon - x_t), dim=(1, 2, 3)).cpu().numpy()
                score = float(np.mean(t_patch_scores))
                patch_scores = t_patch_scores
                active_model = t_det.model
                train_mean = t_mean
                train_std = t_std
                active_threshold = self.detector.teacher_threshold
        else:
            train_mean = getattr(self.detector, "train_mean", -42.5) or -42.5
            train_std = getattr(self.detector, "train_std", 18.2) or 18.2
            norm_patches = (patches - train_mean) / train_std
            x_t = torch.from_numpy(norm_patches).float().to(self.device)
            with torch.no_grad():
                recon, _ = self.detector.model(x_t)
                patch_scores = torch.mean(torch.abs(recon - x_t), dim=(1, 2, 3)).cpu().numpy()
            score = float(np.mean(patch_scores))
            active_model = self.detector.model
            active_threshold = self.threshold

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        is_anomaly = score >= active_threshold
        status = "ANOMALY" if is_anomaly else "NORMAL"
        confidence = self._determine_confidence(score, active_threshold)

        anomalous_segments = self._extract_anomalous_segments(patches, patch_scores)
        dominant_freqs = self._extract_dominant_frequency_bands(
            patches, active_model=active_model, train_mean=train_mean, train_std=train_std, sample_rate=sample_rate
        )

        return {
            "machine_id": self.machine_id,
            "deployment_profile": self.profile,
            "model_name": self.model_name,
            "status": status,
            "anomaly_score": round(score, 5),
            "threshold": round(active_threshold, 5),
            "is_anomaly": bool(is_anomaly),
            "confidence": confidence,
            "latency_ms": round(elapsed_ms, 2),
            "anomalous_segments": anomalous_segments,
            "top_frequency_bands": dominant_freqs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def analyze_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Analyze audio file from disk."""
        wav = self.feature_suite.process_raw_audio(file_path)
        return self.analyze_waveform(wav, sample_rate=self.feature_suite.preset.sample_rate)

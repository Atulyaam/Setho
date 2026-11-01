"""Confidence-Margin Gated Cascade Policy for Industrial Acoustic Inference.

Implements:
- Stage 1: Ultra-fast distilled student model (sub-50ms CPU)
- Decision Margin: |score_student - threshold_student|
- If Margin >= Gamma: Confident decision -> Early-exit immediately
- If Margin < Gamma: Ambiguous region -> Escalate to Teacher (ResNet-AE)
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ml.calibration.statistical_calibration import StatisticalCalibrator
from ml.models.base import BaseAnomalyDetector

logger = logging.getLogger("stetho_margin_cascade")


class MarginGatedCascade(BaseAnomalyDetector):
    """Adaptive margin-gated cascade combining fast student and heavy teacher."""

    def __init__(
        self,
        machine_id: str,
        student_detector: BaseAnomalyDetector,
        teacher_detector: BaseAnomalyDetector,
        margin: float = 0.05,
    ):
        super().__init__(machine_id=machine_id, model_type="margin_gated_cascade")
        self.student = student_detector
        self.teacher = teacher_detector
        self.margin = margin

        self.student_threshold: float = 0.0
        self.teacher_threshold: float = 0.0

        # Runtime telemetry counters
        self.early_exit_count: int = 0
        self.escalation_count: int = 0

    def fit_cascade(self, dev_files: List[Union[str, Path]]) -> None:
        """Calibrate thresholds on normal dev files strictly without leakage."""
        calibrator = StatisticalCalibrator(target_quantile=0.95)

        cal_s = calibrator.calibrate(self.student, dev_files)
        self.student_threshold = cal_s.operating_threshold
        self.student.threshold = self.student_threshold

        cal_t = calibrator.calibrate(self.teacher, dev_files)
        self.teacher_threshold = cal_t.operating_threshold
        self.teacher.threshold = self.teacher_threshold

        logger.info(
            f"Cascade calibrated for {self.machine_id}: "
            f"Student threshold={self.student_threshold:.5f}, "
            f"Teacher threshold={self.teacher_threshold:.5f}, "
            f"Margin={self.margin}"
        )

    def fit(self, train_files: List[Union[str, Path]], dev_files: Optional[List[Union[str, Path]]] = None) -> Dict[str, Any]:
        calib_files = dev_files if dev_files else train_files
        self.fit_cascade(calib_files)
        return {"student_threshold": self.student_threshold, "teacher_threshold": self.teacher_threshold}

    def score_file_with_telemetry(self, file_path: Union[str, Path]) -> Tuple[float, Dict[str, Any]]:
        """Score file, route dynamically, and report latency and routing telemetry."""
        t0 = time.perf_counter()
        s_score = self.student.score_file(file_path)
        t_stage1 = (time.perf_counter() - t0) * 1000.0

        # Distance to operating threshold
        dist = abs(s_score - self.student_threshold)

        if dist >= self.margin:
            # High confidence zone: accept student decision
            self.early_exit_count += 1
            return float(s_score), {
                "stage": 1,
                "escalated": False,
                "stage1_latency_ms": round(t_stage1, 2),
                "total_latency_ms": round(t_stage1, 2),
                "decision": "normal" if s_score < self.student_threshold else "anomaly",
            }

        # Ambiguous boundary: escalate to high-capacity teacher
        self.escalation_count += 1
        t_s2_start = time.perf_counter()
        t_score = self.teacher.score_file(file_path)
        t_stage2 = (time.perf_counter() - t_s2_start) * 1000.0
        total_lat = (time.perf_counter() - t0) * 1000.0

        return float(t_score), {
            "stage": 2,
            "escalated": True,
            "stage1_latency_ms": round(t_stage1, 2),
            "stage2_latency_ms": round(t_stage2, 2),
            "total_latency_ms": round(total_lat, 2),
            "decision": "normal" if t_score < self.teacher_threshold else "anomaly",
        }

    def score_file(self, file_path: Union[str, Path]) -> float:
        score, _ = self.score_file_with_telemetry(file_path)
        return score

    def score_waveform(self, waveform: np.ndarray, sample_rate: int = 16000) -> float:
        s_score = self.student.score_waveform(waveform, sample_rate)
        dist = abs(s_score - self.student_threshold)
        if dist >= self.margin:
            return float(s_score)
        t_score = self.teacher.score_waveform(waveform, sample_rate)
        return float(t_score)

    def save(self, path: Union[str, Path]) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "machine_id": self.machine_id,
            "margin": self.margin,
            "student_threshold": self.student_threshold,
            "teacher_threshold": self.teacher_threshold,
        }
        with open(save_path if save_path.suffix == ".json" else save_path / "margin_cascade.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MarginGatedCascade":
        load_path = Path(path)
        target = load_path if load_path.suffix == ".json" else load_path / "margin_cascade.json"
        with open(target, "r") as f:
            meta = json.load(f)
        cas = cls(machine_id=meta["machine_id"], student_detector=None, teacher_detector=None, margin=meta["margin"])
        cas.student_threshold = meta["student_threshold"]
        cas.teacher_threshold = meta["teacher_threshold"]
        return cas


"""Unit tests for Sprint 6C Few-Shot Machine Calibration Engine."""

from pathlib import Path
import pytest
import numpy as np
import torch

from ml.calibration.few_shot_calibrator import FewShotMachineCalibrator
from ml.models.research.distilled_student import CompactStudentAE, DistilledStudentDetector
from ml.models.research.residual_cnn_ae import ResidualCNNAE
from ml.models.research.research_detector import ResearchNeuralDetector

SAMPLE_FILES = [
    Path("data/mimii/raw/fan/id_00/normal/00000000.wav"),
    Path("data/mimii/raw/fan/id_00/normal/00000001.wav"),
]


def test_few_shot_calibrator_methods():
    """Verify all 4 calibration methods initialize, fit, and score without error."""
    detector = DistilledStudentDetector(machine_id="id_00", base_channels=8, latent_dim=32)
    detector.train_mean = -42.5
    detector.train_std = 18.2

    for method in ["uncalibrated", "mean_var_residual", "mahalanobis_latent", "memory_bank_knn"]:
        calibrator = FewShotMachineCalibrator(
            base_detector=detector,
            method=method,
            n_shots=2,
            target_quantile=0.95,
        )
        res = calibrator.calibrate(SAMPLE_FILES)
        assert res.method == method
        assert res.n_shots == 2
        assert res.threshold > 0.0
        assert res.calibration_time_s < 5.0, f"Calibration took {res.calibration_time_s}s, exceeding 5s limit"

        # Score test file
        score = calibrator.score_file(SAMPLE_FILES[0])
        assert isinstance(score, float)
        assert not np.isnan(score)
        assert not np.isinf(score)

        pred, score2 = calibrator.predict_file(SAMPLE_FILES[0])
        assert pred in (0, 1)
        assert abs(score - score2) < 1e-5


def test_few_shot_sample_sweep():
    """Verify sample sweep behavior with N=1 and N=2 normal recordings."""
    detector = ResearchNeuralDetector(machine_id="id_00", arch_type="residual_cnn_ae", base_channels=16, latent_dim=64)
    detector.train_mean = -42.5
    detector.train_std = 18.2

    for n in [1, 2]:
        calibrator = FewShotMachineCalibrator(
            base_detector=detector,
            method="mean_var_residual",
            n_shots=n,
        )
        res = calibrator.calibrate(SAMPLE_FILES[:n])
        assert res.n_shots == n
        assert calibrator.is_calibrated
        assert calibrator.baseline_residual_mean is not None
        assert calibrator.baseline_residual_std is not None

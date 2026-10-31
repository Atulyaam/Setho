"""Unit tests for Statistical Calibration Engine."""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from ml.calibration.statistical_calibration import CalibrationProfile, StatisticalCalibrator
from ml.models.base import BaseAnomalyDetector


class MockDetector(BaseAnomalyDetector):
    def __init__(self, scores: list[float]):
        super().__init__(machine_id="test_machine", model_type="mock")
        self.scores = scores
        self.call_idx = 0

    def fit(self, train_files, **kwargs):
        self.is_fitted = True
        return {}

    def score_waveform(self, waveform: np.ndarray) -> float:
        val = self.scores[self.call_idx % len(self.scores)]
        self.call_idx += 1
        return val

    def score_file(self, file_path) -> float:
        val = self.scores[self.call_idx % len(self.scores)]
        self.call_idx += 1
        return val

    def save(self, output_dir):
        return {}

    @classmethod
    def load(cls, model_dir):
        return cls([0.5])


def test_calibration_quantile_and_threshold():
    # 20 dev scores uniformly distributed between 0.1 and 1.0
    known_scores = list(np.linspace(0.1, 1.0, 20))
    detector = MockDetector(known_scores)

    calibrator = StatisticalCalibrator(target_quantile=0.95)
    dev_files = [f"dev_{i}.wav" for i in range(20)]

    profile = calibrator.calibrate(detector, dev_files)

    assert not profile.is_provisional
    assert profile.dev_sample_count == 20
    assert profile.operating_threshold > 0.90
    assert detector.threshold == profile.operating_threshold


def test_provisional_flag_for_small_sample():
    # MIMII setup: 2 dev files
    known_scores = [0.32, 0.45]
    detector = MockDetector(known_scores)

    calibrator = StatisticalCalibrator(target_quantile=0.95)
    dev_files = ["dev_0.wav", "dev_1.wav"]

    profile = calibrator.calibrate(detector, dev_files)

    # Must be flagged as provisional with limitation notice
    assert profile.is_provisional
    assert "Provisional" in profile.limitation_notice
    assert "small" in profile.limitation_notice.lower()
    assert profile.dev_sample_count == 2


def test_calibration_profile_json_roundtrip(tmp_path):
    profile = CalibrationProfile(
        machine_id="id_00",
        model_type="isolation_forest",
        target_quantile=0.95,
        operating_threshold=0.55,
        dev_sample_count=2,
        dev_scores=[0.4, 0.5],
        dev_mean=0.45,
        dev_std=0.05,
        dev_min=0.4,
        dev_max=0.5,
        dev_p90=0.49,
        dev_p95=0.495,
        dev_p99=0.499,
        is_provisional=True,
        limitation_notice="Notice",
    )
    save_file = tmp_path / "calibration.json"
    profile.save_json(save_file)

    loaded = CalibrationProfile.load_json(save_file)
    assert loaded.machine_id == "id_00"
    assert loaded.operating_threshold == 0.55
    assert loaded.is_provisional


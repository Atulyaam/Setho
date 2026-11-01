"""Automated verification tests for Phase 6 Ensemble & Model Fusion."""

from pathlib import Path
import numpy as np
import pytest

from ml.models.base import BaseAnomalyDetector
from ml.models.research.ensemble_fusion import CascadedFastRejectDetector, ScoreFusionEnsemble


class DummyDetector(BaseAnomalyDetector):
    """Deterministic dummy detector for testing ensemble fusion logic."""

    def __init__(self, machine_id: str, fixed_score: float):
        super().__init__(machine_id=machine_id, model_type="dummy")
        self.fixed_score = fixed_score

    def fit(self, train_files, dev_files=None):
        return {}

    def score_file(self, file_path) -> float:
        return self.fixed_score

    def score_waveform(self, waveform: np.ndarray, sample_rate: int = 16000) -> float:
        return self.fixed_score

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        return cls("id_00", 0.0)


def test_score_fusion_strategies():
    """Verify weighted average, rank average, and max fusion mathematical behavior."""
    d1 = DummyDetector("id_00", 2.0)
    d2 = DummyDetector("id_00", 4.0)

    # 1. Weighted Average
    ens_w = ScoreFusionEnsemble("id_00", [d1, d2], strategy="weighted_average", weights=[0.25, 0.75])
    # Manually configure normalization for determinism
    ens_w.dev_score_means = [1.0, 2.0]
    ens_w.dev_score_stds = [1.0, 1.0]

    # z1 = (2 - 1)/1 = 1.0, z2 = (4 - 2)/1 = 2.0
    # fuse = 0.25 * 1.0 + 0.75 * 2.0 = 0.25 + 1.5 = 1.75
    score_w = ens_w.score_file("dummy_path.wav")
    assert pytest.approx(score_w, 0.01) == 1.75

    # 2. Max Fusion
    ens_max = ScoreFusionEnsemble("id_00", [d1, d2], strategy="max_fusion")
    ens_max.dev_score_means = [1.0, 2.0]
    ens_max.dev_score_stds = [1.0, 1.0]
    # max(1.0, 2.0) = 2.0
    score_max = ens_max.score_file("dummy_path.wav")
    assert pytest.approx(score_max, 0.01) == 2.0

    # 3. Rank Average
    ens_rank = ScoreFusionEnsemble("id_00", [d1, d2], strategy="rank_average")
    ens_rank.dev_score_pools = [np.array([1.0, 3.0]), np.array([2.0, 5.0])]
    score_rank = ens_rank.score_file("dummy_path.wav")
    assert 0.0 <= score_rank <= 1.0


def test_cascaded_fast_reject():
    """Verify early-exit routing and two-stage escalation in CascadedFastRejectDetector."""
    d_fast = DummyDetector("id_00", 0.10)
    d_heavy = DummyDetector("id_00", 0.90)

    cascade = CascadedFastRejectDetector(
        machine_id="id_00",
        stage1_fast_detector=d_fast,
        stage2_heavy_detector=d_heavy,
        early_exit_quantile=0.50,
    )
    cascade.early_exit_threshold = 0.20
    cascade.stage1_mean = 0.10
    cascade.stage1_std = 1.0
    cascade.stage2_mean = 0.50
    cascade.stage2_std = 1.0

    # Test early exit (d_fast score 0.10 <= threshold 0.20)
    score, info = cascade.score_file_with_telemetry("normal_sample.wav")
    assert info["early_exited"] is True
    assert info["stage"] == 1
    assert cascade.early_exit_count == 1
    assert cascade.heavy_eval_count == 0

    # Now simulate suspect audio exceeding stage 1 threshold
    d_fast.fixed_score = 0.85
    score_heavy, info_heavy = cascade.score_file_with_telemetry("ambiguous_sample.wav")
    assert info_heavy["early_exited"] is False
    assert info_heavy["stage"] == 2
    assert cascade.heavy_eval_count == 1


def test_waveform_scoring_compatibility():
    """Verify score_waveform runs without exception on fusion ensembles."""
    d1 = DummyDetector("id_00", 0.3)
    d2 = DummyDetector("id_00", 0.7)
    ens = ScoreFusionEnsemble("id_00", [d1, d2], strategy="weighted_average")
    ens.dev_score_means = [0.0, 0.0]
    ens.dev_score_stds = [1.0, 1.0]

    dummy_wav = np.zeros(16000, dtype=np.float32)
    score = ens.score_waveform(dummy_wav)
    assert pytest.approx(score, 0.01) == 0.5

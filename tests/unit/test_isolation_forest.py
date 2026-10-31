"""Unit tests for Isolation Forest Baseline Model."""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import soundfile as sf

from ml.models.classical.isolation_forest import IsolationForestDetector


@pytest.fixture
def synthetic_dataset(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    sr = 16000
    t = np.linspace(0, 10.0, int(sr * 10.0), endpoint=False)

    train_files = []
    # 5 normal recordings with similar harmonic pattern
    for i in range(5):
        audio = (0.3 * np.sin(2 * np.pi * 300 * t) + 0.05 * np.sin(2 * np.pi * 600 * t) + 0.01 * np.random.randn(len(t))).astype(np.float32)
        fpath = train_dir / f"train_normal_{i}.wav"
        # Save as 8-channel WAV (MIMII format)
        sf.write(str(fpath), np.repeat(audio[:, np.newaxis], 8, axis=1), sr)
        train_files.append(fpath)

    # 1 normal test clip
    test_normal_audio = (0.3 * np.sin(2 * np.pi * 300 * t) + 0.05 * np.sin(2 * np.pi * 600 * t) + 0.01 * np.random.randn(len(t))).astype(np.float32)
    test_normal_path = tmp_path / "test_normal.wav"
    sf.write(str(test_normal_path), np.repeat(test_normal_audio[:, np.newaxis], 8, axis=1), sr)

    # 1 anomalous test clip (chaotic high-frequency noise & distortion)
    test_abnormal_audio = (0.3 * np.random.randn(len(t)) + 0.5 * np.sin(2 * np.pi * 3500 * t)).astype(np.float32)
    test_abnormal_path = tmp_path / "test_abnormal.wav"
    sf.write(str(test_abnormal_path), np.repeat(test_abnormal_audio[:, np.newaxis], 8, axis=1), sr)

    return train_files, test_normal_path, test_abnormal_path


def test_isolation_forest_fit_and_scoring(synthetic_dataset, tmp_path):
    train_files, test_normal_path, test_abnormal_path = synthetic_dataset

    detector = IsolationForestDetector(machine_id="id_00", n_estimators=50, random_state=42)
    meta = detector.fit(train_files)

    assert detector.is_fitted
    assert meta["train_sample_count"] == len(train_files)

    # Score normal vs abnormal
    score_normal = detector.score_file(test_normal_path)
    score_abnormal = detector.score_file(test_abnormal_path)

    assert not np.isnan(score_normal)
    assert not np.isnan(score_abnormal)
    # HIGHER score must mean MORE ANOMALOUS
    assert score_abnormal > score_normal, f"Expected {score_abnormal} > {score_normal}"


def test_isolation_forest_serialization(synthetic_dataset, tmp_path):
    train_files, test_normal_path, _ = synthetic_dataset

    detector = IsolationForestDetector(machine_id="id_00", n_estimators=50, random_state=42)
    detector.fit(train_files)
    detector.threshold = 0.55

    model_dir = tmp_path / "if_saved"
    detector.save(model_dir)

    loaded = IsolationForestDetector.load(model_dir)
    assert loaded.is_fitted
    assert loaded.threshold == 0.55
    assert loaded.machine_id == "id_00"

    score1 = detector.score_file(test_normal_path)
    score2 = loaded.score_file(test_normal_path)
    assert np.isclose(score1, score2, atol=1e-5)


"""Integration tests for Stetho Baseline Anomaly Detection Pipeline."""

from pathlib import Path
import json
import pytest

from ml.models.baseline_autoencoder.autoencoder_detector import DenseAutoencoderDetector
from ml.models.classical.isolation_forest import IsolationForestDetector

RAW_FAN_DIR = Path("data/mimii/raw/fan")
ARTIFACTS_DIR = Path("ml/artifacts")


@pytest.fixture
def real_mimii_clip():
    clips = list(RAW_FAN_DIR.glob("*/*/*.wav"))
    if not clips:
        pytest.skip("No real MIMII audio clips found")
    return clips[0]


def test_isolation_forest_artifact_load_and_score(real_mimii_clip):
    if_dir = ARTIFACTS_DIR / "isolation_forest" / "id_00"
    if not if_dir.exists():
        pytest.skip("Isolation Forest artifacts for id_00 not yet generated")

    detector = IsolationForestDetector.load(if_dir)
    assert detector.is_fitted
    assert detector.machine_id == "id_00"

    score = detector.score_file(real_mimii_clip)
    assert isinstance(score, float)
    assert not (score != score)  # Not NaN

    if detector.threshold is not None:
        pred = detector.predict(score)
        assert isinstance(pred, (bool, int))


def test_dense_autoencoder_artifact_load_and_score(real_mimii_clip):
    dae_dir = ARTIFACTS_DIR / "dense_autoencoder" / "id_00"
    if not dae_dir.exists():
        pytest.skip("Dense Autoencoder artifacts for id_00 not yet generated")

    detector = DenseAutoencoderDetector.load(dae_dir, device="cpu")
    assert detector.is_fitted
    assert detector.machine_id == "id_00"

    score = detector.score_file(real_mimii_clip)
    assert isinstance(score, float)
    assert not (score != score)  # Not NaN
    assert score > 0.0

    if detector.threshold is not None:
        pred = detector.predict(score)
        assert isinstance(pred, (bool, int))


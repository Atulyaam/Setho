"""Unit tests for Statistical Acoustic Feature Extraction."""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from ml.features.acoustic_features import (
    AcousticFeatureConfig,
    AcousticFeatureExtractor,
    AcousticFeatureScaler,
)


@pytest.fixture
def synthetic_audio():
    sr = 16000
    t = np.linspace(0, 10.0, int(sr * 10.0), endpoint=False)
    # Sinusoid with harmonics and noise
    signal = 0.4 * np.sin(2 * np.pi * 350 * t) + 0.1 * np.sin(2 * np.pi * 700 * t)
    noise = 0.02 * np.random.randn(len(t))
    return (signal + noise).astype(np.float32)


def test_feature_extraction_dimensions(synthetic_audio):
    extractor = AcousticFeatureExtractor()
    features = extractor.extract_from_waveform(synthetic_audio)

    assert features.shape == (38,)
    assert features.dtype == np.float32
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()


def test_feature_repeatability(synthetic_audio):
    extractor = AcousticFeatureExtractor()
    feat1 = extractor.extract_from_waveform(synthetic_audio)
    feat2 = extractor.extract_from_waveform(synthetic_audio)

    assert np.allclose(feat1, feat2, atol=1e-6)


def test_silence_handling():
    extractor = AcousticFeatureExtractor()
    silent = np.zeros(160000, dtype=np.float32)
    features = extractor.extract_from_waveform(silent)

    assert features.shape == (38,)
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()


def test_feature_scaler_train_only_fit():
    scaler = AcousticFeatureScaler()
    X_train = np.random.randn(10, 38).astype(np.float32) * 5.0 + 10.0
    X_test = np.random.randn(5, 38).astype(np.float32) * 5.0 + 10.0

    # Transforming before fitting must fail
    with pytest.raises(RuntimeError):
        scaler.transform(X_test)

    # Fit on train only
    X_train_scaled = scaler.fit_transform(X_train)
    assert np.allclose(np.mean(X_train_scaled, axis=0), 0.0, atol=1e-5)
    assert np.allclose(np.std(X_train_scaled, axis=0), 1.0, atol=1e-5)

    # Transform test without re-fitting
    X_test_scaled = scaler.transform(X_test)
    assert X_test_scaled.shape == (5, 38)
    assert not np.isnan(X_test_scaled).any()


def test_feature_scaler_serialization():
    with tempfile.TemporaryDirectory() as td:
        save_path = Path(td) / "scaler.joblib"
        scaler = AcousticFeatureScaler()
        X_train = np.random.randn(10, 38).astype(np.float32)
        scaler.fit(X_train)
        scaler.save(save_path)

        loaded_scaler = AcousticFeatureScaler.load(save_path)
        assert loaded_scaler.is_fitted

        X_sample = np.random.randn(38).astype(np.float32)
        out1 = scaler.transform(X_sample)
        out2 = loaded_scaler.transform(X_sample)
        assert np.allclose(out1, out2)


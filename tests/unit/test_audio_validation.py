"""Unit tests for Audio Validation Module."""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import soundfile as sf

from stetho_audio.validation import AudioValidator, ValidationResult


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def validator():
    return AudioValidator(
        expected_sample_rate=16000,
        expected_duration_sec=10.0,
        duration_tolerance_sec=0.1,
        silence_threshold_rms=1e-4,
        clipping_threshold=0.999,
    )


def test_validation_non_existent_file(validator):
    res = validator.validate_file("non_existent_file.wav")
    assert not res.is_valid
    assert any("does not exist" in e for e in res.errors)


def test_validation_empty_file(validator, temp_dir):
    empty_file = temp_dir / "empty.wav"
    empty_file.touch()
    res = validator.validate_file(empty_file)
    assert not res.is_valid
    assert any("empty" in e for e in res.errors)


def test_validation_valid_file(validator, temp_dir):
    valid_file = temp_dir / "valid.wav"
    sr = 16000
    duration = 10.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Generate 16 kHz, 8-channel sinusoidal signal
    audio = np.stack([0.3 * np.sin(2 * np.pi * (200 + i * 50) * t) for i in range(8)], axis=1).astype(np.float32)
    sf.write(str(valid_file), audio, sr, subtype="PCM_16")

    res = validator.validate_file(valid_file)
    assert res.is_valid
    assert len(res.errors) == 0
    assert res.properties["sample_rate"] == 16000
    assert res.properties["channels"] == 8
    assert res.properties["duration_sec"] == 10.0


def test_validation_nan_and_inf(validator):
    sr = 16000
    audio_nan = np.zeros(160000, dtype=np.float32)
    audio_nan[100] = np.nan
    res_nan = validator.validate_array(audio_nan, sr)
    assert not res_nan.is_valid
    assert any("NaN" in e for e in res_nan.errors)

    audio_inf = np.zeros(160000, dtype=np.float32)
    audio_inf[200] = np.inf
    res_inf = validator.validate_array(audio_inf, sr)
    assert not res_inf.is_valid
    assert any("Inf" in e for e in res_inf.errors)


def test_validation_silence_detection(validator):
    sr = 16000
    silent_audio = np.zeros(160000, dtype=np.float32)
    res = validator.validate_array(silent_audio, sr)
    assert not res.is_valid
    assert any("silent" in e.lower() for e in res.errors)


def test_validation_clipping_detection(validator):
    sr = 16000
    t = np.linspace(0, 10.0, 160000, endpoint=False)
    # High amplitude signal with clipped flat tops
    audio = np.clip(1.5 * np.sin(2 * np.pi * 300 * t), -1.0, 1.0).astype(np.float32)
    res = validator.validate_array(audio, sr)
    assert res.is_valid  # Still decodable, but flags warnings
    assert len(res.warnings) > 0
    assert any("clipping" in w.lower() for w in res.warnings)
    assert res.properties["clipping_samples"] > 0


def test_validation_sample_rate_mismatch(validator):
    audio = np.random.randn(80000).astype(np.float32) * 0.1
    res = validator.validate_array(audio, sample_rate=8000)
    assert not res.is_valid
    assert any("Sample rate mismatch" in e for e in res.errors)

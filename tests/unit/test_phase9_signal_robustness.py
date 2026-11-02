"""Unit tests for Phase E6 Signal Normalization and Sensor Robustness modules."""

import numpy as np
import pytest

from ml.preprocessing.audio_quality_gate import AudioQualityGate, QualityGateConfig
from ml.preprocessing.gain_normalizer import GainNormalizer, GainNormalizerConfig
from ml.preprocessing.sensor_guard import SensorGuard


def test_gain_normalizer_telemetry_and_scaling():
    normalizer = GainNormalizer(GainNormalizerConfig(target_rms_dbfs=-42.0))

    # Generate a pure sine wave at approx -30 dBFS
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    wav = 0.0447 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    telemetry = normalizer.measure_telemetry(wav)
    assert -31.0 <= telemetry["raw_rms_dbfs"] <= -29.0
    assert 0.04 < telemetry["raw_peak"] < 0.05
    assert telemetry["clipping_ratio"] == 0.0

    # Normalize to -42 dBFS
    norm_wav, post_telemetry = normalizer.normalize(wav)
    assert post_telemetry["normalization_status"] == "NORMALIZED"
    assert -43.0 <= post_telemetry["post_rms_dbfs"] <= -41.0
    assert post_telemetry["gain_applied_db"] < 0.0  # Attenuated from -30 to -42


def test_gain_normalizer_silence_guard():
    normalizer = GainNormalizer(GainNormalizerConfig(target_rms_dbfs=-42.0, silence_threshold_dbfs=-60.0))

    # Near-silent waveform
    wav_silent = 1e-5 * np.ones(16000, dtype=np.float32)
    norm_wav, telemetry = normalizer.normalize(wav_silent)

    assert telemetry["normalization_status"] == "SILENCE_PRESERVED"
    assert telemetry["gain_applied_db"] == 0.0
    assert np.allclose(norm_wav, wav_silent)


def test_audio_quality_gate_sample_rate_rejection():
    gate = AudioQualityGate(QualityGateConfig(min_sample_rate=16000))

    wav = np.random.randn(8000).astype(np.float32) * 0.05
    prepared, result = gate.validate_and_prepare(wav, sample_rate=8000)

    assert prepared is None
    assert not result.is_valid
    assert result.status == "RECORDING_QUALITY_FAILURE"
    assert any("SAMPLE_RATE_UNSUPPORTED" in r for r in result.rejection_reasons)


def test_audio_quality_gate_resampling():
    gate = AudioQualityGate(QualityGateConfig(min_sample_rate=16000, target_sample_rate=16000))

    # 48 kHz input signal (1 second = 48000 samples)
    t = np.linspace(0, 1.0, 48000, endpoint=False)
    wav = 0.05 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)

    prepared, result = gate.validate_and_prepare(wav, sample_rate=48000)
    assert prepared is not None
    assert result.is_valid
    assert result.status == "RESAMPLED"
    assert len(prepared) == 16000


def test_audio_quality_gate_clipping_rejection():
    gate = AudioQualityGate(QualityGateConfig(max_clipping_ratio=0.001))

    # Heavily clipped waveform
    wav = np.ones(16000, dtype=np.float32)
    prepared, result = gate.validate_and_prepare(wav, sample_rate=16000)

    assert prepared is None
    assert not result.is_valid
    assert result.status == "RECORDING_QUALITY_FAILURE"
    assert any("EXCESSIVE_DIGITAL_CLIPPING" in r for r in result.rejection_reasons)


def test_audio_quality_gate_silence_rejection():
    gate = AudioQualityGate(QualityGateConfig(min_rms_dbfs=-60.0))

    wav = np.zeros(16000, dtype=np.float32)
    prepared, result = gate.validate_and_prepare(wav, sample_rate=16000)

    assert prepared is None
    assert not result.is_valid
    assert result.status == "RECORDING_QUALITY_FAILURE"
    assert any("SIGNAL_SILENCE_OR_EMPTY" in r for r in result.rejection_reasons)


def test_audio_quality_gate_noise_floor_warning():
    gate = AudioQualityGate()

    # High white noise
    noise = np.random.randn(16000).astype(np.float32) * 0.05
    prepared, result = gate.validate_and_prepare(noise, sample_rate=16000)

    assert prepared is not None
    assert result.is_valid
    assert result.status == "ADVERSE_ACOUSTIC_ENVIRONMENT"
    assert len(result.warnings) > 0


def test_sensor_guard():
    guard = SensorGuard()
    guard.register_profile(
        machine_id="fan_00",
        sensor_id="mic_array_primary",
        channel_index=0,
        profile_data={"threshold": 0.95, "whitened": True},
    )

    # Compatible sensor
    compat, msg = guard.validate_sensor_compatibility(
        machine_id="fan_00",
        incoming_sensor_id="mic_array_primary",
        incoming_channel=0,
        active_sensor_id="mic_array_primary",
        active_channel=0,
    )
    assert compat
    assert msg == "SENSOR_COMPATIBLE"

    # Mismatched sensor
    compat_bad, msg_bad = guard.validate_sensor_compatibility(
        machine_id="fan_00",
        incoming_sensor_id="mic_array_secondary",
        incoming_channel=0,
        active_sensor_id="mic_array_primary",
        active_channel=0,
    )
    assert not compat_bad
    assert "SENSOR_MISMATCH" in msg_bad

    # Mismatched channel
    compat_ch, msg_ch = guard.validate_sensor_compatibility(
        machine_id="fan_00",
        incoming_sensor_id="mic_array_primary",
        incoming_channel=4,
        active_sensor_id="mic_array_primary",
        active_channel=0,
    )
    assert not compat_ch
    assert "CHANNEL_MISMATCH" in msg_ch

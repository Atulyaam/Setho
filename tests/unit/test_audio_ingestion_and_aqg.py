"""Unit tests for Audio Ingestion Service and Audio Quality Gate (AQG)."""

import io
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from stetho_api.audio import (
    AudioIngestionService,
    AudioQualityGate,
    AudioValidationError,
    QualityStatus,
)


@pytest.fixture
def ingestion_service(tmp_path):
    return AudioIngestionService(storage_dir=tmp_path / "storage")


@pytest.fixture
def quality_gate():
    return AudioQualityGate()


def create_synthetic_wav_bytes(
    duration_sec: float = 10.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
    channels: int = 1,
) -> bytes:
    """Helper to generate in-memory valid WAV file bytes."""
    samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, samples, endpoint=False)
    freq = 440.0
    waveform = amplitude * np.sin(2 * np.pi * freq * t).astype(np.float32)
    if channels > 1:
        waveform = np.tile(waveform[:, np.newaxis], (1, channels))

    buf = io.BytesIO()
    sf.write(buf, waveform, sample_rate, format="WAV")
    return buf.getvalue()


# --- Task 4.3: Audio Ingestion Tests ---

def test_ingest_valid_mono_wav(ingestion_service):
    """Verify clean mono WAV file ingestion succeeds and persists under UUID."""
    wav_bytes = create_synthetic_wav_bytes(duration_sec=5.0)
    meta, data = ingestion_service.ingest_audio(
        file_obj=io.BytesIO(wav_bytes),
        filename="custom_machine_recording.wav",
        machine_id="id_00",
    )
    assert meta.upload_id is not None
    assert meta.channels == 1
    assert meta.duration_sec >= 4.9
    assert Path(meta.stored_path).exists()
    assert meta.sanitized_filename.endswith(".wav")
    assert meta.sanitized_filename != "custom_machine_recording.wav"  # Non-user filename


def test_ingest_valid_multichannel_wav(ingestion_service):
    """Verify multichannel 8-channel WAV (MIMII format) ingests properly."""
    wav_bytes = create_synthetic_wav_bytes(duration_sec=5.0, channels=8)
    meta, data = ingestion_service.ingest_audio(
        file_obj=io.BytesIO(wav_bytes),
        filename="mimii_8ch.wav",
        machine_id="id_04",
    )
    assert meta.channels == 8
    assert data.shape[1] == 8


def test_reject_unsupported_extension(ingestion_service):
    """Verify non-WAV extension is rejected."""
    with pytest.raises(AudioValidationError) as exc:
        ingestion_service.ingest_audio(
            file_obj=io.BytesIO(b"fake data"),
            filename="recording.mp3",
        )
    assert "Unsupported file extension" in str(exc.value)


def test_reject_empty_file(ingestion_service):
    """Verify 0-byte file is rejected."""
    with pytest.raises(AudioValidationError) as exc:
        ingestion_service.ingest_audio(
            file_obj=io.BytesIO(b""),
            filename="empty.wav",
        )
    assert "empty" in str(exc.value)


def test_reject_invalid_magic_bytes(ingestion_service):
    """Verify corrupted/fake WAV header is rejected."""
    fake_bytes = b"NOT_A_RIFF_HEADER_1234567890"
    with pytest.raises(AudioValidationError) as exc:
        ingestion_service.ingest_audio(
            file_obj=io.BytesIO(fake_bytes),
            filename="fake.wav",
        )
    assert "header" in str(exc.value).lower()


def test_reject_too_short_duration(ingestion_service):
    """Verify files under 1.0s duration are rejected."""
    short_bytes = create_synthetic_wav_bytes(duration_sec=0.4)
    with pytest.raises(AudioValidationError) as exc:
        ingestion_service.ingest_audio(
            file_obj=io.BytesIO(short_bytes),
            filename="too_short.wav",
        )
    assert "shorter than minimum" in str(exc.value)


def test_reject_too_long_duration(ingestion_service):
    """Verify files over 30.0s duration are rejected."""
    long_bytes = create_synthetic_wav_bytes(duration_sec=32.0)
    with pytest.raises(AudioValidationError) as exc:
        ingestion_service.ingest_audio(
            file_obj=io.BytesIO(long_bytes),
            filename="too_long.wav",
        )
    assert "exceeds maximum" in str(exc.value)


# --- Task 4.4: Audio Quality Gate Tests ---

def test_aqg_clean_machine_audio(quality_gate):
    """Verify high-fidelity normal audio is ACCEPTED."""
    # 10s audio with clear signal and low noise
    t = np.linspace(0, 10.0, 160000, endpoint=False)
    sig = (0.2 * np.sin(2 * np.pi * 300 * t) + 0.01 * np.random.randn(len(t))).astype(np.float32)

    report = quality_gate.assess_waveform(sig, sample_rate=16000)
    assert report.status == QualityStatus.ACCEPTED
    assert report.is_usable_for_inference is True
    assert report.metrics["clipping_percentage"] == 0.0
    assert report.metrics["duration_sec"] == 10.0


def test_aqg_silence_rejected(quality_gate):
    """Verify silent audio is REJECTED and not marked as machine anomaly."""
    silent_sig = np.zeros(160000, dtype=np.float32)
    report = quality_gate.assess_waveform(silent_sig, sample_rate=16000)
    assert report.status == QualityStatus.REJECTED
    assert report.is_usable_for_inference is False
    assert any("silent" in r.lower() for r in report.reasons)


def test_aqg_severe_clipping_rejected(quality_gate):
    """Verify severely clipped recording (>5% at rail) is REJECTED."""
    sig = np.random.randn(160000).astype(np.float32) * 5.0
    sig = np.clip(sig, -1.0, 1.0)  # Heavy digital saturation

    report = quality_gate.assess_waveform(sig, sample_rate=16000)
    assert report.status == QualityStatus.REJECTED
    assert report.is_usable_for_inference is False
    assert any("severe digital clipping" in r.lower() for r in report.reasons)


def test_aqg_minor_clipping_warning(quality_gate):
    """Verify minor occasional clipping yields WARNING, remaining usable."""
    t = np.linspace(0, 10.0, 160000, endpoint=False)
    sig = (0.3 * np.sin(2 * np.pi * 300 * t) + 0.01 * np.random.randn(len(t))).astype(np.float32)
    # Clip only 200 samples (~0.125%)
    sig[:200] = 1.0

    report = quality_gate.assess_waveform(sig, sample_rate=16000)
    assert report.status == QualityStatus.WARNING
    assert report.is_usable_for_inference is True
    assert any("minor digital clipping" in r.lower() for r in report.reasons)


def test_aqg_excessive_ambient_noise_inconclusive(quality_gate):
    """Verify pure broadband white noise without machine tones yields INCONCLUSIVE."""
    noise = (0.1 * np.random.randn(160000)).astype(np.float32)
    report = quality_gate.assess_waveform(noise, sample_rate=16000)
    assert report.status == QualityStatus.INCONCLUSIVE
    assert report.is_usable_for_inference is False
    assert any("signal-to-noise ratio" in r.lower() for r in report.reasons)


def test_aqg_insufficient_duration_rejected(quality_gate):
    """Verify recording under 2.0s is REJECTED by quality gate."""
    short_sig = 0.2 * np.random.randn(24000).astype(np.float32)  # 1.5s
    report = quality_gate.assess_waveform(short_sig, sample_rate=16000)
    assert report.status == QualityStatus.REJECTED
    assert report.is_usable_for_inference is False
    assert any("too short" in r.lower() for r in report.reasons)


def test_aqg_nan_inf_rejected(quality_gate):
    """Verify NaN/Inf corrupted waveforms are safely REJECTED."""
    sig = 0.2 * np.random.randn(160000).astype(np.float32)
    sig[500] = np.nan
    report = quality_gate.assess_waveform(sig, sample_rate=16000)
    assert report.status == QualityStatus.REJECTED
    assert report.is_usable_for_inference is False
    assert any("non-finite" in r.lower() for r in report.reasons)

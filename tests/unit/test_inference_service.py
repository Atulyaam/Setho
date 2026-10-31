"""Unit and integration tests for AcousticInferenceService."""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from stetho_api.audio import AudioQualityGate
from stetho_api.inference import (
    AcousticInferenceService,
    DecisionZone,
    InferenceResult,
    ModelRegistry,
    ModelType,
)


@pytest.fixture
def inference_service():
    registry = ModelRegistry(artifacts_root="./ml/artifacts", device="cpu", fallback_to_cpu=True)
    return AcousticInferenceService(registry=registry)


def test_inference_real_mimii_normal_audio(inference_service):
    """Verify inference on actual MIMII normal test recording for id_00."""
    wav_path = Path("data/mimii/raw/fan/id_00/normal/00000000.wav")
    assert wav_path.exists(), "MIMII id_00 normal test file must exist"

    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    result = inference_service.analyze_audio(
        audio=audio,
        sample_rate=sr,
        machine_id="id_00",
        model_override=ModelType.STETHO_MSAE,
    )

    assert isinstance(result, InferenceResult)
    assert result.machine_id == "id_00"
    assert result.model_name == "stetho_msae"
    assert result.anomaly_score is not None
    assert result.anomaly_score > 0.0
    assert result.operating_threshold is not None
    assert result.decision in [DecisionZone.NORMAL_LIKE, DecisionZone.INCONCLUSIVE]
    assert result.is_provisional is True
    assert result.processing_duration_ms > 0.0
    assert "reasons" in result.to_dict()


def test_inference_real_mimii_abnormal_audio(inference_service):
    """Verify inference on actual MIMII abnormal recording for id_00."""
    wav_path = Path("data/mimii/raw/fan/id_00/abnormal/00000000.wav")
    assert wav_path.exists(), "MIMII id_00 abnormal file must exist"

    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    result = inference_service.analyze_audio(
        audio=audio,
        sample_rate=sr,
        machine_id="id_00",
        model_override=ModelType.STETHO_MSAE,
    )

    assert result.anomaly_score is not None
    assert result.operating_threshold is not None
    assert len(result.reasons) > 0


def test_inference_with_dense_autoencoder_override(inference_service):
    """Verify running inference with secondary Dense Autoencoder baseline."""
    wav_path = Path("data/mimii/raw/fan/id_00/normal/00000000.wav")
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)

    result = inference_service.analyze_audio(
        audio=audio,
        sample_rate=sr,
        machine_id="id_00",
        model_override=ModelType.DENSE_AUTOENCODER,
    )

    assert result.model_name == "dense_autoencoder"
    assert result.anomaly_score is not None
    assert result.metrics["model_parameters"] > 2_000_000


def test_inference_silence_rejected_without_fake_score(inference_service):
    """Verify silent audio is rejected by AQG and does not produce a fake anomaly score."""
    silent = np.zeros((160000, 1), dtype=np.float32)
    result = inference_service.analyze_audio(
        audio=silent,
        sample_rate=16000,
        machine_id="id_00",
    )

    assert result.decision == DecisionZone.INCONCLUSIVE
    assert result.anomaly_score is None  # Must NOT return misleading score
    assert any("silent" in r.lower() for r in result.reasons)


def test_inference_severe_clipping_rejected(inference_service):
    """Verify severely clipped audio is rejected without running model."""
    clipped = np.clip(np.random.randn(160000, 1).astype(np.float32) * 5.0, -1.0, 1.0)
    result = inference_service.analyze_audio(
        audio=clipped,
        sample_rate=16000,
        machine_id="id_00",
    )

    assert result.decision == DecisionZone.INCONCLUSIVE
    assert result.anomaly_score is None
    assert any("clipping" in r.lower() for r in result.reasons)


def test_inference_pure_noise_inconclusive(inference_service):
    """Verify broadband noise with low SNR yields INCONCLUSIVE decision."""
    noise = (0.1 * np.random.randn(160000, 1)).astype(np.float32)
    result = inference_service.analyze_audio(
        audio=noise,
        sample_rate=16000,
        machine_id="id_00",
    )

    assert result.decision == DecisionZone.INCONCLUSIVE
    assert result.anomaly_score is None
    assert any("signal-to-noise ratio" in r.lower() for r in result.reasons)


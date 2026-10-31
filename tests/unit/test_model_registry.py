"""Unit tests for ModelRegistry and Inference Loader."""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from stetho_api.exceptions import ModelNotReadyException, ResourceNotFoundException
from stetho_api.inference.model_registry import (
    LoadedModel,
    ModelMetadata,
    ModelRegistry,
    ModelType,
)


@pytest.fixture
def registry():
    return ModelRegistry(artifacts_root="./ml/artifacts", device="cpu", fallback_to_cpu=True)


def test_registry_initialization(registry):
    """Verify registry initializes with target CPU device."""
    assert registry.target_device == "cpu"
    status = registry.get_status()
    assert status["loaded_count"] == 0


def test_load_stetho_msae_id_00(registry):
    """Verify loading real Stetho-MSAE checkpoint for id_00."""
    loaded = registry.get_model("id_00", ModelType.STETHO_MSAE)
    assert isinstance(loaded, LoadedModel)
    assert loaded.model_type == ModelType.STETHO_MSAE
    assert loaded.machine_id == "id_00"
    assert loaded.metadata.total_parameters == 559873
    assert len(loaded.metadata.sha256_hash) == 64
    assert loaded.calibration.get("is_provisional") is True
    assert "theta_lower" in loaded.calibration
    assert "theta_upper" in loaded.calibration

    # Test scoring synthetic 10-second normal audio
    audio = np.random.randn(160000).astype(np.float32) * 0.05
    score, diag = loaded.score_waveform(audio)
    assert isinstance(score, float)
    assert score > 0.0
    assert "score" in diag
    assert "threshold" in diag


def test_load_dense_autoencoder_id_00(registry):
    """Verify loading real Dense Autoencoder checkpoint for id_00."""
    loaded = registry.get_model("id_00", ModelType.DENSE_AUTOENCODER)
    assert isinstance(loaded, LoadedModel)
    assert loaded.model_type == ModelType.DENSE_AUTOENCODER
    assert loaded.machine_id == "id_00"
    assert loaded.metadata.total_parameters > 2_000_000
    assert len(loaded.metadata.sha256_hash) == 64
    assert "operating_threshold" in loaded.calibration

    # Test scoring synthetic 10-second audio
    audio = np.random.randn(160000).astype(np.float32) * 0.05
    score, diag = loaded.score_waveform(audio)
    assert isinstance(score, float)
    assert score > 0.0


def test_registry_caches_loaded_models(registry):
    """Verify that multiple calls return the identical cached LoadedModel instance."""
    m1 = registry.get_model("id_02", ModelType.STETHO_MSAE)
    m2 = registry.get_model("id_02", ModelType.STETHO_MSAE)
    assert m1 is m2


def test_default_policy_selects_stetho_msae(registry):
    """Verify default model selection without explicit model_type chooses Stetho-MSAE."""
    loaded = registry.get_model("id_00")
    assert loaded.model_type == ModelType.STETHO_MSAE


def test_invalid_machine_id_raises_not_found(registry):
    """Verify querying an invalid machine raises ResourceNotFoundException."""
    with pytest.raises(ResourceNotFoundException):
        registry.get_model("id_nonexistent_machine_99")


def test_invalid_model_type_raises_not_found(registry):
    """Verify requesting an unsupported model type raises ResourceNotFoundException."""
    with pytest.raises(ResourceNotFoundException):
        registry.get_model("id_00", "unsupported_transformer_arch")


def test_corrupted_model_weights_fail_safely(tmp_path):
    """Verify corrupted checkpoint triggers ModelNotReadyException without crashing."""
    corrupted_root = tmp_path / "artifacts"
    model_dir = corrupted_root / "stetho_msae" / "id_00"
    model_dir.mkdir(parents=True)

    # Write truncated/garbage weights
    (model_dir / "model.pt").write_bytes(b"corrupted_invalid_pytorch_bytes")
    (model_dir / "config.json").write_text('{"machine_id": "id_00"}')
    (model_dir / "calibration.json").write_text('{"machine_id": "id_00", "operating_threshold": 0.5}')

    bad_reg = ModelRegistry(artifacts_root=corrupted_root, device="cpu")
    with pytest.raises(ModelNotReadyException):
        bad_reg.get_model("id_00", ModelType.STETHO_MSAE)

    # Status should report failed model
    status = bad_reg.get_status()
    assert status["failed_count"] == 1


def test_calibration_machine_mismatch_fails_safely(tmp_path):
    """Verify calibration file with mismatched machine_id is rejected."""
    mismatch_root = tmp_path / "artifacts"
    model_dir = mismatch_root / "stetho_msae" / "id_00"
    model_dir.mkdir(parents=True)

    (model_dir / "model.pt").write_bytes(b"dummy_weights")
    (model_dir / "config.json").write_text('{"machine_id": "id_00"}')
    # Write calibration belonging to id_02 instead of id_00
    (model_dir / "calibration.json").write_text('{"machine_id": "id_02", "operating_threshold": 0.5}')

    mismatch_reg = ModelRegistry(artifacts_root=mismatch_root, device="cpu")
    with pytest.raises(ModelNotReadyException) as exc_info:
        mismatch_reg.get_model("id_00", ModelType.STETHO_MSAE)
    assert "mismatch" in str(exc_info.value)


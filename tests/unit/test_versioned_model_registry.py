"""Unit tests for Versioned ModelRegistry in Sprint 7."""

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


def test_list_available_models(registry):
    """Verify registry lists all 6 model architectures with latency and profile metadata."""
    models = registry.list_available_models()
    assert len(models) == 6
    types = {m["type"] for m in models}
    assert ModelType.COMPACT_STUDENT_INT8.value in types
    assert ModelType.RESNET_AE.value in types
    assert ModelType.CASCADE_ENSEMBLE.value in types
    assert ModelType.MULTITASK_RESNET_AE.value in types
    assert ModelType.STETHO_MSAE.value in types
    assert ModelType.DENSE_AUTOENCODER.value in types

    # Verify edge student metadata
    student_meta = next(m for m in models if m["type"] == ModelType.COMPACT_STUDENT_INT8.value)
    assert student_meta["runtime_engine"] == "onnx_int8"
    assert student_meta["disk_size_kb"] < 50.0
    assert student_meta["latency_p50_cpu_ms"] < 50.0


def test_load_compact_student_int8(registry):
    """Verify loading real INT8 ONNX student model and scoring waveform."""
    loaded = registry.get_model("id_00", ModelType.COMPACT_STUDENT_INT8)
    assert isinstance(loaded, LoadedModel)
    assert loaded.metadata.runtime_engine == "onnx_int8"
    assert "universal" in loaded.metadata.supported_categories

    # Score waveform
    waveform = np.random.randn(160000).astype(np.float32) * 0.05
    score, diag = loaded.score_waveform(waveform)
    assert isinstance(score, float)
    assert score > 0.0
    assert diag["runtime_engine"] == "onnx_int8"
    assert diag["threshold"] == 0.37356


def test_load_resnet_ae(registry):
    """Verify loading Optimized ResNet-AE and forward scoring."""
    loaded = registry.get_model("id_00", ModelType.RESNET_AE)
    assert isinstance(loaded, LoadedModel)
    assert loaded.metadata.runtime_engine == "pytorch"
    assert loaded.metadata.total_parameters > 500_000

    waveform = np.random.randn(160000).astype(np.float32) * 0.05
    score, diag = loaded.score_waveform(waveform)
    assert isinstance(score, float)
    assert score > 0.0
    assert diag["threshold"] == 0.35208


def test_load_cascade_ensemble(registry):
    """Verify loading Margin-Gated Cascade and forward scoring."""
    loaded = registry.get_model("id_00", ModelType.CASCADE_ENSEMBLE)
    assert isinstance(loaded, LoadedModel)
    assert loaded.metadata.runtime_engine == "hybrid"

    waveform = np.random.randn(160000).astype(np.float32) * 0.05
    score, diag = loaded.score_waveform(waveform)
    assert isinstance(score, float)
    assert score > 0.0


def test_load_multitask_resnet_ae(registry):
    """Verify loading MultiTask ResNet-AE for zero-shot use cases."""
    loaded = registry.get_model("id_00", ModelType.MULTITASK_RESNET_AE)
    assert isinstance(loaded, LoadedModel)

    waveform = np.random.randn(160000).astype(np.float32) * 0.05
    score, diag = loaded.score_waveform(waveform)
    assert isinstance(score, float)
    assert score > 0.0


def test_model_unloading(registry):
    """Verify dynamic model unloading releases in-memory cache."""
    # Ensure loaded
    _ = registry.get_model("id_04", ModelType.COMPACT_STUDENT_INT8)
    assert registry.unload_model("id_04", ModelType.COMPACT_STUDENT_INT8) is True
    # Second unload returns False because already removed
    assert registry.unload_model("id_04", ModelType.COMPACT_STUDENT_INT8) is False


def test_cpu_fallback_mode():
    """Verify requested CUDA device gracefully falls back to CPU when CUDA is unavailable."""
    reg = ModelRegistry(artifacts_root="./ml/artifacts", device="cuda", fallback_to_cpu=True)
    if not torch.cuda.is_available():
        assert reg.target_device == "cpu"
    else:
        assert reg.target_device == "cuda"

"""Automated verification tests for Phase 8 ONNX Export and Inference Optimization."""

from pathlib import Path
import numpy as np
import pytest
import torch

from ml.models.research.distilled_student import CompactStudentAE
from ml.models.research.onnx_exporter import (
    ORTAnomalyDetector,
    export_to_onnx,
    quantize_to_int8,
    verify_numerical_parity,
)
from ml.models.research.residual_cnn_ae import ResidualCNNAE


def test_onnx_export_and_parity_student(tmp_path: Path):
    """Verify ONNX export and numerical parity on CompactStudentAE."""
    model = CompactStudentAE(in_channels=1, base_channels=8, latent_dim=32)
    dummy_input = torch.randn(2, 1, 64, 64)

    onnx_file = tmp_path / "student.onnx"
    export_to_onnx(model, dummy_input, onnx_file)
    assert onnx_file.exists()

    is_parity, max_diff = verify_numerical_parity(model, onnx_file, dummy_input, tol=1e-4)
    assert is_parity, f"Parity failed: max_diff={max_diff}"
    assert max_diff < 1e-4


def test_onnx_quantization(tmp_path: Path):
    """Verify dynamic INT8 quantization on ONNX model."""
    model = CompactStudentAE(in_channels=1, base_channels=8, latent_dim=32)
    dummy_input = torch.randn(2, 1, 64, 64)

    onnx_fp32 = tmp_path / "student_fp32.onnx"
    export_to_onnx(model, dummy_input, onnx_fp32)

    onnx_int8 = tmp_path / "student_int8.onnx"
    quantize_to_int8(onnx_fp32, onnx_int8)
    assert onnx_int8.exists()
    assert onnx_int8.stat().st_size > 0


def test_ort_detector_scoring(tmp_path: Path):
    """Verify ORTAnomalyDetector scores patches without PyTorch dependency."""
    model = CompactStudentAE(in_channels=1, base_channels=8, latent_dim=32)
    dummy_input = torch.randn(1, 1, 64, 64)
    onnx_file = tmp_path / "detector_test.onnx"
    export_to_onnx(model, dummy_input, onnx_file)

    detector = ORTAnomalyDetector(
        machine_id="id_00",
        onnx_model_path=onnx_file,
        train_mean=0.0,
        train_std=1.0,
        threshold=0.5,
    )

    dummy_patches = np.random.randn(4, 1, 64, 64).astype(np.float32)
    scores = detector.score_patches(dummy_patches)
    assert len(scores) == 4
    assert np.all(scores >= 0.0)


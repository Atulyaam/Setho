"""Automated verification tests for Phase 4 New Research Architectures."""

from pathlib import Path
import numpy as np
import pytest
import torch

from ml.models.research.multiscale_cnn_ae import MultiScaleCNNAE
from ml.models.research.residual_cnn_ae import ResidualCNNAE
from ml.models.research.cnn_gru_attention_ae import CNNGRUAttentionAE
from ml.models.research.embedding_mahalanobis import MahalanobisDetector
from ml.models.research.teacher_student_cnn import TeacherStudentModel


def test_multiscale_cnn_ae_forward_backward():
    """Verify MultiScaleCNNAE forward, backward, shapes, and parameter efficiency."""
    model = MultiScaleCNNAE(in_channels=1, base_channels=24, latent_dim=64)
    x = torch.randn(2, 1, 64, 64)

    recon, z = model(x)
    assert recon.shape == (2, 1, 64, 64), f"Recon shape mismatch: {recon.shape}"
    assert z.shape == (2, 64), f"Latent shape mismatch: {z.shape}"

    loss = torch.nn.functional.l1_loss(recon, x)
    loss.backward()

    # Check gradients
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Gradient missing for {name}"

    param_count = sum(p.numel() for p in model.parameters())
    assert param_count < 1_500_000, f"Model parameter count {param_count} too high"


def test_residual_cnn_ae_forward_backward():
    """Verify ResidualCNNAE forward, backward, shapes, and parameter efficiency."""
    model = ResidualCNNAE(in_channels=1, base_channels=16, latent_dim=64)
    x = torch.randn(2, 1, 64, 64)

    recon, z = model(x)
    assert recon.shape == (2, 1, 64, 64), f"Recon shape mismatch: {recon.shape}"
    assert z.shape == (2, 64), f"Latent shape mismatch: {z.shape}"

    loss = torch.nn.functional.mse_loss(recon, x)
    loss.backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Gradient missing for {name}"

    param_count = sum(p.numel() for p in model.parameters())
    assert param_count < 2_000_000, f"ResNet AE parameter count {param_count} too high"


def test_cnn_gru_attention_ae_forward_backward():
    """Verify CNNGRUAttentionAE forward, backward, and shapes."""
    model = CNNGRUAttentionAE(in_channels=1, freq_bins=64, time_steps=64, cnn_channels=32, gru_hidden_dim=64, latent_dim=64)
    x = torch.randn(2, 1, 64, 64)

    recon, z = model(x)
    assert recon.shape == (2, 1, 64, 64), f"Recon shape mismatch: {recon.shape}"
    assert z.shape == (2, 64), f"Latent shape mismatch: {z.shape}"

    loss = torch.nn.functional.l1_loss(recon, x)
    loss.backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Gradient missing for {name}"


def test_teacher_student_cnn_forward():
    """Verify TeacherStudentModel discrepancy and parameter freezing."""
    model = TeacherStudentModel(in_channels=1, base_channels=16, out_dim=64)
    x = torch.randn(2, 1, 64, 64)

    s_feat, t_feat, error = model(x)
    assert s_feat.shape == (2, 64)
    assert t_feat.shape == (2, 64)
    assert error.shape == (2,)

    # Teacher must have no gradients
    for p in model.teacher.parameters():
        assert not p.requires_grad

    # Student must have gradients
    loss = error.mean()
    loss.backward()
    for name, p in model.student.named_parameters():
        assert p.grad is not None, f"Student gradient missing for {name}"


def test_mahalanobis_detector_fit_and_score():
    """Verify Mahalanobis & k-NN detector fits and scores without errors."""
    from ml.training.train_pipeline import _read_split_csv
    train_rows = _read_split_csv("ml/data/splits/machine_specific/id_00/train_normal.csv")
    train_paths = [r["file_path"] for r in train_rows[:4]]

    det = MahalanobisDetector(machine_id="id_00", feature_preset="compact_stat_38d")
    fit_meta = det.fit(train_paths)
    assert fit_meta["n_samples"] == 4
    assert fit_meta["n_features"] == 38

    score = det.score_file(train_paths[0])
    assert isinstance(score, float)
    assert score >= 0.0

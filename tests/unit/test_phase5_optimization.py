"""Unit and integration tests for Phase 5 Optimization & Ablations."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from ml.models.research.multiscale_cnn_ae import MultiScaleCNNAE
from ml.models.research.research_detector import (
    FrequencyWeightedL1Loss,
    ResearchNeuralDetector,
    _apply_augmentation,
)
from ml.models.research.residual_cnn_ae import ResidualCNNAE


def test_frequency_weighted_loss():
    """Verify FrequencyWeightedL1Loss computes weighted gradients."""
    loss_fn = FrequencyWeightedL1Loss(freq_bins=64, slope=1.5)
    recon = torch.randn(2, 1, 64, 64, requires_grad=True)
    target = torch.randn(2, 1, 64, 64)

    loss = loss_fn(recon, target)
    assert loss.item() > 0.0
    loss.backward()
    assert recon.grad is not None
    # Higher frequency bins should have larger gradients on average
    grad_low = recon.grad[:, :, :10, :].abs().mean()
    grad_high = recon.grad[:, :, -10:, :].abs().mean()
    assert grad_high > grad_low, f"Expected grad_high > grad_low, got {grad_high} vs {grad_low}"


def test_normalization_variants():
    """Verify GroupNorm, BatchNorm, and LayerNorm across architectures."""
    for norm in ["group", "batch", "layer"]:
        m_res = ResidualCNNAE(in_channels=1, base_channels=16, latent_dim=64, norm_type=norm)
        m_msae = MultiScaleCNNAE(in_channels=1, base_channels=16, latent_dim=64, norm_type=norm)

        x = torch.randn(2, 1, 64, 64)
        r_res, z_res = m_res(x)
        r_msae, z_msae = m_msae(x)

        assert r_res.shape == (2, 1, 64, 64)
        assert r_msae.shape == (2, 1, 64, 64)
        assert z_res.shape == (2, 64)
        assert z_msae.shape == (2, 64)


def test_variable_patch_frames():
    """Verify variable temporal context lengths (64, 96, 128 frames)."""
    for patch_len in [64, 96, 128]:
        m_res = ResidualCNNAE(in_channels=1, base_channels=16, latent_dim=64, input_size=(64, patch_len))
        m_msae = MultiScaleCNNAE(in_channels=1, base_channels=16, latent_dim=64, input_size=(64, patch_len))

        x = torch.randn(2, 1, 64, patch_len)
        r_res, z_res = m_res(x)
        r_msae, z_msae = m_msae(x)

        assert r_res.shape == (2, 1, 64, patch_len)
        assert r_msae.shape == (2, 1, 64, patch_len)
        assert z_res.shape == (2, 64)
        assert z_msae.shape == (2, 64)


def test_augmentations():
    """Verify self-supervised normal audio augmentations."""
    x = torch.randn(4, 1, 64, 64)

    # Time shift
    x_shift = _apply_augmentation(x, "time_shift")
    assert x_shift.shape == x.shape

    # Noise injection
    x_noise = _apply_augmentation(x, "noise_injection")
    assert x_noise.shape == x.shape
    assert not torch.allclose(x, x_noise)

    # SpecAugment
    x_mask = _apply_augmentation(x, "spec_augment")
    assert x_mask.shape == x.shape
    assert (x_mask == 0.0).sum() > 0


def test_latent_dimension_sweep():
    """Verify latent dimensions 32, 64, 128."""
    for l_dim in [32, 64, 128]:
        m = ResidualCNNAE(in_channels=1, base_channels=16, latent_dim=l_dim)
        x = torch.randn(2, 1, 64, 64)
        recon, z = m(x)
        assert z.shape == (2, l_dim)
        assert recon.shape == (2, 1, 64, 64)


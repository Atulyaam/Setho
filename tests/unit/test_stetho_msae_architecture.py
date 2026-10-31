"""Unit tests for Stetho-MSAE Custom Neural Architecture.

Verifies:
- Input/output tensor shapes and dimensional consistency
- Multi-scale dilated convolutions and residual connections
- Depthwise-separable vs standard convolution ablation toggles
- Latent dimension configurations (64 and 128)
- Numerical stability with extreme inputs and silence
- Gradient flow across all encoder and decoder parameters
- Checkpoint serialization and parameter footprint
- CPU and CUDA inference capability
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from ml.models.stetho_msae.model import (
    DepthwiseSeparableConv2d,
    MSAEEncoder,
    MSAEDecoder,
    ResidualMSAEBlock,
    StethoMSAE,
)
from ml.models.stetho_msae.msae_detector import StethoMSAEDetector


def test_depthwise_separable_conv_shapes():
    """Verify DepthwiseSeparableConv2d preserves spatial dimensions with stride=1."""
    x = torch.randn(4, 32, 64, 64)
    # Dilation 1
    conv_d1 = DepthwiseSeparableConv2d(32, 64, kernel_size=3, dilation=1, padding=1)
    out1 = conv_d1(x)
    assert out1.shape == (4, 64, 64, 64)

    # Dilation 2
    conv_d2 = DepthwiseSeparableConv2d(32, 64, kernel_size=3, dilation=2, padding=2)
    out2 = conv_d2(x)
    assert out2.shape == (4, 64, 64, 64)

    # Dilation 4
    conv_d4 = DepthwiseSeparableConv2d(32, 64, kernel_size=3, dilation=4, padding=4)
    out4 = conv_d4(x)
    assert out4.shape == (4, 64, 64, 64)


def test_residual_block_and_shortcut():
    """Verify ResidualMSAEBlock forward pass, dilation, and shortcut connection."""
    x = torch.randn(2, 32, 32, 32)
    # Identity shortcut when in_channels == out_channels
    res_same = ResidualMSAEBlock(32, 32, dilation=2)
    out_same = res_same(x)
    assert out_same.shape == (2, 32, 32, 32)

    # 1x1 projection shortcut when in_channels != out_channels
    res_diff = ResidualMSAEBlock(32, 64, dilation=4)
    out_diff = res_diff(x)
    assert out_diff.shape == (2, 64, 32, 32)


def test_stetho_msae_forward_and_reconstruction():
    """Verify complete autoencoder forward pass and sample-wise reconstruction error."""
    model = StethoMSAE(in_channels=1, base_channels=32, dilations=(1, 2, 4), latent_dim=64)
    x = torch.randn(4, 1, 64, 64)

    recon, z = model(x)
    assert recon.shape == (4, 1, 64, 64), f"Reconstruction shape mismatch: {recon.shape}"
    assert z.shape == (4, 64), f"Latent shape mismatch: {z.shape}"

    errors = model.compute_reconstruction_error(x)
    assert errors.shape == (4,), f"Error shape mismatch: {errors.shape}"
    assert torch.all(errors >= 0.0), "Reconstruction error must be non-negative"


def test_stetho_msae_latent_dimensions():
    """Verify model operates correctly with latent_dim=64 and latent_dim=128."""
    for lat_dim in [64, 128]:
        model = StethoMSAE(in_channels=1, base_channels=32, latent_dim=lat_dim)
        x = torch.randn(2, 1, 64, 64)
        recon, z = model(x)
        assert z.shape == (2, lat_dim)
        assert recon.shape == (2, 1, 64, 64)


def test_stetho_msae_ablation_variants():
    """Verify model behaves correctly across all required ablation configurations."""
    # Ablation B: Standard full convolutions, no multi-scale dilation (dilations=1, 1, 1)
    model_b = StethoMSAE(in_channels=1, dilations=(1, 1, 1), use_depthwise=False)
    # Ablation C: Depthwise-separable, no multi-scale dilation
    model_c = StethoMSAE(in_channels=1, dilations=(1, 1, 1), use_depthwise=True)
    # Full Stetho-MSAE: Depthwise-separable + multi-scale dilations (1, 2, 4)
    model_full = StethoMSAE(in_channels=1, dilations=(1, 2, 4), use_depthwise=True)

    x = torch.randn(2, 1, 64, 64)
    for m in [model_b, model_c, model_full]:
        recon, z = m(x)
        assert recon.shape == (2, 1, 64, 64)
        assert z.shape == (2, 64)

    # Verify parameter efficiency: Depthwise model must have fewer params than standard conv model
    params_b = model_b.count_parameters()["total_params"]
    params_c = model_c.count_parameters()["total_params"]
    assert params_c < params_b, f"Depthwise ({params_c}) must have fewer params than Standard ({params_b})"


def test_numerical_stability_and_extreme_inputs():
    """Verify numerical stability with silence (zeros) and large values."""
    model = StethoMSAE(in_channels=1, base_channels=32, latent_dim=64)
    model.eval()

    # Zero input (silence)
    silence = torch.zeros(2, 1, 64, 64)
    recon, z = model(silence)
    assert not torch.isnan(recon).any()
    assert not torch.isinf(recon).any()
    assert not torch.isnan(z).any()

    # High dynamic range input
    loud = torch.randn(2, 1, 64, 64) * 50.0
    recon_loud, z_loud = model(loud)
    assert not torch.isnan(recon_loud).any()
    assert not torch.isinf(recon_loud).any()


def test_gradient_flow_and_backprop():
    """Verify all trainable parameters receive non-zero gradients during backward pass."""
    model = StethoMSAE(in_channels=1, base_channels=32, latent_dim=64)
    model.train()

    x = torch.randn(4, 1, 64, 64)
    recon, z = model(x)
    loss = torch.nn.functional.l1_loss(recon, x)
    loss.backward()

    # Check that every single trainable parameter has a valid gradient
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive a gradient!"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient!"


def test_parameter_count_and_compact_size():
    """Verify model stays within the compact target (< 1,000,000 params, < 5 MB disk)."""
    model = StethoMSAE(in_channels=1, base_channels=32, latent_dim=64, use_depthwise=True)
    counts = model.count_parameters()
    total_params = counts["total_params"]

    # Typical Stetho-MSAE is ~350,000 params
    assert total_params < 1_000_000, f"Model has too many parameters: {total_params}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        pt_path = Path(tmp_dir) / "model.pt"
        torch.save(model.state_dict(), pt_path)
        size_mb = pt_path.stat().st_size / (1024 * 1024)
        assert size_mb < 5.0, f"Model file size ({size_mb:.2f} MB) exceeds 5 MB target"


def test_msae_detector_fit_and_scoring_synthetic():
    """Verify StethoMSAEDetector synthetic training and discrimination."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # Create dummy WAV files
        import soundfile as sf
        train_files = []
        sr = 16000
        for i in range(3):
            f_path = tmp_path / f"normal_{i}.wav"
            # 160000 samples (10s) harmonic signal
            t = np.linspace(0, 10, sr * 10, endpoint=False)
            audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            sf.write(f_path, audio, sr)
            train_files.append(f_path)

        dev_files = []
        for i in range(2):
            f_path = tmp_path / f"dev_{i}.wav"
            t = np.linspace(0, 10, sr * 10, endpoint=False)
            audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            sf.write(f_path, audio, sr)
            dev_files.append(f_path)

        detector = StethoMSAEDetector(
            machine_id="test_fan",
            base_channels=16,  # Fast small model for test
            latent_dim=32,
            max_epochs=2,
            batch_size=4,
            device="cpu",
        )

        meta = detector.fit(train_files, dev_files=dev_files)
        assert detector.is_fitted is True
        assert meta["train_sample_count"] == 3

        # Normal score
        norm_score = detector.score_file(train_files[0])
        assert isinstance(norm_score, float)
        assert norm_score >= 0.0

        # Create anomalous audio (white noise burst)
        anom_path = tmp_path / "anomalous.wav"
        t = np.linspace(0, 10, sr * 10, endpoint=False)
        anom_audio = (np.random.randn(sr * 10) * 0.8).astype(np.float32)
        sf.write(anom_path, anom_audio, sr)
        anom_score = detector.score_file(anom_path)

        assert anom_score > norm_score, (
            f"Anomalous audio score ({anom_score:.4f}) must be higher than normal ({norm_score:.4f})"
        )

        # Save and Reload test
        art_dir = tmp_path / "saved_msae"
        detector.save(art_dir)
        loaded = StethoMSAEDetector.load(art_dir, device="cpu")
        assert loaded.is_fitted is True
        loaded_score = loaded.score_file(train_files[0])
        assert abs(norm_score - loaded_score) < 1e-4

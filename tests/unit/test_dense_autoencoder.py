"""Unit tests for Dense Autoencoder Neural Baseline Model."""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import soundfile as sf
import torch

from ml.models.baseline_autoencoder.autoencoder_detector import DenseAutoencoderDetector
from ml.models.baseline_autoencoder.model import DenseAutoencoder


def test_autoencoder_architecture_dimensions():
    batch_size = 4
    input_dim = 4096
    latent_dim = 32
    hidden_dims = [256, 64]

    model = DenseAutoencoder(input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
    model.eval()

    x = torch.randn(batch_size, input_dim)
    recon, z = model(x)

    assert recon.shape == (batch_size, input_dim)
    assert z.shape == (batch_size, latent_dim)

    # Test reconstruction error
    err = model.compute_reconstruction_error(x)
    assert err.shape == (batch_size,)
    assert (err >= 0.0).all()


@pytest.fixture
def synthetic_audio_files(tmp_path):
    sr = 16000
    t = np.linspace(0, 10.0, int(sr * 10.0), endpoint=False)

    train_files = []
    # 4 normal harmonic recordings
    for i in range(4):
        audio = (0.3 * np.sin(2 * np.pi * 300 * t) + 0.05 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)
        fpath = tmp_path / f"norm_{i}.wav"
        sf.write(str(fpath), np.repeat(audio[:, np.newaxis], 8, axis=1), sr)
        train_files.append(fpath)

    # 1 dev recording
    dev_audio = (0.3 * np.sin(2 * np.pi * 300 * t) + 0.05 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)
    dev_path = tmp_path / "dev_norm.wav"
    sf.write(str(dev_path), np.repeat(dev_audio[:, np.newaxis], 8, axis=1), sr)

    # 1 anomaly recording (high-frequency screech + broadband noise)
    anom_audio = (0.5 * np.sin(2 * np.pi * 3800 * t) + 0.3 * np.random.randn(len(t))).astype(np.float32)
    anom_path = tmp_path / "anom.wav"
    sf.write(str(anom_path), np.repeat(anom_audio[:, np.newaxis], 8, axis=1), sr)

    return train_files, [dev_path], anom_path


def test_autoencoder_training_and_discrimination(synthetic_audio_files, tmp_path):
    train_files, dev_files, anom_path = synthetic_audio_files

    detector = DenseAutoencoderDetector(
        machine_id="id_00",
        hidden_dims=[128, 64],
        latent_dim=16,
        max_epochs=10,
        batch_size=4,
        device="cpu",
        seed=42,
    )

    meta = detector.fit(train_files, dev_files=dev_files)
    assert detector.is_fitted
    assert meta["total_epochs"] > 0
    assert meta["best_val_loss"] < 2.0

    # Score normal dev vs anomaly
    score_normal = detector.score_file(dev_files[0])
    score_anom = detector.score_file(anom_path)

    assert not np.isnan(score_normal)
    assert not np.isnan(score_anom)
    # Anomaly must have higher reconstruction error
    assert score_anom > score_normal, f"Expected anomaly score {score_anom} > normal score {score_normal}"


def test_autoencoder_checkpoint_save_and_load(synthetic_audio_files, tmp_path):
    train_files, dev_files, anom_path = synthetic_audio_files

    detector = DenseAutoencoderDetector(
        machine_id="id_00",
        hidden_dims=[128, 64],
        latent_dim=16,
        max_epochs=5,
        batch_size=4,
        device="cpu",
    )
    detector.fit(train_files, dev_files=dev_files)
    detector.threshold = 0.85

    save_dir = tmp_path / "dae_checkpoint"
    detector.save(save_dir)

    loaded = DenseAutoencoderDetector.load(save_dir, device="cpu")
    assert loaded.is_fitted
    assert loaded.threshold == 0.85

    score1 = detector.score_file(anom_path)
    score2 = loaded.score_file(anom_path)
    assert np.isclose(score1, score2, atol=1e-4)


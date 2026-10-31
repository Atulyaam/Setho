"""Unit and Integration Tests for Stetho-MSAE Synthetic Robustness."""

import math
from pathlib import Path
import tempfile

import numpy as np
import pytest
import soundfile as sf
import torch

from ml.evaluation.robustness_testing import (
    add_gaussian_noise,
    apply_clipping,
    apply_gain_db,
    apply_impulse_clicks,
    apply_packet_drop,
    evaluate_detector_robustness,
)
from ml.models.stetho_msae.msae_detector import StethoMSAEDetector


@pytest.fixture
def dummy_audio_file():
    """Create a temporary clean 16kHz sine wave audio file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = Path(f.name)
    sr = 16000
    t = np.linspace(0, 10, sr * 10, endpoint=False)
    sig = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    sf.write(path, sig, sr)
    yield path
    if path.exists():
        path.unlink()


def test_perturbation_functions():
    """Verify perturbation functions produce expected shape and finite values."""
    sig = np.sin(np.linspace(0, 1, 16000)).astype(np.float32)

    noisy = add_gaussian_noise(sig, snr_db=10.0)
    assert noisy.shape == sig.shape
    assert np.all(np.isfinite(noisy))
    assert not np.array_equal(noisy, sig)

    scaled = apply_gain_db(sig, gain_db=6.0)
    assert scaled.shape == sig.shape
    assert np.all(np.isfinite(scaled))
    assert np.max(np.abs(scaled)) > np.max(np.abs(sig))

    dropped = apply_packet_drop(sig, drop_fraction=0.1)
    assert dropped.shape == sig.shape
    assert np.all(np.isfinite(dropped))

    clipped = apply_clipping(sig, clip_threshold=0.1)
    assert np.max(clipped) <= 0.1001
    assert np.min(clipped) >= -0.1001

    clicked = apply_impulse_clicks(sig, num_clicks=5)
    assert clicked.shape == sig.shape
    assert np.all(np.isfinite(clicked))


def test_detector_robustness_execution(dummy_audio_file):
    """Verify detector handles all perturbations without NaN/Inf or crashing."""
    detector = StethoMSAEDetector(
        machine_id="test_fan",
        base_channels=16,
        latent_dim=32,
        max_epochs=2,
        batch_size=4,
        device="cpu",
    )
    detector.fit([dummy_audio_file], dev_files=[dummy_audio_file])

    results = evaluate_detector_robustness(detector, dummy_audio_file)
    assert "perturbations" in results
    assert len(results["perturbations"]) >= 10

    # Ensure all scores are finite and non-negative
    for name, p_data in results["perturbations"].items():
        assert not math.isnan(p_data["anomaly_score"])
        assert not math.isinf(p_data["anomaly_score"])
        assert p_data["anomaly_score"] >= 0.0

    # Monotonicity check: SNR 0 dB noise should produce higher score than SNR 20 dB noise
    score_20db = results["perturbations"]["gaussian_noise_20db"]["anomaly_score"]
    score_0db = results["perturbations"]["gaussian_noise_0db"]["anomaly_score"]
    assert score_0db > score_20db, "Severe noise (0 dB) must score higher than light noise (20 dB)"

    # Silence check
    silence_score = results["perturbations"]["pure_silence"]["anomaly_score"]
    assert silence_score >= 0.0


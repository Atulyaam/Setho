"""Unit tests for Audio Preprocessing Pipeline."""

import numpy as np
import pytest

from stetho_audio.channel_policy import ChannelPolicy
from stetho_audio.preprocessing import AudioPreprocessor, PreprocessConfig


def test_preprocessing_mono_conversion():
    preprocessor = AudioPreprocessor(PreprocessConfig(channel_policy=ChannelPolicy.CHANNEL_0))
    # 8-channel input
    multichannel = np.random.randn(16000, 8).astype(np.float32) * 0.1
    mono, meta = preprocessor.process_array(multichannel, sample_rate=16000)

    assert mono.ndim == 1
    assert len(mono) == 16000
    assert meta["original_channels"] == 8
    assert not meta["resampled"]
    assert np.allclose(mono, multichannel[:, 0])


def test_preprocessing_resampling():
    preprocessor = AudioPreprocessor(PreprocessConfig(target_sample_rate=16000))
    # Input at 8000 Hz, 1 second = 8000 samples
    t = np.linspace(0, 1.0, 8000, endpoint=False)
    sig_8k = 0.5 * np.sin(2 * np.pi * 300 * t).astype(np.float32)

    mono, meta = preprocessor.process_array(sig_8k, sample_rate=8000)
    assert meta["resampled"]
    assert meta["original_sample_rate"] == 8000
    assert meta["target_sample_rate"] == 16000
    # Length should be approx 16000
    assert abs(len(mono) - 16000) <= 2


def test_preprocessing_peak_normalization():
    preprocessor = AudioPreprocessor(
        PreprocessConfig(normalization="peak", target_peak=0.8)
    )
    # Low amplitude input
    sig = np.random.randn(16000).astype(np.float32) * 0.05
    mono, meta = preprocessor.process_array(sig, sample_rate=16000)

    assert np.isclose(np.max(np.abs(mono)), 0.8, atol=1e-3)
    assert meta["normalization"] == "peak"


def test_preprocessing_rms_normalization():
    preprocessor = AudioPreprocessor(
        PreprocessConfig(normalization="rms", target_rms=0.1)
    )
    sig = np.random.randn(16000).astype(np.float32) * 0.3
    mono, meta = preprocessor.process_array(sig, sample_rate=16000)

    actual_rms = np.sqrt(np.mean(mono**2))
    assert np.isclose(actual_rms, 0.1, atol=1e-3)
    assert meta["normalization"] == "rms"


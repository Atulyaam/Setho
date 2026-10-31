"""Unit tests for Stetho Log-Mel Feature Extraction."""

import numpy as np
import pytest
import torch

from ml.preprocessing.log_mel import LogMelConfig, LogMelExtractor


def test_log_mel_dimensions():
    cfg = LogMelConfig(sample_rate=16000, n_fft=1024, hop_length=256, n_mels=64)
    extractor = LogMelExtractor(cfg)

    # 10 second audio clip = 160,000 samples
    wav_np = np.random.randn(160000).astype(np.float32) * 0.1
    spec_np, meta = extractor.extract_numpy(wav_np)

    # Expected output: (1, 64, frames)
    # Expected frames for 160000 samples with hop 256: 160000 // 256 + 1 = 626 frames
    assert spec_np.ndim == 3
    assert spec_np.shape[0] == 1
    assert spec_np.shape[1] == 64
    assert spec_np.shape[2] == 626
    assert meta["n_mels"] == 64
    assert meta["hop_length"] == 256


def test_log_mel_torch_tensor_batch():
    cfg = LogMelConfig(sample_rate=16000, n_mels=64)
    extractor = LogMelExtractor(cfg)

    # Batch of 4 recordings: shape (4, 16000) (1-second test)
    wav_tensor = torch.randn(4, 16000, dtype=torch.float32) * 0.1
    output = extractor(wav_tensor)

    # Output shape: (4, 1, 64, 63)
    assert output.dim() == 4
    assert output.shape[0] == 4
    assert output.shape[1] == 1
    assert output.shape[2] == 64
    assert output.shape[3] == 63


def test_log_mel_numerical_stability():
    cfg = LogMelConfig(sample_rate=16000, n_mels=64, eps=1e-10)
    extractor = LogMelExtractor(cfg)

    # Test with pure silence (zeros) - ensure no -Inf or NaN occurs due to eps clamp
    silent_wav = np.zeros(16000, dtype=np.float32)
    spec_silent, meta = extractor.extract_numpy(silent_wav)

    assert not np.isnan(spec_silent).any()
    assert not np.isinf(spec_silent).any()
    # decibel value of eps = 1e-10 is 10 * log10(1e-10) = -100.0 dB
    assert np.allclose(spec_silent, -100.0, atol=1e-1)


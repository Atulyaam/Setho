"""Unit tests for Temporal Audio and Spectrogram Windowing."""

import numpy as np
import pytest
import torch

from ml.preprocessing.windowing import AudioWindowing, WindowConfig


def test_spectrogram_windowing_shape():
    cfg = WindowConfig(spec_window_frames=64, spec_hop_frames=32)
    win_tool = AudioWindowing(cfg)

    # Spectrogram with 64 mel bins and 626 time frames (typical 10s recording)
    spec = np.random.randn(64, 626).astype(np.float32)
    windows, ranges = win_tool.slice_spectrogram_numpy(spec)

    # Windows should have shape (num_windows, 64, 64)
    assert windows.ndim == 3
    assert windows.shape[1] == 64
    assert windows.shape[2] == 64

    # Hop is 32. 0:64, 32:96, ...
    # (626 - 64) // 32 + 1 = 562 // 32 + 1 = 17 + 1 = 18 full windows, plus 1 padded window = 19
    assert windows.shape[0] == 19
    assert len(ranges) == 19


def test_spectrogram_windowing_torch():
    cfg = WindowConfig(spec_window_frames=64, spec_hop_frames=32)
    win_tool = AudioWindowing(cfg)

    spec_tensor = torch.randn(1, 64, 626)
    windows_tensor, ranges = win_tool.slice_spectrogram_torch(spec_tensor)

    assert windows_tensor.shape == (19, 1, 64, 64)
    assert len(ranges) == 19


def test_short_spectrogram_padding():
    cfg = WindowConfig(spec_window_frames=64, spec_hop_frames=32)
    win_tool = AudioWindowing(cfg)

    # Very short spectrogram: 20 frames
    short_spec = np.random.randn(64, 20).astype(np.float32)
    windows, ranges = win_tool.slice_spectrogram_numpy(short_spec)

    assert windows.shape == (1, 64, 64)
    assert ranges == [(0, 20)]


def test_waveform_windowing():
    cfg = WindowConfig(wav_window_samples=16384, wav_hop_samples=8192)
    win_tool = AudioWindowing(cfg)

    # 10s audio at 16 kHz = 160,000 samples
    wav = np.random.randn(160000).astype(np.float32)
    windows, ranges = win_tool.slice_waveform(wav)

    assert windows.ndim == 2
    assert windows.shape[1] == 16384
    # (160000 - 16384) // 8192 + 1 = 143616 // 8192 + 1 = 17 + 1 = 18 full + 1 padded = 19
    assert windows.shape[0] == 19

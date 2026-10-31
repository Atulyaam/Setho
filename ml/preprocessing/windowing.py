"""Audio and Spectrogram Windowing Utility for Stetho.

Slices waveforms and log-Mel spectrograms into fixed-length analysis windows:
- Stetho-MSAE target window: 64 Mel bins × 64 time frames (~1.024s at 16ms/frame)
- 50% window overlap (hop = 32 frames / ~0.512s)
- Waveform windowing: 16,384 samples with 8,192 hop (or configurable)
- Strict partition isolation: windows are generated strictly per-recording;
  cross-file contamination is impossible.
"""

from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class WindowConfig:
    """Configuration for temporal window slicing."""
    # Spectrogram domain parameters
    spec_window_frames: int = 64     # 64 time frames
    spec_hop_frames: int = 32        # 50% overlap (32 frames)

    # Waveform domain parameters
    sample_rate: int = 16000
    wav_window_samples: int = 16384  # ~1.024s (exact power of 2 for FFT)
    wav_hop_samples: int = 8192      # 50% overlap (~0.512s)

    drop_last: bool = False          # If False, pads the final window with reflection/zeros


class AudioWindowing:
    """Windowing engine for waveforms and spectrograms."""

    def __init__(self, config: Optional[WindowConfig] = None):
        self.config = config or WindowConfig()

    def slice_spectrogram_numpy(
        self,
        spectrogram: np.ndarray,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Slice a 2D or 3D spectrogram into 64-frame windows.

        Args:
            spectrogram: Array of shape (n_mels, total_frames) or (channels, n_mels, total_frames).

        Returns:
            windows: Array of shape (num_windows, [channels,] n_mels, window_frames).
            frame_ranges: List of (start_frame, end_frame) tuples.
        """
        is_channel_dim = spectrogram.ndim == 3
        if is_channel_dim:
            channels, n_mels, total_frames = spectrogram.shape
        elif spectrogram.ndim == 2:
            n_mels, total_frames = spectrogram.shape
        else:
            raise ValueError(f"Spectrogram must be 2D or 3D, got ndim={spectrogram.ndim}")

        win_frames = self.config.spec_window_frames
        hop_frames = self.config.spec_hop_frames

        if total_frames < win_frames:
            # Pad if shorter than a single window
            pad_len = win_frames - total_frames
            pad_spec = (
                np.pad(spectrogram, ((0, 0), (0, 0), (0, pad_len)), mode="edge")
                if is_channel_dim
                else np.pad(spectrogram, ((0, 0), (0, pad_len)), mode="edge")
            )
            return (
                pad_spec[np.newaxis, ...],
                [(0, total_frames)],
            )

        windows: List[np.ndarray] = []
        frame_ranges: List[Tuple[int, int]] = []

        start = 0
        while start + win_frames <= total_frames:
            end = start + win_frames
            if is_channel_dim:
                windows.append(spectrogram[:, :, start:end])
            else:
                windows.append(spectrogram[:, start:end])
            frame_ranges.append((start, end))
            start += hop_frames

        # Handle remaining frames if not drop_last
        if not self.config.drop_last and start < total_frames:
            rem = total_frames - start
            if rem > 0:
                pad_len = win_frames - rem
                if is_channel_dim:
                    chunk = spectrogram[:, :, start:total_frames]
                    chunk_pad = np.pad(chunk, ((0, 0), (0, 0), (0, pad_len)), mode="edge")
                else:
                    chunk = spectrogram[:, start:total_frames]
                    chunk_pad = np.pad(chunk, ((0, 0), (0, pad_len)), mode="edge")
                windows.append(chunk_pad)
                frame_ranges.append((start, total_frames))

        return np.stack(windows, axis=0), frame_ranges

    def slice_spectrogram_torch(
        self,
        spectrogram: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """PyTorch tensor equivalent of slice_spectrogram_numpy.

        Args:
            spectrogram: Tensor of shape (1, n_mels, total_frames) or (n_mels, total_frames).

        Returns:
            windows: Tensor of shape (num_windows, 1, n_mels, window_frames)
            frame_ranges: List of (start_frame, end_frame).
        """
        orig_dim = spectrogram.dim()
        if orig_dim == 2:
            spec = spectrogram.unsqueeze(0)  # (1, n_mels, T)
        elif orig_dim == 3:
            spec = spectrogram
        else:
            raise ValueError(f"Tensor must be 2D or 3D, got {orig_dim}")

        channels, n_mels, total_frames = spec.shape
        win_frames = self.config.spec_window_frames
        hop_frames = self.config.spec_hop_frames

        windows: List[torch.Tensor] = []
        frame_ranges: List[Tuple[int, int]] = []

        start = 0
        while start + win_frames <= total_frames:
            end = start + win_frames
            windows.append(spec[:, :, start:end])
            frame_ranges.append((start, end))
            start += hop_frames

        if not self.config.drop_last and start < total_frames:
            rem = total_frames - start
            if rem > 0:
                chunk = spec[:, :, start:total_frames]
                chunk_pad = torch.nn.functional.pad(chunk, (0, win_frames - rem), mode="replicate")
                windows.append(chunk_pad)
                frame_ranges.append((start, total_frames))

        stacked = torch.stack(windows, dim=0)
        return stacked, frame_ranges

    def slice_waveform(
        self,
        waveform: np.ndarray,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Slice a 1D waveform array into temporal windows."""
        if waveform.ndim != 1:
            waveform = waveform.squeeze()

        num_samples = len(waveform)
        win_size = self.config.wav_window_samples
        hop_size = self.config.wav_hop_samples

        if num_samples < win_size:
            pad = np.pad(waveform, (0, win_size - num_samples), mode="edge")
            return pad[np.newaxis, :], [(0, num_samples)]

        windows: List[np.ndarray] = []
        ranges: List[Tuple[int, int]] = []

        start = 0
        while start + win_size <= num_samples:
            end = start + win_size
            windows.append(waveform[start:end])
            ranges.append((start, end))
            start += hop_size

        if not self.config.drop_last and start < num_samples:
            rem = num_samples - start
            if rem > 0:
                chunk = waveform[start:num_samples]
                chunk_pad = np.pad(chunk, (0, win_size - rem), mode="edge")
                windows.append(chunk_pad)
                ranges.append((start, num_samples))

        return np.stack(windows, axis=0), ranges


"""Channel Handling Policy Module for Stetho.

MIMII 1.0 recordings contain 8 microphone channels arranged in a circular array.
Stetho's target operating model uses a single acoustic sensor / contact probe.
This module provides flexible, deterministic policies to convert multichannel input
into single-channel audio suitable for the Stetho-MSAE pipeline.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class ChannelPolicy(str, Enum):
    """Channel selection and downmix policies."""
    CHANNEL_0 = "channel_0"          # Primary reference mic (Channel 0) - Stetho standard
    CHANNEL_INDEX = "channel_index"  # Arbitrary specific channel (0-indexed)
    MEAN_DOWNMIX = "mean_downmix"    # Mean average across all channels
    ENERGY_MAX = "energy_max"        # Channel with highest RMS energy


class ChannelHandler:
    """Applies channel policies to multichannel audio arrays."""

    @staticmethod
    def apply(
        audio: np.ndarray,
        policy: Union[ChannelPolicy, str] = ChannelPolicy.CHANNEL_0,
        channel_index: int = 0,
    ) -> np.ndarray:
        """Convert multichannel or mono audio to single-channel 1D array.

        Args:
            audio: Input audio array with shape (num_samples,) or (num_samples, num_channels)
                   or (num_channels, num_samples).
            policy: ChannelPolicy enum or string name.
            channel_index: Used when policy is CHANNEL_INDEX.

        Returns:
            1D numpy array of shape (num_samples,) and float32 dtype.
        """
        if isinstance(policy, str):
            policy = ChannelPolicy(policy)

        # Standardize shape to (num_samples, num_channels)
        if audio.ndim == 1:
            return audio.astype(np.float32)

        # Transpose if shape is (channels, samples) where channels <= 16 and samples > channels
        if audio.shape[0] <= 16 and audio.shape[1] > audio.shape[0]:
            audio = audio.T

        num_samples, num_channels = audio.shape

        if num_channels == 1:
            return audio[:, 0].astype(np.float32)

        if policy == ChannelPolicy.CHANNEL_0:
            return audio[:, 0].astype(np.float32)

        elif policy == ChannelPolicy.CHANNEL_INDEX:
            if channel_index < 0 or channel_index >= num_channels:
                raise ValueError(
                    f"Channel index {channel_index} out of range for {num_channels}-channel audio"
                )
            return audio[:, channel_index].astype(np.float32)

        elif policy == ChannelPolicy.MEAN_DOWNMIX:
            return np.mean(audio, axis=1, dtype=np.float32)

        elif policy == ChannelPolicy.ENERGY_MAX:
            rms_per_channel = np.sqrt(np.mean(audio**2, axis=0))
            best_ch = int(np.argmax(rms_per_channel))
            return audio[:, best_ch].astype(np.float32)

        else:
            raise ValueError(f"Unknown channel policy: {policy}")

    @staticmethod
    def compare_policies(audio: np.ndarray) -> Dict[str, Any]:
        """Compute signal metrics across all available channels and policies."""
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        elif audio.shape[0] <= 16 and audio.shape[1] > audio.shape[0]:
            audio = audio.T

        num_samples, num_channels = audio.shape

        channel_metrics = []
        for ch in range(num_channels):
            sig = audio[:, ch]
            rms = float(np.sqrt(np.mean(sig**2)))
            peak = float(np.max(np.abs(sig)))
            crest_factor = round(peak / (rms + 1e-12), 4)
            channel_metrics.append({
                "channel": ch,
                "rms": round(rms, 6),
                "peak": round(peak, 6),
                "crest_factor": crest_factor,
            })

        # Mean downmix metrics
        downmix = np.mean(audio, axis=1)
        downmix_rms = float(np.sqrt(np.mean(downmix**2)))
        downmix_peak = float(np.max(np.abs(downmix)))

        # Cross-correlation with Channel 0
        ch0 = audio[:, 0]
        correlations = []
        ch0_norm = np.linalg.norm(ch0) + 1e-12
        for ch in range(num_channels):
            ch_sig = audio[:, ch]
            ch_norm = np.linalg.norm(ch_sig) + 1e-12
            corr = float(np.dot(ch0, ch_sig) / (ch0_norm * ch_norm))
            correlations.append(round(corr, 4))

        return {
            "num_channels": num_channels,
            "num_samples": num_samples,
            "channel_metrics": channel_metrics,
            "downmix_metrics": {
                "rms": round(downmix_rms, 6),
                "peak": round(downmix_peak, 6),
                "crest_factor": round(downmix_peak / (downmix_rms + 1e-12), 4),
            },
            "correlation_with_channel_0": correlations,
        }

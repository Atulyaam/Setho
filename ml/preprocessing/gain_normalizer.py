"""Controlled Gain Normalization and Amplitude Telemetry for Stetho.

Standardizes acoustic energy while preserving raw amplitude metrics as
independent diagnostic telemetry to detect energy surges and physical faults.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


@dataclass
class GainNormalizerConfig:
    """Configuration for controlled gain normalization."""

    target_rms_dbfs: float = -42.0  # Empirically validated on MIMII clean baseline
    max_gain_boost_db: float = 24.0  # Prevent blowing up quiet noise floors
    max_gain_attenuation_db: float = 30.0  # Attenuation limit for high amplitude
    silence_threshold_dbfs: float = -60.0  # Audio below this is treated as silence
    clipping_threshold: float = 0.999  # Sample value considered clipped


class GainNormalizer:
    """Applies controlled gain normalization to waveforms."""

    def __init__(self, config: Optional[GainNormalizerConfig] = None):
        self.config = config or GainNormalizerConfig()

    def measure_telemetry(self, waveform: np.ndarray) -> Dict[str, float]:
        """Measure raw amplitude and energy telemetry before normalization."""
        wav_flat = waveform.flatten()
        if len(wav_flat) == 0:
            return {
                "raw_rms_dbfs": -120.0,
                "raw_peak": 0.0,
                "crest_factor_db": 0.0,
                "clipping_ratio": 0.0,
            }

        rms_linear = float(np.sqrt(np.mean(wav_flat**2)))
        rms_dbfs = float(20.0 * np.log10(rms_linear + 1e-12))
        peak_amp = float(np.max(np.abs(wav_flat)))
        crest_factor_db = float(
            20.0 * np.log10((peak_amp + 1e-12) / (rms_linear + 1e-12))
        )
        clipping_ratio = float(
            np.mean(np.abs(wav_flat) >= self.config.clipping_threshold)
        )

        return {
            "raw_rms_dbfs": round(rms_dbfs, 2),
            "raw_peak": round(peak_amp, 4),
            "crest_factor_db": round(crest_factor_db, 2),
            "clipping_ratio": round(clipping_ratio, 6),
        }

    def normalize(
        self, waveform: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Normalize waveform to target RMS with bounding limits.

        Returns:
            Tuple of (normalized_waveform, telemetry_dict)
        """
        telemetry = self.measure_telemetry(waveform)
        raw_rms_dbfs = telemetry["raw_rms_dbfs"]

        # Check for silence or near-zero signal
        if raw_rms_dbfs <= self.config.silence_threshold_dbfs:
            telemetry["gain_applied_db"] = 0.0
            telemetry["normalization_status"] = "SILENCE_PRESERVED"
            telemetry["post_rms_dbfs"] = raw_rms_dbfs
            return waveform.copy(), telemetry

        # Calculate required gain adjustment
        needed_gain_db = self.config.target_rms_dbfs - raw_rms_dbfs

        # Clamp gain adjustment to safe bounds
        clamped_gain_db = float(
            np.clip(
                needed_gain_db,
                -self.config.max_gain_attenuation_db,
                self.config.max_gain_boost_db,
            )
        )

        # Scale waveform
        scale_factor = 10.0 ** (clamped_gain_db / 20.0)
        scaled_waveform = waveform * scale_factor

        # Guard against clipping post-gain
        clipped_count = int(np.sum(np.abs(scaled_waveform) > 1.0))
        post_waveform = np.clip(scaled_waveform, -1.0, 1.0).astype(np.float32)

        post_rms = float(np.sqrt(np.mean(post_waveform**2)))
        post_rms_dbfs = float(20.0 * np.log10(post_rms + 1e-12))

        telemetry["gain_applied_db"] = round(clamped_gain_db, 2)
        telemetry["normalization_status"] = "NORMALIZED"
        telemetry["post_rms_dbfs"] = round(post_rms_dbfs, 2)
        telemetry["post_clipped_samples"] = clipped_count

        return post_waveform, telemetry

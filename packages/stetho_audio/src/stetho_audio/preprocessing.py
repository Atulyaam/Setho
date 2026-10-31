"""Audio Preprocessing Pipeline for Stetho.

Standardizes raw audio recordings into model-ready single-channel float32 waveforms:
- Multi-channel to mono conversion via ChannelPolicy (Channel 0 default)
- Deterministic polyphase resampling to 16,000 Hz
- Amplitude normalization (peak, RMS, or none)
- Rigorous boundary and type checking
"""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy import signal
import soundfile as sf
import torch

from stetho_audio.channel_policy import ChannelHandler, ChannelPolicy
from stetho_audio.validation import AudioValidator, ValidationResult


@dataclass
class PreprocessConfig:
    """Configuration for audio preprocessing."""
    target_sample_rate: int = 16000
    channel_policy: ChannelPolicy = ChannelPolicy.CHANNEL_0
    channel_index: int = 0
    normalization: str = "none"  # "none", "peak", "rms"
    target_peak: float = 0.95
    target_rms: float = 0.1
    validate_input: bool = True


class AudioPreprocessor:
    """Preprocesses raw audio files and arrays into normalized 16 kHz float32 signals."""

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self.validator = AudioValidator()

    def process_file(
        self,
        file_path: Union[str, Path],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load, validate, and preprocess an audio file from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if self.config.validate_input:
            val_res = self.validator.validate_file(path)
            if not val_res.is_valid:
                raise ValueError(f"Input audio validation failed: {'; '.join(val_res.errors)}")

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return self.process_array(data, sample_rate=sr, source_name=path.name)

    def process_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        source_name: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preprocess an in-memory numpy array.

        Returns:
            processed_waveform: 1D float32 numpy array.
            metadata: Dict with signal statistics and transformations applied.
        """
        orig_samples = audio.shape[0] if audio.ndim >= 1 else 0
        orig_channels = audio.shape[1] if audio.ndim == 2 else 1

        # 1. Multi-channel handling -> mono 1D
        mono = ChannelHandler.apply(
            audio,
            policy=self.config.channel_policy,
            channel_index=self.config.channel_index,
        )

        # 2. Resampling if required
        resampled = False
        if sample_rate != self.config.target_sample_rate:
            gcd = math.gcd(sample_rate, self.config.target_sample_rate)
            up = self.config.target_sample_rate // gcd
            down = sample_rate // gcd
            mono = signal.resample_poly(mono, up, down).astype(np.float32)
            resampled = True

        # 3. Amplitude normalization
        scale_factor = 1.0
        norm_type = self.config.normalization.lower()

        if norm_type == "peak":
            current_peak = float(np.max(np.abs(mono)))
            if current_peak > 1e-8:
                scale_factor = self.config.target_peak / current_peak
                mono = mono * scale_factor

        elif norm_type == "rms":
            current_rms = float(np.sqrt(np.mean(mono**2)))
            if current_rms > 1e-8:
                scale_factor = self.config.target_rms / current_rms
                mono = mono * scale_factor

        # Clip safeguards to strictly stay within [-1.0, 1.0]
        mono = np.clip(mono, -1.0, 1.0).astype(np.float32)

        final_peak = float(np.max(np.abs(mono)))
        final_rms = float(np.sqrt(np.mean(mono**2)))

        metadata = {
            "source_name": source_name,
            "original_sample_rate": sample_rate,
            "target_sample_rate": self.config.target_sample_rate,
            "original_channels": orig_channels,
            "channel_policy": str(self.config.channel_policy),
            "resampled": resampled,
            "normalization": norm_type,
            "scale_factor": round(scale_factor, 6),
            "num_samples": len(mono),
            "duration_sec": round(len(mono) / self.config.target_sample_rate, 4),
            "peak_amplitude": round(final_peak, 6),
            "rms_amplitude": round(final_rms, 6),
        }

        return mono, metadata


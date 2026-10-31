"""Single Log-Mel Spectrogram Feature Extraction for Stetho.

Standardized acoustic representation for Stetho-MSAE:
- Sample rate: 16,000 Hz
- FFT size: 1024 (64 ms Hann window)
- Hop size: 256 samples (16 ms step, 75% frame overlap)
- Mel filterbank: 64 bins
- Frequency range: 50 Hz to 8,000 Hz (Nyquist)
- Power spectrogram (|STFT|^2) with logarithmic decibel conversion
- Log clamp safeguard (eps = 1e-10) to prevent numerical divergence
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio.transforms as T


@dataclass
class LogMelConfig:
    """Log-Mel Spectrogram extraction configuration."""
    sample_rate: int = 16000
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    n_mels: int = 64
    f_min: float = 50.0
    f_max: float = 8000.0
    power: float = 2.0
    eps: float = 1e-10
    log_scale: str = "db"  # "db" (10*log10) or "natural" (ln)


class LogMelExtractor(torch.nn.Module):
    """PyTorch-native Log-Mel Spectrogram extractor."""

    def __init__(self, config: Optional[LogMelConfig] = None):
        super().__init__()
        self.config = config or LogMelConfig()

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            n_mels=self.config.n_mels,
            power=self.config.power,
            center=True,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract log-Mel spectrogram from waveform tensor.

        Args:
            waveform: Tensor of shape (time,), (channels, time), or (batch, channels, time).
                      Float32 normalized in [-1.0, 1.0].

        Returns:
            Log-Mel spectrogram tensor.
            If input is 1D (time,): returns (1, n_mels, frames)
            If input is 2D (batch, time): returns (batch, 1, n_mels, frames)
            If input is 3D (batch, channels, time): returns (batch, channels, n_mels, frames)
        """
        # Ensure tensor float32
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32)
        elif waveform.dtype != torch.float32:
            waveform = waveform.to(torch.float32)

        orig_dim = waveform.dim()
        if orig_dim == 1:
            # (time,) -> (1, 1, time)
            wav_input = waveform.unsqueeze(0).unsqueeze(0)
        elif orig_dim == 2:
            # (batch, time) -> (batch, 1, time)
            wav_input = waveform.unsqueeze(1)
        elif orig_dim == 3:
            wav_input = waveform
        else:
            raise ValueError(f"Unsupported waveform tensor dimension: {orig_dim}")

        # Compute power mel spectrogram: shape (batch, channels, n_mels, frames)
        mel_spec = self.mel_spectrogram(wav_input)

        # Apply safe log transformation
        clamped_mel = torch.clamp(mel_spec, min=self.config.eps)
        if self.config.log_scale == "db":
            # 10 * log10(x)
            log_mel = 10.0 * torch.log10(clamped_mel)
        elif self.config.log_scale == "natural":
            log_mel = torch.log(clamped_mel)
        else:
            raise ValueError(f"Unknown log_scale: {self.config.log_scale}")

        if orig_dim == 1:
            return log_mel.squeeze(0)  # (1, n_mels, frames)
        return log_mel

    def extract_numpy(
        self,
        waveform: np.ndarray,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract log-Mel spectrogram from numpy array."""
        self.to(device)
        tensor = torch.from_numpy(waveform.astype(np.float32)).to(device)
        with torch.no_grad():
            output_tensor = self.forward(tensor)

        output_np = output_tensor.cpu().numpy()
        meta = {
            "sample_rate": self.config.sample_rate,
            "n_fft": self.config.n_fft,
            "hop_length": self.config.hop_length,
            "n_mels": self.config.n_mels,
            "f_min": self.config.f_min,
            "f_max": self.config.f_max,
            "output_shape": list(output_np.shape),
            "log_scale": self.config.log_scale,
            "min_val": float(output_np.min()),
            "max_val": float(output_np.max()),
            "mean_val": float(output_np.mean()),
        }
        return output_np, meta

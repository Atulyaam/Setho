"""Compact Statistical Acoustic Feature Extraction for Stetho Baseline Models.

Computes time-averaged statistical descriptors from 16 kHz audio:
- 13 Mel-Frequency Cepstral Coefficients (MFCC) mean & std (26 features)
- Spectral Centroid mean & std (2 features)
- Spectral Bandwidth mean & std (2 features)
- Spectral Rolloff (85%) mean & std (2 features)
- Spectral Flux mean & std (2 features)
- Zero-Crossing Rate mean & std (2 features)
- RMS Energy mean & std (2 features)

Total compact feature vector: 38 dimensions per recording.
Includes train-only scaler fitting to prevent data leakage.
"""

from dataclasses import dataclass
import joblib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal
from scipy.fft import dct
from sklearn.preprocessing import StandardScaler
import soundfile as sf

from stetho_audio.channel_policy import ChannelHandler, ChannelPolicy
from stetho_audio.preprocessing import AudioPreprocessor, PreprocessConfig

logger = logging.getLogger("stetho_features")


@dataclass
class AcousticFeatureConfig:
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 26
    n_mfcc: int = 13
    f_min: float = 50.0
    f_max: float = 8000.0
    rolloff_percent: float = 0.85
    eps: float = 1e-10


class AcousticFeatureExtractor:
    """Extracts compact statistical acoustic features from waveforms."""

    def __init__(self, config: Optional[AcousticFeatureConfig] = None):
        self.config = config or AcousticFeatureConfig()
        self.preprocessor = AudioPreprocessor(
            PreprocessConfig(
                target_sample_rate=self.config.sample_rate,
                channel_policy=ChannelPolicy.CHANNEL_0,
            )
        )
        self._mel_basis = self._build_mel_basis()

    def _build_mel_basis(self) -> np.ndarray:
        """Construct linear triangular Mel filterbank matrix."""
        sr = self.config.sample_rate
        n_fft = self.config.n_fft
        n_mels = self.config.n_mels
        f_min = self.config.f_min
        f_max = self.config.f_max

        # Frequency to mel formula: 2595 * log10(1 + f / 700)
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        min_mel = hz_to_mel(f_min)
        max_mel = hz_to_mel(f_max)
        mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        num_bins = n_fft // 2 + 1
        filterbank = np.zeros((n_mels, num_bins), dtype=np.float32)

        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                if k < num_bins:
                    filterbank[m - 1, k] = (k - bin_points[m - 1]) / max(1, (bin_points[m] - bin_points[m - 1]))
            for k in range(f_m, f_m_plus):
                if k < num_bins:
                    filterbank[m - 1, k] = (bin_points[m + 1] - k) / max(1, (bin_points[m + 1] - bin_points[m]))

        return filterbank

    def extract_from_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """Extract 38-dimensional statistical acoustic feature vector from 1D audio.

        Args:
            waveform: 1D float32 array at target sample rate (16000 Hz).

        Returns:
            1D numpy array of shape (38,) with float32 values.
        """
        if waveform.ndim != 1:
            waveform = waveform.squeeze()

        # Handle pure silence or degenerate signal
        if len(waveform) == 0 or np.max(np.abs(waveform)) < 1e-9:
            return np.zeros(38, dtype=np.float32)

        # 1. Short-Time Fourier Transform (STFT)
        freqs, times, Zxx = signal.stft(
            waveform,
            fs=self.config.sample_rate,
            window="hann",
            nperseg=self.config.n_fft,
            noverlap=self.config.n_fft - self.config.hop_length,
            boundary="zeros",
        )
        mag_spec = np.abs(Zxx)  # (num_bins, num_frames)
        power_spec = mag_spec**2
        num_bins, num_frames = mag_spec.shape

        eps = self.config.eps

        # 2. MFCCs
        # Filterbank energies
        mel_energy = np.dot(self._mel_basis, power_spec)  # (n_mels, num_frames)
        mel_log = np.log(np.maximum(mel_energy, eps))
        mfccs = dct(mel_log, type=2, axis=0, norm="ortho")[: self.config.n_mfcc, :]  # (13, frames)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)

        # 3. Spectral Centroid
        total_mag = np.sum(mag_spec, axis=0) + eps
        centroid_frames = np.sum(freqs[:, np.newaxis] * mag_spec, axis=0) / total_mag
        centroid_mean = float(np.mean(centroid_frames))
        centroid_std = float(np.std(centroid_frames))

        # 4. Spectral Bandwidth
        spread_sq = np.sum(((freqs[:, np.newaxis] - centroid_frames) ** 2) * mag_spec, axis=0) / total_mag
        bandwidth_frames = np.sqrt(np.maximum(spread_sq, 0.0))
        bandwidth_mean = float(np.mean(bandwidth_frames))
        bandwidth_std = float(np.std(bandwidth_frames))

        # 5. Spectral Rolloff (85%)
        cum_energy = np.cumsum(mag_spec, axis=0)
        threshold = self.config.rolloff_percent * np.sum(mag_spec, axis=0)
        rolloff_bins = np.argmax(cum_energy >= threshold, axis=0)
        rolloff_frames = freqs[rolloff_bins]
        rolloff_mean = float(np.mean(rolloff_frames))
        rolloff_std = float(np.std(rolloff_frames))

        # 6. Spectral Flux
        norm_mag = mag_spec / (np.linalg.norm(mag_spec, axis=0, keepdims=True) + eps)
        flux_frames = np.sqrt(np.sum(np.diff(norm_mag, axis=1) ** 2, axis=0))
        if len(flux_frames) > 0:
            flux_mean = float(np.mean(flux_frames))
            flux_std = float(np.std(flux_frames))
        else:
            flux_mean, flux_std = 0.0, 0.0

        # 7. Zero Crossing Rate (framed)
        hop = self.config.hop_length
        win = self.config.n_fft
        n_zcr_frames = max(1, (len(waveform) - win) // hop + 1)
        zcr_list = []
        for i in range(n_zcr_frames):
            frame = waveform[i * hop : i * hop + win]
            zcr = float(np.mean(np.abs(np.diff(np.signbit(frame)))))
            zcr_list.append(zcr)
        zcr_mean = float(np.mean(zcr_list)) if zcr_list else 0.0
        zcr_std = float(np.std(zcr_list)) if zcr_list else 0.0

        # 8. RMS Energy (framed)
        rms_list = []
        for i in range(n_zcr_frames):
            frame = waveform[i * hop : i * hop + win]
            rms = float(np.sqrt(np.mean(frame**2)))
            rms_list.append(rms)
        rms_mean = float(np.mean(rms_list)) if rms_list else 0.0
        rms_std = float(np.std(rms_list)) if rms_list else 0.0

        features = np.concatenate([
            mfcc_means,               # 13
            mfcc_stds,                # 13
            [centroid_mean, centroid_std],    # 2
            [bandwidth_mean, bandwidth_std],  # 2
            [rolloff_mean, rolloff_std],      # 2
            [flux_mean, flux_std],            # 2
            [zcr_mean, zcr_std],              # 2
            [rms_mean, rms_std],              # 2
        ]).astype(np.float32)

        # Replace any potential residual NaN/Inf with 0.0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    def extract_from_file(self, file_path: Union[str, Path]) -> np.ndarray:
        """Process file through standard pipeline and extract feature vector."""
        wav, meta = self.preprocessor.process_file(file_path)
        return self.extract_from_waveform(wav)


class AcousticFeatureScaler:
    """Manages train-only feature normalization to strictly prevent leakage."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X_train: np.ndarray) -> "AcousticFeatureScaler":
        """Fit scaler exclusively on training normal features."""
        if X_train.ndim == 1:
            X_train = X_train.reshape(1, -1)
        self.scaler.fit(X_train)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply learned standardization to any dataset (train/dev/test)."""
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted on training data before transform")
        orig_1d = X.ndim == 1
        if orig_1d:
            X = X.reshape(1, -1)
        scaled = self.scaler.transform(X).astype(np.float32)
        return scaled.squeeze(0) if orig_1d else scaled

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AcousticFeatureScaler":
        inst = cls()
        inst.scaler = joblib.load(path)
        inst.is_fitted = True
        return inst


"""Unified Research Audio Feature Extraction Suite for Stetho.

Provides a unified interface to extract, compare, and benchmark diverse
acoustic representations:
- Single-channel (Ch0) & Multi-channel RMS log-Mel
- Standard Log-Mel (64 bins, N_fft=1024)
- High-Resolution Log-Mel (96 bins, N_fft=1024)
- Multi-scale STFT windows (512, 1024, 2048)
- 38-dim Compact MFCC & Spectral statistical descriptors
- Wavelet decomposition / Multi-resolution filterbank representations
"""

from dataclasses import dataclass
import io
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import torchaudio.transforms as T

from ml.features.acoustic_features import AcousticFeatureConfig, AcousticFeatureExtractor
from ml.preprocessing.log_mel import LogMelConfig, LogMelExtractor
from ml.preprocessing.windowing import AudioWindowing, WindowConfig
from stetho_audio.channel_policy import ChannelPolicy
from stetho_audio.preprocessing import AudioPreprocessor, PreprocessConfig

logger = logging.getLogger("stetho_feature_suite")


@dataclass
class PreprocessingPreset:
    name: str
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 64
    f_min: float = 50.0
    f_max: float = 8000.0
    channel_mode: str = "ch0"  # "ch0" or "multichannel_avg"
    patch_time_frames: int = 64
    patch_hop_frames: int = 32


class UnifiedFeatureSuite:
    """Unified acoustic extraction engine supporting all research representation variants."""

    PRESETS = {
        "standard_64mel": PreprocessingPreset("standard_64mel", n_fft=1024, hop_length=256, n_mels=64),
        "highres_96mel": PreprocessingPreset("highres_96mel", n_fft=1024, hop_length=256, n_mels=96),
        "fast_512fft": PreprocessingPreset("fast_512fft", n_fft=512, hop_length=128, n_mels=64),
        "wide_2048fft": PreprocessingPreset("wide_2048fft", n_fft=2048, hop_length=512, n_mels=64),
        "multichannel_avg": PreprocessingPreset("multichannel_avg", channel_mode="multichannel_avg"),
    }

    def __init__(self, preset_name: str = "standard_64mel"):
        self.preset = self.PRESETS.get(preset_name, self.PRESETS["standard_64mel"])
        self.preprocessor_ch0 = AudioPreprocessor(
            PreprocessConfig(
                target_sample_rate=self.preset.sample_rate,
                channel_policy=ChannelPolicy.CHANNEL_0,
                normalization="peak",
            )
        )
        self.mel_config = LogMelConfig(
            sample_rate=self.preset.sample_rate,
            n_fft=self.preset.n_fft,
            win_length=self.preset.n_fft,
            hop_length=self.preset.hop_length,
            n_mels=self.preset.n_mels,
            f_min=self.preset.f_min,
            f_max=self.preset.f_max,
        )
        self.mel_extractor = LogMelExtractor(self.mel_config)
        self.windowing = AudioWindowing(
            WindowConfig(
                spec_window_frames=self.preset.patch_time_frames,
                spec_hop_frames=self.preset.patch_hop_frames,
            )
        )
        self.stat_extractor = AcousticFeatureExtractor(
            AcousticFeatureConfig(
                sample_rate=self.preset.sample_rate,
                n_fft=self.preset.n_fft,
                hop_length=self.preset.hop_length,
            )
        )

    def process_raw_audio(self, file_path: Union[str, Path]) -> np.ndarray:
        """Load audio according to channel policy."""
        if self.preset.channel_mode == "multichannel_avg":
            data, sr = sf.read(str(file_path), dtype="float32")
            if data.ndim > 1:
                mono = np.mean(data, axis=1)
            else:
                mono = data
            # Normalize peak
            peak = np.max(np.abs(mono))
            if peak > 1e-6:
                mono = mono / peak
            return mono
        else:
            wav, _ = self.preprocessor_ch0.process_file(file_path)
            return wav

    def extract_patches(
        self,
        file_path: Union[str, Path],
        patch_frames: Optional[int] = None,
        hop_frames: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract standardized 4D patches (N, 1, n_mels, patch_time_frames)."""
        t0 = time.perf_counter()
        wav = self.process_raw_audio(file_path)
        spec, meta = self.mel_extractor.extract_numpy(wav)
        # Squeeze channel dim: (1, n_mels, T) -> (n_mels, T)
        win_frames = patch_frames or self.preset.patch_time_frames
        h_frames = hop_frames or self.preset.patch_hop_frames
        if win_frames != self.preset.patch_time_frames or h_frames != self.preset.patch_hop_frames:
            windowing = AudioWindowing(WindowConfig(spec_window_frames=win_frames, spec_hop_frames=h_frames))
        else:
            windowing = self.windowing

        patches_2d, _ = windowing.slice_spectrogram_numpy(spec.squeeze(0))
        # Add channel dim: (N, 1, n_mels, T)
        patches_4d = np.expand_dims(patches_2d, axis=1).astype(np.float32)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        info = {
            "preset": self.preset.name,
            "patch_count": patches_4d.shape[0],
            "patch_shape": list(patches_4d.shape),
            "spectrogram_shape": list(spec.shape),
            "extraction_latency_ms": round(elapsed_ms, 2),
        }
        return patches_4d, info

    def extract_patches_from_waveform(
        self,
        waveform: np.ndarray,
        patch_frames: Optional[int] = None,
        hop_frames: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract standardized 4D patches directly from in-memory waveform."""
        t0 = time.perf_counter()
        spec, meta = self.mel_extractor.extract_numpy(waveform)
        win_frames = patch_frames or self.preset.patch_time_frames
        h_frames = hop_frames or self.preset.patch_hop_frames
        if win_frames != self.preset.patch_time_frames or h_frames != self.preset.patch_hop_frames:
            windowing = AudioWindowing(WindowConfig(spec_window_frames=win_frames, spec_hop_frames=h_frames))
        else:
            windowing = self.windowing

        patches_2d, _ = windowing.slice_spectrogram_numpy(spec.squeeze(0))
        patches_4d = np.expand_dims(patches_2d, axis=1).astype(np.float32)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return patches_4d, {"extraction_latency_ms": round(elapsed_ms, 2)}

    def extract_statistical_vector(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract 38-dim compact statistical descriptor vector."""
        t0 = time.perf_counter()
        vec = self.stat_extractor.extract_from_file(file_path)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return vec, {"dim": len(vec), "latency_ms": round(elapsed_ms, 2)}

    def extract_wavelet_multiscale(self, file_path: Union[str, Path], levels: int = 4) -> Dict[str, Any]:
        """Compute multi-scale discrete wavelet decomposition approximation energies."""
        t0 = time.perf_counter()
        wav = self.process_raw_audio(file_path)
        # Multi-scale subband energy decomposition via cascaded halfband filters
        bands = []
        curr = wav
        for lvl in range(levels):
            # Lowpass approximation (decimate by 2)
            low = scipy.signal.decimate(curr, 2, ftype="fir")
            # Subband detail energy
            high_energy = float(np.sum((curr[:len(low)*2:2] - low) ** 2))
            bands.append({
                "level": lvl + 1,
                "band_length": len(low),
                "detail_energy": round(high_energy, 4),
                "approx_energy": round(float(np.sum(low ** 2)), 4),
            })
            curr = low

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "levels": levels,
            "bands": bands,
            "latency_ms": round(elapsed_ms, 2),
        }


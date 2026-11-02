"""Audio Recording Quality and Acoustic Environment Gate for Stetho.

Validates incoming recordings before inference:
- Rejects sub-Nyquist sample rates (< 16 kHz)
- Resamples supported high-rate streams (22.05k, 44.1k, 48k) to 16 kHz
- Rejects severe digital clipping (> 0.1% clipped samples)
- Rejects silence and extremely low-amplitude recordings (< -60 dBFS)
- Monitors acoustic environment / background noise floor via spectral flatness
  and high-frequency power ratios (distinguishing machine faults from noisy plants).
"""

from dataclasses import dataclass, field
from math import gcd
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.signal as signal


@dataclass
class QualityGateConfig:
    """Configuration thresholds for the recording quality gate."""

    min_sample_rate: int = 16000
    target_sample_rate: int = 16000
    max_clipping_ratio: float = 0.001  # 0.1% of samples
    min_rms_dbfs: float = -60.0  # Signals below this are rejected as silence
    min_peak_amplitude: float = 0.001  # Numerical silence threshold
    adverse_spectral_flatness_threshold: float = 0.12  # White noise threshold
    adverse_hf_ratio_threshold_db: float = -14.0  # High-freq noise threshold


@dataclass
class QualityGateResult:
    """Outcome of recording quality and acoustic environment verification."""

    is_valid: bool
    status: str  # "PASSED", "RECORDING_QUALITY_FAILURE", "ADVERSE_ACOUSTIC_ENVIRONMENT", "RESAMPLED"
    rejection_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    telemetry: Dict[str, Any] = field(default_factory=dict)


class AudioQualityGate:
    """Validates recording format, integrity, and acoustic environment."""

    def __init__(self, config: Optional[QualityGateConfig] = None):
        self.config = config or QualityGateConfig()

    def validate_and_prepare(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        channel: int = 0,
    ) -> Tuple[Optional[np.ndarray], QualityGateResult]:
        """Validate input audio and prepare standardized 16 kHz mono waveform.

        Args:
            waveform: Raw numpy array (1D mono or 2D [samples, channels])
            sample_rate: Ingested sample rate
            channel: Target microphone channel index

        Returns:
            Tuple of (prepared_waveform_or_None, QualityGateResult)
        """
        rejection_reasons: List[str] = []
        warnings: List[str] = []
        telemetry: Dict[str, Any] = {
            "ingested_sample_rate": sample_rate,
            "ingested_ndim": waveform.ndim,
        }

        # 1. Channel Extraction & Shape Verification
        if waveform.ndim == 2:
            num_channels = waveform.shape[1]
            telemetry["available_channels"] = num_channels
            if channel >= num_channels:
                rejection_reasons.append(
                    f"CHANNEL_OUT_OF_BOUNDS: Requested channel {channel} >= available {num_channels}."
                )
                return None, QualityGateResult(
                    is_valid=False,
                    status="RECORDING_QUALITY_FAILURE",
                    rejection_reasons=rejection_reasons,
                    telemetry=telemetry,
                )
            x_mono = np.ascontiguousarray(waveform[:, channel], dtype=np.float32)
        elif waveform.ndim == 1:
            telemetry["available_channels"] = 1
            if channel != 0:
                rejection_reasons.append(
                    f"CHANNEL_OUT_OF_BOUNDS: Requested channel {channel} on 1-channel mono stream."
                )
                return None, QualityGateResult(
                    is_valid=False,
                    status="RECORDING_QUALITY_FAILURE",
                    rejection_reasons=rejection_reasons,
                    telemetry=telemetry,
                )
            x_mono = np.ascontiguousarray(waveform, dtype=np.float32)
        else:
            rejection_reasons.append(f"INVALID_AUDIO_DIMENSION: Expected 1D or 2D array, got {waveform.ndim}D.")
            return None, QualityGateResult(
                is_valid=False,
                status="RECORDING_QUALITY_FAILURE",
                rejection_reasons=rejection_reasons,
                telemetry=telemetry,
            )

        telemetry["num_samples"] = len(x_mono)

        # 2. Sample Rate Validation Gate
        if sample_rate < self.config.min_sample_rate:
            rejection_reasons.append(
                f"SAMPLE_RATE_UNSUPPORTED: Sample rate {sample_rate} Hz < minimum {self.config.min_sample_rate} Hz. "
                f"Nyquist limit ({sample_rate / 2:.0f} Hz) truncates essential diagnostic high-frequency fan acoustics."
            )
            return None, QualityGateResult(
                is_valid=False,
                status="RECORDING_QUALITY_FAILURE",
                rejection_reasons=rejection_reasons,
                telemetry=telemetry,
            )

        # 3. Resampling to Target Sample Rate (e.g. 22.05k, 44.1k, 48k -> 16k)
        if sample_rate != self.config.target_sample_rate:
            g = gcd(self.config.target_sample_rate, sample_rate)
            up = self.config.target_sample_rate // g
            down = sample_rate // g
            x_resampled = signal.resample_poly(x_mono, up, down).astype(np.float32)
            telemetry["resampled_from"] = sample_rate
            telemetry["resampled_to"] = self.config.target_sample_rate
            x_eval = x_resampled
        else:
            x_eval = x_mono

        # 4. Silence and Low Amplitude Gate
        rms_linear = float(np.sqrt(np.mean(x_eval**2)))
        rms_dbfs = float(20.0 * np.log10(rms_linear + 1e-12))
        peak_amp = float(np.max(np.abs(x_eval)))
        telemetry["rms_dbfs"] = round(rms_dbfs, 2)
        telemetry["peak_amplitude"] = round(peak_amp, 4)

        if rms_dbfs < self.config.min_rms_dbfs or peak_amp < self.config.min_peak_amplitude:
            rejection_reasons.append(
                f"SIGNAL_SILENCE_OR_EMPTY: RMS energy {rms_dbfs:.1f} dBFS < {self.config.min_rms_dbfs:.1f} dBFS "
                f"or peak {peak_amp:.4f} < {self.config.min_peak_amplitude}."
            )
            return None, QualityGateResult(
                is_valid=False,
                status="RECORDING_QUALITY_FAILURE",
                rejection_reasons=rejection_reasons,
                telemetry=telemetry,
            )

        # 5. Clipping Gate
        clipping_ratio = float(np.mean(np.abs(x_eval) >= 0.999))
        telemetry["clipping_ratio"] = round(clipping_ratio, 6)
        if clipping_ratio > self.config.max_clipping_ratio:
            rejection_reasons.append(
                f"EXCESSIVE_DIGITAL_CLIPPING: Clipping ratio {clipping_ratio * 100:.2f}% > "
                f"allowed {self.config.max_clipping_ratio * 100:.2f}%."
            )
            return None, QualityGateResult(
                is_valid=False,
                status="RECORDING_QUALITY_FAILURE",
                rejection_reasons=rejection_reasons,
                telemetry=telemetry,
            )

        # 6. Acoustic Environment / Background Noise Floor Assessment
        # Compute Welch PSD for spectral flatness & high-frequency ratio
        f, pxx = signal.welch(x_eval, fs=self.config.target_sample_rate, nperseg=1024)
        pxx = pxx + 1e-12
        geo_mean = float(np.exp(np.mean(np.log(pxx))))
        ari_mean = float(np.mean(pxx))
        spectral_flatness = geo_mean / ari_mean

        low_mask = f < 4000.0
        high_mask = f >= 4000.0
        p_low = float(np.mean(pxx[low_mask]))
        p_high = float(np.mean(pxx[high_mask]))
        hf_ratio_db = float(10.0 * np.log10(p_high / p_low))

        telemetry["spectral_flatness"] = round(spectral_flatness, 4)
        telemetry["high_frequency_ratio_db"] = round(hf_ratio_db, 2)

        status = "PASSED"
        if sample_rate != self.config.target_sample_rate:
            status = "RESAMPLED"

        if (
            spectral_flatness > self.config.adverse_spectral_flatness_threshold
            or hf_ratio_db > self.config.adverse_hf_ratio_threshold_db
        ):
            warnings.append(
                f"ADVERSE_ACOUSTIC_ENVIRONMENT: High background noise floor detected "
                f"(spectral flatness {spectral_flatness:.3f} > {self.config.adverse_spectral_flatness_threshold}, "
                f"HF ratio {hf_ratio_db:.1f} dB > {self.config.adverse_hf_ratio_threshold_db} dB)."
            )
            status = "ADVERSE_ACOUSTIC_ENVIRONMENT"

        return x_eval, QualityGateResult(
            is_valid=True,
            status=status,
            rejection_reasons=rejection_reasons,
            warnings=warnings,
            telemetry=telemetry,
        )

"""Audio Quality Gate (AQG) Module for Stetho API.

Performs deterministic acoustic quality inspection prior to model inference:
- Differentiates sensor/recording faults from mechanical machine deviations.
- Prevents corrupt, silent, clipped, or noise-dominated recordings from triggering false anomaly alarms.
- Returns structured quality grades: ACCEPTED, WARNING, REJECTED, INCONCLUSIVE.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("stetho.audio_quality_gate")


class QualityStatus(str, Enum):
    """Overall outcome of the Audio Quality Gate evaluation."""
    ACCEPTED = "ACCEPTED"          # Clean, high-fidelity machine recording ready for inference
    WARNING = "WARNING"            # Minor imperfections (minor clipping/short duration); usable with caution
    REJECTED = "REJECTED"          # Fatally flawed (silence, severe clipping, NaN); CANNOT be scored
    INCONCLUSIVE = "INCONCLUSIVE"  # High ambient noise or uncertain conditions; anomaly decision unsupported


@dataclass
class QualityAssessment:
    """Structured quality assessment report produced by AudioQualityGate."""
    status: QualityStatus
    is_usable_for_inference: bool
    reasons: List[str]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "is_usable_for_inference": self.is_usable_for_inference,
            "reasons": self.reasons,
            "metrics": self.metrics,
        }


class AudioQualityGate:
    """Evaluates raw waveforms to certify audio suitability for acoustic inference."""

    def __init__(
        self,
        silence_rms_threshold: float = 1e-4,
        clipping_threshold: float = 0.999,
        severe_clipping_pct: float = 5.0,
        minor_clipping_pct: float = 0.1,
        min_usable_duration_sec: float = 2.0,
        min_recommended_duration_sec: float = 5.0,
        min_acceptable_snr_db: float = 3.0,
    ):
        self.silence_rms_threshold = silence_rms_threshold
        self.clipping_threshold = clipping_threshold
        self.severe_clipping_pct = severe_clipping_pct
        self.minor_clipping_pct = minor_clipping_pct
        self.min_usable_duration_sec = min_usable_duration_sec
        self.min_recommended_duration_sec = min_recommended_duration_sec
        self.min_acceptable_snr_db = min_acceptable_snr_db

    @staticmethod
    def estimate_snr_db(waveform: np.ndarray) -> float:
        """Estimate Signal-to-Noise Ratio (dB) via Spectral Peak-to-Median Power Ratio.

        In industrial machinery (fans, pumps, motors), harmonic peaks rise significantly
        above the broadband acoustic noise floor. In pure random background noise,
        the spectral peak-to-median ratio is bounded by chi-square statistics (~12-14 dB).
        """
        if len(waveform) < 1024:
            return 20.0

        fft = np.abs(np.fft.rfft(waveform)) ** 2
        median_val = float(np.median(fft))
        max_val = float(np.max(fft))

        if median_val <= 1e-12:
            return 50.0  # Clean recording with negligible noise floor

        peak_to_noise_db = float(10.0 * np.log10(max(max_val / (median_val + 1e-12), 1.0)))
        # Subtract the theoretical random white noise baseline (~12.5 dB)
        snr_est = max(0.0, peak_to_noise_db - 12.5)
        return round(snr_est, 2)

    def assess_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
    ) -> QualityAssessment:
        """Run comprehensive quality gate assessment on mono float32 waveform."""
        # Ensure 1D
        if waveform.ndim > 1:
            waveform = waveform.flatten()

        reasons: List[str] = []
        is_rejected = False
        is_inconclusive = False
        has_warnings = False

        total_samples = len(waveform)
        duration_sec = total_samples / sample_rate if sample_rate > 0 else 0.0

        # 1. Numerical Sanity
        nan_count = int(np.isnan(waveform).sum())
        inf_count = int(np.isinf(waveform).sum())
        if nan_count > 0 or inf_count > 0:
            reasons.append(f"Recording contains non-finite numerical artifacts ({nan_count} NaNs, {inf_count} Infs)")
            return QualityAssessment(
                status=QualityStatus.REJECTED,
                is_usable_for_inference=False,
                reasons=reasons,
                metrics={
                    "duration_sec": duration_sec,
                    "nan_count": nan_count,
                    "inf_count": inf_count,
                },
            )

        if total_samples == 0:
            reasons.append("Recording contains 0 audio samples")
            return QualityAssessment(
                status=QualityStatus.REJECTED,
                is_usable_for_inference=False,
                reasons=reasons,
                metrics={"duration_sec": 0.0},
            )

        # 2. Basic Signal Metrics
        peak_amp = float(np.max(np.abs(waveform)))
        rms_amp = float(np.sqrt(np.mean(waveform ** 2)))
        crest_factor = round(peak_amp / (rms_amp + 1e-12), 3)

        # 3. Silence / Signal Dropout
        if rms_amp < self.silence_rms_threshold:
            is_rejected = True
            reasons.append(
                f"Audio is silent or near-silent: RMS amplitude {rms_amp:.2e} is below threshold {self.silence_rms_threshold:.2e}"
            )

        # 4. Digital Clipping Check
        clipping_samples = int(np.sum(np.abs(waveform) >= self.clipping_threshold))
        clipping_pct = round((clipping_samples / total_samples) * 100.0, 3)

        if clipping_pct >= self.severe_clipping_pct:
            is_rejected = True
            reasons.append(
                f"Severe digital clipping detected: {clipping_pct}% of samples ({clipping_samples:,}) saturated at rail (>= {self.clipping_threshold})"
            )
        elif clipping_pct >= self.minor_clipping_pct:
            has_warnings = True
            reasons.append(
                f"Minor digital clipping detected: {clipping_pct}% of samples ({clipping_samples:,}) saturated at rail"
            )

        # 5. Duration Bounds
        if duration_sec < self.min_usable_duration_sec:
            is_rejected = True
            reasons.append(
                f"Recording duration ({duration_sec:.2f}s) is too short to extract periodic acoustic cycles (min: {self.min_usable_duration_sec:.1f}s)"
            )
        elif duration_sec < self.min_recommended_duration_sec:
            has_warnings = True
            reasons.append(
                f"Short recording duration ({duration_sec:.2f}s); recommended minimum is {self.min_recommended_duration_sec:.1f}s"
            )

        # 6. SNR & Environmental Noise Floor
        estimated_snr = self.estimate_snr_db(waveform)
        if estimated_snr < 0.0:
            is_inconclusive = True
            reasons.append(
                f"Severe ambient background noise: Estimated SNR is negative ({estimated_snr} dB); machine sound is completely masked"
            )
        elif estimated_snr < self.min_acceptable_snr_db:
            is_inconclusive = True
            reasons.append(
                f"Low Signal-to-Noise Ratio ({estimated_snr} dB < {self.min_acceptable_snr_db} dB); ambient noise may corrupt anomaly scoring"
            )

        # Determine Final Status
        if is_rejected:
            final_status = QualityStatus.REJECTED
            is_usable = False
        elif is_inconclusive:
            final_status = QualityStatus.INCONCLUSIVE
            is_usable = False
        elif has_warnings:
            final_status = QualityStatus.WARNING
            is_usable = True
        else:
            final_status = QualityStatus.ACCEPTED
            is_usable = True

        metrics = {
            "duration_sec": round(duration_sec, 3),
            "sample_rate": sample_rate,
            "peak_amplitude": round(peak_amp, 5),
            "rms_amplitude": round(rms_amp, 5),
            "clipping_percentage": clipping_pct,
            "clipping_samples": clipping_samples,
            "estimated_snr_db": estimated_snr,
            "crest_factor": crest_factor,
        }

        if len(reasons) == 0:
            reasons.append("Recording satisfies all audio quality criteria for machine acoustic analysis")

        return QualityAssessment(
            status=final_status,
            is_usable_for_inference=is_usable,
            reasons=reasons,
            metrics=metrics,
        )

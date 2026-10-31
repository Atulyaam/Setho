"""Audio Validation Module for Stetho.

Performs robust sanity and integrity checks on raw and processed audio recordings:
- File existence and format readability
- Expected sample rate and duration
- Zero-byte/empty file detection
- Numerical anomalies (NaN, Inf)
- Complete silence and signal dropout
- Digital clipping and amplitude saturation
- Multichannel structure and properties
"""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf


@dataclass
class ValidationResult:
    """Structured result of audio validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "properties": self.properties,
        }


class AudioValidator:
    """Validates audio files and waveforms against Stetho operational specifications."""

    def __init__(
        self,
        expected_sample_rate: Optional[int] = 16000,
        expected_duration_sec: Optional[float] = 10.0,
        duration_tolerance_sec: float = 0.5,
        silence_threshold_rms: float = 1e-4,
        clipping_threshold: float = 0.999,
        allow_multichannel: bool = True,
    ):
        self.expected_sample_rate = expected_sample_rate
        self.expected_duration_sec = expected_duration_sec
        self.duration_tolerance_sec = duration_tolerance_sec
        self.silence_threshold_rms = silence_threshold_rms
        self.clipping_threshold = clipping_threshold
        self.allow_multichannel = allow_multichannel

    def validate_file(self, file_path: Union[str, Path]) -> ValidationResult:
        """Validate an audio file on disk."""
        path = Path(file_path)
        errors: List[str] = []
        warnings: List[str] = []
        props: Dict[str, Any] = {"path": str(path)}

        # 1. Existence check
        if not path.exists():
            return ValidationResult(
                is_valid=False,
                errors=[f"File does not exist: {path}"],
                properties=props,
            )

        if not path.is_file():
            return ValidationResult(
                is_valid=False,
                errors=[f"Path is not a regular file: {path}"],
                properties=props,
            )

        file_size = path.stat().st_size
        props["file_size_bytes"] = file_size
        if file_size == 0:
            return ValidationResult(
                is_valid=False,
                errors=[f"File is empty (0 bytes): {path}"],
                properties=props,
            )

        # 2. Format and header readability check
        try:
            info = sf.info(str(path))
            props["sample_rate"] = info.samplerate
            props["channels"] = info.channels
            props["duration_sec"] = info.duration
            props["frames"] = info.frames
            props["format"] = info.format
            props["subtype"] = info.subtype
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Failed to read audio file format or header: {str(e)}"],
                properties=props,
            )

        # 3. Audio decoding and numerical checks
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Failed to decode audio data: {str(e)}"],
                properties=props,
            )

        return self.validate_array(data, sr, base_properties=props)

    def validate_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        base_properties: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate an in-memory numpy audio array."""
        errors: List[str] = []
        warnings: List[str] = []
        props = dict(base_properties or {})

        if audio.ndim == 1:
            audio = audio[:, np.newaxis]

        num_samples, num_channels = audio.shape
        duration = num_samples / sample_rate if sample_rate > 0 else 0.0

        props.update({
            "sample_rate": sample_rate,
            "channels": num_channels,
            "duration_sec": round(duration, 4),
            "samples": num_samples,
        })

        # Check empty array
        if num_samples == 0:
            errors.append("Audio signal has 0 samples")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings, properties=props)

        # Check sample rate
        if self.expected_sample_rate is not None and sample_rate != self.expected_sample_rate:
            errors.append(
                f"Sample rate mismatch: expected {self.expected_sample_rate} Hz, got {sample_rate} Hz"
            )

        # Check duration
        if self.expected_duration_sec is not None:
            diff = abs(duration - self.expected_duration_sec)
            if diff > self.duration_tolerance_sec:
                errors.append(
                    f"Duration mismatch: expected {self.expected_duration_sec}s "
                    f"(±{self.duration_tolerance_sec}s), got {duration:.3f}s"
                )

        # Check channels
        if not self.allow_multichannel and num_channels > 1:
            errors.append(f"Multichannel audio not allowed: found {num_channels} channels")
        elif num_channels > 1:
            props["channel_layout"] = f"{num_channels}-channel"

        # Check NaN / Inf
        nan_count = int(np.isnan(audio).sum())
        inf_count = int(np.isinf(audio).sum())
        if nan_count > 0:
            errors.append(f"Audio contains {nan_count} NaN values")
        if inf_count > 0:
            errors.append(f"Audio contains {inf_count} Inf values")

        if nan_count > 0 or inf_count > 0:
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings, properties=props)

        # Signal level statistics
        peak_amp = float(np.max(np.abs(audio)))
        rms_amp = float(np.sqrt(np.mean(audio**2)))
        props["peak_amplitude"] = round(peak_amp, 6)
        props["rms_amplitude"] = round(rms_amp, 6)

        # Check silence
        if rms_amp < self.silence_threshold_rms:
            errors.append(
                f"Audio is silent or near-silent: RMS amplitude {rms_amp:.2e} < {self.silence_threshold_rms:.2e}"
            )

        # Check clipping
        clipping_samples = int(np.sum(np.abs(audio) >= self.clipping_threshold))
        props["clipping_samples"] = clipping_samples
        if clipping_samples > 0:
            clip_percent = (clipping_samples / audio.size) * 100.0
            props["clipping_percentage"] = round(clip_percent, 4)
            if clip_percent > 1.0:
                warnings.append(
                    f"Severe clipping detected: {clipping_samples} samples ({clip_percent:.2f}%) >= {self.clipping_threshold}"
                )
            else:
                warnings.append(
                    f"Minor clipping detected: {clipping_samples} samples ({clip_percent:.2f}%) >= {self.clipping_threshold}"
                )

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            properties=props,
        )

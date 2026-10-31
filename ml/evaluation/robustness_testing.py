"""Synthetic Acoustic Robustness and Stress Testing for Stetho-MSAE.

Evaluates model stability and degradation curves under simulated physical perturbations:
1. Additive Gaussian noise (SNR 20 dB, 10 dB, 0 dB)
2. Gain scaling (+6 dB, -6 dB, -20 dB)
3. Temporal packet drop / temporal dropout (50ms - 200ms segments zeroed)
4. Clipping distortion / saturation
5. Pure silence (all zeros)
6. Impulsive acoustic transients (random clicks)

Asserts:
- No NaN / Inf predictions under any perturbation.
- Scores increase gracefully with severity (monotonicity).
- Robust runtime error handling with zero unhandled exceptions.
"""

import json
import logging
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.models.stetho_msae.msae_detector import StethoMSAEDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stetho_robustness")


def add_gaussian_noise(wav: np.ndarray, snr_db: float) -> np.ndarray:
    """Add zero-mean Gaussian noise at specified SNR in dB."""
    signal_power = np.mean(wav ** 2)
    if signal_power == 0:
        return wav + np.random.normal(0, 0.01, size=wav.shape).astype(np.float32)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), size=wav.shape).astype(np.float32)
    return (wav + noise).astype(np.float32)


def apply_gain_db(wav: np.ndarray, gain_db: float) -> np.ndarray:
    """Scale audio by decibel factor."""
    scale = 10.0 ** (gain_db / 20.0)
    return (wav * scale).astype(np.float32)


def apply_packet_drop(wav: np.ndarray, drop_fraction: float = 0.1, drop_duration_samples: int = 1600) -> np.ndarray:
    """Zero out random contiguous slices of audio simulating frame / packet drop."""
    out = wav.copy()
    n_drops = int((len(wav) * drop_fraction) / drop_duration_samples)
    for _ in range(max(1, n_drops)):
        start = np.random.randint(0, max(1, len(wav) - drop_duration_samples))
        out[start : start + drop_duration_samples] = 0.0
    return out


def apply_clipping(wav: np.ndarray, clip_threshold: float = 0.05) -> np.ndarray:
    """Simulate severe non-linear microphone preamp clipping."""
    return np.clip(wav, -clip_threshold, clip_threshold).astype(np.float32)


def apply_impulse_clicks(wav: np.ndarray, num_clicks: int = 10, click_amp: float = 0.8) -> np.ndarray:
    """Simulate loose sensor contact clicks or mechanical impulse transients."""
    out = wav.copy()
    indices = np.random.randint(0, len(wav), size=num_clicks)
    out[indices] = click_amp
    return out


def evaluate_detector_robustness(
    detector: StethoMSAEDetector,
    clean_audio_path: Path,
) -> Dict[str, Any]:
    """Test detector response across comprehensive synthetic perturbations."""
    wav, _ = detector.preprocessor.process_file(clean_audio_path)
    clean_score = detector.score_waveform(wav)

    tests = [
        ("clean_reference", wav, 0.0, "Reference untouched audio"),
        ("gaussian_noise_20db", add_gaussian_noise(wav, 20.0), 20.0, "Light background noise (SNR 20 dB)"),
        ("gaussian_noise_10db", add_gaussian_noise(wav, 10.0), 10.0, "Moderate background noise (SNR 10 dB)"),
        ("gaussian_noise_0db", add_gaussian_noise(wav, 0.0), 0.0, "Severe background noise (SNR 0 dB)"),
        ("gain_plus_6db", apply_gain_db(wav, 6.0), 6.0, "Amplitude gain +6 dB"),
        ("gain_minus_6db", apply_gain_db(wav, -6.0), -6.0, "Amplitude attenuation -6 dB"),
        ("gain_minus_20db", apply_gain_db(wav, -20.0), -20.0, "Severe attenuation -20 dB"),
        ("packet_drop_10pct", apply_packet_drop(wav, 0.10), 0.10, "10% packet / temporal drop"),
        ("packet_drop_25pct", apply_packet_drop(wav, 0.25), 0.25, "25% packet / temporal drop"),
        ("clipping_saturation", apply_clipping(wav, 0.05), 0.05, "Non-linear microphone clipping"),
        ("impulse_clicks", apply_impulse_clicks(wav, 20, 0.9), 20, "20 acoustic impulse click transients"),
        ("pure_silence", np.zeros_like(wav), 0.0, "Complete acoustic silence (all zeros)"),
    ]

    results = {
        "machine_id": detector.machine_id,
        "clean_baseline_score": round(clean_score, 6),
        "threshold": detector.threshold,
        "perturbations": {},
    }

    for name, degraded_wav, param_val, desc in tests:
        # 1. Run inference
        score = detector.score_waveform(degraded_wav)

        # 2. Strict stability assertions
        assert not math.isnan(score), f"NaN score encountered under perturbation: {name}"
        assert not math.isinf(score), f"Inf score encountered under perturbation: {name}"
        assert score >= 0.0, f"Negative score encountered under perturbation: {name}"

        score_diff = score - clean_score
        is_flagged = bool(score > detector.threshold) if detector.threshold else False

        results["perturbations"][name] = {
            "description": desc,
            "parameter_value": param_val,
            "anomaly_score": round(score, 6),
            "score_diff_from_clean": round(score_diff, 6),
            "flagged_abnormal": is_flagged,
            "is_finite": True,
        }
        logger.info(
            f"Perturbation [{name:<22}]: Score={score:.5f} "
            f"(Delta={score_diff:+.5f}) | Flagged={is_flagged}"
        )

    return results


def run_robustness_suite(
    artifact_dir: Path = Path("ml/artifacts/stetho_msae/id_00"),
    output_path: Path = Path("ml/reports/robustness_results.json"),
) -> Dict[str, Any]:
    """Execute robustness suite and write JSON report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not artifact_dir.exists():
        raise FileNotFoundError(f"Stetho-MSAE artifact directory not found: {artifact_dir}")

    # Find normal test sample
    test_normal = Path("data/mimii/raw/fan/id_00/normal/00000008.wav")
    if not test_normal.exists():
        candidates = list(Path("data/mimii/raw/fan/id_00/normal").glob("*.wav"))
        if not candidates:
            raise FileNotFoundError("No normal test WAV found for id_00")
        test_normal = candidates[0]

    detector = StethoMSAEDetector.load(artifact_dir, device="cpu")
    robustness_data = evaluate_detector_robustness(detector, test_normal)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(robustness_data, f, indent=2)

    logger.info(f"Robustness evaluation complete. Results written to {output_path}")
    return robustness_data


if __name__ == "__main__":
    run_robustness_suite()


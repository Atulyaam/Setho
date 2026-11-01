"""Automated verification tests for Phase 9 Generalization & Robustness."""

from pathlib import Path
import numpy as np
import pytest

from ml.training.train_pipeline import _read_split_csv


def inject_noise_at_snr(waveform: np.ndarray, snr_db: float, seed: int = 42) -> np.ndarray:
    """Inject stationary factory noise at a specified Signal-to-Noise Ratio (dB)."""
    np.random.seed(seed)
    p_sig = np.mean(waveform ** 2)
    if p_sig < 1e-9:
        return waveform

    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    noise = np.random.randn(*waveform.shape).astype(np.float32)
    noise = noise * np.sqrt(p_noise / (np.mean(noise ** 2) + 1e-9))
    return (waveform + noise).astype(np.float32)


def test_noise_injection_snr_calculation():
    """Verify that injected noise matches requested SNR within 0.2 dB tolerance."""
    # Synthetic 1kHz sine wave
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)

    for target_snr in [6.0, 0.0, -6.0]:
        noisy = inject_noise_at_snr(sig, snr_db=target_snr)
        noise = noisy - sig
        measured_snr = 10.0 * np.log10(np.mean(sig ** 2) / np.mean(noise ** 2))
        assert pytest.approx(measured_snr, abs=0.2) == target_snr


def test_protocol_b_splits_integrity():
    """Verify Protocol B (Unseen Machines) splits exist and contain disjoint machines."""
    for m_id in ["id_00", "id_02", "id_04", "id_06"]:
        split_dir = Path(f"ml/data/splits/unseen_machine/fan/heldout_{m_id}")
        assert split_dir.exists(), f"Split directory missing for heldout_{m_id}"

        train_rows = _read_split_csv(split_dir / "train_known_machines.csv")
        test_rows = _read_split_csv(split_dir / "test_unseen_machine.csv")

        train_files = [r["file_path"] for r in train_rows]
        test_files = [r["file_path"] for r in test_rows]

        # Verify zero machine leakage
        for tf in train_files:
            assert m_id not in tf, f"Leaked held-out machine {m_id} found in train set!"
        for test_f in test_files:
            assert m_id in test_f, f"Non-target machine found in test set: {test_f}"


def test_protocol_c_cross_category_splits_integrity():
    """Verify Protocol C (Cross-Category) splits exist and contain distinct categories."""
    for pair in ["fan_to_pump", "fan_to_valve"]:
        split_dir = Path(f"ml/data/splits/cross_category/{pair}")
        assert split_dir.exists(), f"Split directory missing for {pair}"

        train_rows = _read_split_csv(split_dir / "train_source_category.csv")
        test_rows = _read_split_csv(split_dir / "test_target_category.csv")

        assert len(train_rows) > 0
        assert len(test_rows) > 0

        # Verify source is fan and target is pump or valve
        for r in train_rows:
            assert "fan" in r["file_path"].lower()
        target_name = pair.split("_to_")[1]
        for r in test_rows:
            assert target_name in r["file_path"].lower()


"""Unit tests for MachineCalibrationEngine and dynamic threshold recalibration (Task 4.5.2)."""

import io
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from stetho_api.db import DatabaseManager, EnrollmentRepository, MachineRepository
from stetho_api.inference import (
    CalibrationResult,
    MachineCalibrationEngine,
    ModelRegistry,
    ModelType,
)


@pytest.fixture
def setup_calib(tmp_path):
    db_file = tmp_path / "calib_test.db"
    db = DatabaseManager(db_file)
    registry = ModelRegistry(artifacts_root="./ml/artifacts", device="cpu", fallback_to_cpu=True)
    engine = MachineCalibrationEngine(db=db, registry=registry)
    machine_repo = MachineRepository(db)
    enrollment_repo = EnrollmentRepository(db)
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir(parents=True)
    return {
        "engine": engine,
        "machine_repo": machine_repo,
        "enrollment_repo": enrollment_repo,
        "registry": registry,
        "storage_dir": storage_dir,
    }


def save_test_audio(path: Path, amplitude: float = 0.2, duration_sec: float = 5.0) -> str:
    samples = int(duration_sec * 16000)
    t = np.linspace(0, duration_sec, samples, endpoint=False)
    wav = (amplitude * np.sin(2 * np.pi * 300 * t) + 0.005 * np.random.randn(samples)).astype(np.float32)
    sf.write(str(path), wav, 16000, format="WAV")
    return str(path)


def test_recalibrate_with_zero_enrollments(setup_calib):
    """Verify machine with zero enrollments returns uncalibrated fallback status."""
    engine = setup_calib["engine"]
    res = engine.recalibrate_machine("id_00")
    assert isinstance(res, CalibrationResult)
    assert res.enrolled_sample_count == 0
    assert res.calibration_status == "uncalibrated"
    assert res.is_provisional is True


def test_recalibrate_dynamic_threshold_update(setup_calib):
    """Verify enrolling distinct confirmed normal recordings genuinely recalculates thresholds."""
    engine = setup_calib["engine"]
    m_repo = setup_calib["machine_repo"]
    e_repo = setup_calib["enrollment_repo"]
    s_dir = setup_calib["storage_dir"]

    # Initial baseline
    p0 = m_repo.get_or_raise("id_00")
    initial_threshold = p0.calibration_profile.get("operating_threshold") if p0.calibration_profile else None

    # Enroll 1st normal clip (amplitude 0.15)
    f1 = save_test_audio(s_dir / "norm_01.wav", amplitude=0.15)
    e_repo.record_enrollment("id_00", "upl-1", f1, 5.0, 16000, 1, "ACCEPTED", True)
    res1 = engine.recalibrate_machine("id_00")
    assert res1.enrolled_sample_count == 1
    assert res1.calibration_status == "provisional"
    assert res1.is_provisional is True
    assert res1.calibration_version == "1.1.0"
    theta_1 = res1.operating_threshold

    # Enroll 2nd normal clip (different amplitude 0.25 -> different acoustic score)
    f2 = save_test_audio(s_dir / "norm_02.wav", amplitude=0.25)
    e_repo.record_enrollment("id_00", "upl-2", f2, 5.0, 16000, 1, "ACCEPTED", True)
    res2 = engine.recalibrate_machine("id_00")
    assert res2.enrolled_sample_count == 2
    assert res2.calibration_version == "1.2.0"
    theta_2 = res2.operating_threshold

    # Verify thresholds genuinely changed
    assert theta_1 != theta_2, "Threshold must genuinely recalculate from enrollment audio"

    # Verify live model in registry was updated with new threshold
    model = setup_calib["registry"].get_model("id_00", ModelType.STETHO_MSAE)
    assert model.calibration["operating_threshold"] == theta_2


def test_transition_to_calibrated_at_30_recordings(setup_calib):
    """Verify that reaching N >= 30 confirmed normal recordings transitions status to calibrated."""
    engine = setup_calib["engine"]
    e_repo = setup_calib["enrollment_repo"]
    s_dir = setup_calib["storage_dir"]

    # Enroll 30 distinct synthetic normal clips
    for i in range(30):
        f = save_test_audio(s_dir / f"batch_norm_{i:02d}.wav", amplitude=0.1 + (i % 5) * 0.02)
        e_repo.record_enrollment("id_02", f"upl-{i}", f, 5.0, 16000, 1, "ACCEPTED", True)

    res30 = engine.recalibrate_machine("id_02")
    assert res30.enrolled_sample_count == 30
    assert res30.calibration_status == "calibrated"
    assert res30.is_provisional is False
    assert "Statistically stabilized" in res30.notice


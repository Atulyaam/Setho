"""Unit tests for Stetho Field Data Collection, Annotation, and Dataset Governance (Sprint 7.7 / E8)."""

import io
from pathlib import Path
import tempfile
import pytest
import numpy as np
import soundfile as sf

from stetho_api.db.database import DatabaseManager
from stetho_api.db.field_data_repo import (
    DatasetRoleConflictException,
    DuplicateRecordingException,
    FieldDataRepo,
    InvalidAudioQualityException,
    ManifestSealedException,
)


def create_test_wav(
    duration_sec: float = 2.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    amplitude: float = 0.2,
    channels: int = 1,
) -> bytes:
    """Generate a clean in-memory PCM_16 WAV byte buffer."""
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False, dtype=np.float32)
    sig = amplitude * np.sin(2 * np.pi * frequency * t)
    if channels > 1:
        sig = np.column_stack([sig] * channels)
    buf = io.BytesIO()
    sf.write(buf, sig, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def temp_repo():
    """Create a temporary FieldDataRepo with an isolated SQLite DB and storage directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_path = tmp_path / "test_stetho.db"
        storage_dir = tmp_path / "storage"
        db = DatabaseManager(db_path)
        repo = FieldDataRepo(db=db, storage_base_dir=storage_dir)
        try:
            yield repo, db, tmp_path
        finally:
            db.close()


def test_import_valid_field_recording(temp_repo):
    repo, _, _ = temp_repo
    wav_bytes = create_test_wav(duration_sec=3.0, sample_rate=16000, frequency=300.0)

    rec = repo.import_recording(
        file_bytes=wav_bytes,
        original_filename="fan_bearing_intake.wav",
        machine_category="fan",
        machine_id="fan_unit_4a",
        sensor_id="mic_sensor_091",
        operator_id="inspector_bob",
        channel_index=0,
        microphone_position="primary_bearing",
        operating_condition="nominal",
        load_level="full_load",
    )

    assert rec["recording_id"].startswith("fld_rec_")
    assert rec["machine_category"] == "fan"
    assert rec["machine_id"] == "fan_unit_4a"
    assert rec["sample_rate"] == 16000
    assert rec["channels"] == 1
    assert rec["duration_sec"] == 3.0
    assert rec["quality_status"] == "PASSED"
    assert rec["current_label"] == "unlabelled"
    assert rec["verification_status"] == "PENDING"
    assert rec["dataset_role"] == "unassigned"

    # Verify physical file existence
    assert Path(rec["file_path"]).exists()


def test_import_duplicate_recording_rejection(temp_repo):
    repo, _, _ = temp_repo
    wav_bytes = create_test_wav(duration_sec=2.0, sample_rate=16000, frequency=500.0)

    # First import: success
    rec1 = repo.import_recording(
        file_bytes=wav_bytes,
        original_filename="pump_original.wav",
        machine_category="pump",
        machine_id="pump_01",
        sensor_id="mic_01",
        operator_id="operator_alice",
    )

    # Second import with identical bytes: must raise DuplicateRecordingException
    with pytest.raises(DuplicateRecordingException) as exc_info:
        repo.import_recording(
            file_bytes=wav_bytes,
            original_filename="pump_renamed_copy.wav",
            machine_category="pump",
            machine_id="pump_01",
            sensor_id="mic_01",
            operator_id="operator_alice",
        )

    assert exc_info.value.code == "DUPLICATE_RECORDING"
    assert exc_info.value.existing_id == rec1["recording_id"]

    # Verify import history logged duplicate
    stats = repo.get_dashboard_stats()
    assert stats["duplicates_detected"] == 1
    assert stats["total_imported"] == 1


def test_import_unsupported_sample_rate_rejection(temp_repo):
    repo, _, _ = temp_repo
    # Sub-Nyquist 8 kHz audio must be rejected
    sub_nyquist_bytes = create_test_wav(duration_sec=2.0, sample_rate=8000, frequency=200.0)

    with pytest.raises(InvalidAudioQualityException) as exc_info:
        repo.import_recording(
            file_bytes=sub_nyquist_bytes,
            original_filename="low_rate_audio.wav",
            machine_category="valve",
            machine_id="valve_01",
            sensor_id="mic_01",
            operator_id="operator_alice",
        )

    assert exc_info.value.code == "INVALID_AUDIO_QUALITY"
    assert "SAMPLE_RATE_UNSUPPORTED" in exc_info.value.message

    stats = repo.get_dashboard_stats()
    assert stats["rejections_count"] == 1
    assert stats["total_imported"] == 0


def test_ground_truth_annotation_and_revision_history(temp_repo):
    repo, _, _ = temp_repo
    wav_bytes = create_test_wav(duration_sec=2.5, sample_rate=16000, frequency=120.0)
    rec = repo.import_recording(
        file_bytes=wav_bytes,
        original_filename="valve_test.wav",
        machine_category="valve",
        machine_id="valve_v12",
        sensor_id="mic_valve_01",
        operator_id="inspector_01",
    )

    # Revision 1: initial diagnosis
    ann1 = repo.add_annotation(
        recording_id=rec["recording_id"],
        label="uncertain",
        reason="Acoustic hiss detected but vibration accelerometer inconclusive",
        evidence_source="OPERATIONAL_LOG",
        annotator_id="inspector_01",
    )
    assert ann1["revision_number"] == 1
    assert ann1["label"] == "uncertain"

    updated_rec = repo.get_recording(rec["recording_id"])
    assert updated_rec["current_label"] == "uncertain"
    assert updated_rec["verification_status"] == "PENDING"

    # Revision 2: confirmed after teardown inspection
    ann2 = repo.add_annotation(
        recording_id=rec["recording_id"],
        label="abnormal",
        reason="Teardown inspection confirmed torn diaphragm seal",
        evidence_source="TEARDOWN_INSPECTION",
        annotator_id="lead_specialist_jane",
        inspection_notes="Replaced seal kit #99281",
    )
    assert ann2["revision_number"] == 2
    assert ann2["label"] == "abnormal"

    updated_rec2 = repo.get_recording(rec["recording_id"])
    assert updated_rec2["current_label"] == "abnormal"
    assert updated_rec2["verification_status"] == "VERIFIED"

    # Verify audit history
    history = repo.get_annotation_history(rec["recording_id"])
    assert len(history) == 2
    assert history[0]["verification_status"] == "SUPERSEDED"
    assert history[1]["verification_status"] == "VERIFIED"


def test_dataset_role_assignment_and_calibration_restrictions(temp_repo):
    repo, _, _ = temp_repo
    wav_bytes = create_test_wav(duration_sec=2.0, sample_rate=16000, frequency=250.0)
    rec = repo.import_recording(
        file_bytes=wav_bytes,
        original_filename="fan_healthy.wav",
        machine_category="fan",
        machine_id="fan_01",
        sensor_id="mic_01",
        operator_id="inspector_01",
    )

    # Label as abnormal first
    repo.add_annotation(
        recording_id=rec["recording_id"],
        label="abnormal",
        reason="Confirmed bearing cage defect",
        evidence_source="VIBRATION_ORDER_ANALYSIS",
        annotator_id="analyst_01",
    )

    # Attempting to assign 'calibration_enrollment' to abnormal recording must fail!
    with pytest.raises(DatasetRoleConflictException) as exc_info:
        repo.assign_dataset_role(rec["recording_id"], "calibration_enrollment", "admin_01")

    assert "abnormal recording" in exc_info.value.message.lower()

    # Re-annotating as normal enables calibration enrollment
    repo.add_annotation(
        recording_id=rec["recording_id"],
        label="normal",
        reason="Corrected: Healthy test baseline recording",
        evidence_source="VISUAL_INSPECTION",
        annotator_id="analyst_01",
    )
    repo.assign_dataset_role(rec["recording_id"], "calibration_enrollment", "admin_01")
    updated = repo.get_recording(rec["recording_id"])
    assert updated["dataset_role"] == "calibration_enrollment"

    # Now attempting to annotate a calibration recording as abnormal must be blocked!
    with pytest.raises(DatasetRoleConflictException):
        repo.add_annotation(
            recording_id=rec["recording_id"],
            label="abnormal",
            reason="Contradicting label on calibration audio",
            evidence_source="VIBRATION_ORDER_ANALYSIS",
            annotator_id="analyst_01",
        )


def test_manifest_creation_and_sealing(temp_repo):
    repo, _, _ = temp_repo
    ids = []
    for i in range(3):
        wav = create_test_wav(duration_sec=2.0, sample_rate=16000, frequency=300.0 + i * 50)
        r = repo.import_recording(
            file_bytes=wav,
            original_filename=f"eval_clip_{i}.wav",
            machine_category="fan",
            machine_id="fan_pilot_01",
            sensor_id="mic_01",
            operator_id="inspector_01",
        )
        repo.assign_dataset_role(r["recording_id"], "sealed_evaluation", "admin_01")
        ids.append(r["recording_id"])

    manifest = repo.create_or_seal_manifest(
        name="Pilot Fan Evaluation Benchmark v1",
        dataset_role="sealed_evaluation",
        machine_category="fan",
        recording_ids=ids,
        created_by="lead_engineer",
        seal_now=True,
    )

    assert manifest["is_sealed"] is True
    assert manifest["sha256_checksum"] is not None
    assert manifest["recording_count"] == 3

    # Attempting to re-assign role of a recording in a sealed manifest must be blocked!
    with pytest.raises(ManifestSealedException):
        repo.assign_dataset_role(ids[0], "training", "admin_01")


def test_dashboard_stats_and_independent_readiness_warning(temp_repo):
    repo, _, _ = temp_repo
    # When 0 sealed evaluation recordings exist, readiness must show clear warning
    stats = repo.get_dashboard_stats()
    assert stats["total_imported"] == 0
    assert stats["independent_evaluation_readiness"]["is_ready"] is False
    assert stats["independent_evaluation_readiness"]["verdict"] == "INSUFFICIENT_INDEPENDENT_EVIDENCE"
    assert "INSUFFICIENT INDEPENDENT EVIDENCE" in stats["independent_evaluation_readiness"]["warning"]

    # Import 1 recording
    wav = create_test_wav(duration_sec=2.0, sample_rate=16000)
    repo.import_recording(
        file_bytes=wav,
        original_filename="clip1.wav",
        machine_category="fan",
        machine_id="fan_01",
        sensor_id="mic_01",
        operator_id="admin",
    )
    stats2 = repo.get_dashboard_stats()
    assert stats2["total_imported"] == 1
    assert stats2["by_category"]["fan"] == 1
    assert stats2["unlabelled_count"] == 1
    assert stats2["independent_evaluation_readiness"]["is_ready"] is False

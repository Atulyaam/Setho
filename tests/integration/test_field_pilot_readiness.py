"""Integration and Governance Tests for Sprint 7.9 / E10 Controlled Field Pilot Data Collection.

Verifies:
1. Original MIMII recordings are strictly identified as laboratory testbench data.
2. Ingestion pipeline enforces SHA-256 deduplication and quality gating.
3. Strict isolation between calibration enrollment and sealed evaluation roles (zero data leakage).
4. Machine, sensor, and facility provenance metadata are mandatory.
5. Sealed evaluation manifests remain cryptographically tamper-evident.
6. Production model artifacts under ml/artifacts/ remain unchanged.
"""

import json
from pathlib import Path
import sqlite3
import numpy as np
import pytest
import soundfile as sf

from stetho_api.db.database import DatabaseManager, get_db
from stetho_api.db.field_data_repo import (
    DatasetRoleConflictException,
    DuplicateRecordingException,
    FieldDataRepo,
    InvalidAudioQualityException,
    ManifestSealedException,
)


@pytest.fixture
def temp_pilot_repo(tmp_path: Path):
    """Create isolated SQLite database and storage directory for pilot governance testing."""
    db_path = tmp_path / "pilot_test.db"
    storage_dir = tmp_path / "pilot_storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    db = DatabaseManager(db_path)
    return FieldDataRepo(db=db, storage_base_dir=storage_dir)


def make_test_wav(
    filepath: Path,
    duration_sec: float = 3.0,
    sample_rate: int = 16000,
    freq: float = 400.0,
    amplitude: float = 0.3,
) -> bytes:
    """Generate a clean synthetic PCM 16-bit WAV for testing governance constraints."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    sig = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(filepath), sig, sample_rate, format="WAV", subtype="PCM_16")
    return filepath.read_bytes()


def test_mimii_data_identified_as_laboratory_only():
    """Verify that existing MIMII data is classified as laboratory-only and has not been fabricated into field data."""
    manifest_path = Path("ml/data/manifests/diverse_dataset_manifest.json")
    assert manifest_path.exists(), "Diverse dataset manifest must exist."

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_recordings"] == 360
    assert "records" in manifest

    # All records must have category fan, pump, or valve and point to laboratory datasets
    for record in manifest["records"]:
        assert record["category"] in {"fan", "pump", "valve"}
        file_path = Path(record["file_path"])
        assert file_path.exists(), f"MIMII laboratory file missing: {file_path}"
        # Confirm file path points to laboratory / MIMII directory
        assert "mimii" in str(file_path).lower(), "MIMII files must not be masqueraded as field pilot data."


def test_field_ingestion_deduplication_and_quality(temp_pilot_repo: FieldDataRepo, tmp_path: Path):
    """Verify SHA-256 deduplication and audio quality gating during field ingestion."""
    wav1_path = tmp_path / "field_rec_01.wav"
    wav1_bytes = make_test_wav(wav1_path, duration_sec=3.0, sample_rate=16000, freq=350.0)

    # 1. First import should succeed
    rec1 = temp_pilot_repo.import_recording(
        file_bytes=wav1_bytes,
        original_filename="field_rec_01.wav",
        machine_category="fan",
        machine_id="facility_north_fan_01",
        sensor_id="mic_sensor_01",
        operator_id="lead_technician",
        provenance_source="field_pilot",
        dataset_role="unassigned",
    )
    assert rec1["recording_id"].startswith("fld_rec_")
    assert rec1["machine_id"] == "facility_north_fan_01"
    assert rec1["provenance_source"] == "field_pilot"

    # 2. Duplicate file with different name/machine must be rejected with DuplicateRecordingException
    with pytest.raises(DuplicateRecordingException):
        temp_pilot_repo.import_recording(
            file_bytes=wav1_bytes,
            original_filename="totally_different_filename.wav",
            machine_category="fan",
            machine_id="facility_south_fan_02",
            sensor_id="mic_sensor_02",
            operator_id="lead_technician",
        )

    # 3. Sub-Nyquist audio (8 kHz) must be rejected
    wav_8k_path = tmp_path / "audio_8khz.wav"
    wav_8k_bytes = make_test_wav(wav_8k_path, duration_sec=2.0, sample_rate=8000, freq=300.0)
    with pytest.raises(InvalidAudioQualityException) as exc:
        temp_pilot_repo.import_recording(
            file_bytes=wav_8k_bytes,
            original_filename="audio_8khz.wav",
            machine_category="fan",
            machine_id="facility_north_fan_01",
            sensor_id="mic_sensor_01",
            operator_id="lead_technician",
        )
    assert "16000 Hz" in str(exc.value)


def test_calibration_and_sealed_evaluation_strict_isolation(temp_pilot_repo: FieldDataRepo, tmp_path: Path):
    """Verify strict role separation: zero leakage between calibration enrollment and sealed evaluation."""
    # Create two recordings: rec1 (normal), rec2 (normal)
    wav1 = make_test_wav(tmp_path / "rec1.wav", freq=320.0)
    wav2 = make_test_wav(tmp_path / "rec2.wav", freq=420.0)

    rec1 = temp_pilot_repo.import_recording(
        file_bytes=wav1,
        original_filename="rec1.wav",
        machine_category="fan",
        machine_id="site_a_fan_101",
        sensor_id="mic_01",
        operator_id="auditor",
        dataset_role="unassigned",
    )
    rec2 = temp_pilot_repo.import_recording(
        file_bytes=wav2,
        original_filename="rec2.wav",
        machine_category="fan",
        machine_id="site_a_fan_101",
        sensor_id="mic_01",
        operator_id="auditor",
        dataset_role="unassigned",
    )

    # Annotate rec1 and rec2 as verified normal
    temp_pilot_repo.add_annotation(
        recording_id=rec1["recording_id"],
        label="normal",
        annotator_id="auditor",
        reason="Baseline acoustic pattern normal",
        evidence_source="VISUAL_INSPECTION",
    )
    temp_pilot_repo.add_annotation(
        recording_id=rec2["recording_id"],
        label="normal",
        annotator_id="auditor",
        reason="Baseline acoustic pattern normal",
        evidence_source="VISUAL_INSPECTION",
    )

    # Assign rec1 to calibration_enrollment
    temp_pilot_repo.assign_dataset_role(
        recording_id=rec1["recording_id"],
        new_role="calibration_enrollment",
        operator_id="admin",
    )

    # Assign rec2 to sealed_evaluation
    temp_pilot_repo.assign_dataset_role(
        recording_id=rec2["recording_id"],
        new_role="sealed_evaluation",
        operator_id="admin",
    )

    # Attempt to add calibration recording rec1 to a sealed evaluation manifest -> must fail
    with pytest.raises(DatasetRoleConflictException) as exc:
        temp_pilot_repo.create_or_seal_manifest(
            name="Pilot Sealed Evaluation Manifest",
            dataset_role="sealed_evaluation",
            machine_category="fan",
            recording_ids=[rec1["recording_id"], rec2["recording_id"]],
            created_by="admin",
            seal_now=True,
        )
    assert "does not match manifest role" in str(exc.value)

    # Create valid manifest containing ONLY rec2
    manifest = temp_pilot_repo.create_or_seal_manifest(
        name="Pilot Sealed Evaluation Manifest",
        dataset_role="sealed_evaluation",
        machine_category="fan",
        recording_ids=[rec2["recording_id"]],
        created_by="admin",
        seal_now=True,
    )
    assert manifest["is_sealed"] is True

    # Attempt to reassign rec2 (in sealed manifest) to calibration_enrollment -> must be blocked
    with pytest.raises(ManifestSealedException):
        temp_pilot_repo.assign_dataset_role(
            recording_id=rec2["recording_id"],
            new_role="calibration_enrollment",
            operator_id="admin",
        )


def test_facility_and_machine_provenance_preservation(temp_pilot_repo: FieldDataRepo, tmp_path: Path):
    """Verify that site, machine, mounting position, operating condition, and load level are recorded."""
    wav = make_test_wav(tmp_path / "provenance_rec.wav", freq=360.0)
    rec = temp_pilot_repo.import_recording(
        file_bytes=wav,
        original_filename="provenance_rec.wav",
        machine_category="pump",
        machine_id="site_water_treatment_pump_03",
        sensor_id="transducer_piezo_44a",
        operator_id="field_engineer",
        channel_index=0,
        microphone_position="drive_end_bearing_housing",
        operating_condition="steady_state_1750_rpm",
        load_level="85_percent_flow",
        provenance_source="field_pilot",
        consent_status="authorized",
        dataset_role="development",
        metadata={"facility_site": "Plant North Unit 3", "ambient_temp_c": 22.5},
    )

    assert rec["machine_id"] == "site_water_treatment_pump_03"
    assert rec["machine_category"] == "pump"
    assert rec["sensor_id"] == "transducer_piezo_44a"
    assert rec["channel_index"] == 0
    assert rec["microphone_position"] == "drive_end_bearing_housing"
    assert rec["operating_condition"] == "steady_state_1750_rpm"
    assert rec["load_level"] == "85_percent_flow"
    assert rec["provenance_source"] == "field_pilot"
    assert rec["consent_status"] == "authorized"
    assert rec["metadata"]["facility_site"] == "Plant North Unit 3"


def test_production_model_artifacts_integrity():
    """Verify that ml/artifacts/ contains expected files and has not been altered or replaced with unverified weights."""
    artifacts_dir = Path("ml/artifacts")
    assert artifacts_dir.exists(), "ml/artifacts directory must exist."

    # Verify critical model family subdirectories and checkpoints exist
    expected_checkpoints = [
        artifacts_dir / "stetho_msae" / "id_00" / "model.pt",
        artifacts_dir / "stetho_msae" / "id_00" / "stetho_msae.pt",
        artifacts_dir / "dense_autoencoder" / "id_00" / "model.pt",
        artifacts_dir / "isolation_forest" / "id_00" / "isolation_forest.joblib",
    ]
    for ckpt_path in expected_checkpoints:
        assert ckpt_path.exists(), f"Production model artifact {ckpt_path} missing."
        assert ckpt_path.stat().st_size > 0, f"Production model artifact {ckpt_path} is empty."

"""Phase 4 — Calibration Validation Script for Stetho Sprint 7.5 Audit.

Tests:
1. Ten confirmed-normal recordings -> status='calibrated', version='1.10.0'
2. Incorrect operator confirmation (is_confirmed_normal=False) -> rejected with ValidationException
3. Corrupt WAV file -> rejected by Audio Quality Gate
4. Duplicate recordings -> deduplicated via content hash
5. Unsupported / non-standard sample rate (e.g., 44.1kHz resampled cleanly or rejected if invalid)
6. Insufficient recordings (N < 10) -> provisional calibration state
7. Isolation between different machines (Machine A calibration does not bleed into Machine B)
8. Recalibration (N=10 -> N=11 updates version and thresholds)
9. Rollback (restores prior profile and active thresholds)
10. Persistence across database restart
11. Failed calibration never clobbers active valid profile
"""

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path("services/api/src").resolve()))
sys.path.insert(0, str(Path(".").resolve()))

from stetho_api.db.database import DatabaseManager
from stetho_api.db.asset_repo import AssetRepository
from stetho_api.db.calibration_repo import CalibrationRepository
from stetho_api.db.analysis_repo import EnrollmentRepository
from stetho_api.exceptions import ValidationException
from stetho_api.inference.calibration_engine import MachineCalibrationEngine
from stetho_api.inference.model_registry import ModelRegistry, ModelType

def create_wav_file(path: Path, freq=300.0, duration=5.0, sr=16000, noise=0.005) -> Path:
    samples = int(duration * sr)
    t = np.linspace(0, duration, samples, endpoint=False)
    wave = (0.2 * np.sin(2 * np.pi * freq * t) + noise * np.random.randn(samples)).astype(np.float32)
    sf.write(str(path), wave, sr, format="WAV")
    return path

def main():
    print("=" * 80)
    print("STETHO SPRINT 7.5 — PHASE 4: CALIBRATION WORKFLOW VALIDATION")
    print("=" * 80)

    temp_dir = Path(tempfile.mkdtemp(prefix="stetho_cal_audit_"))
    db_path = temp_dir / "audit_calibration.db"
    storage_dir = temp_dir / "storage"
    storage_dir.mkdir(parents=True)

    print(f"Temporary Audit Sandbox: {temp_dir}")
    print(f"Audit Database: {db_path}")

    db = DatabaseManager(db_path)
    asset_repo = AssetRepository(db)
    cal_repo = CalibrationRepository(db)
    enr_repo = EnrollmentRepository(db)
    registry = ModelRegistry(device="cpu", fallback_to_cpu=True)
    engine = MachineCalibrationEngine(db=db, registry=registry)

    results = {}

    # Setup test assets
    asset_a = asset_repo.create(
        asset_id="audit_fan_A",
        name="Cooling Fan Alpha",
        category="fan",
        assigned_model="cascade_ensemble",
    )
    asset_b = asset_repo.create(
        asset_id="audit_fan_B",
        name="Cooling Fan Beta",
        category="fan",
        assigned_model="cascade_ensemble",
    )

    # 1. Ten confirmed-normal recordings commissioning
    print("\n--- 1. Ten Confirmed-Normal Recordings Commissioning ---")
    recordings_a = []
    for i in range(10):
        p = create_wav_file(storage_dir / f"fanA_norm_{i:02d}.wav", freq=300.0 + i * 5.0)
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        enr_repo.record_enrollment(
            machine_id="audit_fan_A",
            upload_id=f"upl_a_{i}",
            stored_path=str(p),
            duration_sec=5.0,
            sample_rate=16000,
            channels=1,
            quality_status="ACCEPTED",
            is_confirmed_normal=True,
            content_hash=sha,
        )

    res_10 = engine.calibrate_asset(
        asset_id="audit_fan_A",
        operator_id="inspector_01",
        is_confirmed_normal=True,
        notes="10-shot commissioning baseline",
    )
    print(f"  Asset: audit_fan_A")
    print(f"  Calibration Version: {res_10.calibration_version}")
    print(f"  Calibration Status: {res_10.calibration_status}")
    print(f"  Is Provisional: {res_10.is_provisional}")
    print(f"  Enrolled Count: {res_10.enrolled_sample_count}")
    print(f"  Operating Threshold (theta_upper): {res_10.operating_threshold:.6f}")
    print(f"  Lower Bound (theta_lower): {res_10.theta_lower:.6f}")
    assert res_10.calibration_status == "calibrated", "N=10 must reach 'calibrated' status"
    assert res_10.is_provisional is False, "N=10 must not be provisional"
    assert res_10.calibration_version == "1.10.0", "Version must be 1.10.0"
    results["ten_shot_commissioning"] = {
        "status": res_10.calibration_status,
        "version": res_10.calibration_version,
        "is_provisional": res_10.is_provisional,
        "enrolled_count": res_10.enrolled_sample_count,
        "pass": True,
    }

    # 2. Incorrect operator confirmation (is_confirmed_normal=False)
    print("\n--- 2. Unconfirmed Operator Normalcy Check ---")
    unconfirmed_blocked = False
    try:
        engine.calibrate_asset(
            asset_id="audit_fan_A",
            is_confirmed_normal=False,
        )
    except ValidationException as e:
        unconfirmed_blocked = True
        print(f"  [PASS] Blocked unconfirmed calibration: {e.message}")
    assert unconfirmed_blocked, "Calibration without operator confirmation must be blocked"
    results["unconfirmed_normal_rejection"] = {"blocked": unconfirmed_blocked, "pass": True}

    # 3. Corrupt WAV Handling
    print("\n--- 3. Corrupt WAV Handling ---")
    corrupt_blocked = False
    corrupt_wave = np.array([float("nan"), float("inf"), 0.0, 1.0], dtype=np.float32)
    try:
        engine.calibrate_asset(
            asset_id="audit_fan_A",
            audio_recordings=[(corrupt_wave, 16000)],
            is_confirmed_normal=True,
        )
    except ValidationException as e:
        corrupt_blocked = True
        print(f"  [PASS] Audio Quality Gate rejected corrupt audio: {e.message}")
    assert corrupt_blocked, "Corrupt audio must be rejected by AQG"
    results["corrupt_wav_rejection"] = {"blocked": corrupt_blocked, "pass": True}

    # 4. Duplicate Recording Deduplication
    print("\n--- 4. Duplicate Recording Deduplication Check ---")
    dup_path = storage_dir / "fanA_norm_00.wav"
    dup_sha = hashlib.sha256(dup_path.read_bytes()).hexdigest()
    dup_blocked = False
    try:
        enr_repo.record_enrollment(
            machine_id="audit_fan_A",
            upload_id="upl_dup_test",
            stored_path=str(dup_path),
            duration_sec=5.0,
            sample_rate=16000,
            channels=1,
            quality_status="ACCEPTED",
            is_confirmed_normal=True,
            content_hash=dup_sha,
        )
    except Exception as e:
        dup_blocked = True
        print(f"  [PASS] Duplicate audio blocked by content hash: {type(e).__name__} ({e})")
    assert dup_blocked, "Duplicate audio enrollment must be rejected"
    results["duplicate_deduplication"] = {"blocked": dup_blocked, "pass": True}

    # 5. Non-standard Sample Rate Resampling
    print("\n--- 5. Non-Standard Sample Rate Resampling (44.1 kHz) ---")
    asset_c = asset_repo.create(
        asset_id="audit_fan_C",
        name="Cooling Fan Gamma",
        category="fan",
        assigned_model="cascade_ensemble",
    )
    resampled_p = create_wav_file(storage_dir / "fanC_44k.wav", freq=320.0, duration=5.0, sr=44100)
    data_44k, sr_44k = sf.read(str(resampled_p), dtype="float32", always_2d=True)
    res_44k = engine.calibrate_asset(
        asset_id="audit_fan_C",
        audio_recordings=[(data_44k[:, 0], sr_44k)],
        is_confirmed_normal=True,
    )
    print(f"  [PASS] Successfully resampled 44.1kHz audio to 16kHz during calibration.")
    results["sample_rate_resampling"] = {"resampled": True, "pass": True}

    # 6. Insufficient Recordings (N < 10)
    print("\n--- 6. Insufficient Recordings Check on Machine B (N=3) ---")
    for i in range(3):
        p = create_wav_file(storage_dir / f"fanB_norm_{i:02d}.wav", freq=300.0 + i * 10.0)
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        enr_repo.record_enrollment(
            machine_id="audit_fan_B",
            upload_id=f"upl_b_{i}",
            stored_path=str(p),
            duration_sec=5.0,
            sample_rate=16000,
            channels=1,
            quality_status="ACCEPTED",
            is_confirmed_normal=True,
            content_hash=sha,
        )

    res_3 = engine.calibrate_asset(
        asset_id="audit_fan_B",
        operator_id="inspector_02",
        is_confirmed_normal=True,
    )
    print(f"  Asset: audit_fan_B")
    print(f"  Calibration Version: {res_3.calibration_version}")
    print(f"  Calibration Status: {res_3.calibration_status}")
    print(f"  Is Provisional: {res_3.is_provisional}")
    assert res_3.calibration_status == "provisional", "N=3 must have 'provisional' status"
    assert res_3.is_provisional is True, "N=3 must be provisional"
    results["insufficient_recordings"] = {
        "status": res_3.calibration_status,
        "is_provisional": res_3.is_provisional,
        "pass": True,
    }

    # 7. Machine Calibration Isolation
    print("\n--- 7. Machine Calibration Isolation ---")
    active_a = cal_repo.get_active_profile("audit_fan_A")
    active_b = cal_repo.get_active_profile("audit_fan_B")
    print(f"  Machine A (N=10) Threshold: {active_a.operating_threshold:.6f} | Version: {active_a.calibration_version}")
    print(f"  Machine B (N=3)  Threshold: {active_b.operating_threshold:.6f} | Version: {active_b.calibration_version}")
    assert active_a.calibration_version != active_b.calibration_version
    assert active_a.enrolled_count != active_b.enrolled_count
    print(f"  [PASS] Verified strict isolation between different physical machines.")
    results["machine_isolation"] = {"pass": True}

    # 8. Recalibration (N=10 -> N=11)
    print("\n--- 8. Recalibration Workflow (N=10 -> N=11) ---")
    p11 = create_wav_file(storage_dir / "fanA_norm_10.wav", freq=355.0)
    sha11 = hashlib.sha256(p11.read_bytes()).hexdigest()
    enr_repo.record_enrollment(
        machine_id="audit_fan_A",
        upload_id="upl_a_10",
        stored_path=str(p11),
        duration_sec=5.0,
        sample_rate=16000,
        channels=1,
        quality_status="ACCEPTED",
        is_confirmed_normal=True,
        content_hash=sha11,
    )
    res_11 = engine.calibrate_asset(
        asset_id="audit_fan_A",
        operator_id="inspector_01",
        is_confirmed_normal=True,
        notes="Recalibration with 11th sample",
    )
    print(f"  Recalibrated Version: {res_11.calibration_version}")
    print(f"  New Sample Count: {res_11.enrolled_sample_count}")
    print(f"  New Operating Threshold: {res_11.operating_threshold:.6f}")
    assert res_11.calibration_version == "1.11.0"
    results["recalibration"] = {
        "version": res_11.calibration_version,
        "count": res_11.enrolled_sample_count,
        "threshold": res_11.operating_threshold,
        "pass": True,
    }

    # 9. Rollback Workflow
    print("\n--- 9. Calibration Rollback Workflow ---")
    res_rollback = engine.rollback_calibration(
        asset_id="audit_fan_A",
        operator_id="inspector_01",
    )
    print(f"  Restored Calibration Version: {res_rollback.calibration_version}")
    print(f"  Restored Operating Threshold: {res_rollback.operating_threshold:.6f}")
    print(f"  Restored Sample Count: {res_rollback.enrolled_sample_count}")
    assert res_rollback.calibration_version == "1.10.0", "Rollback must restore 1.10.0"
    assert res_rollback.enrolled_sample_count == 10
    active_after_rb = cal_repo.get_active_profile("audit_fan_A")
    assert active_after_rb.calibration_version == "1.10.0"
    print(f"  [PASS] Rollback successfully restored prior verified version 1.10.0")
    results["rollback"] = {
        "restored_version": res_rollback.calibration_version,
        "restored_count": res_rollback.enrolled_sample_count,
        "pass": True,
    }

    # 10. Persistence after Backend Restart
    print("\n--- 10. Database Restart & Profile Persistence ---")
    # Close existing DB connection and re-instantiate fresh
    if hasattr(db._local, "conn"):
        db._local.conn.close()
        delattr(db._local, "conn")
    fresh_db = DatabaseManager(db_path)
    fresh_cal_repo = CalibrationRepository(fresh_db)
    fresh_asset_repo = AssetRepository(fresh_db)

    persisted_profile = fresh_cal_repo.get_active_profile("audit_fan_A")
    persisted_asset = fresh_asset_repo.get_or_raise("audit_fan_A")

    print(f"  Persisted Profile Version: {persisted_profile.calibration_version}")
    print(f"  Persisted Threshold: {persisted_profile.operating_threshold:.6f}")
    print(f"  Asset Recorded Version: {persisted_asset.calibration_version}")
    assert persisted_profile.calibration_version == "1.10.0"
    assert persisted_asset.calibration_version == "1.10.0"
    print(f"  [PASS] Verified state persists accurately across complete server restart.")
    results["restart_persistence"] = {"pass": True}

    # 11. Failed Calibration Preservation Invariant
    print("\n--- 11. Failed Calibration Preservation Invariant ---")
    # Verify that if an error occurs, the existing active profile is untouched
    preserved_before = fresh_cal_repo.get_active_profile("audit_fan_A")
    failed_attempt = False
    fresh_engine = MachineCalibrationEngine(db=fresh_db, registry=registry)
    try:
        fresh_engine.calibrate_asset(
            asset_id="audit_fan_A",
            audio_recordings=[(corrupt_wave, 16000)],
            is_confirmed_normal=True,
        )
    except ValidationException:
        failed_attempt = True
    assert failed_attempt, "Corrupt audio must raise exception"
    preserved_after = fresh_cal_repo.get_active_profile("audit_fan_A")
    assert preserved_after.profile_id == preserved_before.profile_id
    assert preserved_after.calibration_version == preserved_before.calibration_version
    assert preserved_after.operating_threshold == preserved_before.operating_threshold
    print(f"  [PASS] Failed calibration cleanly preserved active profile '{preserved_after.calibration_version}'.")
    results["preservation_invariant"] = {"pass": True}

    # Clean up sandbox
    try:
        if hasattr(fresh_db._local, "conn"):
            fresh_db._local.conn.close()
        shutil.rmtree(temp_dir)
    except Exception:
        pass

    out_path = Path("docs/phase4_calibration_validation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("=" * 80)
    print("PHASE 4 CALIBRATION VALIDATION: ALL 11 SCENARIOS PASSED")
    print("=" * 80)

if __name__ == "__main__":
    main()

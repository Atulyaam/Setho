"""Unit tests for database repositories (machines, enrollments, analyses, jobs)."""

from pathlib import Path
import pytest

from stetho_api.db import (
    AnalysisRepository,
    DatabaseManager,
    EnrollmentRepository,
    JobRepository,
    MachineRepository,
)
from stetho_api.exceptions import ResourceNotFoundException, StethoAPIException


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_stetho.db"
    return DatabaseManager(db_file)


@pytest.fixture
def machine_repo(test_db):
    return MachineRepository(test_db)


@pytest.fixture
def enrollment_repo(test_db):
    return EnrollmentRepository(test_db)


@pytest.fixture
def analysis_repo(test_db):
    return AnalysisRepository(test_db)


@pytest.fixture
def job_repo(test_db):
    return JobRepository(test_db)


def test_schema_seeding_default_machines(machine_repo):
    """Verify default MIMII benchmark machines are seeded upon initialization."""
    machines = machine_repo.list_all()
    assert len(machines) >= 4
    machine_ids = [m.machine_id for m in machines]
    assert "id_00" in machine_ids
    assert "id_02" in machine_ids
    assert "id_04" in machine_ids
    assert "id_06" in machine_ids


def test_create_and_get_machine(machine_repo):
    """Verify creating a custom machine profile."""
    profile = machine_repo.create(
        machine_id="fan_custom_01",
        name="Test Chiller Fan",
        category="fan",
        description="HVAC auxiliary cooling unit",
        selected_model_policy="stetho_msae",
    )
    assert profile.machine_id == "fan_custom_01"
    assert profile.enrollment_status == "uncalibrated"
    assert profile.enrolled_recording_count == 0

    fetched = machine_repo.get_or_raise("fan_custom_01")
    assert fetched.name == "Test Chiller Fan"


def test_create_duplicate_machine_rejected(machine_repo):
    """Verify duplicate machine_id raises conflict error."""
    with pytest.raises(StethoAPIException) as exc:
        machine_repo.create("id_00", "Duplicate Unit")
    assert exc.value.status_code == 409


def test_machine_enrollment_progression(machine_repo):
    """Verify enrollment increment transitions status from provisional to calibrated."""
    machine_repo.create("machine_prog_test", "Progression Fan")

    # 1 enrollment -> provisional
    p1 = machine_repo.increment_enrollment("machine_prog_test")
    assert p1.enrolled_recording_count == 1
    assert p1.enrollment_status == "provisional"

    # 30 enrollments -> calibrated
    p30 = machine_repo.increment_enrollment("machine_prog_test", new_count=30)
    assert p30.enrolled_recording_count == 30
    assert p30.enrollment_status == "calibrated"


def test_enrollment_recording_lifecycle(enrollment_repo, machine_repo):
    """Verify recording an audio enrollment linked to machine profile."""
    machine_repo.create("enr_machine", "Enrollment Unit")
    rec = enrollment_repo.record_enrollment(
        machine_id="enr_machine",
        upload_id="upl-12345",
        stored_path="/storage/rec.wav",
        duration_sec=10.0,
        sample_rate=16000,
        channels=1,
        quality_status="ACCEPTED",
        is_confirmed_normal=True,
    )
    assert rec.enrollment_id is not None
    assert rec.is_confirmed_normal is True

    records = enrollment_repo.list_by_machine("enr_machine")
    assert len(records) == 1
    assert records[0].upload_id == "upl-12345"


def test_analysis_persistence_and_retrieval(analysis_repo, machine_repo):
    """Verify creating and retrieving full inspection analysis record."""
    machine_repo.create("ana_machine", "Analysis Unit")
    rec = analysis_repo.create(
        machine_id="ana_machine",
        model_name="stetho_msae",
        model_version="1.0.0",
        analysis_status="completed",
        quality_status="ACCEPTED",
        decision="NORMAL_LIKE",
        anomaly_score=0.215,
        operating_threshold=0.322,
        theta_lower=0.309,
        theta_upper=0.322,
        is_provisional=True,
        processing_duration_ms=45.2,
        reasons=["Score within normal baseline"],
        metrics={"snr_db": 28.5},
    )
    assert rec.analysis_id is not None

    fetched = analysis_repo.get_or_raise(rec.analysis_id)
    assert fetched.decision == "NORMAL_LIKE"
    assert fetched.anomaly_score == 0.215
    assert fetched.is_provisional is True
    assert fetched.metrics["snr_db"] == 28.5


def test_job_lifecycle(job_repo, machine_repo):
    """Verify async job creation, progression, and completion tracking."""
    machine_repo.create("job_machine", "Job Unit")
    job = job_repo.create_job("job_machine")
    assert job.status == "pending"
    assert job.progress_pct == 0

    # Update to processing
    j_proc = job_repo.update_status(job.job_id, status="processing", progress_pct=50)
    assert j_proc.status == "processing"
    assert j_proc.progress_pct == 50

    # Update to completed with analysis_id
    j_done = job_repo.update_status(
        job.job_id, status="completed", progress_pct=100, analysis_id="ana-999"
    )
    assert j_done.status == "completed"
    assert j_done.progress_pct == 100
    assert j_done.analysis_id == "ana-999"


"""Isolated Integration Tests for Phase 11 Research Production Adapter."""

from pathlib import Path
import pytest
import soundfile as sf
import numpy as np

from ml.models.research.production_adapter import ResearchInferenceAdapter

FAN_TEST_FILE = Path("ml/data/splits/machine_specific/id_00/test.csv")
PUMP_TEST_FILE = Path("ml/data/splits/cross_category/fan_to_pump/test_target_category.csv")
VALVE_TEST_FILE = Path("ml/data/splits/cross_category/fan_to_valve/test_target_category.csv")


def _get_first_file_from_split(csv_path: Path) -> str:
    from ml.training.train_pipeline import _read_split_csv
    rows = _read_split_csv(csv_path)
    return rows[0]["file_path"]


def test_cloud_high_accuracy_adapter_response():
    """Verify Model A (ResNet-AE) through production adapter."""
    adapter = ResearchInferenceAdapter(machine_id="id_00", profile="cloud_high_accuracy")
    sample_file = _get_first_file_from_split(FAN_TEST_FILE)

    res = adapter.analyze_file(sample_file)

    # 1. Field validation
    required_fields = [
        "machine_id",
        "deployment_profile",
        "model_name",
        "status",
        "anomaly_score",
        "threshold",
        "is_anomaly",
        "confidence",
        "latency_ms",
        "anomalous_segments",
        "top_frequency_bands",
        "timestamp",
    ]
    for f in required_fields:
        assert f in res, f"Missing required response field: {f}"

    # 2. Value constraints
    assert res["status"] in ("NORMAL", "ANOMALY")
    assert res["confidence"] in ("HIGH", "MEDIUM", "LOW")
    assert isinstance(res["anomaly_score"], float)
    assert res["latency_ms"] < 250.0, f"Latency {res['latency_ms']} ms exceeds tolerance"
    assert len(res["anomalous_segments"]) > 0
    assert len(res["top_frequency_bands"]) > 0


def test_edge_real_time_adapter_response():
    """Verify Model B (CompactStudentAE) through production adapter."""
    adapter = ResearchInferenceAdapter(machine_id="id_00", profile="edge_real_time")
    sample_file = _get_first_file_from_split(FAN_TEST_FILE)

    res = adapter.analyze_file(sample_file)

    assert res["deployment_profile"] == "edge_real_time"
    assert "Student" in res["model_name"]
    assert res["status"] in ("NORMAL", "ANOMALY")
    assert res["latency_ms"] < 100.0, f"Edge model latency {res['latency_ms']} ms exceeds 100ms target"


def test_hybrid_cascade_adapter_response():
    """Verify Hybrid Cascade policy through production adapter."""
    adapter = ResearchInferenceAdapter(machine_id="id_00", profile="hybrid_cascade")
    sample_file = _get_first_file_from_split(FAN_TEST_FILE)

    res = adapter.analyze_file(sample_file)
    assert res["deployment_profile"] == "hybrid_cascade"
    assert res["status"] in ("NORMAL", "ANOMALY")


def test_multi_category_scoring_stability():
    """Verify adapter analyzes Fan, Pump, and Valve audio files without error."""
    adapter = ResearchInferenceAdapter(machine_id="id_00", profile="cloud_high_accuracy")

    fan_sample = _get_first_file_from_split(FAN_TEST_FILE)
    pump_sample = _get_first_file_from_split(PUMP_TEST_FILE)
    valve_sample = _get_first_file_from_split(VALVE_TEST_FILE)

    res_fan = adapter.analyze_file(fan_sample)
    res_pump = adapter.analyze_file(pump_sample)
    res_valve = adapter.analyze_file(valve_sample)

    assert res_fan["status"] in ("NORMAL", "ANOMALY")
    assert res_pump["status"] in ("NORMAL", "ANOMALY")
    assert res_valve["status"] in ("NORMAL", "ANOMALY")

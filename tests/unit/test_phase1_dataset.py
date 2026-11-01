"""Unit and integrity tests for Phase 1 Diverse Industrial Dataset Expansion."""

import json
from pathlib import Path
import pytest
import soundfile as sf


def test_diverse_manifest_exists_and_valid():
    manifest_path = Path("ml/data/manifests/diverse_dataset_manifest.json")
    assert manifest_path.exists(), "Manifest file missing"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_recordings"] == 360
    assert manifest["total_categories"] == 3
    assert set(manifest["category_counts"].keys()) == {"fan", "pump", "valve"}
    assert manifest["duplicate_count"] == 0
    assert manifest["corrupted_count"] == 0


def test_audio_file_attributes_and_integrity():
    manifest_path = Path("ml/data/manifests/diverse_dataset_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Check a sample across each category
    categories_checked = set()
    for rec in manifest["records"]:
        cat = rec["category"]
        if cat not in categories_checked:
            file_path = Path(rec["file_path"])
            assert file_path.exists(), f"Audio file {file_path} not found on disk"
            assert file_path.stat().st_size == rec["file_size_bytes"]

            info = sf.info(str(file_path))
            assert info.samplerate == 16000
            assert info.channels == 8
            assert 9.9 <= info.duration <= 10.1
            categories_checked.add(cat)

    assert categories_checked == {"fan", "pump", "valve"}


def test_balanced_machine_id_distribution():
    manifest_path = Path("ml/data/manifests/diverse_dataset_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    machine_tallies = {}
    for rec in manifest["records"]:
        key = (rec["category"], rec["machine_id"], rec["label"])
        machine_tallies[key] = machine_tallies.get(key, 0) + 1

    expected_machines = ["id_00", "id_02", "id_04", "id_06"]
    expected_categories = ["fan", "pump", "valve"]

    for cat in expected_categories:
        for m_id in expected_machines:
            for label in ["normal", "abnormal"]:
                count = machine_tallies.get((cat, m_id, label), 0)
                assert count == 15, f"Expected 15 files for ({cat}, {m_id}, {label}), got {count}"


"""Unit tests for Dataset Splitting and Leakage Prevention."""

from pathlib import Path
import tempfile
import pytest

from ml.data.splits import create_machine_specific_split, create_cross_machine_splits


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


def test_machine_specific_splits_leakage_free(temp_dir):
    # Create synthetic records
    normal_files = [
        {"file_path": f"/data/norm_{i:04d}.wav", "machine_id": "id_00", "label": "normal", "channels": 8, "duration_sec": 10.0}
        for i in range(20)
    ]
    abnormal_files = [
        {"file_path": f"/data/abnorm_{i:04d}.wav", "machine_id": "id_00", "label": "abnormal", "channels": 8, "duration_sec": 10.0}
        for i in range(10)
    ]

    splits = create_machine_specific_split(
        machine_id="id_00",
        normal_files=normal_files,
        abnormal_files=abnormal_files,
        output_dir=temp_dir,
        train_ratio=0.7,
        dev_ratio=0.15,
    )

    train_paths = {x["file_path"] for x in splits["train_normal"]}
    dev_paths = {x["file_path"] for x in splits["dev_normal"]}
    test_paths = {x["file_path"] for x in splits["test"]}

    # Zero overlap
    assert len(train_paths.intersection(dev_paths)) == 0
    assert len(train_paths.intersection(test_paths)) == 0
    assert len(dev_paths.intersection(test_paths)) == 0

    # Total normal files conserved
    assert len(train_paths) + len(dev_paths) + len([x for x in splits["test"] if x["label"] == "normal"]) == 20

    # Zero abnormal in train or dev
    assert all(x["label"] == "normal" for x in splits["train_normal"])
    assert all(x["label"] == "normal" for x in splits["dev_normal"])

    # All abnormal are in test
    abnorm_in_test = [x for x in splits["test"] if x["label"] == "abnormal"]
    assert len(abnorm_in_test) == 10

    # CSV files written
    assert (temp_dir / "id_00" / "train_normal.csv").exists()
    assert (temp_dir / "id_00" / "dev_normal.csv").exists()
    assert (temp_dir / "id_00" / "test.csv").exists()

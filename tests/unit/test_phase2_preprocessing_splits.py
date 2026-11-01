"""Automated verification tests for Phase 2 Preprocessing & Split Protocols."""

import csv
import glob
from pathlib import Path
import numpy as np
import pytest

from ml.preprocessing.feature_suite import UnifiedFeatureSuite


def test_splits_no_overlap_and_no_contamination():
    """Verify Protocol A splits have zero file overlap and zero abnormal contamination in train/dev."""
    base_dir = Path("ml/data/splits/diverse_machine_specific")
    categories = ["fan", "pump", "valve"]
    machine_ids = ["id_00", "id_02", "id_04", "id_06"]

    for cat in categories:
        for m in machine_ids:
            m_dir = base_dir / cat / m
            assert m_dir.exists(), f"Split directory missing for {cat}/{m}"

            def read_paths_and_labels(split_name):
                with open(m_dir / f"{split_name}.csv", "r", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                    return {r["file_path"] for r in rows}, {r["label"] for r in rows}

            train_paths, train_labels = read_paths_and_labels("train_normal")
            dev_paths, dev_labels = read_paths_and_labels("dev_normal")
            test_paths, test_labels = read_paths_and_labels("test")

            # 1. Counts
            assert len(train_paths) == 10, f"Expected 10 train files for {cat}/{m}, got {len(train_paths)}"
            assert len(dev_paths) == 2, f"Expected 2 dev files for {cat}/{m}, got {len(dev_paths)}"
            assert len(test_paths) == 18, f"Expected 18 test files for {cat}/{m}, got {len(test_paths)}"

            # 2. No abnormal in train or dev
            assert train_labels == {"normal"}, f"Abnormal contamination in train for {cat}/{m}: {train_labels}"
            assert dev_labels == {"normal"}, f"Abnormal contamination in dev for {cat}/{m}: {dev_labels}"
            assert test_labels == {"normal", "abnormal"}, f"Test must contain both labels for {cat}/{m}"

            # 3. Zero file overlap
            assert len(train_paths.intersection(dev_paths)) == 0, f"train-dev overlap in {cat}/{m}"
            assert len(train_paths.intersection(test_paths)) == 0, f"train-test overlap in {cat}/{m}"
            assert len(dev_paths.intersection(test_paths)) == 0, f"dev-test overlap in {cat}/{m}"


def test_protocol_b_unseen_machine_splits():
    """Verify Protocol B (leave-one-machine-out) holds out the target machine completely."""
    base_dir = Path("ml/data/splits/unseen_machine")
    categories = ["fan", "pump", "valve"]
    machine_ids = ["id_00", "id_02", "id_04", "id_06"]

    for cat in categories:
        for heldout_m in machine_ids:
            b_dir = base_dir / cat / f"heldout_{heldout_m}"
            assert b_dir.exists(), f"Protocol B split directory missing: {b_dir}"

            with open(b_dir / "train_known_machines.csv", "r", encoding="utf-8") as f:
                train_rows = list(csv.DictReader(f))
            with open(b_dir / "test_unseen_machine.csv", "r", encoding="utf-8") as f:
                test_rows = list(csv.DictReader(f))

            assert len(train_rows) == 30  # 3 other machines * 10
            assert len(test_rows) == 18  # 3 normal + 15 abnormal

            # Check that heldout machine never appears in train_known
            train_machines = {r["machine_id"] for r in train_rows}
            assert heldout_m not in train_machines, f"Heldout machine {heldout_m} leaked into training set!"
            # Check test set contains only the heldout machine
            test_machines = {r["machine_id"] for r in test_rows}
            assert test_machines == {heldout_m}


def test_protocol_c_cross_category_splits():
    """Verify Protocol C cross-category splits preserve strict category separation."""
    base_dir = Path("ml/data/splits/cross_category")
    assert (base_dir / "fan_to_pump").exists()

    with open(base_dir / "fan_to_pump" / "train_source_category.csv", "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    with open(base_dir / "fan_to_pump" / "test_target_category.csv", "r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))

    assert len(train_rows) == 40  # 4 machines * 10
    assert len(test_rows) == 72   # 4 machines * 18

    assert all(r["category"] == "fan" for r in train_rows)
    assert all(r["category"] == "pump" for r in test_rows)


def test_feature_suite_presets():
    """Verify that all feature presets extract valid representations across categories."""
    presets = ["standard_64mel", "highres_96mel", "fast_512fft", "wide_2048fft", "multichannel_avg"]
    test_files = [
        "data/mimii/raw/fan/id_00/normal/00000000.wav",
        "data/mimii/raw/pump/id_00/normal/00000000.wav",
        "data/mimii/raw/valve/id_00/normal/00000000.wav",
    ]

    for p in presets:
        suite = UnifiedFeatureSuite(p)
        for tf in test_files:
            patches, info = suite.extract_patches(tf)
            assert patches.ndim == 4, f"Preset {p} output shape invalid: {patches.shape}"
            assert patches.shape[0] > 0, f"Preset {p} extracted 0 patches from {tf}"
            assert not np.isnan(patches).any(), f"NaN detected in {p} for {tf}"
            assert not np.isinf(patches).any(), f"Inf detected in {p} for {tf}"

    # Test statistical features
    suite_std = UnifiedFeatureSuite("standard_64mel")
    for tf in test_files:
        vec, stat_info = suite_std.extract_statistical_vector(tf)
        assert len(vec) == 38, f"Expected 38-dim vector, got {len(vec)}"
        assert not np.isnan(vec).any()

        # Test wavelet decomposition
        wav_res = suite_std.extract_wavelet_multiscale(tf, levels=4)
        assert len(wav_res["bands"]) == 4


def test_feature_determinism():
    """Verify bitwise determinism: repeated extraction yields identical floating point arrays."""
    suite = UnifiedFeatureSuite("standard_64mel")
    tf = "data/mimii/raw/fan/id_00/normal/00000000.wav"

    patches1, _ = suite.extract_patches(tf)
    patches2, _ = suite.extract_patches(tf)
    np.testing.assert_array_equal(patches1, patches2)

    vec1, _ = suite.extract_statistical_vector(tf)
    vec2, _ = suite.extract_statistical_vector(tf)
    np.testing.assert_array_equal(vec1, vec2)


"""Dataset Splitting and Data Leakage Prevention Engine for Stetho.

Produces reproducible, mathematically verified train/dev/test partitions:
1. Machine-specific splits (standard unsupervised anomaly detection setup):
   - train_normal.csv: ~70% normal recordings for self-supervised representation learning.
   - dev_normal.csv: ~15% normal recordings for threshold calibration and validation.
   - test.csv: remaining ~15% normal recordings + ALL abnormal recordings for final evaluation.
2. Cross-machine splits (leave-one-machine-out domain transfer evaluation):
   - train_source.csv: normal recordings from all machine IDs EXCEPT target.
   - test_target.csv: all test recordings from target machine ID.

LEAKAGE PREVENTION GUARANTEES:
- Splitting is performed exclusively at the FILE LEVEL prior to any temporal windowing.
- Abnormal files are strictly prohibited from train and dev sets (asserted).
- Disjoint intersection between train, dev, and test sets (asserted zero overlap).
"""

import argparse
import csv
import logging
from pathlib import Path
import random
from typing import Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stetho_splits")


def create_machine_specific_split(
    machine_id: str,
    normal_files: List[dict],
    abnormal_files: List[dict],
    output_dir: Path,
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[dict]]:
    """Create disjoint train_normal, dev_normal, and test partitions for a specific machine ID."""
    rng = random.Random(seed)
    norm_sorted = sorted(normal_files, key=lambda x: x["file_path"])
    rng.shuffle(norm_sorted)

    n_total = len(norm_sorted)
    n_train = int(n_total * train_ratio)
    n_dev = int(n_total * dev_ratio)
    # Ensure at least 1 file in dev and test if possible
    if n_total >= 3:
        n_train = max(1, n_train)
        n_dev = max(1, n_dev)
        if n_train + n_dev >= n_total:
            n_train = n_total - 2
            n_dev = 1

    train_norm = norm_sorted[:n_train]
    dev_norm = norm_sorted[n_train : n_train + n_dev]
    test_norm = norm_sorted[n_train + n_dev :]

    # Annotate partitions
    for item in train_norm:
        item["split"] = "train"
    for item in dev_norm:
        item["split"] = "dev"
    for item in test_norm:
        item["split"] = "test"

    abnorm_sorted = sorted(abnormal_files, key=lambda x: x["file_path"])
    test_abnorm = []
    for item in abnorm_sorted:
        c = dict(item)
        c["split"] = "test"
        test_abnorm.append(c)

    test_all = test_norm + test_abnorm

    # --- LEAKAGE PREVENTION AUDIT ---
    train_paths: Set[str] = {x["file_path"] for x in train_norm}
    dev_paths: Set[str] = {x["file_path"] for x in dev_norm}
    test_paths: Set[str] = {x["file_path"] for x in test_all}

    # 1. Zero overlap assertions
    train_dev_overlap = train_paths.intersection(dev_paths)
    train_test_overlap = train_paths.intersection(test_paths)
    dev_test_overlap = dev_paths.intersection(test_paths)
    assert not train_dev_overlap, f"LEAKAGE DETECTED: train and dev overlap: {train_dev_overlap}"
    assert not train_test_overlap, f"LEAKAGE DETECTED: train and test overlap: {train_test_overlap}"
    assert not dev_test_overlap, f"LEAKAGE DETECTED: dev and test overlap: {dev_test_overlap}"

    # 2. No abnormal in train or dev assertions
    assert all(x["label"] == "normal" for x in train_norm), "LEAKAGE: abnormal found in train"
    assert all(x["label"] == "normal" for x in dev_norm), "LEAKAGE: abnormal found in dev"

    # Save to disk
    m_dir = output_dir / machine_id
    m_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["file_path", "machine_id", "label", "split", "channels", "duration_sec"]

    def write_csv(path: Path, rows: List[dict]):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    write_csv(m_dir / "train_normal.csv", train_norm)
    write_csv(m_dir / "dev_normal.csv", dev_norm)
    write_csv(m_dir / "test.csv", test_all)

    logger.info(
        f"Machine {machine_id} split created: {len(train_norm)} train_normal, "
        f"{len(dev_norm)} dev_normal, {len(test_all)} test ({len(test_norm)} norm + {len(test_abnorm)} abnorm)"
    )

    return {
        "train_normal": train_norm,
        "dev_normal": dev_norm,
        "test": test_all,
    }


def create_cross_machine_splits(
    all_files_by_machine: Dict[str, Dict[str, List[dict]]],
    output_dir: Path,
) -> None:
    """Create leave-one-machine-out cross-machine evaluation partitions."""
    fieldnames = ["file_path", "machine_id", "label", "split", "channels", "duration_sec"]

    for target_m_id in all_files_by_machine:
        cross_dir = output_dir / target_m_id
        cross_dir.mkdir(parents=True, exist_ok=True)

        train_source = []
        for other_m_id, groups in all_files_by_machine.items():
            if other_m_id != target_m_id:
                for item in groups.get("normal", []):
                    c = dict(item)
                    c["split"] = "train"
                    train_source.append(c)

        test_target = []
        for item in all_files_by_machine[target_m_id].get("normal", []):
            c = dict(item)
            c["split"] = "test"
            test_target.append(c)
        for item in all_files_by_machine[target_m_id].get("abnormal", []):
            c = dict(item)
            c["split"] = "test"
            test_target.append(c)

        # Audit
        source_paths = {x["file_path"] for x in train_source}
        target_paths = {x["file_path"] for x in test_target}
        assert not source_paths.intersection(target_paths), "Cross-machine split leak!"

        def write_csv(path: Path, rows: List[dict]):
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)

        write_csv(cross_dir / "train_source.csv", train_source)
        write_csv(cross_dir / "test_target.csv", test_target)

        logger.info(
            f"Cross-machine split for target={target_m_id}: {len(train_source)} source train, "
            f"{len(test_target)} target test"
        )


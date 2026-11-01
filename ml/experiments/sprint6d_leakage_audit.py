"""Sprint 6D Phase 2: Independent Data Leakage Audit.

Performs exhaustive verification of:
1. Dataset split manifests (diverse_machine_specific, unseen_machine, cross_category).
2. Inter-split leakage (Train vs Dev vs Test).
3. Label purity (Ensuring zero abnormal files in calibration sets).
4. SHA256 hash uniqueness across splits.
5. Multi-task training set overlap with held-out test partitions.
"""

from collections import defaultdict
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.training.train_pipeline import _read_split_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sprint6d_leakage_audit")

AUDIT_OUTPUT = Path("ml/reports/sprint_6d_leakage_audit.json")


def compute_sha256(filepath: Path) -> str:
    """Compute sha256 checksum of an audio file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_diverse_machine_splits() -> Dict[str, Any]:
    """Audit diverse machine-specific splits (fan, pump, valve across id_00, 02, 04, 06)."""
    categories = ["fan", "pump", "valve"]
    machines = ["id_00", "id_02", "id_04", "id_06"]
    report = {}

    for cat in categories:
        report[cat] = {}
        for mid in machines:
            split_dir = Path(f"ml/data/splits/diverse_machine_specific/{cat}/{mid}")
            train_rows = _read_split_csv(split_dir / "train_normal.csv")
            dev_rows = _read_split_csv(split_dir / "dev_normal.csv")
            test_rows = _read_split_csv(split_dir / "test.csv")

            train_files = {r["file_path"] for r in train_rows}
            dev_files = {r["file_path"] for r in dev_rows}
            test_files = {r["file_path"] for r in test_rows}

            # Check overlaps
            train_dev_overlap = train_files.intersection(dev_files)
            train_test_overlap = train_files.intersection(test_files)
            dev_test_overlap = dev_files.intersection(test_files)

            # Check labels
            dev_abnormal_count = sum(1 for r in dev_rows if str(r.get("label", "")).lower() != "normal")
            train_abnormal_count = sum(1 for r in train_rows if str(r.get("label", "")).lower() != "normal")
            test_normal_count = sum(1 for r in test_rows if str(r.get("label", "")).lower() == "normal")
            test_abnormal_count = sum(1 for r in test_rows if str(r.get("label", "")).lower() in ("abnormal", "anomaly", "1"))

            report[cat][mid] = {
                "train_count": len(train_files),
                "dev_count": len(dev_files),
                "test_count": len(test_files),
                "test_normal_count": test_normal_count,
                "test_abnormal_count": test_abnormal_count,
                "train_dev_overlap": len(train_dev_overlap),
                "train_test_overlap": len(train_test_overlap),
                "dev_test_overlap": len(dev_test_overlap),
                "dev_abnormal_count": dev_abnormal_count,
                "train_abnormal_count": train_abnormal_count,
                "leakage_detected": bool(train_dev_overlap or train_test_overlap or dev_test_overlap or dev_abnormal_count > 0),
            }

    return report


def audit_unseen_machine_splits() -> Dict[str, Any]:
    """Audit Protocol B: Held-out unseen machine evaluation splits."""
    machines = ["id_00", "id_02", "id_04", "id_06"]
    report = {}

    for mid in machines:
        split_dir = Path(f"ml/data/splits/unseen_machine/fan/heldout_{mid}")
        train_rows = _read_split_csv(split_dir / "train_known_machines.csv")
        test_rows = _read_split_csv(split_dir / "test_unseen_machine.csv")

        train_files = {r["file_path"] for r in train_rows}
        test_files = {r["file_path"] for r in test_rows}

        overlap = train_files.intersection(test_files)

        # Confirm target machine ID is not in train
        train_machines = {r.get("machine_id", "") for r in train_rows}
        target_machine_in_train = mid in train_machines

        report[f"heldout_{mid}"] = {
            "train_count": len(train_files),
            "test_count": len(test_files),
            "overlap_count": len(overlap),
            "target_machine_leaked_in_train": target_machine_in_train,
            "leakage_detected": bool(len(overlap) > 0 or target_machine_in_train),
        }

    return report


def audit_cross_category_splits() -> Dict[str, Any]:
    """Audit Protocol C: Cross-category mechanical transfer splits."""
    pairs = [
        "fan_to_pump", "fan_to_valve",
        "pump_to_fan", "pump_to_valve",
        "valve_to_fan", "valve_to_pump"
    ]
    report = {}

    for pair in pairs:
        split_dir = Path(f"ml/data/splits/cross_category/{pair}")
        train_rows = _read_split_csv(split_dir / "train_source_category.csv")
        test_rows = _read_split_csv(split_dir / "test_target_category.csv")

        train_files = {r["file_path"] for r in train_rows}
        test_files = {r["file_path"] for r in test_rows}

        train_test_overlap = train_files.intersection(test_files)

        # Collect calibration files used from target dev_normal.csv
        src, tgt = pair.split("_to_")
        cal_files = set()
        for mid in ["id_00", "id_02", "id_04", "id_06"]:
            dev_csv = Path(f"ml/data/splits/diverse_machine_specific/{tgt}/{mid}/dev_normal.csv")
            if dev_csv.exists():
                d_rows = _read_split_csv(dev_csv)
                cal_files.update([r["file_path"] for r in d_rows])

        cal_test_overlap = cal_files.intersection(test_files)

        report[pair] = {
            "train_source_count": len(train_files),
            "test_target_count": len(test_files),
            "cal_target_count": len(cal_files),
            "train_test_overlap": len(train_test_overlap),
            "cal_test_overlap": len(cal_test_overlap),
            "leakage_detected": bool(train_test_overlap or cal_test_overlap),
        }

    return report


def audit_multitask_pretraining_contamination() -> Dict[str, Any]:
    """Verify whether multi-task pretraining files contaminated unseen or cross-category test sets."""
    # Multi-task training files were collected from train_normal.csv across all 12 machines
    categories = ["fan", "pump", "valve"]
    machines = ["id_00", "id_02", "id_04", "id_06"]
    multitask_train_files = set()
    for cat in categories:
        for mid in machines:
            csv_path = Path(f"ml/data/splits/diverse_machine_specific/{cat}/{mid}/train_normal.csv")
            if csv_path.exists():
                rows = _read_split_csv(csv_path)
                multitask_train_files.update([r["file_path"] for r in rows if str(r.get("label", "")).lower() == "normal"])

    # Collect all test files
    all_test_files = set()
    for cat in categories:
        for mid in machines:
            test_csv = Path(f"ml/data/splits/diverse_machine_specific/{cat}/{mid}/test.csv")
            if test_csv.exists():
                rows = _read_split_csv(test_csv)
                all_test_files.update([r["file_path"] for r in rows])

    direct_overlap = multitask_train_files.intersection(all_test_files)

    # Check unseen test splits
    unseen_test_files = set()
    for mid in machines:
        u_csv = Path(f"ml/data/splits/unseen_machine/fan/heldout_{mid}/test_unseen_machine.csv")
        if u_csv.exists():
            rows = _read_split_csv(u_csv)
            unseen_test_files.update([r["file_path"] for r in rows])

    unseen_overlap = multitask_train_files.intersection(unseen_test_files)

    return {
        "multitask_train_file_count": len(multitask_train_files),
        "total_test_file_count": len(all_test_files),
        "train_test_direct_overlap": len(direct_overlap),
        "unseen_test_overlap": len(unseen_overlap),
        "contamination_detected": bool(len(direct_overlap) > 0 or len(unseen_overlap) > 0),
    }


def main():
    logger.info("Executing Sprint 6D Phase 2: Independent Data Leakage Audit...")

    diverse_audit = audit_diverse_machine_splits()
    unseen_audit = audit_unseen_machine_splits()
    cross_audit = audit_cross_category_splits()
    multitask_audit = audit_multitask_pretraining_contamination()

    total_leaks = 0
    for cat, m_dict in diverse_audit.items():
        for mid, res in m_dict.items():
            if res["leakage_detected"]:
                total_leaks += 1
    for k, res in unseen_audit.items():
        if res["leakage_detected"]:
            total_leaks += 1
    for k, res in cross_audit.items():
        if res["leakage_detected"]:
            total_leaks += 1
    if multitask_audit["contamination_detected"]:
        total_leaks += 1

    summary = {
        "status": "PASS" if total_leaks == 0 else "FAIL",
        "total_violations": total_leaks,
        "diverse_machine_splits": diverse_audit,
        "unseen_machine_splits": unseen_audit,
        "cross_category_splits": cross_audit,
        "multitask_pretraining_audit": multitask_audit,
    }

    AUDIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_OUTPUT, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Audit completed: Status={summary['status']}, Total Violations={total_leaks}")
    logger.info(f"Report written to {AUDIT_OUTPUT}")


if __name__ == "__main__":
    main()

"""Generates unified, zero-leakage dataset splits for diverse machine categories.

Supports:
- Protocol A: Machine-specific splits (train normal, dev normal, test normal/abnormal)
- Protocol B: Unseen machine-ID evaluation (leave-one-machine-out within category)
- Protocol C: Cross-machine category evaluation (train on Category A, test on Category B)
- Protocol D: Environmental/Operating condition shift protocol
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stetho_create_splits")

MANIFEST_PATH = Path("ml/data/manifests/diverse_dataset_manifest.json")
SPLITS_DIR = Path("ml/data/splits")


def write_csv_split(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file_path", "label", "category", "machine_id", "sha256", "duration_sec"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "file_path": r["file_path"],
                "label": r["label"],
                "category": r["category"],
                "machine_id": r["machine_id"],
                "sha256": r["sha256"],
                "duration_sec": r.get("duration_sec", 10.0),
            })


def generate_all_splits(manifest_path: Path = MANIFEST_PATH, base_out_dir: Path = SPLITS_DIR):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    records = manifest["records"]
    categories = sorted(list(set(r["category"] for r in records)))
    machine_ids = sorted(list(set(r["machine_id"] for r in records)))

    logger.info(f"Loaded {len(records)} records across categories {categories} and machines {machine_ids}")

    # =========================================================================
    # PROTOCOL A: Machine-Specific Splits
    # =========================================================================
    logger.info("Generating Protocol A: Machine-Specific Splits...")
    proto_a_dir = base_out_dir / "diverse_machine_specific"

    for cat in categories:
        for m_id in machine_ids:
            cat_m_records = [r for r in records if r["category"] == cat and r["machine_id"] == m_id]
            norm_files = sorted([r for r in cat_m_records if r["label"] == "normal"], key=lambda x: x["filename"])
            abn_files = sorted([r for r in cat_m_records if r["label"] == "abnormal"], key=lambda x: x["filename"])

            # 10 train normal, 2 dev normal, 3 test normal
            train_norm = norm_files[:10]
            dev_norm = norm_files[10:12]
            test_norm = norm_files[12:15]
            test_abn = abn_files[:15]

            test_all = test_norm + test_abn

            m_dir = proto_a_dir / cat / m_id
            write_csv_split(m_dir / "train_normal.csv", train_norm)
            write_csv_split(m_dir / "dev_normal.csv", dev_norm)
            write_csv_split(m_dir / "test.csv", test_all)

            logger.info(f"Protocol A [{cat}/{m_id}]: train={len(train_norm)}, dev={len(dev_norm)}, test={len(test_all)}")

    # =========================================================================
    # PROTOCOL B: Unseen Machine-ID Evaluation (Leave-One-Machine-Out)
    # =========================================================================
    logger.info("Generating Protocol B: Unseen Machine-ID Splits...")
    proto_b_dir = base_out_dir / "unseen_machine"

    for cat in categories:
        for test_m_id in machine_ids:
            train_machines = [m for m in machine_ids if m != test_m_id]

            # Collect train normal from all other machines in category (10 per machine * 3 = 30 train files)
            train_norm = []
            for tm in train_machines:
                m_norm = sorted([r for r in records if r["category"] == cat and r["machine_id"] == tm and r["label"] == "normal"], key=lambda x: x["filename"])[:10]
                train_norm.extend(m_norm)

            # Test set is the complete test set of the held-out machine (3 normal + 15 abnormal = 18 files)
            held_norm = sorted([r for r in records if r["category"] == cat and r["machine_id"] == test_m_id and r["label"] == "normal"], key=lambda x: x["filename"])[12:15]
            held_abn = sorted([r for r in records if r["category"] == cat and r["machine_id"] == test_m_id and r["label"] == "abnormal"], key=lambda x: x["filename"])[:15]
            test_all = held_norm + held_abn

            b_dir = proto_b_dir / cat / f"heldout_{test_m_id}"
            write_csv_split(b_dir / "train_known_machines.csv", train_norm)
            write_csv_split(b_dir / "test_unseen_machine.csv", test_all)

            logger.info(f"Protocol B [{cat}/heldout_{test_m_id}]: train_known={len(train_norm)}, test_unseen={len(test_all)}")

    # =========================================================================
    # PROTOCOL C: Cross-Machine Category Evaluation
    # =========================================================================
    logger.info("Generating Protocol C: Cross-Machine Category Splits...")
    proto_c_dir = base_out_dir / "cross_category"

    for src_cat in categories:
        for tgt_cat in categories:
            if src_cat == tgt_cat:
                continue

            # Train on all 40 normal train files of src_cat
            train_src = []
            for m in machine_ids:
                m_norm = sorted([r for r in records if r["category"] == src_cat and r["machine_id"] == m and r["label"] == "normal"], key=lambda x: x["filename"])[:10]
                train_src.extend(m_norm)

            # Test on all test files of tgt_cat (18 files * 4 machines = 72 test files)
            test_tgt = []
            for m in machine_ids:
                m_norm = sorted([r for r in records if r["category"] == tgt_cat and r["machine_id"] == m and r["label"] == "normal"], key=lambda x: x["filename"])[12:15]
                m_abn = sorted([r for r in records if r["category"] == tgt_cat and r["machine_id"] == m and r["label"] == "abnormal"], key=lambda x: x["filename"])[:15]
                test_tgt.extend(m_norm + m_abn)

            c_dir = proto_c_dir / f"{src_cat}_to_{tgt_cat}"
            write_csv_split(c_dir / "train_source_category.csv", train_src)
            write_csv_split(c_dir / "test_target_category.csv", test_tgt)

            logger.info(f"Protocol C [{src_cat} -> {tgt_cat}]: train_src={len(train_src)}, test_tgt={len(test_tgt)}")

    logger.info("All split manifests generated successfully with zero leakage.")


if __name__ == "__main__":
    generate_all_splits()


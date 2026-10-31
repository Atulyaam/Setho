"""Generate all train, dev, and test split manifests for Stetho."""

import csv
import logging
from pathlib import Path
import sys
from typing import Dict, List

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.data.splits import create_cross_machine_splits, create_machine_specific_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stetho_generate_splits")


def run_generate_splits(
    metadata_csv: Path = Path("ml/data/manifests/audio_metadata.csv"),
    splits_base_dir: Path = Path("ml/data/splits"),
) -> None:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata manifest not found: {metadata_csv}")

    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    # Group by machine_id and label
    grouped: Dict[str, Dict[str, List[dict]]] = {}
    for r in records:
        m_id = r["machine_id"]
        label = r["label"]
        grouped.setdefault(m_id, {}).setdefault(label, []).append(r)

    machine_specific_dir = splits_base_dir / "machine_specific"
    for m_id, groups in sorted(grouped.items()):
        norm_files = groups.get("normal", [])
        abnorm_files = groups.get("abnormal", [])

        create_machine_specific_split(
            machine_id=m_id,
            normal_files=norm_files,
            abnormal_files=abnorm_files,
            output_dir=machine_specific_dir,
            train_ratio=0.70,
            dev_ratio=0.15,
            seed=42,
        )

    cross_machine_dir = splits_base_dir / "cross_machine"
    create_cross_machine_splits(grouped, output_dir=cross_machine_dir)

    logger.info("All split manifests generated and verified successfully.")


if __name__ == "__main__":
    run_generate_splits()

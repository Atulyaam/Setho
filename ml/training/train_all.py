"""Master Training and Evaluation Pipeline Runner for Stetho Sprint 2.

Orchestrates sequential training and evaluation:
1. Isolation Forest for all 4 fan IDs (id_00, id_02, id_04, id_06)
2. Dense Autoencoder for all 4 fan IDs (id_00, id_02, id_04, id_06)
3. Cross-Machine leave-one-machine-out evaluation
4. Computational benchmarking (latency, RAM, VRAM, model size)
5. Experiment leaderboard and baseline comparison report generation
"""

import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.benchmarking.benchmark_models import run_all_benchmarks
from ml.configs.experiment_config import ExperimentConfig
from ml.evaluation.cross_machine_evaluation import run_cross_machine_fold
from ml.reports.generate_leaderboard import generate_leaderboard_and_report
from ml.training.train_pipeline import run_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stetho_train_all")


def main():
    machine_ids = ["id_00", "id_02", "id_04", "id_06"]
    if_config_path = Path("ml/configs/isolation_forest_default.yaml")
    dae_config_path = Path("ml/configs/dense_autoencoder_default.yaml")

    # 1. Train Machine-Specific Isolation Forests
    logger.info("==================================================")
    logger.info("PHASE 1: Machine-Specific Isolation Forest Models")
    logger.info("==================================================")
    for m_id in machine_ids:
        cfg = ExperimentConfig.from_yaml(if_config_path)
        cfg.set_machine_id(m_id)
        run_experiment(cfg)

    # 2. Train Machine-Specific Dense Autoencoders
    logger.info("==================================================")
    logger.info("PHASE 2: Machine-Specific Dense Autoencoder Models")
    logger.info("==================================================")
    for m_id in machine_ids:
        cfg = ExperimentConfig.from_yaml(dae_config_path)
        cfg.set_machine_id(m_id)
        run_experiment(cfg)

    # 3. Cross-Machine Evaluations
    logger.info("==================================================")
    logger.info("PHASE 3: Cross-Machine Evaluation (Leave-One-Out)")
    logger.info("==================================================")
    for m_id in machine_ids:
        run_cross_machine_fold(target_machine_id=m_id, model_type="isolation_forest")
        run_cross_machine_fold(target_machine_id=m_id, model_type="dense_autoencoder")

    # 4. Benchmarking
    logger.info("==================================================")
    logger.info("PHASE 4: Latency & Memory Benchmarking")
    logger.info("==================================================")
    run_all_benchmarks()

    # 5. Generate Leaderboard and Report
    logger.info("==================================================")
    logger.info("PHASE 5: Leaderboard & Comparison Report")
    logger.info("==================================================")
    generate_leaderboard_and_report()

    logger.info("All Sprint 2 training and evaluation tasks completed successfully.")


if __name__ == "__main__":
    main()

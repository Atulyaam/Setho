"""Training and Evaluation Orchestration Pipeline for Stetho Sprint 2.

Manages machine-specific detector fitting, statistical threshold calibration,
and final-test evaluation with strict partition isolation and reproducibility.
"""

import csv
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ml.calibration.statistical_calibration import StatisticalCalibrator
from ml.configs.experiment_config import ExperimentConfig
from ml.evaluation.metrics import compute_anomaly_metrics
from ml.models.base import BaseAnomalyDetector
from ml.models.baseline_autoencoder.autoencoder_detector import DenseAutoencoderDetector
from ml.models.classical.isolation_forest import IsolationForestDetector

logger = logging.getLogger("stetho_pipeline")


def _read_split_csv(csv_path: Union[str, Path]) -> List[Dict[str, str]]:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Split manifest not found: {csv_path}")
    with open(p, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Execute complete machine-specific baseline experiment."""
    logger.info(
        f"=== Starting Experiment '{config.experiment_id}' "
        f"[Model: {config.model.model_type}, Machine: {config.machine_id}] ==="
    )
    start_total_time = time.perf_counter()

    # 1. Load splits
    train_rows = _read_split_csv(config.train_split)
    dev_rows = _read_split_csv(config.dev_split)
    test_rows = _read_split_csv(config.test_split)

    train_paths = [r["file_path"] for r in train_rows]
    dev_paths = [r["file_path"] for r in dev_rows]
    test_paths = [r["file_path"] for r in test_rows]

    # --- ZERO LEAKAGE ASSERTIONS ---
    train_set = set(train_paths)
    dev_set = set(dev_paths)
    test_set = set(test_paths)

    assert not train_set.intersection(dev_set), f"CRITICAL: Train and Dev overlap in {config.machine_id}"
    assert not train_set.intersection(test_set), f"CRITICAL: Train and Test overlap in {config.machine_id}"
    assert not dev_set.intersection(test_set), f"CRITICAL: Dev and Test overlap in {config.machine_id}"

    assert all(r["label"] == "normal" for r in train_rows), "CRITICAL: Non-normal sample in train_split"
    assert all(r["label"] == "normal" for r in dev_rows), "CRITICAL: Non-normal sample in dev_split"

    logger.info(
        f"Partitions verified: {len(train_paths)} train (normal), "
        f"{len(dev_paths)} dev (normal), {len(test_paths)} test"
    )

    # 2. Instantiate detector
    detector: BaseAnomalyDetector
    if config.model.model_type == "isolation_forest":
        detector = IsolationForestDetector(
            machine_id=config.machine_id,
            n_estimators=config.model.n_estimators,
            contamination=config.model.contamination,
            max_features=config.model.max_features,
            random_state=config.seed,
        )
    elif config.model.model_type == "dense_autoencoder":
        detector = DenseAutoencoderDetector(
            machine_id=config.machine_id,
            latent_dim=config.model.latent_dim,
            hidden_dims=config.model.hidden_dims,
            learning_rate=config.model.learning_rate,
            weight_decay=config.model.weight_decay,
            batch_size=config.model.batch_size,
            max_epochs=config.model.max_epochs,
            early_stopping_patience=config.model.early_stopping_patience,
            device=config.device,
            seed=config.seed,
        )
    else:
        raise ValueError(f"Unknown model_type: {config.model.model_type}")

    # 3. Fit on train files (and dev for validation loss)
    train_meta = detector.fit(train_paths, dev_files=dev_paths)

    # 4. Calibrate threshold on dev files only
    calibrator = StatisticalCalibrator(target_quantile=0.95)
    calibration_profile = calibrator.calibrate(detector, dev_paths)

    # 5. Evaluate on untouched final-test set
    logger.info(f"Evaluating detector on {len(test_paths)} final-test recordings...")
    test_scores = []
    test_labels = []

    for r in test_rows:
        score = detector.score_file(r["file_path"])
        label_int = 1 if r["label"] == "abnormal" else 0
        test_scores.append(score)
        test_labels.append(label_int)

    metrics = compute_anomaly_metrics(
        y_true=test_labels,
        y_scores=test_scores,
        threshold=detector.threshold,
    )

    total_duration = time.perf_counter() - start_total_time

    # 6. Save artifacts
    artifact_dir = Path(config.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    detector.save(artifact_dir)
    calibration_profile.save_json(artifact_dir / "calibration.json")
    config.save_yaml(artifact_dir / "config.yaml")

    eval_results = {
        "experiment_id": config.experiment_id,
        "machine_id": config.machine_id,
        "model_type": config.model.model_type,
        "metrics": metrics,
        "test_scores": [round(s, 6) for s in test_scores],
        "test_labels": test_labels,
        "calibration": calibration_profile.to_dict(),
        "training_metadata": train_meta,
        "total_experiment_duration_sec": round(total_duration, 4),
    }
    with open(artifact_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    logger.info(
        f"Experiment '{config.experiment_id}' completed in {total_duration:.2f}s | "
        f"ROC-AUC: {metrics['roc_auc']} | pAUC: {metrics['pauc']} | "
        f"F1: {metrics['f1']} | Recall: {metrics['recall']} | FPR: {metrics['fpr']}"
    )

    return eval_results


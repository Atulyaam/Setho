"""Cross-Machine (Leave-One-Machine-Out) Evaluation Framework for Stetho.

Evaluates domain transfer and zero-shot vs few-shot generalization:
- Train on pooled normal recordings from 3 source machines.
- Evaluate zero-shot discrimination (ROC-AUC, pAUC) on the held-out machine.
- Evaluate few-shot threshold calibration using normal development audio from the target machine.
"""

import csv
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

from ml.calibration.statistical_calibration import StatisticalCalibrator
from ml.configs.experiment_config import ExperimentConfig, ModelHyperparameters, FeatureConfig
from ml.evaluation.metrics import compute_anomaly_metrics
from ml.models.base import BaseAnomalyDetector
from ml.models.baseline_autoencoder.autoencoder_detector import DenseAutoencoderDetector
from ml.models.classical.isolation_forest import IsolationForestDetector

logger = logging.getLogger("stetho_cross_machine")


def run_cross_machine_fold(
    target_machine_id: str,
    model_type: str = "isolation_forest",
    splits_dir: Path = Path("ml/data/splits"),
    output_base_dir: Path = Path("ml/artifacts/cross_machine"),
    seed: int = 42,
    device: str = "auto",
) -> Dict[str, Any]:
    """Execute one leave-one-machine-out cross-machine fold."""
    artifact_dir = output_base_dir / model_type / target_machine_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    res_file = artifact_dir / "cross_evaluation_results.json"
    if res_file.exists():
        logger.info(f"Cross fold {target_machine_id} [{model_type}] already exists at {res_file}. Skipping retraining.")
        with open(res_file, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info(f"=== Cross-Machine Fold: Target={target_machine_id}, Model={model_type} ===")
    start_time = time.perf_counter()

    cross_dir = splits_dir / "cross_machine" / target_machine_id
    src_csv = cross_dir / "train_source.csv"
    tgt_csv = cross_dir / "test_target.csv"
    dev_csv = splits_dir / "machine_specific" / target_machine_id / "dev_normal.csv"

    with open(src_csv, "r", encoding="utf-8") as f:
        src_rows = list(csv.DictReader(f))
    with open(tgt_csv, "r", encoding="utf-8") as f:
        tgt_rows = list(csv.DictReader(f))
    with open(dev_csv, "r", encoding="utf-8") as f:
        dev_rows = list(csv.DictReader(f))

    train_paths = [r["file_path"] for r in src_rows]
    test_paths = [r["file_path"] for r in tgt_rows]
    dev_paths = [r["file_path"] for r in dev_rows]

    # Verify zero target leakage into train
    assert all(r["machine_id"] != target_machine_id for r in src_rows), "Target machine in train source!"
    assert all(r["label"] == "normal" for r in src_rows), "Abnormal in train source!"

    # 1. Instantiate model
    detector: BaseAnomalyDetector
    if model_type == "isolation_forest":
        detector = IsolationForestDetector(
            machine_id=f"cross_{target_machine_id}",
            n_estimators=200,
            random_state=seed,
        )
    elif model_type == "dense_autoencoder":
        detector = DenseAutoencoderDetector(
            machine_id=f"cross_{target_machine_id}",
            latent_dim=32,
            hidden_dims=[512, 128],
            learning_rate=0.001,
            max_epochs=40,
            early_stopping_patience=8,
            device=device,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 2. Train on source machines
    detector.fit(train_paths)

    # 3. Score target test recordings
    test_scores = []
    test_labels = []
    for r in tgt_rows:
        score = detector.score_file(r["file_path"])
        label_int = 1 if r["label"] == "abnormal" else 0
        test_scores.append(score)
        test_labels.append(label_int)

    # 4. Zero-shot metrics (using median threshold)
    zero_shot_metrics = compute_anomaly_metrics(
        y_true=test_labels,
        y_scores=test_scores,
        threshold=None,
    )

    # 5. Few-shot calibration using target machine dev files
    calibrator = StatisticalCalibrator(target_quantile=0.95)
    few_shot_profile = calibrator.calibrate(detector, dev_paths)

    few_shot_metrics = compute_anomaly_metrics(
        y_true=test_labels,
        y_scores=test_scores,
        threshold=detector.threshold,
    )

    duration = time.perf_counter() - start_time
    artifact_dir = output_base_dir / model_type / target_machine_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment_id": f"cross_{model_type}_{target_machine_id}",
        "target_machine_id": target_machine_id,
        "model_type": model_type,
        "train_source_count": len(train_paths),
        "test_target_count": len(test_paths),
        "zero_shot_metrics": zero_shot_metrics,
        "few_shot_metrics": few_shot_metrics,
        "few_shot_calibration": few_shot_profile.to_dict(),
        "total_duration_sec": round(duration, 4),
    }

    with open(artifact_dir / "cross_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(
        f"Cross fold {target_machine_id} [{model_type}] finished in {duration:.2f}s | "
        f"Zero-shot ROC-AUC: {zero_shot_metrics['roc_auc']} | "
        f"Few-shot F1: {few_shot_metrics['f1']} | Few-shot Recall: {few_shot_metrics['recall']}"
    )
    return results

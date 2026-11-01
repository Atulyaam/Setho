"""Sprint 6C Phase 3: Generalization Experiments Runner.

Evaluates two architectural/training approaches targeting generalization:
- Approach 1: Multi-Task / Multi-Category Representation (MultiTaskResNetDetector)
- Approach 2: Regularized Variational Latent Space (VariationalResNetDetector)
Compared against the single-task Residual CNN baseline across:
1. Seen Diverse Machine Split
2. Unseen Machine Split (Zero-Shot & Few-Shot Calibrated N=5)
3. Cross-Category Transfer Split (Zero-Shot & Few-Shot Calibrated N=5)
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from ml.calibration.few_shot_calibrator import FewShotMachineCalibrator
from ml.calibration.statistical_calibration import StatisticalCalibrator
from ml.evaluation.metrics import compute_anomaly_metrics
from ml.models.research.multitask_resnet_ae import MultiTaskResNetDetector
from ml.models.research.research_detector import ResearchNeuralDetector
from ml.models.research.variational_resnet_ae import VariationalResNetDetector
from ml.training.train_pipeline import _read_split_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sprint6c_generalization_runner")

PILOT_MACHINES = ["id_00", "id_02", "id_04", "id_06"]
CATEGORIES = ["fan", "pump", "valve"]
OUTPUT_FILE = Path("ml/reports/sprint_6c_generalization_results.json")


def load_multi_task_train_files() -> List[str]:
    """Collect all normal training files across all categories and machines."""
    files = []
    for cat in CATEGORIES:
        for m_id in PILOT_MACHINES:
            split_dir = Path(f"ml/data/splits/diverse_machine_specific/{cat}/{m_id}")
            csv_path = split_dir / "train_normal.csv"
            if csv_path.exists():
                rows = _read_split_csv(csv_path)
                files.extend([r["file_path"] for r in rows if str(r.get("label", "normal")).lower() == "normal"])
    return files


def evaluate_seen_machines(detector, category: str = "fan") -> Dict[str, Any]:
    """Evaluate detector on seen machines within a category."""
    per_machine = {}
    aucs, paucs, f1s = [], [], []

    for m_id in PILOT_MACHINES:
        split_dir = Path(f"ml/data/splits/diverse_machine_specific/{category}/{m_id}")
        test_rows = _read_split_csv(split_dir / "test.csv")
        test_paths = [r["file_path"] for r in test_rows]
        test_labels = [1 if str(r["label"]).lower() in ("abnormal", "anomaly", "1") else 0 for r in test_rows]

        # Use dev set to set 95% operating threshold
        dev_rows = _read_split_csv(split_dir / "dev_normal.csv")
        dev_normal_paths = [r["file_path"] for r in dev_rows if str(r.get("label", "normal")).lower() == "normal"]
        if dev_normal_paths:
            calibrator = StatisticalCalibrator(target_quantile=0.95)
            prof = calibrator.calibrate(detector, dev_normal_paths)
            thresh = prof.operating_threshold
        else:
            thresh = detector.threshold

        scores = [detector.score_file(p) for p in test_paths]
        m = compute_anomaly_metrics(test_labels, scores, thresh)
        per_machine[m_id] = m
        aucs.append(m["roc_auc"])
        paucs.append(m["pauc"])
        f1s.append(m["f1"])

    return {
        "macro_roc_auc": round(float(np.mean(aucs)), 4),
        "macro_pauc": round(float(np.mean(paucs)), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "per_machine": per_machine,
    }


def evaluate_unseen_machines(detector, calibrate: bool = False, n_shots: int = 5) -> Dict[str, Any]:
    """Evaluate detector on unseen machines, with or without Few-Shot Mahalanobis calibration."""
    per_machine = {}
    aucs, paucs, f1s = [], [], []

    for m_id in PILOT_MACHINES:
        split_dir = Path(f"ml/data/splits/unseen_machine/fan/heldout_{m_id}")
        test_rows = _read_split_csv(split_dir / "test_unseen_machine.csv")

        # Select n_shots from normal clips of the heldout machine (from diverse dev split)
        div_dev_path = Path(f"ml/data/splits/diverse_machine_specific/fan/{m_id}/dev_normal.csv")
        dev_rows = _read_split_csv(div_dev_path)
        cal_paths = [r["file_path"] for r in dev_rows if str(r.get("label", "normal")).lower() == "normal"][:n_shots]

        test_paths = [r["file_path"] for r in test_rows]
        test_labels = [1 if str(r["label"]).lower() in ("abnormal", "anomaly", "1") else 0 for r in test_rows]

        if calibrate and cal_paths:
            calibrator = FewShotMachineCalibrator(
                base_detector=detector,
                method="mahalanobis_latent",
                n_shots=n_shots,
                target_quantile=0.95,
            )
            calibrator.calibrate(cal_paths)
            scores = [calibrator.score_file(p) for p in test_paths]
            thresh = calibrator.operating_threshold
        else:
            thresh = detector.threshold
            scores = [detector.score_file(p) for p in test_paths]

        m = compute_anomaly_metrics(test_labels, scores, thresh)
        per_machine[m_id] = m
        aucs.append(m["roc_auc"])
        paucs.append(m["pauc"])
        f1s.append(m["f1"])

    return {
        "macro_roc_auc": round(float(np.mean(aucs)), 4),
        "macro_pauc": round(float(np.mean(paucs)), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "per_machine": per_machine,
    }


def evaluate_cross_category(detector, calibrate: bool = False, n_shots: int = 5) -> Dict[str, Any]:
    """Evaluate cross-category transfer across all 6 pairs."""
    pair_results = {}
    pairs = [
        ("fan", "pump"),
        ("fan", "valve"),
        ("pump", "fan"),
        ("pump", "valve"),
        ("valve", "fan"),
        ("valve", "pump"),
    ]

    for src, tgt in pairs:
        pair_key = f"{src}_to_{tgt}"
        split_dir = Path(f"ml/data/splits/cross_category/{pair_key}")
        test_rows = _read_split_csv(split_dir / "test_target_category.csv")
        test_paths = [r["file_path"] for r in test_rows]
        test_labels = [1 if str(r["label"]).lower() in ("abnormal", "anomaly", "1") else 0 for r in test_rows]

        # Draw normal calibration files from target category
        cal_paths = []
        for mid in PILOT_MACHINES:
            dev_csv = Path(f"ml/data/splits/diverse_machine_specific/{tgt}/{mid}/dev_normal.csv")
            if dev_csv.exists():
                d_rows = _read_split_csv(dev_csv)
                if d_rows:
                    cal_paths.append(d_rows[0]["file_path"])
        cal_paths = cal_paths[:n_shots]

        if calibrate and cal_paths:
            calibrator = FewShotMachineCalibrator(
                base_detector=detector,
                method="mahalanobis_latent",
                n_shots=len(cal_paths),
                target_quantile=0.95,
            )
            calibrator.calibrate(cal_paths)
            scores = [calibrator.score_file(p) for p in test_paths]
            thresh = calibrator.operating_threshold
        else:
            thresh = detector.threshold
            scores = [detector.score_file(p) for p in test_paths]

        m = compute_anomaly_metrics(test_labels, scores, thresh)
        pair_results[pair_key] = {
            "roc_auc": round(m["roc_auc"], 4),
            "pauc": round(m["pauc"], 4),
            "f1": round(m["f1"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "fpr": round(m["fpr"], 4),
        }

    macro_auc = round(float(np.mean([p["roc_auc"] for p in pair_results.values()])), 4)
    macro_pauc = round(float(np.mean([p["pauc"] for p in pair_results.values()])), 4)
    macro_f1 = round(float(np.mean([p["f1"] for p in pair_results.values()])), 4)

    return {
        "macro_roc_auc": macro_auc,
        "macro_pauc": macro_pauc,
        "macro_f1": macro_f1,
        "pairs": pair_results,
    }


def main():
    logger.info("Starting Sprint 6C Phase 3 Generalization Experiments...")
    multi_task_train_files = load_multi_task_train_files()
    logger.info(f"Loaded {len(multi_task_train_files)} multi-task training files across Fan, Pump, and Valve.")

    # Fan single-task files
    fan_train_rows = _read_split_csv("ml/data/splits/diverse_machine_specific/fan/id_00/train_normal.csv")
    fan_train_files = [r["file_path"] for r in fan_train_rows if str(r.get("label", "normal")).lower() == "normal"]

    results = {}

    # ----------------------------------------------------
    # Baseline: Single-Task ResNet-AE (Trained on fan id_00)
    # ----------------------------------------------------
    logger.info("1/3 Evaluating Baseline: Single-Task ResNet-AE...")
    baseline_detector = ResearchNeuralDetector(
        machine_id="id_00",
        arch_type="residual_cnn_ae",
        base_channels=16,
        latent_dim=64,
        max_epochs=15,
        seed=42,
    )
    baseline_detector.fit(fan_train_files)
    calibrator = StatisticalCalibrator(target_quantile=0.95)
    baseline_detector.threshold = calibrator.calibrate(baseline_detector, fan_train_files[:3]).operating_threshold

    baseline_seen = evaluate_seen_machines(baseline_detector, "fan")
    baseline_unseen_zero = evaluate_unseen_machines(baseline_detector, calibrate=False)
    baseline_unseen_cal = evaluate_unseen_machines(baseline_detector, calibrate=True, n_shots=5)
    baseline_cross_zero = evaluate_cross_category(baseline_detector, calibrate=False)
    baseline_cross_cal = evaluate_cross_category(baseline_detector, calibrate=True, n_shots=5)

    results["single_task_resnet"] = {
        "seen_machines": baseline_seen,
        "unseen_machines_zero_shot": baseline_unseen_zero,
        "unseen_machines_few_shot_cal": baseline_unseen_cal,
        "cross_category_zero_shot": baseline_cross_zero,
        "cross_category_few_shot_cal": baseline_cross_cal,
    }
    logger.info(
        f"[Baseline ResNet] Seen: {baseline_seen['macro_roc_auc']:.4f} | "
        f"Unseen Zero: {baseline_unseen_zero['macro_roc_auc']:.4f} | "
        f"Unseen Cal(N=5): {baseline_unseen_cal['macro_roc_auc']:.4f} | "
        f"Cross Zero: {baseline_cross_zero['macro_roc_auc']:.4f} | "
        f"Cross Cal(N=5): {baseline_cross_cal['macro_roc_auc']:.4f}"
    )

    # ----------------------------------------------------
    # Approach 1: Multi-Task Unified ResNet-AE (Trained on all 12 machines across 3 categories)
    # ----------------------------------------------------
    logger.info("2/3 Training & Evaluating Approach 1: Multi-Task ResNet-AE...")
    multitask_detector = MultiTaskResNetDetector(
        base_channels=16,
        latent_dim=64,
        max_epochs=15,
        seed=42,
    )
    multitask_detector.fit(multi_task_train_files)
    multitask_detector.threshold = 0.5  # reasonable default

    multitask_seen = evaluate_seen_machines(multitask_detector, "fan")
    multitask_unseen_zero = evaluate_unseen_machines(multitask_detector, calibrate=False)
    multitask_unseen_cal = evaluate_unseen_machines(multitask_detector, calibrate=True, n_shots=5)
    multitask_cross_zero = evaluate_cross_category(multitask_detector, calibrate=False)
    multitask_cross_cal = evaluate_cross_category(multitask_detector, calibrate=True, n_shots=5)

    results["multitask_resnet"] = {
        "seen_machines": multitask_seen,
        "unseen_machines_zero_shot": multitask_unseen_zero,
        "unseen_machines_few_shot_cal": multitask_unseen_cal,
        "cross_category_zero_shot": multitask_cross_zero,
        "cross_category_few_shot_cal": multitask_cross_cal,
    }
    logger.info(
        f"[MultiTask ResNet] Seen: {multitask_seen['macro_roc_auc']:.4f} | "
        f"Unseen Zero: {multitask_unseen_zero['macro_roc_auc']:.4f} | "
        f"Unseen Cal(N=5): {multitask_unseen_cal['macro_roc_auc']:.4f} | "
        f"Cross Zero: {multitask_cross_zero['macro_roc_auc']:.4f} | "
        f"Cross Cal(N=5): {multitask_cross_cal['macro_roc_auc']:.4f}"
    )

    # ----------------------------------------------------
    # Approach 2: Regularized Variational ResNet-AE (VAE with KL bottleneck penalty & dropout)
    # ----------------------------------------------------
    logger.info("3/3 Training & Evaluating Approach 2: Variational ResNet-AE...")
    vae_detector = VariationalResNetDetector(
        machine_id="id_00",
        base_channels=16,
        latent_dim=64,
        beta_kl=0.001,
        dropout=0.1,
        max_epochs=15,
        seed=42,
    )
    vae_detector.fit(fan_train_files)
    vae_detector.threshold = 0.5

    vae_seen = evaluate_seen_machines(vae_detector, "fan")
    vae_unseen_zero = evaluate_unseen_machines(vae_detector, calibrate=False)
    vae_unseen_cal = evaluate_unseen_machines(vae_detector, calibrate=True, n_shots=5)
    vae_cross_zero = evaluate_cross_category(vae_detector, calibrate=False)
    vae_cross_cal = evaluate_cross_category(vae_detector, calibrate=True, n_shots=5)

    results["variational_resnet"] = {
        "seen_machines": vae_seen,
        "unseen_machines_zero_shot": vae_unseen_zero,
        "unseen_machines_few_shot_cal": vae_unseen_cal,
        "cross_category_zero_shot": vae_cross_zero,
        "cross_category_few_shot_cal": vae_cross_cal,
    }
    logger.info(
        f"[Variational ResNet] Seen: {vae_seen['macro_roc_auc']:.4f} | "
        f"Unseen Zero: {vae_unseen_zero['macro_roc_auc']:.4f} | "
        f"Unseen Cal(N=5): {vae_unseen_cal['macro_roc_auc']:.4f} | "
        f"Cross Zero: {vae_cross_zero['macro_roc_auc']:.4f} | "
        f"Cross Cal(N=5): {vae_cross_cal['macro_roc_auc']:.4f}"
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Phase 3 Generalization Results recorded to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

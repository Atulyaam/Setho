"""Sprint 6C: Few-Shot Machine Calibration & Cross-Category Generalization Runner.

Evaluates:
- Uncalibrated vs Calibrated Unseen Machine Generalization (Fan, Pump, Valve)
- Comparison of Calibration Methods:
  1. uncalibrated
  2. mean_var_residual (spatial-spectral residual whitening)
  3. mahalanobis_latent (covariance-regularized bottleneck distance)
  4. memory_bank_knn (k-NN reference embeddings)
- Sample size sweep: N in {2, 5, 10} normal recordings
- Cross-category transfer with and without few-shot calibration
"""

import csv
import json
import logging
import os
import sys
from pathlib import Path
import time

sys.path.insert(0, os.getcwd())
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch

from ml.calibration.few_shot_calibrator import FewShotMachineCalibrator
from ml.models.research.distilled_student import DistilledStudentDetector
from ml.models.research.research_detector import ResearchNeuralDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sprint6c_runner")

RESEARCH_ARTIFACTS_DIR = Path("ml/experiments/research_artifacts/sprint6c")
SPLITS_DIR = Path("ml/data/splits")


def _read_csv(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_metrics(y_true: List[int], y_score: List[float], threshold: float) -> Dict[str, Any]:
    auc = float(roc_auc_score(y_true, y_score))
    pauc = float(roc_auc_score(y_true, y_score, max_fpr=0.10))
    y_pred = (np.array(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return {
        "roc_auc": round(auc, 4),
        "pauc": round(pauc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }


def run_few_shot_calibration_benchmark() -> Dict[str, Any]:
    """Execute complete Phase 2 few-shot calibration benchmark on unseen Fan machines."""
    logger.info("Executing Phase 2 Few-Shot Machine Calibration Benchmark...")
    machines = ["id_00", "id_02", "id_04", "id_06"]
    methods = ["uncalibrated", "mean_var_residual", "mahalanobis_latent", "memory_bank_knn"]
    n_shots_list = [2, 5, 10]

    # Use trained ResNet-AE teacher from Phase 5/Phase 4
    results_by_method_shots: Dict[str, Any] = {}

    for method in methods:
        results_by_method_shots[method] = {}
        for n_shots in n_shots_list:
            if method == "uncalibrated" and n_shots != 5:
                continue  # Uncalibrated baseline does not vary with shots except for default threshold

            key = f"{method}_shots_{n_shots}" if method != "uncalibrated" else "uncalibrated"
            fold_results = {}
            aucs, paucs, precs, recs, f1s, fprs, cal_times = [], [], [], [], [], [], []

            for heldout_m in machines:
                fold_dir = SPLITS_DIR / "unseen_machine" / "fan" / f"heldout_{heldout_m}"
                train_known_file = fold_dir / "train_known_machines.csv"
                test_unseen_file = fold_dir / "test_unseen_machine.csv"

                # Candidate detector trained on known machines or general ResNet-AE
                detector = ResearchNeuralDetector(
                    machine_id=f"known_ex_{heldout_m}",
                    arch_type="residual_cnn_ae",
                    base_channels=16,
                    latent_dim=64,
                )
                detector.train_mean = -42.5
                detector.train_std = 18.2

                # Load trained weights if checkpoint exists, or use phase 4 ResNet-AE weights
                sample_ckpt = Path("ml/experiments/research_artifacts/phase4/residual_cnn_ae/id_00/model.pt")
                if sample_ckpt.exists():
                    ckpt = torch.load(sample_ckpt, map_location="cpu", weights_only=False)
                    if isinstance(ckpt, dict) and "state_dict" in ckpt:
                        detector.model.load_state_dict(ckpt["state_dict"])
                    else:
                        detector.model.load_state_dict(ckpt)
                detector.model.eval()

                # Get available calibration normal recordings for the heldout machine
                # Drawn strictly from diverse_machine_specific train_normal (10 available normal clips)
                cal_source_csv = SPLITS_DIR / "diverse_machine_specific" / "fan" / heldout_m / "train_normal.csv"
                cal_rows = _read_csv(cal_source_csv)
                cal_files = [r["file_path"] for r in cal_rows[:n_shots]]

                # Calibrate
                calibrator = FewShotMachineCalibrator(
                    base_detector=detector,
                    method=method,
                    n_shots=n_shots,
                    target_quantile=0.95,
                )
                t_c0 = time.perf_counter()
                cal_res = calibrator.calibrate(cal_files)
                cal_time_s = time.perf_counter() - t_c0
                cal_times.append(cal_time_s)

                # Evaluate on untouched test_unseen_machine.csv (18 files: 3 norm, 15 abn)
                test_rows = _read_csv(test_unseen_file)
                test_files = [r["file_path"] for r in test_rows]
                y_true = [1 if r["label"].lower() in ("abnormal", "anomaly", "1") else 0 for r in test_rows]

                y_scores = [calibrator.score_file(fp) for fp in test_files]
                m_metrics = compute_metrics(y_true, y_scores, calibrator.operating_threshold)
                fold_results[heldout_m] = {
                    "metrics": m_metrics,
                    "threshold": calibrator.operating_threshold,
                    "cal_time_s": round(cal_time_s, 4),
                }
                aucs.append(m_metrics["roc_auc"])
                paucs.append(m_metrics["pauc"])
                precs.append(m_metrics["precision"])
                recs.append(m_metrics["recall"])
                f1s.append(m_metrics["f1"])
                fprs.append(m_metrics["fpr"])

            macro_res = {
                "macro_roc_auc": round(float(np.mean(aucs)), 4),
                "macro_pauc": round(float(np.mean(paucs)), 4),
                "macro_precision": round(float(np.mean(precs)), 4),
                "macro_recall": round(float(np.mean(recs)), 4),
                "macro_f1": round(float(np.mean(f1s)), 4),
                "macro_fpr": round(float(np.mean(fprs)), 4),
                "avg_cal_time_s": round(float(np.mean(cal_times)), 4),
                "per_machine": fold_results,
            }
            results_by_method_shots[method][key] = macro_res
            logger.info(
                f"[{key:<28}] Macro ROC-AUC: {macro_res['macro_roc_auc']:.4f} | "
                f"pAUC: {macro_res['macro_pauc']:.4f} | "
                f"F1: {macro_res['macro_f1']:.4f} | "
                f"Cal Time: {macro_res['avg_cal_time_s']:.3f}s"
            )

    return results_by_method_shots


def run_cross_category_benchmark() -> Dict[str, Any]:
    """Execute complete Phase 4 cross-category transfer benchmark with & without few-shot calibration."""
    logger.info("Executing Phase 4 Cross-Category Transfer Benchmark...")
    pairs = [
        ("fan_to_pump", "pump"),
        ("fan_to_valve", "valve"),
        ("pump_to_fan", "fan"),
        ("pump_to_valve", "valve"),
        ("valve_to_fan", "fan"),
        ("valve_to_pump", "pump"),
    ]

    cross_results = {}
    for pair_name, tgt_cat in pairs:
        pair_dir = SPLITS_DIR / "cross_category" / pair_name
        test_file = pair_dir / "test_target_category.csv"
        test_rows = _read_csv(test_file)
        test_files = [r["file_path"] for r in test_rows]
        y_true = [1 if r["label"].lower() in ("abnormal", "anomaly", "1") else 0 for r in test_rows]

        # Detector
        detector = ResearchNeuralDetector(
            machine_id=pair_name, arch_type="residual_cnn_ae", base_channels=16, latent_dim=64
        )
        detector.train_mean = -42.5
        detector.train_std = 18.2
        sample_ckpt = Path("ml/experiments/research_artifacts/phase4/residual_cnn_ae/id_00/model.pt")
        if sample_ckpt.exists():
            ckpt = torch.load(sample_ckpt, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                detector.model.load_state_dict(ckpt["state_dict"])
            else:
                detector.model.load_state_dict(ckpt)
        detector.model.eval()

        # 1. Zero-shot uncalibrated
        cal_zero = FewShotMachineCalibrator(base_detector=detector, method="uncalibrated", n_shots=5)
        # Calibrate threshold on source category train normal clips
        src_train_csv = pair_dir / "train_source_category.csv"
        src_train_files = [r["file_path"] for r in _read_csv(src_train_csv)[:5]]
        cal_zero.calibrate(src_train_files)
        scores_zero = [cal_zero.score_file(fp) for fp in test_files]
        metrics_zero = compute_metrics(y_true, scores_zero, cal_zero.operating_threshold)

        # 2. Few-shot calibrated with N=5 normal recordings from target category
        # Draw from target machines normal dev/train clips
        tgt_sample_files = []
        for mid in ["id_00", "id_02", "id_04", "id_06"]:
            m_train_csv = SPLITS_DIR / "diverse_machine_specific" / tgt_cat / mid / "train_normal.csv"
            m_rows = _read_csv(m_train_csv)
            if m_rows:
                tgt_sample_files.append(m_rows[0]["file_path"])
        tgt_cal_files = tgt_sample_files[:5]

        cal_adapted = FewShotMachineCalibrator(
            base_detector=detector, method="mean_var_residual", n_shots=len(tgt_cal_files)
        )
        t0 = time.perf_counter()
        cal_adapted.calibrate(tgt_cal_files)
        cal_time_s = time.perf_counter() - t0
        scores_adapted = [cal_adapted.score_file(fp) for fp in test_files]
        metrics_adapted = compute_metrics(y_true, scores_adapted, cal_adapted.operating_threshold)

        cross_results[pair_name] = {
            "zero_shot_uncalibrated": metrics_zero,
            "few_shot_adapted": metrics_adapted,
            "delta_roc_auc": round(metrics_adapted["roc_auc"] - metrics_zero["roc_auc"], 4),
            "delta_pauc": round(metrics_adapted["pauc"] - metrics_zero["pauc"], 4),
            "cal_time_s": round(cal_time_s, 4),
        }
        logger.info(
            f"[{pair_name:<16}] Zero-shot AUC: {metrics_zero['roc_auc']:.4f} -> "
            f"Few-shot AUC: {metrics_adapted['roc_auc']:.4f} (Delta: {cross_results[pair_name]['delta_roc_auc']:+.4f})"
        )

    return cross_results


def main():
    RESEARCH_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = Path("ml/reports/sprint_6c_few_shot_results.json")

    few_shot_results = run_few_shot_calibration_benchmark()
    cross_cat_results = run_cross_category_benchmark()

    all_results = {
        "few_shot_calibration_benchmark": few_shot_results,
        "cross_category_benchmark": cross_cat_results,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Sprint 6C benchmarks successfully recorded to {out_file}")


if __name__ == "__main__":
    main()

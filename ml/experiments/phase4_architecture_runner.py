"""Phase 4 New Architecture Pilot & Comparative Evaluation Engine.

Evaluates candidate architectures:
1. multiscale_cnn_ae (3x3, 3x5, 5x3 depthwise-separable multi-kernel branches)
2. residual_cnn_ae (ResNet Autoencoder)
3. cnn_gru_attention_ae (Spatial-temporal CNN-GRU with attention)
4. embedding_mahalanobis (Statistical Mahalanobis + k-NN)
5. teacher_student_cnn (Discrepancy CNN)

Runs training pilot, statistical calibration, latency benchmarking, and test evaluation.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import psutil
import torch

from ml.calibration.statistical_calibration import StatisticalCalibrator
from ml.evaluation.metrics import compute_anomaly_metrics
from ml.experiments.experiment_manager import ExperimentManager
from ml.models.research.embedding_mahalanobis import MahalanobisDetector
from ml.models.research.research_detector import ResearchNeuralDetector
from ml.training.train_pipeline import _read_split_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stetho_phase4_runner")

PILOT_MACHINES = ["id_00", "id_02", "id_04", "id_06"]
RESEARCH_ARTIFACTS_DIR = Path("ml/experiments/research_artifacts/phase4")


def benchmark_model_latency(detector, sample_file: Path, iterations: int = 5) -> Dict[str, Any]:
    """Measure single-file inference latency on CPU."""
    # Warmup
    for _ in range(2):
        detector.score_file(sample_file)

    cpu_latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        detector.score_file(sample_file)
        cpu_latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "cpu_mean_ms": round(float(np.mean(cpu_latencies)), 2),
        "cpu_median_ms": round(float(np.median(cpu_latencies)), 2),
        "cpu_p95_ms": round(float(np.percentile(cpu_latencies, 95)), 2),
        "cpu_min_ms": round(float(np.min(cpu_latencies)), 2),
        "cpu_max_ms": round(float(np.max(cpu_latencies)), 2),
    }


def evaluate_candidate(
    arch_type: str,
    machines: List[str] = PILOT_MACHINES,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train, calibrate, benchmark, and evaluate a candidate architecture."""
    out_dir = RESEARCH_ARTIFACTS_DIR / arch_type
    out_dir.mkdir(parents=True, exist_ok=True)
    results_by_machine = {}

    for idx, m_id in enumerate(machines):
        logger.info(f"Evaluating {arch_type} on {m_id} ({idx + 1}/{len(machines)})...")
        m_dir = out_dir / m_id
        m_dir.mkdir(parents=True, exist_ok=True)

        train_split = f"ml/data/splits/machine_specific/{m_id}/train_normal.csv"
        dev_split = f"ml/data/splits/machine_specific/{m_id}/dev_normal.csv"
        test_split = f"ml/data/splits/machine_specific/{m_id}/test.csv"

        train_rows = _read_split_csv(train_split)
        dev_rows = _read_split_csv(dev_split)
        test_rows = _read_split_csv(test_split)

        train_paths = [r["file_path"] for r in train_rows]
        dev_paths = [r["file_path"] for r in dev_rows]
        test_paths = [r["file_path"] for r in test_rows]

        # 1. Instantiate
        if arch_type == "embedding_mahalanobis":
            detector = MahalanobisDetector(machine_id=m_id, feature_preset="compact_stat_38d")
            param_count = 38 * 38 + 38
        else:
            detector = ResearchNeuralDetector(
                machine_id=m_id,
                arch_type=arch_type,
                base_channels=16,
                latent_dim=64,
                learning_rate=0.001,
                max_epochs=20,
                early_stopping_patience=5,
                seed=seed,
            )
            param_count = sum(p.numel() for p in detector.model.parameters())

        # 2. Fit
        t0 = time.perf_counter()
        fit_meta = detector.fit(train_paths, dev_files=dev_paths)
        fit_duration = round(time.perf_counter() - t0, 2)

        # 3. Calibrate
        calibrator = StatisticalCalibrator(target_quantile=0.95)
        cal_prof = calibrator.calibrate(detector, dev_paths)
        detector.threshold = cal_prof.operating_threshold

        # 4. Serialize
        detector.save(m_dir)
        checkpoint_file = m_dir / ("model.pt" if arch_type != "embedding_mahalanobis" else "mahalanobis_model.json")
        checkpoint_size_kb = round(checkpoint_file.stat().st_size / 1024, 1)

        # 5. Benchmark Latency
        latency = benchmark_model_latency(detector, Path(test_paths[0]), iterations=5)

        # 6. Evaluate on Test Set
        test_scores = [detector.score_file(p) for p in test_paths]
        test_labels = [1 if r["label"] == "abnormal" else 0 for r in test_rows]
        metrics = compute_anomaly_metrics(test_labels, test_scores, detector.threshold)

        machine_res = {
            "machine_id": m_id,
            "arch_type": arch_type,
            "train_duration_sec": fit_duration,
            "parameter_count": param_count,
            "checkpoint_size_kb": checkpoint_size_kb,
            "threshold": detector.threshold,
            "latency": latency,
            "metrics": metrics,
            "test_scores": [round(s, 6) for s in test_scores],
            "test_labels": test_labels,
        }

        with open(m_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(machine_res, f, indent=2)

        results_by_machine[m_id] = machine_res
        logger.info(
            f"[{arch_type} | {m_id}] ROC-AUC={metrics['roc_auc']:.4f}, pAUC={metrics['pauc']:.4f}, "
            f"F1={metrics['f1']:.4f}, CPU Latency={latency['cpu_median_ms']:.1f}ms"
        )

    macro_roc = round(float(np.mean([results_by_machine[m]["metrics"]["roc_auc"] for m in machines])), 4)
    macro_pauc = round(float(np.mean([results_by_machine[m]["metrics"]["pauc"] for m in machines])), 4)
    macro_f1 = round(float(np.mean([results_by_machine[m]["metrics"]["f1"] for m in machines])), 4)
    macro_recall = round(float(np.mean([results_by_machine[m]["metrics"]["recall"] for m in machines])), 4)
    macro_fpr = round(float(np.mean([results_by_machine[m]["metrics"]["fpr"] for m in machines])), 4)
    avg_latency = round(float(np.mean([results_by_machine[m]["latency"]["cpu_median_ms"] for m in machines])), 2)

    total_summary = {
        "arch_type": arch_type,
        "macro_roc_auc": macro_roc,
        "macro_pauc": macro_pauc,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "macro_fpr": macro_fpr,
        "average_median_cpu_latency_ms": avg_latency,
        "parameter_count": results_by_machine[machines[0]]["parameter_count"],
        "checkpoint_size_kb": results_by_machine[machines[0]]["checkpoint_size_kb"],
        "per_machine": results_by_machine,
    }

    with open(out_dir / "macro_summary.json", "w", encoding="utf-8") as f:
        json.dump(total_summary, f, indent=2)

    return total_summary


def main():
    manager = ExperimentManager()
    candidates = [
        ("embedding_mahalanobis", "Compact Feature Mahalanobis & k-NN Detector"),
        ("multiscale_cnn_ae", "Multi-Kernel Multi-Scale Depthwise-Separable CNN-AE"),
        ("teacher_student_cnn", "Compact Teacher-Student Discrepancy CNN"),
        ("residual_cnn_ae", "Pure Residual CNN Autoencoder"),
    ]

    all_summaries = {}
    for arch_type, desc in candidates:
        exp_id = f"phase4_{arch_type}"
        manager.register_experiment(
            experiment_id=exp_id,
            name=f"Phase 4 {desc}",
            description=f"Evaluation of candidate {arch_type}",
            model_type=arch_type,
            config={"architecture": arch_type, "preprocessing": "standard_64mel", "seed": 42},
            seed=42,
        )
        manager.update_status(exp_id, status="RUNNING", pid=os.getpid())
        try:
            summary = evaluate_candidate(arch_type)
            manager.update_status(exp_id, status="COMPLETED", metrics=summary)
            all_summaries[arch_type] = summary
        except Exception as e:
            logger.exception(f"Candidate {arch_type} failed: {e}")
            manager.update_status(exp_id, status="FAILED", error=str(e))

    print("\n================== PHASE 4 ARCHITECTURE SUMMARY ==================")
    for arch, s in all_summaries.items():
        print(f"{arch:25s} | Macro AUC: {s['macro_roc_auc']:.4f} | pAUC: {s['macro_pauc']:.4f} | F1: {s['macro_f1']:.4f} | Latency: {s['average_median_cpu_latency_ms']:.1f}ms | Params: {s['parameter_count']}")
    print("==================================================================")


if __name__ == "__main__":
    main()


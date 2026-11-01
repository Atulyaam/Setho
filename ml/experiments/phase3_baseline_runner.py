"""Phase 3 Baseline Reproduction and Rigorous Benchmark Engine.

Trains and evaluates:
1. Dense Autoencoder Baseline
2. Stetho-MSAE Champion

Uses identical splits across all 4 benchmark machines (id_00, id_02, id_04, id_06)
and records comprehensive metrics, latency profiles, parameter footprints, and memory telemetry.
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
from ml.configs.experiment_config import ExperimentConfig
from ml.evaluation.metrics import compute_anomaly_metrics
from ml.experiments.experiment_manager import ExperimentManager
from ml.models.baseline_autoencoder.autoencoder_detector import DenseAutoencoderDetector
from ml.models.stetho_msae.msae_detector import StethoMSAEDetector
from ml.training.train_pipeline import _read_split_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stetho_phase3_baselines")

ALL_MACHINES = ["id_00", "id_02", "id_04", "id_06"]
RESEARCH_ARTIFACTS_DIR = Path("ml/experiments/research_artifacts/phase3")


def benchmark_model_latency(detector, sample_file: Path, iterations: int = 5) -> Dict[str, Any]:
    """Measure single-file inference latency on CPU and GPU (if available)."""
    # CPU Warmup
    for _ in range(2):
        detector.score_file(sample_file)

    cpu_latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        detector.score_file(sample_file)
        cpu_latencies.append((time.perf_counter() - t0) * 1000.0)

    latency_result = {
        "cpu_mean_ms": round(float(np.mean(cpu_latencies)), 2),
        "cpu_median_ms": round(float(np.median(cpu_latencies)), 2),
        "cpu_p95_ms": round(float(np.percentile(cpu_latencies, 95)), 2),
        "cpu_min_ms": round(float(np.min(cpu_latencies)), 2),
        "cpu_max_ms": round(float(np.max(cpu_latencies)), 2),
        "gpu_available": torch.cuda.is_available(),
        "gpu_median_ms": None,
    }

    if torch.cuda.is_available():
        try:
            # Measure GPU latency for raw forward pass
            dev = torch.device("cuda:0")
            detector.model.to(dev)
            detector.model.eval()
            # Generate dummy batch
            if hasattr(detector, "base_channels"): # MSAE takes (B, 1, F, T)
                dummy = torch.randn(1, 1, 64, 128, device=dev)
            else: # Dense takes (B, input_dim)
                dummy = torch.randn(1, 4096, device=dev)

            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = detector.model(dummy)
            torch.cuda.synchronize()

            gpu_lats = []
            for _ in range(iterations):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = detector.model(dummy)
                torch.cuda.synchronize()
                gpu_lats.append((time.perf_counter() - t0) * 1000.0)

            latency_result["gpu_median_ms"] = round(float(np.median(gpu_lats)), 3)
            latency_result["gpu_p95_ms"] = round(float(np.percentile(gpu_lats, 95)), 3)
            # Revert model to cpu
            detector.model.to("cpu")
        except Exception as e:
            logger.warning(f"Could not benchmark GPU latency: {e}")
            detector.model.to("cpu")

    return latency_result


def run_baseline_training(
    model_type: str,
    exp_id: str,
    manager: ExperimentManager,
    machine_ids: List[str] = ALL_MACHINES,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate specified baseline across all target machines."""
    results_by_machine = {}
    out_dir = RESEARCH_ARTIFACTS_DIR / model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, m_id in enumerate(machine_ids):
        logger.info(f"[{exp_id}] Training {model_type} for machine {m_id} ({idx + 1}/{len(machine_ids)})...")
        m_artifact_dir = out_dir / m_id
        m_artifact_dir.mkdir(parents=True, exist_ok=True)

        train_split = f"ml/data/splits/machine_specific/{m_id}/train_normal.csv"
        dev_split = f"ml/data/splits/machine_specific/{m_id}/dev_normal.csv"
        test_split = f"ml/data/splits/machine_specific/{m_id}/test.csv"

        train_rows = _read_split_csv(train_split)
        dev_rows = _read_split_csv(dev_split)
        test_rows = _read_split_csv(test_split)

        train_paths = [r["file_path"] for r in train_rows]
        dev_paths = [r["file_path"] for r in dev_rows]
        test_paths = [r["file_path"] for r in test_rows]

        # Initialize detector
        if model_type == "dense_autoencoder":
            detector = DenseAutoencoderDetector(
                machine_id=m_id,
                input_dim=4096,
                latent_dim=32,
                hidden_dims=[512, 128],
                learning_rate=0.001,
                max_epochs=40,
                early_stopping_patience=8,
                seed=seed,
            )
        elif model_type == "stetho_msae":
            detector = StethoMSAEDetector(
                machine_id=m_id,
                base_channels=32,
                dilations=(1, 2, 4),
                latent_dim=64,
                learning_rate=0.0005,
                max_epochs=40,
                early_stopping_patience=8,
                loss_type="l1",
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Train model
        t_start = time.perf_counter()
        train_meta = detector.fit(train_paths, dev_paths)
        train_duration = round(time.perf_counter() - t_start, 2)

        # Calibrate threshold on dev_normal
        calibrator = StatisticalCalibrator(target_quantile=0.95)
        cal_prof = calibrator.calibrate(detector, dev_paths)
        threshold = cal_prof.operating_threshold
        detector.threshold = threshold

        # Save checkpoint and config
        detector.save(m_artifact_dir)
        checkpoint_size_kb = round((m_artifact_dir / "model.pt").stat().st_size / 1024, 1)

        # Benchmark inference latency
        latency_stats = benchmark_model_latency(detector, Path(test_paths[0]), iterations=5)

        # Evaluate on test set
        test_scores = [detector.score_file(p) for p in test_paths]
        test_labels = [1 if r["label"] == "abnormal" else 0 for r in test_rows]
        metrics = compute_anomaly_metrics(test_labels, test_scores, threshold)

        param_count = sum(p.numel() for p in detector.model.parameters())

        # Track memory
        process = psutil.Process()
        ram_peak_mb = round(process.memory_info().rss / (1024 * 1024), 2)
        gpu_peak_mb = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0

        machine_summary = {
            "machine_id": m_id,
            "model_type": model_type,
            "train_duration_sec": train_duration,
            "parameter_count": param_count,
            "checkpoint_size_kb": checkpoint_size_kb,
            "threshold": threshold,
            "latency": latency_stats,
            "peak_ram_mb": ram_peak_mb,
            "peak_vram_mb": gpu_peak_mb,
            "metrics": metrics,
            "test_scores": [round(s, 6) for s in test_scores],
            "test_labels": test_labels,
        }

        # Save machine result
        with open(m_artifact_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(machine_summary, f, indent=2)

        results_by_machine[m_id] = machine_summary
        logger.info(
            f"[{exp_id}] {m_id} complete: ROC-AUC={metrics['roc_auc']:.4f}, pAUC={metrics['pauc']:.4f}, "
            f"F1={metrics['f1']:.4f}, Latency={latency_stats['cpu_median_ms']:.1f}ms (GPU: {latency_stats['gpu_median_ms']}ms)"
        )

    # Compute macro metrics
    macro_roc = round(float(np.mean([results_by_machine[m]["metrics"]["roc_auc"] for m in machine_ids])), 4)
    macro_pauc = round(float(np.mean([results_by_machine[m]["metrics"]["pauc"] for m in machine_ids])), 4)
    macro_f1 = round(float(np.mean([results_by_machine[m]["metrics"]["f1"] for m in machine_ids])), 4)
    macro_recall = round(float(np.mean([results_by_machine[m]["metrics"]["recall"] for m in machine_ids])), 4)
    macro_fpr = round(float(np.mean([results_by_machine[m]["metrics"]["fpr"] for m in machine_ids])), 4)
    avg_latency = round(float(np.mean([results_by_machine[m]["latency"]["cpu_median_ms"] for m in machine_ids])), 2)
    avg_gpu_latency = None
    if torch.cuda.is_available():
        gpu_lats = [results_by_machine[m]["latency"]["gpu_median_ms"] for m in machine_ids if results_by_machine[m]["latency"]["gpu_median_ms"] is not None]
        if gpu_lats:
            avg_gpu_latency = round(float(np.mean(gpu_lats)), 3)

    total_summary = {
        "model_type": model_type,
        "macro_roc_auc": macro_roc,
        "macro_pauc": macro_pauc,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "macro_fpr": macro_fpr,
        "average_median_cpu_latency_ms": avg_latency,
        "average_median_gpu_latency_ms": avg_gpu_latency,
        "parameter_count": results_by_machine[machine_ids[0]]["parameter_count"],
        "checkpoint_size_kb": results_by_machine[machine_ids[0]]["checkpoint_size_kb"],
        "per_machine": results_by_machine,
    }

    # Save summary
    with open(out_dir / "macro_summary.json", "w", encoding="utf-8") as f:
        json.dump(total_summary, f, indent=2)

    return total_summary


def main():
    manager = ExperimentManager()

    # 1. Dense Autoencoder Baseline
    exp_id_dae = "phase3_dense_autoencoder"
    manager.register_experiment(
        experiment_id=exp_id_dae,
        name="Phase 3 Dense Autoencoder Reproduction",
        description="Independent baseline training and benchmarking of Dense Autoencoder",
        model_type="dense_autoencoder",
        config={"architecture": "dense_autoencoder", "preprocessing": "standard_64mel", "seed": 42},
        seed=42,
    )
    manager.update_status(exp_id_dae, status="RUNNING", pid=os.getpid())
    try:
        dae_results = run_baseline_training("dense_autoencoder", exp_id_dae, manager)
        manager.update_status(exp_id_dae, status="COMPLETED", metrics=dae_results)
    except Exception as e:
        logger.exception(f"Dense Autoencoder failed: {e}")
        manager.update_status(exp_id_dae, status="FAILED", error=str(e))
        return

    # 2. Stetho-MSAE Champion
    exp_id_msae = "phase3_stetho_msae"
    manager.register_experiment(
        experiment_id=exp_id_msae,
        name="Phase 3 Stetho-MSAE Reproduction",
        description="Independent baseline training and benchmarking of Stetho-MSAE",
        model_type="stetho_msae",
        config={"architecture": "stetho_msae", "preprocessing": "standard_64mel", "seed": 42},
        seed=42,
    )
    manager.update_status(exp_id_msae, status="RUNNING", pid=os.getpid())
    try:
        msae_results = run_baseline_training("stetho_msae", exp_id_msae, manager)
        manager.update_status(exp_id_msae, status="COMPLETED", metrics=msae_results)
    except Exception as e:
        logger.exception(f"Stetho-MSAE failed: {e}")
        manager.update_status(exp_id_msae, status="FAILED", error=str(e))
        return

    logger.info("Phase 3 Baseline reproduction completed successfully!")
    print("\n================== PHASE 3 REPRODUCTION SUMMARY ==================")
    print(f"Dense Autoencoder: Macro AUC={dae_results['macro_roc_auc']}, pAUC={dae_results['macro_pauc']}, Latency={dae_results['average_median_cpu_latency_ms']}ms, Params={dae_results['parameter_count']}")
    print(f"Stetho-MSAE:       Macro AUC={msae_results['macro_roc_auc']}, pAUC={msae_results['macro_pauc']}, Latency={msae_results['average_median_cpu_latency_ms']}ms, Params={msae_results['parameter_count']}")
    print("==================================================================")


if __name__ == "__main__":
    main()

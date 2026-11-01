"""Phase 8 Inference & System Performance Optimization Benchmark Runner.

Benchmarks:
1. Latency Breakdown (Audio Loading -> Mel Extraction -> Patching -> Inference -> Scoring)
2. Precision Formats:
   - PyTorch FP32
   - ONNX Runtime FP32 (Graph Optimized)
   - ONNX Runtime INT8 (Dynamic Quantized)
3. Model Candidates:
   - ResNet-AE (Accuracy Champion candidate)
   - Distilled Student (Real-Time Champion candidate)
   - MultiScale CNN-AE (Efficiency Baseline)
4. System Targets:
   - End-to-end latency (< 100ms on CPU)
   - Cold-start latency (< 500ms)
   - RAM peak (< 500MB), VRAM peak (< 1GB)
   - Throughput (> 2 files/sec on CPU)
   - Parity (max abs diff < 1e-4) & Quantization degradation (< 1.0% ROC-AUC)
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
import psutil
import soundfile as sf
import torch

from ml.calibration.statistical_calibration import StatisticalCalibrator
from ml.evaluation.metrics import compute_anomaly_metrics
from ml.experiments.experiment_manager import ExperimentManager, get_system_telemetry
from ml.models.research.distilled_student import CompactStudentAE, DistilledStudentDetector
from ml.models.research.multiscale_cnn_ae import MultiScaleCNNAE
from ml.models.research.onnx_exporter import (
    ORTAnomalyDetector,
    export_to_onnx,
    quantize_to_int8,
    verify_numerical_parity,
)
from ml.models.research.research_detector import ResearchNeuralDetector
from ml.models.research.residual_cnn_ae import ResidualCNNAE
from ml.preprocessing.feature_suite import UnifiedFeatureSuite
from ml.training.train_pipeline import _read_split_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase8_inference_runner")

RESEARCH_ARTIFACTS_DIR = Path("ml/experiments/research_artifacts/phase8")
BENCHMARK_MACHINE = "id_00"
EVAL_MACHINES = ["id_00", "id_02", "id_04", "id_06"]


def profile_pipeline_latency(detector: Any, sample_file: Path) -> Dict[str, float]:
    """Measure granular latency for each step of the acoustic inference pipeline."""
    suite = UnifiedFeatureSuite(preset_name="standard_64mel")

    # Step 1: Audio loading and normalization
    t0 = time.perf_counter()
    wav = suite.process_raw_audio(sample_file)
    t_load = (time.perf_counter() - t0) * 1000.0

    # Step 2: Log-Mel extraction
    t0 = time.perf_counter()
    spec, _ = suite.mel_extractor.extract_numpy(wav)
    t_mel = (time.perf_counter() - t0) * 1000.0

    # Step 3: Windowing / Patch slicing
    t0 = time.perf_counter()
    patches_2d, _ = suite.windowing.slice_spectrogram_numpy(spec.squeeze(0))
    patches_4d = np.expand_dims(patches_2d, axis=1).astype(np.float32)
    t_patch = (time.perf_counter() - t0) * 1000.0

    # Step 4: Model forward pass (Inference)
    t0 = time.perf_counter()
    scores = detector.score_patches(patches_4d)
    t_infer = (time.perf_counter() - t0) * 1000.0

    # Step 5: Aggregation and decision thresholding
    t0 = time.perf_counter()
    final_score = float(np.mean(scores))
    decision = final_score >= detector.threshold
    t_post = (time.perf_counter() - t0) * 1000.0

    total_e2e = t_load + t_mel + t_patch + t_infer + t_post

    return {
        "audio_loading_ms": round(t_load, 2),
        "mel_extraction_ms": round(t_mel, 2),
        "patch_slicing_ms": round(t_patch, 2),
        "model_inference_ms": round(t_infer, 2),
        "postprocessing_decision_ms": round(t_post, 2),
        "total_e2e_latency_ms": round(total_e2e, 2),
    }


def benchmark_model_execution(
    detector: Any,
    test_paths: List[str],
    iterations: int = 10,
) -> Dict[str, Any]:
    """Measure warm latency stats, cold-start latency, and throughput."""
    sample = Path(test_paths[0])

    # Cold start (first time loading and scoring)
    t_cold_start_0 = time.perf_counter()
    _ = detector.score_file(sample)
    cold_start_ms = (time.perf_counter() - t_cold_start_0) * 1000.0

    # Warm inference iterations
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        detector.score_file(sample)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    # Throughput measurement over all test files
    t_tp_start = time.perf_counter()
    for p in test_paths:
        detector.score_file(p)
    elapsed_tp = time.perf_counter() - t_tp_start
    throughput_fps = len(test_paths) / elapsed_tp if elapsed_tp > 0 else 0.0

    return {
        "cold_start_ms": round(cold_start_ms, 2),
        "warm_mean_ms": round(float(np.mean(latencies)), 2),
        "warm_median_ms": round(float(np.median(latencies)), 2),
        "warm_p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "warm_min_ms": round(float(np.min(latencies)), 2),
        "warm_max_ms": round(float(np.max(latencies)), 2),
        "throughput_files_per_sec": round(throughput_fps, 2),
    }


def run_phase8_benchmarks() -> Dict[str, Any]:
    """Execute full Phase 8 optimization benchmark suite."""
    logger.info("Initializing Phase 8 Inference & System Performance Optimization...")
    RESEARCH_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    models_dir = RESEARCH_ARTIFACTS_DIR / "exported_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    benchmark_summary: Dict[str, Any] = {}

    # Read benchmark data
    train_split = f"ml/data/splits/machine_specific/{BENCHMARK_MACHINE}/train_normal.csv"
    dev_split = f"ml/data/splits/machine_specific/{BENCHMARK_MACHINE}/dev_normal.csv"
    test_split = f"ml/data/splits/machine_specific/{BENCHMARK_MACHINE}/test.csv"

    train_rows = _read_split_csv(train_split)
    dev_rows = _read_split_csv(dev_split)
    test_rows = _read_split_csv(test_split)

    train_paths = [r["file_path"] for r in train_rows]
    dev_paths = [r["file_path"] for r in dev_rows]
    test_paths = [r["file_path"] for r in test_rows]
    test_labels = [1 if str(r["label"]).lower() in ("abnormal", "anomaly", "1") else 0 for r in test_rows]

    calibrator = StatisticalCalibrator(target_quantile=0.95)
    dummy_input = torch.randn(2, 1, 64, 64)

    # -------------------------------------------------------------
    # 1. ResNet-AE (Accuracy Champion Candidate)
    # -------------------------------------------------------------
    logger.info("Training and exporting ResNet-AE (Model A Candidate)...")
    resnet_pt = ResearchNeuralDetector(
        machine_id=BENCHMARK_MACHINE,
        arch_type="residual_cnn_ae",
        base_channels=16,
        latent_dim=64,
        optimizer_type="adamw",
        lr_schedule="cosine",
        max_epochs=20,
    )
    resnet_pt.fit(train_paths, dev_files=dev_paths)
    cal_prof = calibrator.calibrate(resnet_pt, dev_paths)
    resnet_pt.threshold = cal_prof.operating_threshold

    # Export to ONNX FP32
    resnet_onnx_fp32 = models_dir / "resnet_ae_fp32.onnx"
    export_to_onnx(resnet_pt.model, dummy_input, resnet_onnx_fp32)
    parity_resnet, diff_resnet = verify_numerical_parity(resnet_pt.model, resnet_onnx_fp32, dummy_input)

    # Quantize to INT8
    resnet_onnx_int8 = models_dir / "resnet_ae_int8.onnx"
    quantize_to_int8(resnet_onnx_fp32, resnet_onnx_int8)

    # Instantiate ORT detectors
    resnet_ort_fp32 = ORTAnomalyDetector(
        BENCHMARK_MACHINE, resnet_onnx_fp32, resnet_pt.train_mean, resnet_pt.train_std, resnet_pt.threshold
    )
    resnet_ort_int8 = ORTAnomalyDetector(
        BENCHMARK_MACHINE, resnet_onnx_int8, resnet_pt.train_mean, resnet_pt.train_std, resnet_pt.threshold
    )

    # Benchmark accuracy
    scores_pt = [resnet_pt.score_file(p) for p in test_paths]
    scores_ort = [resnet_ort_fp32.score_file(p) for p in test_paths]
    scores_int8 = [resnet_ort_int8.score_file(p) for p in test_paths]

    auc_pt = compute_anomaly_metrics(test_labels, scores_pt, resnet_pt.threshold)["roc_auc"]
    auc_ort = compute_anomaly_metrics(test_labels, scores_ort, resnet_pt.threshold)["roc_auc"]
    auc_int8 = compute_anomaly_metrics(test_labels, scores_int8, resnet_pt.threshold)["roc_auc"]

    bench_pt = benchmark_model_execution(resnet_pt, test_paths)
    bench_ort = benchmark_model_execution(resnet_ort_fp32, test_paths)
    bench_int8 = benchmark_model_execution(resnet_ort_int8, test_paths)
    breakdown_resnet = profile_pipeline_latency(resnet_ort_fp32, Path(test_paths[0]))

    benchmark_summary["resnet_ae"] = {
        "parameters": sum(p.numel() for p in resnet_pt.model.parameters()),
        "parity_passed": parity_resnet,
        "max_absolute_diff": diff_resnet,
        "pytorch_fp32": {"roc_auc": auc_pt, **bench_pt, "size_kb": resnet_onnx_fp32.stat().st_size / 1024},
        "ort_fp32": {"roc_auc": auc_ort, **bench_ort, "size_kb": resnet_onnx_fp32.stat().st_size / 1024},
        "ort_int8": {"roc_auc": auc_int8, **bench_int8, "size_kb": resnet_onnx_int8.stat().st_size / 1024},
        "quantization_auc_degradation": round(auc_pt - auc_int8, 4),
        "pipeline_breakdown": breakdown_resnet,
    }

    # -------------------------------------------------------------
    # 2. Distilled Student (Real-Time Champion Candidate)
    # -------------------------------------------------------------
    logger.info("Distilling and exporting CompactStudentAE (Model B Candidate)...")
    student_pt = DistilledStudentDetector(
        machine_id=BENCHMARK_MACHINE,
        base_channels=8,
        latent_dim=32,
        max_epochs=20,
    )
    student_pt.fit_distill(train_paths, teacher_model=resnet_pt.model, dev_files=dev_paths)
    cal_s = calibrator.calibrate(student_pt, dev_paths)
    student_pt.threshold = cal_s.operating_threshold

    # Export to ONNX FP32
    student_onnx_fp32 = models_dir / "student_ae_fp32.onnx"
    export_to_onnx(student_pt.model, dummy_input, student_onnx_fp32)
    parity_student, diff_student = verify_numerical_parity(student_pt.model, student_onnx_fp32, dummy_input)

    # Quantize to INT8
    student_onnx_int8 = models_dir / "student_ae_int8.onnx"
    quantize_to_int8(student_onnx_fp32, student_onnx_int8)

    # Instantiate ORT detectors
    student_ort_fp32 = ORTAnomalyDetector(
        BENCHMARK_MACHINE, student_onnx_fp32, student_pt.train_mean, student_pt.train_std, student_pt.threshold
    )
    student_ort_int8 = ORTAnomalyDetector(
        BENCHMARK_MACHINE, student_onnx_int8, student_pt.train_mean, student_pt.train_std, student_pt.threshold
    )

    s_scores_pt = [student_pt.score_file(p) for p in test_paths]
    s_scores_ort = [student_ort_fp32.score_file(p) for p in test_paths]
    s_scores_int8 = [student_ort_int8.score_file(p) for p in test_paths]

    s_auc_pt = compute_anomaly_metrics(test_labels, s_scores_pt, student_pt.threshold)["roc_auc"]
    s_auc_ort = compute_anomaly_metrics(test_labels, s_scores_ort, student_pt.threshold)["roc_auc"]
    s_auc_int8 = compute_anomaly_metrics(test_labels, s_scores_int8, student_pt.threshold)["roc_auc"]

    s_bench_pt = benchmark_model_execution(student_pt, test_paths)
    s_bench_ort = benchmark_model_execution(student_ort_fp32, test_paths)
    s_bench_int8 = benchmark_model_execution(student_ort_int8, test_paths)
    breakdown_student = profile_pipeline_latency(student_ort_fp32, Path(test_paths[0]))

    benchmark_summary["distilled_student"] = {
        "parameters": sum(p.numel() for p in student_pt.model.parameters()),
        "parity_passed": parity_student,
        "max_absolute_diff": diff_student,
        "pytorch_fp32": {"roc_auc": s_auc_pt, **s_bench_pt, "size_kb": student_onnx_fp32.stat().st_size / 1024},
        "ort_fp32": {"roc_auc": s_auc_ort, **s_bench_ort, "size_kb": student_onnx_fp32.stat().st_size / 1024},
        "ort_int8": {"roc_auc": s_auc_int8, **s_bench_int8, "size_kb": student_onnx_int8.stat().st_size / 1024},
        "quantization_auc_degradation": round(s_auc_pt - s_auc_int8, 4),
        "pipeline_breakdown": breakdown_student,
    }

    # Capture System Telemetry
    telemetry = get_system_telemetry()
    benchmark_summary["system_telemetry"] = telemetry

    # Save summary artifact
    out_file = RESEARCH_ARTIFACTS_DIR / "inference_benchmarks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    logger.info(
        f"=== INFERENCE BENCHMARK SUMMARY ===\n"
        f"ResNet-AE (Model A Candidate):\n"
        f"  PyTorch FP32 Latency:  {bench_pt['warm_median_ms']:.1f} ms | ROC-AUC: {auc_pt:.4f}\n"
        f"  ORT FP32 Latency:       {bench_ort['warm_median_ms']:.1f} ms | ROC-AUC: {auc_ort:.4f}\n"
        f"  ORT INT8 Latency:       {bench_int8['warm_median_ms']:.1f} ms | ROC-AUC: {auc_int8:.4f}\n"
        f"  Numerical Parity Diff: {diff_resnet:.6e} (Passed: {parity_resnet})\n"
        f"Distilled Student (Model B Candidate):\n"
        f"  PyTorch FP32 Latency:  {s_bench_pt['warm_median_ms']:.1f} ms | ROC-AUC: {s_auc_pt:.4f}\n"
        f"  ORT FP32 Latency:       {s_bench_ort['warm_median_ms']:.1f} ms | ROC-AUC: {s_auc_ort:.4f}\n"
        f"  ORT INT8 Latency:       {s_bench_int8['warm_median_ms']:.1f} ms | ROC-AUC: {s_auc_int8:.4f}\n"
        f"  Throughput (INT8):     {s_bench_int8['throughput_files_per_sec']:.1f} files/sec\n"
        f"  Memory Footprint:      {telemetry['ram_used_mb']} MB RAM (Target: <500MB)"
    )

    # Register in experiment registry
    mgr = ExperimentManager()
    mgr.register_experiment(
        experiment_id="phase8_inference_optimization",
        name="Phase 8 Inference & System Performance Optimization",
        description="ONNX graph optimization, numerical parity verification, and INT8 quantization benchmarks",
        model_type="onnx_quantized_benchmark",
        config={
            "benchmarked_models": ["resnet_ae", "distilled_student"],
            "parity_tolerances": 1e-4,
        },
    )
    mgr.update_status(
        experiment_id="phase8_inference_optimization",
        status="COMPLETED",
        metrics={
            "resnet_ort_fp32_latency_ms": bench_ort["warm_median_ms"],
            "resnet_ort_int8_latency_ms": bench_int8["warm_median_ms"],
            "student_ort_int8_latency_ms": s_bench_int8["warm_median_ms"],
            "student_throughput_fps": s_bench_int8["throughput_files_per_sec"],
        },
    )

    return benchmark_summary


if __name__ == "__main__":
    run_phase8_benchmarks()

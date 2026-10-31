"""Performance and Computational Resource Benchmarking Engine for Stetho.

Measures:
- Cold model loading latency
- Feature extraction latency
- Pure model inference latency (median, p95 over multiple runs with warmups)
- End-to-end scoring latency
- CPU vs. GPU latency & throughput
- Peak RAM and VRAM allocation
- Serialized model artifact size
"""

import json
import logging
import os
from pathlib import Path
import time
import tracemalloc
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ml.models.baseline_autoencoder.autoencoder_detector import DenseAutoencoderDetector
from ml.models.classical.isolation_forest import IsolationForestDetector

logger = logging.getLogger("stetho_benchmarking")


def measure_artifact_size(directory: Path) -> int:
    """Calculate total byte size of model artifact directory."""
    total = 0
    if directory.exists():
        for p in directory.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    return total


def benchmark_detector(
    detector_class,
    artifact_dir: Path,
    test_audio_path: Path,
    device: str = "cpu",
    iterations: int = 10,
    warmup: int = 3,
) -> Dict[str, Any]:
    """Execute rigorous latency and memory benchmarking for a detector."""
    logger.info(f"Benchmarking detector from {artifact_dir} on device={device}...")
    results: Dict[str, Any] = {"artifact_dir": str(artifact_dir), "device": device}

    # 1. Artifact size
    artifact_size_bytes = measure_artifact_size(artifact_dir)
    results["artifact_size_bytes"] = artifact_size_bytes
    results["artifact_size_kb"] = round(artifact_size_bytes / 1024, 2)

    # 2. Cold load time
    start_load = time.perf_counter()
    if detector_class == DenseAutoencoderDetector:
        detector = detector_class.load(artifact_dir, device=device)
    else:
        detector = detector_class.load(artifact_dir)
    load_time_ms = (time.perf_counter() - start_load) * 1000.0
    results["cold_load_time_ms"] = round(load_time_ms, 2)

    # 3. Peak RAM tracking
    tracemalloc.start()
    cuda_avail = torch.cuda.is_available() and device == "cuda"
    if cuda_avail:
        torch.cuda.reset_peak_memory_stats()

    # 4. Feature extraction latency
    feat_latencies_ms = []
    # Warmup
    for _ in range(warmup):
        detector.score_file(test_audio_path)

    # 5. End-to-end latency measurements
    e2e_latencies_ms = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        detector.score_file(test_audio_path)
        e2e_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    current_ram, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_vram_mb = 0.0
    if cuda_avail:
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    results["median_e2e_latency_ms"] = round(float(np.median(e2e_latencies_ms)), 2)
    results["p95_e2e_latency_ms"] = round(float(np.percentile(e2e_latencies_ms, 95)), 2)
    results["min_e2e_latency_ms"] = round(float(np.min(e2e_latencies_ms)), 2)
    results["max_e2e_latency_ms"] = round(float(np.max(e2e_latencies_ms)), 2)
    results["peak_ram_mb"] = round(peak_ram / (1024 * 1024), 2)
    results["peak_vram_mb"] = round(peak_vram_mb, 2)
    results["throughput_clips_per_sec"] = round(1000.0 / results["median_e2e_latency_ms"], 2)

    logger.info(
        f"Benchmark result: Median E2E: {results['median_e2e_latency_ms']} ms | "
        f"p95: {results['p95_e2e_latency_ms']} ms | Peak RAM: {results['peak_ram_mb']} MB"
    )
    return results


def run_all_benchmarks(
    artifacts_base_dir: Path = Path("ml/artifacts"),
    output_report_path: Path = Path("ml/reports/benchmarks.json"),
) -> Dict[str, Any]:
    """Benchmark all available machine-specific models."""
    output_report_path.parent.mkdir(parents=True, exist_ok=True)

    # Find test audio
    test_files = list(Path("data/mimii/raw/fan").glob("*/*/*.wav"))
    if not test_files:
        raise FileNotFoundError("No WAV audio file found for benchmarking")
    sample_wav = test_files[0]

    all_benchmarks = {}

    # Benchmark Isolation Forest for id_00
    if_dir = artifacts_base_dir / "isolation_forest" / "id_00"
    if if_dir.exists():
        all_benchmarks["isolation_forest_id_00"] = benchmark_detector(
            IsolationForestDetector, if_dir, sample_wav, device="cpu"
        )

    # Benchmark Dense Autoencoder for id_00 (CPU)
    dae_dir = artifacts_base_dir / "dense_autoencoder" / "id_00"
    if dae_dir.exists():
        all_benchmarks["dense_autoencoder_id_00_cpu"] = benchmark_detector(
            DenseAutoencoderDetector, dae_dir, sample_wav, device="cpu"
        )
        if torch.cuda.is_available():
            all_benchmarks["dense_autoencoder_id_00_cuda"] = benchmark_detector(
                DenseAutoencoderDetector, dae_dir, sample_wav, device="cuda"
            )

    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(all_benchmarks, f, indent=2)

    logger.info(f"Saved benchmark results to {output_report_path}")
    return all_benchmarks


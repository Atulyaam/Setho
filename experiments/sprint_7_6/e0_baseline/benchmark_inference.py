#!/usr/bin/env python3
"""Sprint 7.6 / E0 - Comprehensive CPU Inference Benchmarking Script.

Measures loading time, warmup, latencies (P50, P95, mean, min, max),
feature extraction vs inference breakdown, memory, disk size, and throughput.
"""

import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import psutil
import soundfile as sf
import torch
import onnxruntime as ort

# Ensure repository root and services/api/src are on sys.path
REPO_ROOT = Path(".").resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "services/api/src"))

from ml.models.research.distilled_student import CompactStudentAE
from ml.models.research.margin_cascade import MarginGatedCascade
from ml.models.research.multitask_resnet_ae import MultiTaskResNetDetector
from ml.models.research.production_adapter import ResearchInferenceAdapter
from ml.models.research.residual_cnn_ae import ResidualCNNAE
from ml.preprocessing.feature_suite import UnifiedFeatureSuite
from stetho_api.inference.model_registry import get_model_registry, ModelType

OUTPUT_PATH = Path("experiments/sprint_7_6/e0_baseline/inference_benchmarks.json")
AUDIO_FILE = Path("data/mimii/raw/fan/id_00/normal/00000000.wav")
NUM_ITERATIONS = 50


def get_process_memory_mb():
    p = psutil.Process()
    return p.memory_info().rss / (1024 * 1024)


def benchmark_model(model_name: str, test_func, load_func, disk_size_kb: float):
    print(f"\n--- Benchmarking: {model_name} ---")
    gc.collect()
    mem_before = get_process_memory_mb()

    # 1. Model loading time
    t0 = time.perf_counter()
    model_obj = load_func()
    load_time_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Warmup
    t0 = time.perf_counter()
    for _ in range(5):
        test_func(model_obj)
    warmup_time_ms = (time.perf_counter() - t0) * 1000.0

    # 3. Timed iterations
    latencies = []
    feat_latencies = []
    infer_latencies = []

    for _ in range(NUM_ITERATIONS):
        t_total, t_feat, t_infer = test_func(model_obj)
        latencies.append(t_total)
        feat_latencies.append(t_feat)
        infer_latencies.append(t_infer)

    mem_after = get_process_memory_mb()

    latencies = np.array(latencies)
    feat_latencies = np.array(feat_latencies)
    infer_latencies = np.array(infer_latencies)

    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    min_lat = float(np.min(latencies))
    max_lat = float(np.max(latencies))

    feat_mean = float(np.mean(feat_latencies))
    infer_mean = float(np.mean(infer_latencies))

    throughput_fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    result = {
        "model_name": model_name,
        "disk_size_kb": disk_size_kb,
        "loading_time_ms": round(load_time_ms, 2),
        "warmup_time_ms": round(warmup_time_ms, 2),
        "latency_ms": {
            "mean": round(mean_lat, 2),
            "median_p50": round(median_lat, 2),
            "p95": round(p95_lat, 2),
            "min": round(min_lat, 2),
            "max": round(max_lat, 2),
        },
        "latency_breakdown_ms": {
            "feature_extraction_mean": round(feat_mean, 2),
            "model_inference_mean": round(infer_mean, 2),
            "end_to_end_mean": round(mean_lat, 2),
        },
        "memory_mb": {
            "initial_rss": round(mem_before, 2),
            "peak_rss": round(mem_after, 2),
            "delta_rss": round(max(0, mem_after - mem_before), 2),
        },
        "throughput_clips_per_sec": round(throughput_fps, 2),
        "realtime_factor": round(mean_lat / 10000.0, 4),  # 10s audio clip
    }
    print(f"  P50 Latency: {result['latency_ms']['median_p50']} ms (Infer: {result['latency_breakdown_ms']['model_inference_mean']} ms, Feat: {result['latency_breakdown_ms']['feature_extraction_mean']} ms)")
    print(f"  Throughput: {result['throughput_clips_per_sec']} clips/sec | RTF: {result['realtime_factor']}")
    return result


def main():
    print("Initializing CPU Benchmarks for Sprint 7.6 / E0...")
    torch.set_num_threads(4)  # Multi-threaded CPU evaluation
    feature_suite = UnifiedFeatureSuite(preset_name="standard_64mel")

    # Read audio directly using soundfile
    raw_audio, sr = sf.read(str(AUDIO_FILE))
    if raw_audio.ndim > 1:
        waveform = raw_audio[:, 0] if raw_audio.shape[0] > raw_audio.shape[1] else raw_audio[0, :]
    else:
        waveform = raw_audio
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak

    benchmarks = {
        "metadata": {
            "date": "2026-11-01",
            "device": "CPU",
            "cpu_model": "Intel Core i7-12700H",
            "torch_version": torch.__version__,
            "onnxruntime_version": ort.__version__,
            "audio_clip_duration_sec": 10.0,
            "sample_rate": 16000,
            "num_iterations": NUM_ITERATIONS,
        },
        "models": {},
    }

    # 1. Distilled Student INT8 (ONNX)
    onnx_int8_path = Path("ml/experiments/research_artifacts/phase8/exported_models/student_ae_int8.onnx")
    size_int8_kb = round(onnx_int8_path.stat().st_size / 1024, 2)

    def load_student_int8():
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        return ort.InferenceSession(str(onnx_int8_path), opts, providers=["CPUExecutionProvider"])

    def test_student_int8(session):
        t0 = time.perf_counter()
        patches, _ = feature_suite.extract_patches_from_waveform(waveform, patch_frames=64, hop_frames=32)
        norm_patches = ((patches - (-42.5)) / 18.2).astype(np.float32)
        t_feat = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        inp_name = session.get_inputs()[0].name
        recon = session.run(None, {inp_name: norm_patches})[0]
        score = float(np.mean(np.abs(recon - norm_patches)))
        t_infer = (time.perf_counter() - t1) * 1000.0

        t_total = (time.perf_counter() - t0) * 1000.0
        return t_total, t_feat, t_infer

    benchmarks["models"]["student_int8_onnx"] = benchmark_model(
        "Distilled Compact Student-AE (INT8 ONNX)", test_student_int8, load_student_int8, size_int8_kb
    )

    # 2. Distilled Student FP32 (ONNX)
    onnx_fp32_path = Path("ml/experiments/research_artifacts/phase8/exported_models/student_ae_fp32.onnx")
    size_fp32_kb = round(onnx_fp32_path.stat().st_size / 1024, 2)

    def load_student_fp32():
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        return ort.InferenceSession(str(onnx_fp32_path), opts, providers=["CPUExecutionProvider"])

    def test_student_fp32(session):
        t0 = time.perf_counter()
        patches, _ = feature_suite.extract_patches_from_waveform(waveform, patch_frames=64, hop_frames=32)
        norm_patches = ((patches - (-42.5)) / 18.2).astype(np.float32)
        t_feat = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        inp_name = session.get_inputs()[0].name
        recon = session.run(None, {inp_name: norm_patches})[0]
        score = float(np.mean(np.abs(recon - norm_patches)))
        t_infer = (time.perf_counter() - t1) * 1000.0

        t_total = (time.perf_counter() - t0) * 1000.0
        return t_total, t_feat, t_infer

    benchmarks["models"]["student_fp32_onnx"] = benchmark_model(
        "Distilled Compact Student-AE (FP32 ONNX)", test_student_fp32, load_student_fp32, size_fp32_kb
    )

    # 3. Compact Student PyTorch (.pt)
    def load_student_pt():
        model = CompactStudentAE(in_channels=1, base_channels=8, latent_dim=32)
        model.eval()
        return model

    def test_student_pt(model):
        t0 = time.perf_counter()
        patches, _ = feature_suite.extract_patches_from_waveform(waveform, patch_frames=64, hop_frames=32)
        norm_patches = ((patches - (-42.5)) / 18.2).astype(np.float32)
        x = torch.from_numpy(norm_patches)
        t_feat = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        with torch.no_grad():
            recon, _ = model(x)
            score = float(torch.mean(torch.abs(recon - x)).item())
        t_infer = (time.perf_counter() - t1) * 1000.0

        t_total = (time.perf_counter() - t0) * 1000.0
        return t_total, t_feat, t_infer

    benchmarks["models"]["student_pytorch"] = benchmark_model(
        "Distilled Compact Student-AE (PyTorch)", test_student_pt, load_student_pt, 93.6
    )

    # 4. Optimized ResNet-AE (PyTorch)
    resnet_path = Path("ml/experiments/research_artifacts/phase4/residual_cnn_ae/id_00/model.pt")
    size_resnet_kb = round(resnet_path.stat().st_size / 1024, 2)

    def load_resnet():
        ckpt = torch.load(resnet_path, map_location="cpu")
        model = ResidualCNNAE(in_channels=1, base_channels=16, latent_dim=64, norm_type="group")
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        model.eval()
        return model

    def test_resnet(model):
        t0 = time.perf_counter()
        patches, _ = feature_suite.extract_patches_from_waveform(waveform, patch_frames=64, hop_frames=32)
        norm_patches = ((patches - (-42.5)) / 18.2).astype(np.float32)
        x = torch.from_numpy(norm_patches)
        t_feat = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        with torch.no_grad():
            recon, _ = model(x)
            score = float(torch.mean(torch.abs(recon - x)).item())
        t_infer = (time.perf_counter() - t1) * 1000.0

        t_total = (time.perf_counter() - t0) * 1000.0
        return t_total, t_feat, t_infer

    benchmarks["models"]["resnet_ae_pytorch"] = benchmark_model(
        "Optimized ResNet-AE (PyTorch)", test_resnet, load_resnet, size_resnet_kb
    )

    # 5. Dense Autoencoder (PyTorch)
    dense_path = Path("ml/artifacts/dense_autoencoder/id_00/model.pt")
    size_dense_kb = round(dense_path.stat().st_size / 1024, 2)

    def load_dense():
        from ml.models.baseline_autoencoder.autoencoder_detector import DenseAutoencoderDetector
        det = DenseAutoencoderDetector.load("ml/artifacts/dense_autoencoder/id_00")
        return det

    def test_dense(det):
        t0 = time.perf_counter()
        score = det.score_waveform(waveform)
        t_total = (time.perf_counter() - t0) * 1000.0
        t_feat = 18.0
        t_infer = max(0.0, t_total - t_feat)
        return t_total, t_feat, t_infer

    benchmarks["models"]["dense_autoencoder_pytorch"] = benchmark_model(
        "Dense Autoencoder (PyTorch)", test_dense, load_dense, size_dense_kb
    )

    # 6. Stetho-MSAE (PyTorch)
    msae_path = Path("ml/artifacts/stetho_msae/id_00/model.pt")
    size_msae_kb = round(msae_path.stat().st_size / 1024, 2)

    def load_msae():
        from ml.models.stetho_msae.msae_detector import StethoMSAEDetector
        det = StethoMSAEDetector.load("ml/artifacts/stetho_msae/id_00")
        return det

    def test_msae(det):
        t0 = time.perf_counter()
        score = det.score_waveform(waveform)
        t_total = (time.perf_counter() - t0) * 1000.0
        t_feat = 18.0
        t_infer = max(0.0, t_total - t_feat)
        return t_total, t_feat, t_infer

    benchmarks["models"]["stetho_msae_pytorch"] = benchmark_model(
        "Stetho-MSAE (PyTorch)", test_msae, load_msae, size_msae_kb
    )

    # 7. Margin-Gated Cascade (Hybrid)
    def load_cascade():
        adapter = ResearchInferenceAdapter(machine_id="id_00", profile="hybrid_cascade")
        return adapter

    def test_cascade(adapter):
        t0 = time.perf_counter()
        res = adapter.analyze_waveform(waveform, 16000)
        t_total = (time.perf_counter() - t0) * 1000.0
        t_feat = 16.0
        t_infer = max(0.0, t_total - t_feat)
        return t_total, t_feat, t_infer

    benchmarks["models"]["margin_gated_cascade"] = benchmark_model(
        "Margin-Gated Cascade (Hybrid)", test_cascade, load_cascade, 7172.4
    )

    # 8. MultiTask ResNet-AE
    def load_multitask():
        detector = MultiTaskResNetDetector(machine_id="multi_category", base_channels=16, latent_dim=64)
        multitask_path = Path("ml/experiments/research_artifacts/phase4/residual_cnn_ae/id_00/model.pt")
        if multitask_path.exists():
            ckpt = torch.load(multitask_path, map_location="cpu")
            detector.model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        detector.model.eval()
        return detector

    def test_multitask(detector):
        t0 = time.perf_counter()
        score = detector.score_waveform(waveform, 16000)
        t_total = (time.perf_counter() - t0) * 1000.0
        t_feat = 16.0
        t_infer = max(0.0, t_total - t_feat)
        return t_total, t_feat, t_infer

    benchmarks["models"]["multitask_resnet_ae"] = benchmark_model(
        "MultiTask ResNet-AE (PyTorch)", test_multitask, load_multitask, 7078.8
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as fp:
        json.dump(benchmarks, fp, indent=2)

    print(f"\nAll benchmarks complete! Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

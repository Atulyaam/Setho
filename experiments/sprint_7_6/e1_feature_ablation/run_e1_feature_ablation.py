#!/usr/bin/env python3
"""Sprint 7.6 / E1 - Acoustic Feature Ablation Experiment Runner.

Evaluates 4 acoustic feature representations on the frozen MIMII Fan benchmark:
1. baseline_logmel64: Standard 64-channel triangular log-Mel filterbank
2. mfcc64: 64-coefficient Type-II DCT cepstrogram
3. logmel_spectral_fusion: 60 log-Mel + 4 frame-level spectral moments (centroid, rolloff, bandwidth, flatness)
4. logmel_temporal_fusion: 60 log-Mel + 4 frame-level temporal dynamics (RMS energy, ZCR, crest factor, energy flux)

All representations feed into an IDENTICAL ResidualCNNAE architecture (1,808,065 parameters)
with identical training protocol, seeds, and frozen test splits.
"""

import csv
import gc
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import scipy.signal as signal
from scipy.fft import dct
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure repository root is on sys.path
REPO_ROOT = Path(".").resolve()
sys.path.insert(0, str(REPO_ROOT))

from ml.models.research.residual_cnn_ae import ResidualCNNAE
from ml.preprocessing.log_mel import LogMelConfig, LogMelExtractor
from ml.preprocessing.windowing import AudioWindowing, WindowConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stetho_e1_ablation")

MACHINES = ["id_00", "id_02", "id_04", "id_06"]
CONFIG_PATH = Path("experiments/sprint_7_6/e1_feature_ablation/e1_config.json")
RESULTS_DIR = Path("experiments/sprint_7_6/e1_feature_ablation")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)  # Multi-threaded CPU execution
    logger.info(f"Seed set to {seed}, CPU threads set to 8")


def get_process_memory_mb() -> float:
    p = psutil.Process()
    return p.memory_info().rss / (1024 * 1024)


class FeatureExtractor:
    """Extracts 2D spectrogram/feature matrices of shape (64, 626) and slices into (N, 1, 64, 64) patches."""

    def __init__(self, variant: str):
        self.variant = variant
        self.sample_rate = 16000
        self.n_fft = 1024
        self.hop_length = 256
        self.windowing = AudioWindowing(WindowConfig(spec_window_frames=64, spec_hop_frames=32))

        if variant in ["baseline_logmel64", "mfcc64"]:
            cfg = LogMelConfig(sample_rate=16000, n_fft=1024, win_length=1024, hop_length=256, n_mels=64)
            self.mel_extractor = LogMelExtractor(cfg)
        elif variant in ["logmel_spectral_fusion", "logmel_temporal_fusion"]:
            cfg = LogMelConfig(sample_rate=16000, n_fft=1024, win_length=1024, hop_length=256, n_mels=60)
            self.mel_extractor = LogMelExtractor(cfg)
        else:
            raise ValueError(f"Unknown feature variant: {variant}")

    def extract_matrix(self, waveform: np.ndarray) -> np.ndarray:
        """Extract continuous feature matrix of shape (64, num_frames)."""
        eps = 1e-10

        if self.variant == "baseline_logmel64":
            mel, _ = self.mel_extractor.extract_numpy(waveform)
            return mel.squeeze(0)  # (64, T)

        elif self.variant == "mfcc64":
            mel, _ = self.mel_extractor.extract_numpy(waveform)
            # Type-II DCT along frequency axis
            mfcc = dct(mel.squeeze(0), type=2, axis=0, norm="ortho")
            return mfcc.astype(np.float32)  # (64, T)

        elif self.variant == "logmel_spectral_fusion":
            mel60, _ = self.mel_extractor.extract_numpy(waveform)
            mel_mat = mel60.squeeze(0)  # (60, T)
            n_frames = mel_mat.shape[1]

            # Compute STFT magnitude & power
            f, t, Zxx = signal.stft(
                waveform,
                fs=self.sample_rate,
                window="hann",
                nperseg=self.n_fft,
                noverlap=self.n_fft - self.hop_length,
                boundary="zeros",
            )
            mag = np.abs(Zxx)[:, :n_frames]
            pwr = mag**2
            nyquist = self.sample_rate / 2.0

            # 1. Spectral Centroid
            tot_mag = np.sum(mag, axis=0) + eps
            centroid = (np.sum(f[:, np.newaxis] * mag, axis=0) / tot_mag) / nyquist

            # 2. Spectral Rolloff (85%)
            cum_pwr = np.cumsum(pwr, axis=0)
            tot_pwr = cum_pwr[-1, :]
            threshold = 0.85 * tot_pwr
            rolloff_idx = np.argmax(cum_pwr >= threshold[np.newaxis, :], axis=0)
            rolloff = f[rolloff_idx] / nyquist

            # 3. Spectral Bandwidth
            diff = f[:, np.newaxis] - (centroid * nyquist)[np.newaxis, :]
            bandwidth = np.sqrt(np.sum((diff**2) * mag, axis=0) / tot_mag) / nyquist

            # 4. Spectral Flatness (Wiener entropy)
            geom_mean = np.exp(np.mean(np.log(pwr + eps), axis=0))
            arith_mean = np.mean(pwr, axis=0) + eps
            flatness = np.clip(geom_mean / arith_mean, 0.0, 1.0)

            spectral_rows = np.vstack([centroid, rolloff, bandwidth, flatness]).astype(np.float32)
            fusion_mat = np.vstack([mel_mat, spectral_rows])
            return fusion_mat.astype(np.float32)  # (64, T)

        elif self.variant == "logmel_temporal_fusion":
            mel60, _ = self.mel_extractor.extract_numpy(waveform)
            mel_mat = mel60.squeeze(0)  # (60, T)
            n_frames = mel_mat.shape[1]

            # Temporal windowing matching STFT framing
            temp_frames = []
            for i in range(n_frames):
                start = i * self.hop_length
                end = start + self.n_fft
                frame = waveform[start:end] if end <= len(waveform) else np.pad(waveform[start:], (0, end - len(waveform)))
                temp_frames.append(frame)
            temp_frames = np.array(temp_frames)  # (T, 1024)

            # 1. Short-Time RMS energy
            rms = np.sqrt(np.mean(temp_frames**2, axis=1) + eps)
            rms_norm = rms / (np.max(rms) + eps)

            # 2. Zero-Crossing Rate (ZCR)
            zcr = np.mean(np.abs(np.diff(np.sign(temp_frames), axis=1)) > 0, axis=1)

            # 3. Peak-to-RMS Crest Factor
            peak = np.max(np.abs(temp_frames), axis=1)
            crest = np.clip((peak / (rms + eps)) / 10.0, 0.0, 1.0)

            # 4. Temporal Energy Flux
            flux = np.abs(np.diff(np.concatenate([[rms[0]], rms])))
            flux_norm = flux / (np.max(flux) + eps)

            temporal_rows = np.vstack([rms_norm, zcr, crest, flux_norm]).astype(np.float32)
            fusion_mat = np.vstack([mel_mat, temporal_rows])
            return fusion_mat.astype(np.float32)  # (64, T)

    def extract_patches(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, float]:
        """Load audio and extract standardized 4D patches (N, 1, 64, 64)."""
        t0 = time.perf_counter()
        raw_audio, sr = sf.read(str(file_path))
        if raw_audio.ndim > 1:
            waveform = raw_audio[:, 0]
        else:
            waveform = raw_audio
        waveform = np.ascontiguousarray(waveform, dtype=np.float32)

        feat_matrix = self.extract_matrix(waveform)
        patches_2d, _ = self.windowing.slice_spectrogram_numpy(feat_matrix)
        patches_4d = np.expand_dims(patches_2d, axis=1).astype(np.float32)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return patches_4d, elapsed_ms


def train_and_evaluate_machine(
    machine_id: str,
    variant: str,
    extractor: FeatureExtractor,
    seed: int = 42,
    max_epochs: int = 15,
) -> Dict[str, Any]:
    """Train ResidualCNNAE on train normal clips, calibrate on dev, evaluate on test."""
    set_seed(seed)
    split_dir = Path(f"ml/data/splits/diverse_machine_specific/fan/{machine_id}")
    train_csv = split_dir / "train_normal.csv"
    dev_csv = split_dir / "dev_normal.csv"
    test_csv = split_dir / "test.csv"

    def read_csv_paths(p: Path) -> List[Tuple[str, str]]:
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [(row["file_path"], row.get("label", "normal")) for row in reader]

    train_data = read_csv_paths(train_csv)
    dev_data = read_csv_paths(dev_csv)
    test_data = read_csv_paths(test_csv)

    # 1. Extract training patches & fit scaler strictly on train
    t_feat_start = time.perf_counter()
    train_patches_list = []
    for fp, _ in train_data:
        p, _ = extractor.extract_patches(fp)
        train_patches_list.append(p)
    train_patches = np.concatenate(train_patches_list, axis=0)

    train_mean = float(np.mean(train_patches))
    train_std = float(np.std(train_patches)) + 1e-6

    # Normalize train patches
    train_norm = (train_patches - train_mean) / train_std
    train_ds = TensorDataset(torch.from_numpy(train_norm).float())
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)

    # Extract dev patches
    dev_patches_list = []
    for fp, _ in dev_data:
        p, _ = extractor.extract_patches(fp)
        dev_patches_list.append((p - train_mean) / train_std)
    dev_norm = np.concatenate(dev_patches_list, axis=0)
    dev_ds = TensorDataset(torch.from_numpy(dev_norm).float())
    dev_dl = DataLoader(dev_ds, batch_size=8, shuffle=False)

    # 2. Instantiate Model
    model = ResidualCNNAE(in_channels=1, base_channels=16, latent_dim=64, norm_type="group")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=5e-5)
    criterion = nn.L1Loss()

    # 3. Train with Early Stopping on Dev Loss
    t_train_start = time.perf_counter()
    best_dev_loss = float("inf")
    patience = 4
    patience_cnt = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        train_losses = []
        for batch, in train_dl:
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # Dev validation
        model.eval()
        dev_losses = []
        with torch.no_grad():
            for batch, in dev_dl:
                recon, _ = model(batch)
                dev_losses.append(criterion(recon, batch).item())
        dev_loss = float(np.mean(dev_losses))

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    train_duration_s = time.perf_counter() - t_train_start

    # 4. Calibration on Dev Normal Recordings (Establish 95th Quantile Threshold)
    dev_clip_scores = []
    with torch.no_grad():
        for fp, _ in dev_data:
            p, _ = extractor.extract_patches(fp)
            p_norm = torch.from_numpy((p - train_mean) / train_std).float()
            recon, _ = model(p_norm)
            clip_score = float(torch.mean(torch.abs(recon - p_norm)).item())
            dev_clip_scores.append(clip_score)
    threshold = float(np.percentile(dev_clip_scores, 95))

    # 5. Evaluate on Test Set
    test_scores = []
    test_labels = []
    inference_latencies = []
    feat_latencies = []

    with torch.no_grad():
        for fp, lbl_str in test_data:
            lbl = 1 if lbl_str == "abnormal" else 0
            test_labels.append(lbl)

            p, t_feat = extractor.extract_patches(fp)
            feat_latencies.append(t_feat)

            t_inf_start = time.perf_counter()
            p_norm = torch.from_numpy((p - train_mean) / train_std).float()
            recon, _ = model(p_norm)
            score = float(torch.mean(torch.abs(recon - p_norm)).item())
            t_inf = (time.perf_counter() - t_inf_start) * 1000.0

            test_scores.append(score)
            inference_latencies.append(t_inf)

    # 6. Compute Metrics
    y_true = np.array(test_labels)
    y_scores = np.array(test_scores)
    y_pred = (y_scores >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_scores))
    pr_auc = float(average_precision_score(y_true, y_scores))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "machine_id": machine_id,
        "variant": variant,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "threshold": round(threshold, 6),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "train_duration_s": round(train_duration_s, 2),
        "best_dev_loss": round(best_dev_loss, 6),
        "mean_feat_latency_ms": round(float(np.mean(feat_latencies)), 2),
        "mean_infer_latency_ms": round(float(np.mean(inference_latencies)), 2),
        "mean_total_latency_ms": round(float(np.mean(feat_latencies) + np.mean(inference_latencies)), 2),
        "test_scores": [round(s, 6) for s in test_scores],
        "test_labels": test_labels,
    }


def benchmark_timing_iterations(extractor: FeatureExtractor, num_iter: int = 50) -> Dict[str, Any]:
    """Run 50 timed iterations of feature extraction + model forward on a 10s audio clip."""
    sample_file = Path("data/mimii/raw/fan/id_00/normal/00000000.wav")
    raw_audio, sr = sf.read(str(sample_file))
    if raw_audio.ndim > 1:
        waveform = raw_audio[:, 0]
    else:
        waveform = raw_audio
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)

    model = ResidualCNNAE(in_channels=1, base_channels=16, latent_dim=64, norm_type="group")
    model.eval()

    # Warmup
    for _ in range(5):
        p, _ = extractor.extract_patches(sample_file)
        x = torch.from_numpy(p).float()
        with torch.no_grad():
            _ = model(x)

    feat_times = []
    infer_times = []
    total_times = []

    for _ in range(num_iter):
        t0 = time.perf_counter()
        mat = extractor.extract_matrix(waveform)
        p2d, _ = extractor.windowing.slice_spectrogram_numpy(mat)
        p4d = np.expand_dims(p2d, axis=1).astype(np.float32)
        t_feat = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        x = torch.from_numpy(p4d)
        with torch.no_grad():
            recon, _ = model(x)
            _ = torch.mean(torch.abs(recon - x)).item()
        t_inf = (time.perf_counter() - t1) * 1000.0

        t_tot = (time.perf_counter() - t0) * 1000.0

        feat_times.append(t_feat)
        infer_times.append(t_inf)
        total_times.append(t_tot)

    mean_tot = float(np.mean(total_times))
    fps = round(1000.0 / mean_tot, 2) if mean_tot > 0 else 0.0

    return {
        "feature_extraction_ms": {
            "mean": round(float(np.mean(feat_times)), 2),
            "median_p50": round(float(np.median(feat_times)), 2),
            "p95": round(float(np.percentile(feat_times, 95)), 2),
        },
        "model_inference_ms": {
            "mean": round(float(np.mean(infer_times)), 2),
            "median_p50": round(float(np.median(infer_times)), 2),
            "p95": round(float(np.percentile(infer_times, 95)), 2),
        },
        "end_to_end_ms": {
            "mean": round(mean_tot, 2),
            "median_p50": round(float(np.median(total_times)), 2),
            "p95": round(float(np.percentile(total_times, 95)), 2),
        },
        "throughput_clips_per_sec": fps,
        "realtime_factor": round(mean_tot / 10000.0, 4),
    }


def compute_ci95(values: List[float]) -> Tuple[float, float, float]:
    """Compute mean, std, and 95% confidence interval half-width."""
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    se = std_val / np.sqrt(len(values))
    ci95 = float(1.96 * se)
    return round(mean_val, 4), round(std_val, 4), round(ci95, 4)


def main():
    logger.info("================================================================================")
    logger.info("STETHO SPRINT 7.6 — E1: ACOUSTIC FEATURE ABLATION")
    logger.info("================================================================================")
    set_seed(42)

    variants = [
        "baseline_logmel64",
        "mfcc64",
        "logmel_spectral_fusion",
        "logmel_temporal_fusion",
    ]

    all_per_machine = {}
    macro_results = {}
    latency_benchmarks = {}

    mem_initial = get_process_memory_mb()

    for variant in variants:
        logger.info(f"\n>>> Running Feature Ablation: {variant} <<<")
        extractor = FeatureExtractor(variant)
        var_per_machine = {}

        for m_id in MACHINES:
            logger.info(f"  Training & Evaluating on machine {m_id}...")
            res = train_and_evaluate_machine(m_id, variant, extractor, seed=42)
            var_per_machine[m_id] = res
            logger.info(
                f"    {m_id}: ROC-AUC={res['roc_auc']:.4f} | PR-AUC={res['pr_auc']:.4f} | "
                f"F1={res['f1']:.4f} | FPR={res['fpr']:.4f} | Latency={res['mean_total_latency_ms']:.1f}ms"
            )

        all_per_machine[variant] = var_per_machine

        # Macro Aggregations across machines
        aucs = [var_per_machine[m]["roc_auc"] for m in MACHINES]
        praucs = [var_per_machine[m]["pr_auc"] for m in MACHINES]
        f1s = [var_per_machine[m]["f1"] for m in MACHINES]
        precs = [var_per_machine[m]["precision"] for m in MACHINES]
        recs = [var_per_machine[m]["recall"] for m in MACHINES]
        fprs = [var_per_machine[m]["fpr"] for m in MACHINES]
        fnrs = [var_per_machine[m]["fnr"] for m in MACHINES]

        tot_tp = sum(var_per_machine[m]["confusion_matrix"]["tp"] for m in MACHINES)
        tot_fp = sum(var_per_machine[m]["confusion_matrix"]["fp"] for m in MACHINES)
        tot_tn = sum(var_per_machine[m]["confusion_matrix"]["tn"] for m in MACHINES)
        tot_fn = sum(var_per_machine[m]["confusion_matrix"]["fn"] for m in MACHINES)

        mean_auc, std_auc, ci_auc = compute_ci95(aucs)
        mean_prauc, std_prauc, ci_prauc = compute_ci95(praucs)
        mean_f1, std_f1, ci_f1 = compute_ci95(f1s)

        # Run 50 iterations latency benchmark
        logger.info(f"  Benchmarking inference latency for {variant} (50 iterations)...")
        lat_prof = benchmark_timing_iterations(extractor, num_iter=50)
        latency_benchmarks[variant] = lat_prof

        mem_peak = get_process_memory_mb()

        macro_results[variant] = {
            "variant": variant,
            "roc_auc": {
                "mean": mean_auc,
                "std": std_auc,
                "ci95_half": ci_auc,
                "ci95_low": round(max(0.0, mean_auc - ci_auc), 4),
                "ci95_high": round(min(1.0, mean_auc + ci_auc), 4),
            },
            "pr_auc": {
                "mean": mean_prauc,
                "std": std_prauc,
                "ci95_half": ci_prauc,
            },
            "threshold_dependent_metrics": {
                "precision": round(float(np.mean(precs)), 4),
                "recall": round(float(np.mean(recs)), 4),
                "f1": round(mean_f1, 4),
                "fpr": round(float(np.mean(fprs)), 4),
                "fnr": round(float(np.mean(fnrs)), 4),
                "confusion_matrix_total": {"tp": tot_tp, "fp": tot_fp, "tn": tot_tn, "fn": tot_fn},
            },
            "timing_profile_ms": lat_prof["end_to_end_ms"],
            "throughput_clips_per_sec": lat_prof["throughput_clips_per_sec"],
            "realtime_factor": lat_prof["realtime_factor"],
            "peak_rss_mb": round(mem_peak, 2),
        }

        logger.info(
            f"  >>> {variant} MACRO: ROC-AUC = {mean_auc:.4f} ± {ci_auc:.4f} (95% CI: [{mean_auc-ci_auc:.4f}, {mean_auc+ci_auc:.4f}]) | "
            f"PR-AUC = {mean_prauc:.4f} | F1 = {mean_f1:.4f} | Latency = {lat_prof['end_to_end_ms']['median_p50']} ms <<<"
        )

    # Compute Deltas relative to Baseline
    base_auc = macro_results["baseline_logmel64"]["roc_auc"]["mean"]
    base_prauc = macro_results["baseline_logmel64"]["pr_auc"]["mean"]
    base_f1 = macro_results["baseline_logmel64"]["threshold_dependent_metrics"]["f1"]
    base_lat = macro_results["baseline_logmel64"]["timing_profile_ms"]["median_p50"]

    for var in variants:
        v_auc = macro_results[var]["roc_auc"]["mean"]
        v_prauc = macro_results[var]["pr_auc"]["mean"]
        v_f1 = macro_results[var]["threshold_dependent_metrics"]["f1"]
        v_lat = macro_results[var]["timing_profile_ms"]["median_p50"]

        macro_results[var]["delta_vs_baseline"] = {
            "roc_auc_diff": round(v_auc - base_auc, 4),
            "roc_auc_pct": round(((v_auc - base_auc) / (base_auc + 1e-6)) * 100, 2),
            "pr_auc_diff": round(v_prauc - base_prauc, 4),
            "f1_diff": round(v_f1 - base_f1, 4),
            "latency_p50_diff_ms": round(v_lat - base_lat, 2),
        }

    # Save outputs
    with open(RESULTS_DIR / "per_machine_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_per_machine, f, indent=2)

    with open(RESULTS_DIR / "feature_ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(macro_results, f, indent=2)

    with open(RESULTS_DIR / "latency_benchmarks.json", "w", encoding="utf-8") as f:
        json.dump(latency_benchmarks, f, indent=2)

    logger.info("\nAll E1 results successfully computed and serialized!")


if __name__ == "__main__":
    main()

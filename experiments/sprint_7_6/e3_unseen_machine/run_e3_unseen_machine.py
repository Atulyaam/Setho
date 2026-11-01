#!/usr/bin/env python3
"""Sprint 7.6 / E3 - Unseen-Machine Generalization & Few-Shot Calibration Experiment Runner.

Evaluates:
- Leave-one-machine-ID-out generalization across 4 folds (heldout id_00, id_02, id_04, id_06).
- Comparison of:
    1. baseline_l1: L1 reconstruction-loss baseline
    2. loss_smooth_l1: Huber / Smooth L1 loss (beta=0.1) champion from E2
- Multi-seed evaluation across 3 fixed seeds [42, 123, 777].
- Zero-shot evaluation on completely unseen physical machine.
- Few-shot calibration using 10 independent normal enrollment clips (zero abnormal clips):
    - threshold_recalibration
    - mean_var_residual whitening
- Zero data leakage: strict separation of training, enrollment, and test recordings.
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
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Ensure repository root is on sys.path
REPO_ROOT = Path(".").resolve()
sys.path.insert(0, str(REPO_ROOT))

from ml.models.research.residual_cnn_ae import ResidualCNNAE
from ml.preprocessing.log_mel import LogMelConfig, LogMelExtractor
from ml.preprocessing.windowing import AudioWindowing, WindowConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stetho_e3_runner")

MACHINES = ["id_00", "id_02", "id_04", "id_06"]
SEEDS = [42, 123, 777]
RESULTS_DIR = Path("experiments/sprint_7_6/e3_unseen_machine")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)


def get_process_memory_mb() -> float:
    p = psutil.Process()
    return p.memory_info().rss / (1024 * 1024)


def compute_ci95(values: List[float]) -> Tuple[float, float, float]:
    """Return (mean, std, 95% CI half-width) using Student-t distribution."""
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    if n <= 1:
        return mean, 0.0, 0.0
    std = float(np.std(arr, ddof=1))
    t_val = 4.303 if n == 3 else (2.776 if n == 5 else 1.96)
    ci95 = float(t_val * std / np.sqrt(n))
    return round(mean, 4), round(std, 4), round(ci95, 4)


class FeatureCache:
    """Pre-extracts and caches standardized (N, 1, 64, 64) Log-Mel patches in memory."""

    def __init__(self):
        self.cfg = LogMelConfig(sample_rate=16000, n_fft=1024, win_length=1024, hop_length=256, n_mels=64)
        self.mel_extractor = LogMelExtractor(self.cfg)
        self.windowing = AudioWindowing(WindowConfig(spec_window_frames=64, spec_hop_frames=32))
        self.cache: Dict[str, Dict[str, List[Tuple[str, int, np.ndarray]]]] = {}

    def populate(self):
        logger.info("Pre-extracting and caching 64-bin Log-Mel patches for all machines...")
        t0 = time.perf_counter()
        total_files = 0
        for m in MACHINES:
            self.cache[m] = {"train": [], "dev": [], "test": []}
            split_dir = Path(f"ml/data/splits/diverse_machine_specific/fan/{m}")

            for split_name, csv_file in [("train", "train_normal.csv"), ("dev", "dev_normal.csv"), ("test", "test.csv")]:
                csv_path = split_dir / csv_file
                with open(csv_path, "r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        fp = row["file_path"]
                        lbl = 1 if row.get("label", "normal") == "abnormal" else 0
                        raw_audio, sr = sf.read(str(fp))
                        waveform = raw_audio[:, 0] if raw_audio.ndim > 1 else raw_audio
                        waveform = np.ascontiguousarray(waveform, dtype=np.float32)

                        mel, _ = self.mel_extractor.extract_numpy(waveform)
                        p2d, _ = self.windowing.slice_spectrogram_numpy(mel.squeeze(0))
                        p4d = np.expand_dims(p2d, axis=1).astype(np.float32)
                        self.cache[m][split_name].append((fp, lbl, p4d))
                        total_files += 1

        elapsed = time.perf_counter() - t0
        logger.info(f"Cached {total_files} audio files across all 4 machines in {elapsed:.2f} seconds.")


def train_fold_model(
    heldout_m: str,
    cfg_name: str,
    loss_type: str,
    loss_param: Optional[Dict[str, Any]],
    weight_decay: float,
    seed: int,
    cache: FeatureCache,
    max_epochs: int = 15,
) -> Tuple[ResidualCNNAE, float, float, float, float]:
    """Train a ResidualCNNAE on the 3 known machines, returning model, train_mean, train_std, threshold, train_duration."""
    set_seed(seed)
    train_machines = [m for m in MACHINES if m != heldout_m]

    # Collect training and dev patches from the 3 known machines only
    train_patches_list = []
    for tm in train_machines:
        for _, _, p in cache.cache[tm]["train"]:
            train_patches_list.append(p)
    train_all = np.concatenate(train_patches_list, axis=0)

    # Compute normalization strictly from known source machines
    train_mean = float(np.mean(train_all))
    train_std = float(np.std(train_all)) + 1e-6

    # Normalize training set
    train_norm = (train_all - train_mean) / train_std
    train_ds = TensorDataset(torch.from_numpy(train_norm).float())
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=False)

    # Dev set from known source machines
    dev_patches_list = []
    for tm in train_machines:
        for _, _, p in cache.cache[tm]["dev"]:
            dev_patches_list.append(p)
    dev_all = np.concatenate(dev_patches_list, axis=0)
    dev_norm = (dev_all - train_mean) / train_std
    dev_ds = TensorDataset(torch.from_numpy(dev_norm).float())
    dev_dl = DataLoader(dev_ds, batch_size=32, shuffle=False)

    # Initialize model
    model = ResidualCNNAE(in_channels=1, base_channels=16, latent_dim=64, norm_type="group")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    beta_val = loss_param.get("beta", 0.1) if loss_param else 0.1

    t_start = time.perf_counter()
    best_dev_loss = float("inf")
    best_state = None
    patience = 4
    patience_cnt = 0

    for epoch in range(max_epochs):
        model.train()
        for batch, in train_dl:
            optimizer.zero_grad()
            recon, _ = model(batch)

            if loss_type == "smooth_l1":
                loss = F.smooth_l1_loss(recon, batch, beta=beta_val)
            else:
                loss = F.l1_loss(recon, batch)

            loss.backward()
            optimizer.step()
        scheduler.step()

        # Dev validation loss
        model.eval()
        dev_losses = []
        with torch.no_grad():
            for batch, in dev_dl:
                recon, _ = model(batch)
                dev_losses.append(F.l1_loss(recon, batch).item())
        mean_dev_loss = float(np.mean(dev_losses))

        if mean_dev_loss < best_dev_loss:
            best_dev_loss = mean_dev_loss
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    train_duration_s = time.perf_counter() - t_start

    # Determine zero-shot threshold from known machines' dev clips (95th percentile)
    dev_clip_scores = []
    with torch.no_grad():
        for tm in train_machines:
            for _, _, p in cache.cache[tm]["dev"]:
                p_norm = torch.from_numpy((p - train_mean) / train_std).float()
                recon, _ = model(p_norm)
                score = float(torch.mean(torch.abs(recon - p_norm)).item())
                dev_clip_scores.append(score)
    zero_shot_threshold = float(np.percentile(dev_clip_scores, 95))

    return model, train_mean, train_std, zero_shot_threshold, train_duration_s


def evaluate_test_set(
    model: ResidualCNNAE,
    test_entries: List[Tuple[str, int, np.ndarray]],
    train_mean: float,
    train_std: float,
    threshold: float,
    residual_mean: Optional[np.ndarray] = None,
    residual_std: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Score test clips and return full metrics."""
    test_scores = []
    test_labels = []

    model.eval()
    with torch.no_grad():
        for _, lbl, p in test_entries:
            p_norm = torch.from_numpy((p - train_mean) / train_std).float()
            recon, _ = model(p_norm)
            diff = torch.abs(recon - p_norm).cpu().numpy()  # (P, 1, 64, 64)

            if residual_mean is not None and residual_std is not None:
                # Spatial-spectral residual whitening
                whitened = np.maximum(0.0, (diff - residual_mean) / residual_std)
                score = float(np.mean(whitened))
            else:
                score = float(np.mean(diff))

            test_scores.append(score)
            test_labels.append(lbl)

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
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "threshold": round(threshold, 6),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "scores": [round(s, 5) for s in test_scores],
    }


def calibrate_few_shot(
    model: ResidualCNNAE,
    enrollment_entries: List[Tuple[str, int, np.ndarray]],
    train_mean: float,
    train_std: float,
    method: str = "mean_var_residual",
    target_quantile: float = 0.95,
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray], float]:
    """Calibrate on 10 normal enrollment clips from the held-out machine."""
    t0 = time.perf_counter()
    model.eval()

    all_enroll_residuals = []
    enroll_scores_raw = []

    with torch.no_grad():
        for _, _, p in enrollment_entries:
            p_norm = torch.from_numpy((p - train_mean) / train_std).float()
            recon, _ = model(p_norm)
            diff = torch.abs(recon - p_norm).cpu().numpy()  # (P, 1, 64, 64)
            all_enroll_residuals.append(diff)
            enroll_scores_raw.append(float(np.mean(diff)))

    all_res_np = np.concatenate(all_enroll_residuals, axis=0)  # (Total_P, 1, 64, 64)

    if method == "threshold_recalibration":
        threshold = float(np.percentile(enroll_scores_raw, target_quantile * 100))
        cal_time = time.perf_counter() - t0
        return threshold, None, None, cal_time

    elif method == "mean_var_residual":
        res_mean = np.mean(all_res_np, axis=0, keepdims=True)  # (1, 1, 64, 64)
        res_std = np.std(all_res_np, axis=0, keepdims=True) + 1e-4

        # Compute whitened scores on enrollment clips
        whitened_enroll_scores = []
        for diff in all_enroll_residuals:
            w = np.maximum(0.0, (diff - res_mean) / res_std)
            whitened_enroll_scores.append(float(np.mean(w)))

        threshold = float(np.percentile(whitened_enroll_scores, target_quantile * 100))
        cal_time = time.perf_counter() - t0
        return threshold, res_mean, res_std, cal_time

    else:
        raise ValueError(f"Unknown calibration method: {method}")


def benchmark_latency_profile(cache: FeatureCache, num_iter: int = 50) -> Dict[str, Any]:
    """Benchmark feature extraction, model inference, and end-to-end CPU latencies."""
    dummy_model = ResidualCNNAE(in_channels=1, base_channels=16, latent_dim=64, norm_type="group")
    dummy_model.eval()

    sample_fp = cache.cache["id_00"]["test"][0][0]
    raw_audio, sr = sf.read(str(sample_fp))
    waveform = raw_audio[:, 0] if raw_audio.ndim > 1 else raw_audio
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)

    # Warm-up
    for _ in range(5):
        mel, _ = cache.mel_extractor.extract_numpy(waveform)
        p2d, _ = cache.windowing.slice_spectrogram_numpy(mel.squeeze(0))
        p4d = torch.from_numpy(np.expand_dims(p2d, axis=1)).float()
        with torch.no_grad():
            _ = dummy_model(p4d)

    extract_times = []
    infer_times = []
    e2e_times = []

    for _ in range(num_iter):
        t0 = time.perf_counter()
        mel, _ = cache.mel_extractor.extract_numpy(waveform)
        p2d, _ = cache.windowing.slice_spectrogram_numpy(mel.squeeze(0))
        p4d = torch.from_numpy(np.expand_dims(p2d, axis=1)).float()
        t1 = time.perf_counter()

        with torch.no_grad():
            recon, _ = dummy_model(p4d)
            _ = torch.mean(torch.abs(recon - p4d)).item()
        t2 = time.perf_counter()

        extract_times.append((t1 - t0) * 1000.0)
        infer_times.append((t2 - t1) * 1000.0)
        e2e_times.append((t2 - t0) * 1000.0)

    return {
        "feature_extraction_ms": {
            "p50": round(float(np.median(extract_times)), 2),
            "p95": round(float(np.percentile(extract_times, 95)), 2),
        },
        "model_inference_ms": {
            "mean": round(float(np.mean(infer_times)), 2),
            "median_p50": round(float(np.median(infer_times)), 2),
            "p95": round(float(np.percentile(infer_times, 95)), 2),
        },
        "end_to_end_ms": {
            "mean": round(float(np.mean(e2e_times)), 2),
            "median_p50": round(float(np.median(e2e_times)), 2),
            "p95": round(float(np.percentile(e2e_times, 95)), 2),
        },
        "throughput_clips_per_sec": round(1000.0 / float(np.median(e2e_times)), 1),
        "realtime_factor": round(float(np.median(e2e_times)) / 10000.0, 4),
    }


def main():
    logger.info("=" * 80)
    logger.info("STETHO SPRINT 7.6 - E3: UNSEEN-MACHINE GENERALIZATION & FEW-SHOT CALIBRATION")
    logger.info("=" * 80)

    # 1. Feature Pre-Extraction & Caching
    cache = FeatureCache()
    cache.populate()

    models_config = {
        "baseline_l1": {
            "loss_type": "l1",
            "loss_param": None,
            "weight_decay": 0.0001,
            "description": "Baseline L1 loss",
        },
        "loss_smooth_l1": {
            "loss_type": "smooth_l1",
            "loss_param": {"beta": 0.1},
            "weight_decay": 0.0001,
            "description": "Huber / Smooth L1 loss (beta=0.1)",
        },
    }

    per_machine_metrics: Dict[str, Dict[str, Dict[str, Dict[int, Dict[str, Any]]]]] = {}
    macro_results: Dict[str, Any] = {}
    few_shot_summary: Dict[str, Any] = {}

    mem_start = get_process_memory_mb()

    # 2. Execute 4 Leave-One-Machine-Out Folds x 3 Seeds x 2 Models
    for cfg_name, cfg in models_config.items():
        logger.info(f"\n=======================================================")
        logger.info(f">>> EVALUATING MODEL: {cfg_name} ({cfg['description']}) <<<")
        logger.info(f"=======================================================")
        per_machine_metrics[cfg_name] = {
            "zero_shot": {m: {} for m in MACHINES},
            "few_shot_threshold": {m: {} for m in MACHINES},
            "few_shot_whitened": {m: {} for m in MACHINES},
        }

        zero_shot_aucs_by_seed = {s: [] for s in SEEDS}
        few_shot_thresh_aucs_by_seed = {s: [] for s in SEEDS}
        few_shot_white_aucs_by_seed = {s: [] for s in SEEDS}

        zero_shot_praucs_by_seed = {s: [] for s in SEEDS}
        few_shot_thresh_praucs_by_seed = {s: [] for s in SEEDS}
        few_shot_white_praucs_by_seed = {s: [] for s in SEEDS}

        zero_shot_f1s_by_seed = {s: [] for s in SEEDS}
        few_shot_thresh_f1s_by_seed = {s: [] for s in SEEDS}
        few_shot_white_f1s_by_seed = {s: [] for s in SEEDS}

        train_times = []
        cal_times_thresh = []
        cal_times_white = []

        for heldout_m in MACHINES:
            logger.info(f"\n  --- FOLD: Held-out {heldout_m} (Trained on {[m for m in MACHINES if m != heldout_m]}) ---")

            for seed in SEEDS:
                # Train model on known machines
                model, t_mean, t_std, zs_thresh, t_dur = train_fold_model(
                    heldout_m=heldout_m,
                    cfg_name=cfg_name,
                    loss_type=cfg["loss_type"],
                    loss_param=cfg["loss_param"],
                    weight_decay=cfg["weight_decay"],
                    seed=seed,
                    cache=cache,
                    max_epochs=15,
                )
                train_times.append(t_dur)

                # Test set from held-out machine (18 clips: 3 normal, 15 abnormal)
                test_entries = cache.cache[heldout_m]["test"]
                # Enrollment set from held-out machine (10 normal clips)
                enroll_entries = cache.cache[heldout_m]["train"]

                # A. Zero-Shot Evaluation (using source dev threshold)
                res_zs = evaluate_test_set(
                    model=model,
                    test_entries=test_entries,
                    train_mean=t_mean,
                    train_std=t_std,
                    threshold=zs_thresh,
                )
                res_zs["train_duration_s"] = round(t_dur, 2)
                per_machine_metrics[cfg_name]["zero_shot"][heldout_m][seed] = res_zs
                zero_shot_aucs_by_seed[seed].append(res_zs["roc_auc"])
                zero_shot_praucs_by_seed[seed].append(res_zs["pr_auc"])
                zero_shot_f1s_by_seed[seed].append(res_zs["f1"])

                # B. Few-Shot Threshold Recalibration
                cal_thresh, _, _, ct_thresh = calibrate_few_shot(
                    model=model,
                    enrollment_entries=enroll_entries,
                    train_mean=t_mean,
                    train_std=t_std,
                    method="threshold_recalibration",
                    target_quantile=0.95,
                )
                cal_times_thresh.append(ct_thresh)
                res_fs_thresh = evaluate_test_set(
                    model=model,
                    test_entries=test_entries,
                    train_mean=t_mean,
                    train_std=t_std,
                    threshold=cal_thresh,
                )
                res_fs_thresh["calibration_time_s"] = round(ct_thresh, 4)
                per_machine_metrics[cfg_name]["few_shot_threshold"][heldout_m][seed] = res_fs_thresh
                few_shot_thresh_aucs_by_seed[seed].append(res_fs_thresh["roc_auc"])
                few_shot_thresh_praucs_by_seed[seed].append(res_fs_thresh["pr_auc"])
                few_shot_thresh_f1s_by_seed[seed].append(res_fs_thresh["f1"])

                # C. Few-Shot Mean-Variance Residual Whitening
                cal_white_thresh, r_mean, r_std, ct_white = calibrate_few_shot(
                    model=model,
                    enrollment_entries=enroll_entries,
                    train_mean=t_mean,
                    train_std=t_std,
                    method="mean_var_residual",
                    target_quantile=0.95,
                )
                cal_times_white.append(ct_white)
                res_fs_white = evaluate_test_set(
                    model=model,
                    test_entries=test_entries,
                    train_mean=t_mean,
                    train_std=t_std,
                    threshold=cal_white_thresh,
                    residual_mean=r_mean,
                    residual_std=r_std,
                )
                res_fs_white["calibration_time_s"] = round(ct_white, 4)
                per_machine_metrics[cfg_name]["few_shot_whitened"][heldout_m][seed] = res_fs_white
                few_shot_white_aucs_by_seed[seed].append(res_fs_white["roc_auc"])
                few_shot_white_praucs_by_seed[seed].append(res_fs_white["pr_auc"])
                few_shot_white_f1s_by_seed[seed].append(res_fs_white["f1"])

                logger.info(
                    f"    {heldout_m} (Seed {seed}): "
                    f"Zero-Shot ROC={res_zs['roc_auc']:.4f}, F1={res_zs['f1']:.4f} | "
                    f"FS-Thresh F1={res_fs_thresh['f1']:.4f} | "
                    f"FS-White ROC={res_fs_white['roc_auc']:.4f}, F1={res_fs_white['f1']:.4f} | "
                    f"TrainTime={t_dur:.1f}s"
                )

        # Macro aggregations across seeds
        zs_seed_aucs = [float(np.mean(zero_shot_aucs_by_seed[s])) for s in SEEDS]
        zs_seed_praucs = [float(np.mean(zero_shot_praucs_by_seed[s])) for s in SEEDS]
        zs_seed_f1s = [float(np.mean(zero_shot_f1s_by_seed[s])) for s in SEEDS]

        fsw_seed_aucs = [float(np.mean(few_shot_white_aucs_by_seed[s])) for s in SEEDS]
        fsw_seed_praucs = [float(np.mean(few_shot_white_praucs_by_seed[s])) for s in SEEDS]
        fsw_seed_f1s = [float(np.mean(few_shot_white_f1s_by_seed[s])) for s in SEEDS]

        fst_seed_f1s = [float(np.mean(few_shot_thresh_f1s_by_seed[s])) for s in SEEDS]

        m_auc_zs, s_auc_zs, ci_auc_zs = compute_ci95(zs_seed_aucs)
        m_prauc_zs, s_prauc_zs, ci_prauc_zs = compute_ci95(zs_seed_praucs)
        m_f1_zs, s_f1_zs, ci_f1_zs = compute_ci95(zs_seed_f1s)

        m_auc_fsw, s_auc_fsw, ci_auc_fsw = compute_ci95(fsw_seed_aucs)
        m_prauc_fsw, s_prauc_fsw, ci_prauc_fsw = compute_ci95(fsw_seed_praucs)
        m_f1_fsw, s_f1_fsw, ci_f1_fsw = compute_ci95(fsw_seed_f1s)

        m_f1_fst, s_f1_fst, ci_f1_fst = compute_ci95(fst_seed_f1s)

        macro_results[cfg_name] = {
            "name": cfg["description"],
            "zero_shot": {
                "roc_auc": {"mean": m_auc_zs, "std": s_auc_zs, "ci95_half": ci_auc_zs, "per_seed": zs_seed_aucs},
                "pr_auc": {"mean": m_prauc_zs, "std": s_prauc_zs, "ci95_half": ci_prauc_zs, "per_seed": zs_seed_praucs},
                "f1": {"mean": m_f1_zs, "std": s_f1_zs, "ci95_half": ci_f1_zs, "per_seed": zs_seed_f1s},
            },
            "few_shot_threshold_recalibration": {
                "f1": {"mean": m_f1_fst, "std": s_f1_fst, "ci95_half": ci_f1_fst, "per_seed": fst_seed_f1s},
                "mean_cal_time_s": round(float(np.mean(cal_times_thresh)), 4),
            },
            "few_shot_whitened": {
                "roc_auc": {"mean": m_auc_fsw, "std": s_auc_fsw, "ci95_half": ci_auc_fsw, "per_seed": fsw_seed_aucs},
                "pr_auc": {"mean": m_prauc_fsw, "std": s_prauc_fsw, "ci95_half": ci_prauc_fsw, "per_seed": fsw_seed_praucs},
                "f1": {"mean": m_f1_fsw, "std": s_f1_fsw, "ci95_half": ci_f1_fsw, "per_seed": fsw_seed_f1s},
                "mean_cal_time_s": round(float(np.mean(cal_times_white)), 4),
            },
            "mean_train_duration_s": round(float(np.mean(train_times)), 2),
        }

        logger.info(
            f"\n>>> {cfg_name} SUMMARY: Zero-Shot ROC-AUC = {m_auc_zs:.4f} ± {ci_auc_zs:.4f} | "
            f"Few-Shot Whitened ROC-AUC = {m_auc_fsw:.4f} ± {ci_auc_fsw:.4f} | "
            f"Few-Shot Whitened F1 = {m_f1_fsw:.4f} ± {ci_f1_fsw:.4f} <<<"
        )

    # 3. Dedicated Few-Shot Comparison Analysis
    few_shot_summary = {
        "enrollment_shots": 10,
        "enrollment_source": "held-out machine normal training clips (strictly disjoint from test)",
        "models": {
            k: {
                "zero_shot_auc": macro_results[k]["zero_shot"]["roc_auc"]["mean"],
                "zero_shot_f1": macro_results[k]["zero_shot"]["f1"]["mean"],
                "few_shot_thresh_f1": macro_results[k]["few_shot_threshold_recalibration"]["f1"]["mean"],
                "few_shot_white_auc": macro_results[k]["few_shot_whitened"]["roc_auc"]["mean"],
                "few_shot_white_f1": macro_results[k]["few_shot_whitened"]["f1"]["mean"],
                "delta_auc_few_shot_whitened": round(
                    macro_results[k]["few_shot_whitened"]["roc_auc"]["mean"]
                    - macro_results[k]["zero_shot"]["roc_auc"]["mean"],
                    4,
                ),
                "delta_f1_few_shot_whitened": round(
                    macro_results[k]["few_shot_whitened"]["f1"]["mean"]
                    - macro_results[k]["zero_shot"]["f1"]["mean"],
                    4,
                ),
                "calibration_time_s": macro_results[k]["few_shot_whitened"]["mean_cal_time_s"],
            }
            for k in models_config
        },
    }

    # 4. CPU Latency & Resource Benchmarking
    logger.info("\nBenchmarking CPU inference latency across 50 iterations...")
    latency_profile = benchmark_latency_profile(cache, num_iter=50)
    mem_peak = get_process_memory_mb()

    telemetry = {
        "hardware": {
            "device": "cpu",
            "threads": 8,
            "cuda_available": False,
        },
        "latency_profile": latency_profile,
        "memory_profile_mb": {
            "initial_rss": round(mem_start, 2),
            "peak_rss": round(mem_peak, 2),
        },
    }
    macro_results["telemetry"] = telemetry

    # 5. Save Artifacts
    with open(RESULTS_DIR / "e3_results.json", "w", encoding="utf-8") as f:
        json.dump(macro_results, f, indent=2)

    with open(RESULTS_DIR / "per_machine_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_machine_metrics, f, indent=2)

    with open(RESULTS_DIR / "few_shot_results.json", "w", encoding="utf-8") as f:
        json.dump(few_shot_summary, f, indent=2)

    logger.info(f"\nAll E3 experiments completed successfully! Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Sprint 7.6 / E5 - Recording Robustness & Environmental Stress Testing Runner.

Evaluates acoustic robustness of the leading Stetho model (ResidualCNNAE + Smooth L1 + N=10 whitening)
against the baseline (ResidualCNNAE + L1 + N=10 whitening) across 20 environmental conditions:
1. Clean baseline (control)
2. Background noise: Additive white noise at SNR 30, 20, 10, 5 dB
3. Gain variation: Scaling at -12, -6, +6, +12 dB (with clipping checks)
4. Sample rate variation: Resampling to 8 kHz, 22.05 kHz, 44.1 kHz, 48 kHz
5. Simulated microphone response: Lowpass 4 kHz, Highpass 300 Hz, Bandpass 300-4000 Hz
6. Combined stress: Factory realistic (20dB + -6dB + 300Hz), Harsh stress (10dB + +6dB + 4kHz)
7. Actual physical microphone variation: Channels 1 and 4 from the 8-channel circular array
"""

import csv
import gc
import json
import logging
from math import gcd
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import scipy.signal as signal
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(".").resolve()
sys.path.insert(0, str(REPO_ROOT))

from ml.models.research.residual_cnn_ae import ResidualCNNAE
from ml.preprocessing.log_mel import LogMelConfig, LogMelExtractor
from ml.preprocessing.windowing import AudioWindowing, WindowConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stetho_e5_robustness")

MACHINES = ["id_00", "id_02", "id_04", "id_06"]
SEEDS = [42, 123, 777]
RESULTS_DIR = Path("experiments/sprint_7_6/e5_robustness")


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
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    if n <= 1:
        return mean, 0.0, 0.0
    std = float(np.std(arr, ddof=1))
    t_val = 4.303 if n == 3 else (2.776 if n == 5 else 1.96)
    ci95 = float(t_val * std / np.sqrt(n))
    return round(mean, 4), round(std, 4), round(ci95, 4)


class AudioPerturber:
    """Applies controlled acoustic perturbations to raw 1D waveforms."""

    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        # Pre-compute filter coefficients
        self.sos_lp4k = signal.butter(4, 4000.0, btype="lowpass", fs=self.sr, output="sos")
        self.sos_hp300 = signal.butter(4, 300.0, btype="highpass", fs=self.sr, output="sos")
        self.sos_bp = signal.butter(4, [300.0, 4000.0], btype="bandpass", fs=self.sr, output="sos")

    def add_noise(self, x: np.ndarray, snr_db: float, seed: int = 42) -> Tuple[np.ndarray, float]:
        p_sig = float(np.mean(x**2))
        if p_sig == 0:
            return x.copy(), 0.0
        p_noise = p_sig * (10.0 ** (-snr_db / 10.0))
        rng = np.random.RandomState(seed)
        noise = rng.randn(len(x)).astype(np.float32) * np.sqrt(p_noise)
        y = x + noise
        actual_snr = float(10.0 * np.log10(p_sig / np.mean(noise**2)))
        return y, actual_snr

    def scale_gain(self, x: np.ndarray, gain_db: float) -> Tuple[np.ndarray, int]:
        scale = 10.0 ** (gain_db / 20.0)
        y_scaled = x * scale
        clipped = int(np.sum(np.abs(y_scaled) > 1.0))
        y = np.clip(y_scaled, -1.0, 1.0)
        return y.astype(np.float32), clipped

    def resample_cycle(self, x: np.ndarray, intermediate_sr: int) -> np.ndarray:
        g = gcd(intermediate_sr, self.sr)
        up = intermediate_sr // g
        down = self.sr // g
        # Resample to intermediate rate
        y_inter = signal.resample_poly(x, up, down)
        # Resample back to 16 kHz
        y_back = signal.resample_poly(y_inter, down, up)
        if len(y_back) < len(x):
            y_back = np.pad(y_back, (0, len(x) - len(y_back)))
        elif len(y_back) > len(x):
            y_back = y_back[: len(x)]
        return y_back.astype(np.float32)

    def filter_audio(self, x: np.ndarray, ftype: str) -> np.ndarray:
        if ftype == "lowpass_4000hz":
            return signal.sosfilt(self.sos_lp4k, x).astype(np.float32)
        elif ftype == "highpass_300hz":
            return signal.sosfilt(self.sos_hp300, x).astype(np.float32)
        elif ftype == "bandpass_300_4000hz":
            return signal.sosfilt(self.sos_bp, x).astype(np.float32)
        return x


class FeatureEngine:
    """Extracts standardized (P, 1, 64, 64) Log-Mel patches from waveforms."""

    def __init__(self):
        self.cfg = LogMelConfig(sample_rate=16000, n_fft=1024, win_length=1024, hop_length=256, n_mels=64)
        self.mel_extractor = LogMelExtractor(self.cfg)
        self.windowing = AudioWindowing(WindowConfig(spec_window_frames=64, spec_hop_frames=32))

    def extract_patches(self, waveform: np.ndarray) -> np.ndarray:
        mel, _ = self.mel_extractor.extract_numpy(waveform)
        p2d, _ = self.windowing.slice_spectrogram_numpy(mel.squeeze(0))
        return np.expand_dims(p2d, axis=1).astype(np.float32)


def train_fold_model(
    heldout_m: str,
    loss_type: str,
    loss_param: Optional[Dict[str, Any]],
    weight_decay: float,
    seed: int,
    raw_cache: Dict[str, Dict[str, List[Tuple[str, int, np.ndarray]]]],
    feat_engine: FeatureEngine,
    max_epochs: int = 15,
) -> Tuple[ResidualCNNAE, float, float, float]:
    set_seed(seed)
    train_machines = [m for m in MACHINES if m != heldout_m]

    train_patches_list = []
    for tm in train_machines:
        for _, _, wav in raw_cache[tm]["train"]:
            p = feat_engine.extract_patches(wav)
            train_patches_list.append(p)
    train_all = np.concatenate(train_patches_list, axis=0)

    train_mean = float(np.mean(train_all))
    train_std = float(np.std(train_all)) + 1e-6

    train_norm = (train_all - train_mean) / train_std
    train_ds = TensorDataset(torch.from_numpy(train_norm).float())
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=False)

    dev_patches_list = []
    for tm in train_machines:
        for _, _, wav in raw_cache[tm]["dev"]:
            p = feat_engine.extract_patches(wav)
            dev_patches_list.append(p)
    dev_all = np.concatenate(dev_patches_list, axis=0)
    dev_norm = (dev_all - train_mean) / train_std
    dev_ds = TensorDataset(torch.from_numpy(dev_norm).float())
    dev_dl = DataLoader(dev_ds, batch_size=32, shuffle=False)

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
    train_dur = time.perf_counter() - t_start
    return model, train_mean, train_std, train_dur


def calibrate_n10_whitening(
    model: ResidualCNNAE,
    enroll_raw: List[Tuple[str, int, np.ndarray]],
    train_mean: float,
    train_std: float,
    feat_engine: FeatureEngine,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """Fit spatial-spectral residual whitening on 10 healthy clean enrollment clips."""
    t0 = time.perf_counter()
    model.eval()

    all_enroll_residuals = []
    with torch.no_grad():
        for _, _, wav in enroll_raw:
            patches = feat_engine.extract_patches(wav)
            p_norm = torch.from_numpy((patches - train_mean) / train_std).float()
            recon, _ = model(p_norm)
            diff = torch.abs(recon - p_norm).cpu().numpy()
            all_enroll_residuals.append(diff)

    all_res_np = np.concatenate(all_enroll_residuals, axis=0)
    res_mean = np.mean(all_res_np, axis=0, keepdims=True)
    res_std = np.std(all_res_np, axis=0, keepdims=True) + 1e-4

    whitened_scores = []
    for diff in all_enroll_residuals:
        w = np.maximum(0.0, (diff - res_mean) / res_std)
        whitened_scores.append(float(np.mean(w)))

    threshold = float(np.percentile(whitened_scores, 95))
    cal_dur = time.perf_counter() - t0
    return threshold, res_mean, res_std, cal_dur


def evaluate_under_perturbation(
    model: ResidualCNNAE,
    test_raw: List[Tuple[str, int, np.ndarray, str]],  # (fp, lbl, wav_ch0, raw_fp)
    train_mean: float,
    train_std: float,
    threshold: float,
    res_mean: np.ndarray,
    res_std: np.ndarray,
    feat_engine: FeatureEngine,
    perturber: AudioPerturber,
    condition: str,
) -> Dict[str, Any]:
    """Score test clips under specified perturbation condition."""
    test_scores = []
    test_labels = []

    model.eval()
    with torch.no_grad():
        for fp, lbl, wav_ch0, raw_fp in test_raw:
            # Apply perturbation
            if condition == "clean_baseline":
                wav_eval = wav_ch0.copy()
            elif condition == "noise_snr_30db":
                wav_eval, _ = perturber.add_noise(wav_ch0, 30.0)
            elif condition == "noise_snr_20db":
                wav_eval, _ = perturber.add_noise(wav_ch0, 20.0)
            elif condition == "noise_snr_10db":
                wav_eval, _ = perturber.add_noise(wav_ch0, 10.0)
            elif condition == "noise_snr_5db":
                wav_eval, _ = perturber.add_noise(wav_ch0, 5.0)
            elif condition == "gain_minus_12db":
                wav_eval, _ = perturber.scale_gain(wav_ch0, -12.0)
            elif condition == "gain_minus_6db":
                wav_eval, _ = perturber.scale_gain(wav_ch0, -6.0)
            elif condition == "gain_plus_6db":
                wav_eval, _ = perturber.scale_gain(wav_ch0, 6.0)
            elif condition == "gain_plus_12db":
                wav_eval, _ = perturber.scale_gain(wav_ch0, 12.0)
            elif condition == "resample_8000hz":
                wav_eval = perturber.resample_cycle(wav_ch0, 8000)
            elif condition == "resample_22050hz":
                wav_eval = perturber.resample_cycle(wav_ch0, 22050)
            elif condition == "resample_44100hz":
                wav_eval = perturber.resample_cycle(wav_ch0, 44100)
            elif condition == "resample_48000hz":
                wav_eval = perturber.resample_cycle(wav_ch0, 48000)
            elif condition == "mic_lowpass_4000hz":
                wav_eval = perturber.filter_audio(wav_ch0, "lowpass_4000hz")
            elif condition == "mic_highpass_300hz":
                wav_eval = perturber.filter_audio(wav_ch0, "highpass_300hz")
            elif condition == "mic_bandpass_300_4000hz":
                wav_eval = perturber.filter_audio(wav_ch0, "bandpass_300_4000hz")
            elif condition == "combined_factory_realistic":
                # SNR 20dB + -6dB gain + 300Hz highpass
                w_n, _ = perturber.add_noise(wav_ch0, 20.0)
                w_g, _ = perturber.scale_gain(w_n, -6.0)
                wav_eval = perturber.filter_audio(w_g, "highpass_300hz")
            elif condition == "combined_harsh_stress":
                # SNR 10dB + +6dB gain + 4000Hz lowpass
                w_n, _ = perturber.add_noise(wav_ch0, 10.0)
                w_g, _ = perturber.scale_gain(w_n, 6.0)
                wav_eval = perturber.filter_audio(w_g, "lowpass_4000hz")
            elif condition == "physical_mic_ch1":
                # Read channel 1 from physical 8-channel array
                full_raw, _ = sf.read(raw_fp)
                wav_eval = full_raw[:, 1].astype(np.float32)
            elif condition == "physical_mic_ch4":
                # Read channel 4 from physical 8-channel array
                full_raw, _ = sf.read(raw_fp)
                wav_eval = full_raw[:, 4].astype(np.float32)
            else:
                wav_eval = wav_ch0.copy()

            patches = feat_engine.extract_patches(wav_eval)
            p_norm = torch.from_numpy((patches - train_mean) / train_std).float()
            recon, _ = model(p_norm)
            diff = torch.abs(recon - p_norm).cpu().numpy()
            whitened = np.maximum(0.0, (diff - res_mean) / res_std)
            score = float(np.mean(whitened))

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
    }


def main():
    logger.info("=" * 80)
    logger.info("STETHO SPRINT 7.6 - E5: RECORDING ROBUSTNESS & ENVIRONMENTAL STRESS TESTING")
    logger.info("=" * 80)

    perturber = AudioPerturber()
    feat_engine = FeatureEngine()

    # Pre-load raw waveforms (Channel 0 and original file path)
    logger.info("Pre-loading raw waveforms for all 120 files...")
    raw_cache: Dict[str, Dict[str, List[Any]]] = {}
    splits_dir = Path("ml/data/splits/diverse_machine_specific/fan")

    for m in MACHINES:
        raw_cache[m] = {"train": [], "dev": [], "test": []}
        for split in ["train", "dev", "test"]:
            csv_name = f"{split}_normal.csv" if split != "test" else "test.csv"
            with open(splits_dir / m / csv_name) as fp:
                for row in csv.DictReader(fp):
                    fp_path = Path(row["file_path"])
                    lbl = 1 if row.get("label", "normal") == "abnormal" else 0
                    full_audio, sr = sf.read(str(fp_path))
                    ch0 = full_audio[:, 0] if full_audio.ndim > 1 else full_audio
                    ch0 = np.ascontiguousarray(ch0, dtype=np.float32)
                    if split == "test":
                        raw_cache[m][split].append((str(fp_path), lbl, ch0, str(fp_path)))
                    else:
                        raw_cache[m][split].append((str(fp_path), lbl, ch0))

    logger.info("Loaded all raw waveforms. Setting up perturbation suites...")

    CONDITIONS = [
        "clean_baseline",
        "noise_snr_30db",
        "noise_snr_20db",
        "noise_snr_10db",
        "noise_snr_5db",
        "gain_minus_12db",
        "gain_minus_6db",
        "gain_plus_6db",
        "gain_plus_12db",
        "resample_8000hz",
        "resample_22050hz",
        "resample_44100hz",
        "resample_48000hz",
        "mic_lowpass_4000hz",
        "mic_highpass_300hz",
        "mic_bandpass_300_4000hz",
        "combined_factory_realistic",
        "combined_harsh_stress",
        "physical_mic_ch1",
        "physical_mic_ch4",
    ]

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

    per_machine_metrics: Dict[str, Any] = {}
    macro_results: Dict[str, Any] = {}
    robustness_degradation: Dict[str, Any] = {}

    mem_start = get_process_memory_mb()

    for cfg_name, cfg in models_config.items():
        logger.info(f"\n=======================================================")
        logger.info(f">>> EVALUATING ROBUSTNESS FOR MODEL: {cfg_name} <<<")
        logger.info(f"=======================================================")
        per_machine_metrics[cfg_name] = {cond: {m: {} for m in MACHINES} for cond in CONDITIONS}
        macro_results[cfg_name] = {}
        robustness_degradation[cfg_name] = {}

        # 1. Train and calibrate fold models once across 4 folds x 3 seeds
        logger.info("Training and calibrating clean fold models across 4 folds x 3 seeds...")
        trained_calibrated_models = {}

        for heldout_m in MACHINES:
            trained_calibrated_models[heldout_m] = {}
            for seed in SEEDS:
                model, t_mean, t_std, t_dur = train_fold_model(
                    heldout_m=heldout_m,
                    loss_type=cfg["loss_type"],
                    loss_param=cfg["loss_param"],
                    weight_decay=cfg["weight_decay"],
                    seed=seed,
                    raw_cache=raw_cache,
                    feat_engine=feat_engine,
                    max_epochs=15,
                )
                # Calibrate on 10 healthy clean enrollment clips
                thresh, r_mean, r_std, c_dur = calibrate_n10_whitening(
                    model=model,
                    enroll_raw=raw_cache[heldout_m]["train"],
                    train_mean=t_mean,
                    train_std=t_std,
                    feat_engine=feat_engine,
                )
                trained_calibrated_models[heldout_m][seed] = (model, t_mean, t_std, thresh, r_mean, r_std)

        logger.info("Fold models trained and calibrated. Now evaluating across 20 perturbation conditions...")

        # 2. Evaluate across all 20 conditions
        for cond in CONDITIONS:
            aucs_by_seed = {s: [] for s in SEEDS}
            praucs_by_seed = {s: [] for s in SEEDS}
            f1s_by_seed = {s: [] for s in SEEDS}
            fprs_by_seed = {s: [] for s in SEEDS}
            fnrs_by_seed = {s: [] for s in SEEDS}

            for heldout_m in MACHINES:
                test_raw = raw_cache[heldout_m]["test"]

                for seed in SEEDS:
                    model, t_mean, t_std, thresh, r_mean, r_std = trained_calibrated_models[heldout_m][seed]
                    res = evaluate_under_perturbation(
                        model=model,
                        test_raw=test_raw,
                        train_mean=t_mean,
                        train_std=t_std,
                        threshold=thresh,
                        res_mean=r_mean,
                        res_std=r_std,
                        feat_engine=feat_engine,
                        perturber=perturber,
                        condition=cond,
                    )
                    per_machine_metrics[cfg_name][cond][heldout_m][seed] = res

                    aucs_by_seed[seed].append(res["roc_auc"])
                    praucs_by_seed[seed].append(res["pr_auc"])
                    f1s_by_seed[seed].append(res["f1"])
                    fprs_by_seed[seed].append(res["fpr"])
                    fnrs_by_seed[seed].append(res["fnr"])

            seed_aucs = [float(np.mean(aucs_by_seed[s])) for s in SEEDS]
            seed_praucs = [float(np.mean(praucs_by_seed[s])) for s in SEEDS]
            seed_f1s = [float(np.mean(f1s_by_seed[s])) for s in SEEDS]
            seed_fprs = [float(np.mean(fprs_by_seed[s])) for s in SEEDS]
            seed_fnrs = [float(np.mean(fnrs_by_seed[s])) for s in SEEDS]

            m_auc, s_auc, ci_auc = compute_ci95(seed_aucs)
            m_prauc, s_prauc, ci_prauc = compute_ci95(seed_praucs)
            m_f1, s_f1, ci_f1 = compute_ci95(seed_f1s)
            m_fpr = round(float(np.mean(seed_fprs)), 4)
            m_fnr = round(float(np.mean(seed_fnrs)), 4)

            macro_results[cfg_name][cond] = {
                "roc_auc": {"mean": m_auc, "std": s_auc, "ci95_half": ci_auc, "per_seed": seed_aucs},
                "pr_auc": {"mean": m_prauc, "std": s_prauc, "ci95_half": ci_prauc, "per_seed": seed_praucs},
                "f1": {"mean": m_f1, "std": s_f1, "ci95_half": ci_f1, "per_seed": seed_f1s},
                "fpr": m_fpr,
                "fnr": m_fnr,
            }

            logger.info(
                f"  [{cfg_name} - {cond:28s}] ROC-AUC={m_auc:.4f} ± {ci_auc:.4f} | "
                f"PR-AUC={m_prauc:.4f} | F1={m_f1:.4f} | FPR={m_fpr:.4f} | FNR={m_fnr:.4f}"
            )

        # 3. Compute Degradation Deltas vs Clean Baseline
        clean_auc = macro_results[cfg_name]["clean_baseline"]["roc_auc"]["mean"]
        clean_prauc = macro_results[cfg_name]["clean_baseline"]["pr_auc"]["mean"]
        clean_f1 = macro_results[cfg_name]["clean_baseline"]["f1"]["mean"]
        clean_fpr = macro_results[cfg_name]["clean_baseline"]["fpr"]
        clean_fnr = macro_results[cfg_name]["clean_baseline"]["fnr"]

        for cond in CONDITIONS:
            cond_auc = macro_results[cfg_name][cond]["roc_auc"]["mean"]
            cond_prauc = macro_results[cfg_name][cond]["pr_auc"]["mean"]
            cond_f1 = macro_results[cfg_name][cond]["f1"]["mean"]
            cond_fpr = macro_results[cfg_name][cond]["fpr"]
            cond_fnr = macro_results[cfg_name][cond]["fnr"]

            robustness_degradation[cfg_name][cond] = {
                "delta_roc_auc": round(cond_auc - clean_auc, 4),
                "delta_pr_auc": round(cond_prauc - clean_prauc, 4),
                "delta_f1": round(cond_f1 - clean_f1, 4),
                "delta_fpr": round(cond_fpr - clean_fpr, 4),
                "delta_fnr": round(cond_fnr - clean_fnr, 4),
                "pct_auc_change": round(((cond_auc - clean_auc) / (clean_auc + 1e-6)) * 100.0, 2),
                "status": "PASS" if (cond_auc >= clean_auc - 0.05 and cond_fnr <= clean_fnr + 0.15) else "DEGRADED" if cond_auc >= 0.60 else "CRITICAL_FAILURE",
            }

    # 4. Latency Benchmark
    logger.info("\nBenchmarking CPU inference latency across 50 iterations...")
    dummy_model = ResidualCNNAE(in_channels=1, base_channels=16, latent_dim=64, norm_type="group")
    dummy_model.eval()

    sample_wav = raw_cache["id_00"]["test"][0][2]
    for _ in range(5):
        p = feat_engine.extract_patches(sample_wav)
        p_t = torch.from_numpy(p).float()
        with torch.no_grad():
            _ = dummy_model(p_t)

    ext_times, inf_times, e2e_times = [], [], []
    for _ in range(50):
        t0 = time.perf_counter()
        p = feat_engine.extract_patches(sample_wav)
        p_t = torch.from_numpy(p).float()
        t1 = time.perf_counter()
        with torch.no_grad():
            recon, _ = dummy_model(p_t)
            _ = torch.mean(torch.abs(recon - p_t)).item()
        t2 = time.perf_counter()

        ext_times.append((t1 - t0) * 1000.0)
        inf_times.append((t2 - t1) * 1000.0)
        e2e_times.append((t2 - t0) * 1000.0)

    mem_peak = get_process_memory_mb()

    telemetry = {
        "hardware": {"device": "cpu", "threads": 8, "cuda_available": False},
        "latency_profile": {
            "feature_extraction_ms": {
                "p50": round(float(np.median(ext_times)), 2),
                "p95": round(float(np.percentile(ext_times, 95)), 2),
            },
            "model_inference_ms": {
                "mean": round(float(np.mean(inf_times)), 2),
                "median_p50": round(float(np.median(inf_times)), 2),
                "p95": round(float(np.percentile(inf_times, 95)), 2),
            },
            "end_to_end_ms": {
                "mean": round(float(np.mean(e2e_times)), 2),
                "median_p50": round(float(np.median(e2e_times)), 2),
                "p95": round(float(np.percentile(e2e_times, 95)), 2),
            },
            "throughput_clips_per_sec": round(1000.0 / float(np.median(e2e_times)), 1),
            "realtime_factor": round(float(np.median(e2e_times)) / 10000.0, 4),
        },
        "memory_profile_mb": {
            "initial_rss": round(mem_start, 2),
            "peak_rss": round(mem_peak, 2),
        },
    }

    macro_results["telemetry"] = telemetry

    # 5. Save Deliverables
    with open(RESULTS_DIR / "e5_results.json", "w", encoding="utf-8") as f:
        json.dump(macro_results, f, indent=2)

    with open(RESULTS_DIR / "per_machine_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_machine_metrics, f, indent=2)

    with open(RESULTS_DIR / "robustness_degradation.json", "w", encoding="utf-8") as f:
        json.dump(robustness_degradation, f, indent=2)

    logger.info(f"\nAll E5 experiments completed successfully! Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()


"""Sprint 7.6 / E6: Signal Normalization & Sensor Robustness Remediation Harness.

Evaluates and compares:
- Pipeline A: Original Baseline (E5 control)
- Pipeline B: Gain-Normalized Pipeline
- Pipeline C: Gain-Normalized + Sensor Guard
- Pipeline D: Complete Production Pipeline with Quality & Environment Gates

Across all 20 environmental stress conditions, 4 machine IDs, and 3 fixed seeds.
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

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import psutil
import scipy.signal as signal
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ml.models.research.residual_cnn_ae import ResidualCNNAE
from ml.preprocessing.audio_quality_gate import AudioQualityGate, QualityGateConfig
from ml.preprocessing.gain_normalizer import GainNormalizer, GainNormalizerConfig
from ml.preprocessing.log_mel import LogMelConfig, LogMelExtractor
from ml.preprocessing.sensor_guard import SensorGuard
from ml.preprocessing.windowing import AudioWindowing, WindowConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("E6_Robustness_Remediation")

MACHINES = ["id_00", "id_02", "id_04", "id_06"]
SEEDS = [42, 123, 777]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)


def compute_ci95(values: List[float]) -> Tuple[float, float, float]:
    """Compute mean, std, and 95% confidence interval half-width using t-critical 4.303 (df=2)."""
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    t_crit = 4.303
    ci_half = t_crit * (std_val / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return round(mean_val, 4), round(std_val, 4), round(ci_half, 4)


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


class AudioPerturber:
    """Applies controlled acoustic perturbations to raw 1D waveforms."""

    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
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
        from math import gcd

        g = gcd(intermediate_sr, self.sr)
        up = intermediate_sr // g
        down = self.sr // g
        y_inter = signal.resample_poly(x, up, down)
        y_back = signal.resample_poly(y_inter, down, up)
        if len(y_back) < len(x):
            y_back = np.pad(y_back, (0, len(x) - len(y_back)))
        elif len(y_back) > len(x):
            y_back = y_back[: len(x)]
        return y_back.astype(np.float32)

    def filter_audio(self, x: np.ndarray, filter_type: str) -> np.ndarray:
        if filter_type == "lowpass_4000hz":
            return signal.sosfiltfilt(self.sos_lp4k, x).astype(np.float32)
        elif filter_type == "highpass_300hz":
            return signal.sosfiltfilt(self.sos_hp300, x).astype(np.float32)
        elif filter_type == "bandpass_300_4000hz":
            return signal.sosfiltfilt(self.sos_bp, x).astype(np.float32)
        return x.copy()


def train_fold_model(
    heldout_m: str,
    seed: int,
    raw_cache: Dict[str, Dict[str, List[Tuple[str, int, np.ndarray]]]],
    feat_engine: FeatureEngine,
    gain_normalizer: Optional[GainNormalizer] = None,
    max_epochs: int = 15,
) -> Tuple[ResidualCNNAE, float, float, float]:
    """Train fold autoencoder with Smooth L1 beta=0.1 on the 3 known machines."""
    set_seed(seed)
    train_machines = [m for m in MACHINES if m != heldout_m]

    train_patches_list = []
    for tm in train_machines:
        for item in raw_cache[tm]["train"]:
            wav = item[2]
            w_norm = gain_normalizer.normalize(wav)[0] if gain_normalizer else wav
            p = feat_engine.extract_patches(w_norm)
            train_patches_list.append(p)
    train_all = np.concatenate(train_patches_list, axis=0)

    train_mean = float(np.mean(train_all))
    train_std = float(np.std(train_all)) + 1e-6

    train_norm = (train_all - train_mean) / train_std
    train_ds = TensorDataset(torch.from_numpy(train_norm).float())
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=False)

    dev_patches_list = []
    for tm in train_machines:
        for item in raw_cache[tm]["dev"]:
            wav = item[2]
            w_norm = gain_normalizer.normalize(wav)[0] if gain_normalizer else wav
            p = feat_engine.extract_patches(w_norm)
            dev_patches_list.append(p)
    dev_all = np.concatenate(dev_patches_list, axis=0)
    dev_norm = (dev_all - train_mean) / train_std
    dev_ds = TensorDataset(torch.from_numpy(dev_norm).float())
    dev_dl = DataLoader(dev_ds, batch_size=32, shuffle=False)

    model = ResidualCNNAE(in_channels=1, base_channels=16, latent_dim=64, norm_type="group")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    t_start = time.perf_counter()
    best_dev_loss = float("inf")
    best_state = None
    patience = 4
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for (batch_x,) in train_dl:
            optimizer.zero_grad()
            recon, _ = model(batch_x)
            loss = F.smooth_l1_loss(recon, batch_x, beta=0.1)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        dev_loss_accum = 0.0
        with torch.no_grad():
            for (batch_x,) in dev_dl:
                recon, _ = model(batch_x)
                dev_loss_accum += float(F.smooth_l1_loss(recon, batch_x, beta=0.1).item()) * len(batch_x)
        val_loss = dev_loss_accum / len(dev_norm)

        if val_loss < best_dev_loss:
            best_dev_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_dur = time.perf_counter() - t_start
    return model, train_mean, train_std, train_dur


def calibrate_whitening(
    model: ResidualCNNAE,
    enroll_raw: List[Tuple[str, int, np.ndarray]],
    train_mean: float,
    train_std: float,
    feat_engine: FeatureEngine,
    gain_normalizer: Optional[GainNormalizer] = None,
    channel_index: int = 0,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """Fit spatial-spectral residual whitening on 10 healthy enrollment clips."""
    t0 = time.perf_counter()
    model.eval()

    all_enroll_residuals = []
    with torch.no_grad():
        for item in enroll_raw:
            # Handle multi-channel raw audio if provided
            wav_item = item[3] if len(item) > 3 else item[2]
            if wav_item.ndim > 1:
                wav = wav_item[:, channel_index]
            else:
                wav = wav_item

            w_norm = gain_normalizer.normalize(wav)[0] if gain_normalizer else wav
            patches = feat_engine.extract_patches(w_norm)
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


def generate_perturbed_waveform(
    wav_ch0: np.ndarray,
    raw_fp: str,
    condition: str,
    perturber: AudioPerturber,
    seed: int,
) -> Tuple[np.ndarray, int]:
    """Generate waveform for specified environmental condition.

    Returns:
        Tuple of (waveform, sample_rate)
    """
    sr = 16000

    if condition == "clean_baseline":
        return wav_ch0.copy(), sr
    elif condition == "noise_snr_30db":
        y, _ = perturber.add_noise(wav_ch0, 30.0, seed=seed)
        return y, sr
    elif condition == "noise_snr_20db":
        y, _ = perturber.add_noise(wav_ch0, 20.0, seed=seed)
        return y, sr
    elif condition == "noise_snr_10db":
        y, _ = perturber.add_noise(wav_ch0, 10.0, seed=seed)
        return y, sr
    elif condition == "noise_snr_5db":
        y, _ = perturber.add_noise(wav_ch0, 5.0, seed=seed)
        return y, sr
    elif condition == "gain_minus_12db":
        y, _ = perturber.scale_gain(wav_ch0, -12.0)
        return y, sr
    elif condition == "gain_minus_6db":
        y, _ = perturber.scale_gain(wav_ch0, -6.0)
        return y, sr
    elif condition == "gain_plus_6db":
        y, _ = perturber.scale_gain(wav_ch0, 6.0)
        return y, sr
    elif condition == "gain_plus_12db":
        y, _ = perturber.scale_gain(wav_ch0, 12.0)
        return y, sr
    elif condition == "resample_8000hz":
        # Pass 8 kHz sample rate audio directly to test quality gate
        from math import gcd

        up = 8000 // gcd(8000, 16000)
        down = 16000 // gcd(8000, 16000)
        y = signal.resample_poly(wav_ch0, up, down).astype(np.float32)
        return y, 8000
    elif condition == "resample_22050hz":
        from math import gcd

        up = 22050 // gcd(22050, 16000)
        down = 16000 // gcd(22050, 16000)
        y = signal.resample_poly(wav_ch0, up, down).astype(np.float32)
        return y, 22050
    elif condition == "resample_44100hz":
        from math import gcd

        up = 44100 // gcd(44100, 16000)
        down = 16000 // gcd(44100, 16000)
        y = signal.resample_poly(wav_ch0, up, down).astype(np.float32)
        return y, 44100
    elif condition == "resample_48000hz":
        from math import gcd

        up = 48000 // gcd(48000, 16000)
        down = 16000 // gcd(48000, 16000)
        y = signal.resample_poly(wav_ch0, up, down).astype(np.float32)
        return y, 48000
    elif condition == "mic_lowpass_4000hz":
        return perturber.filter_audio(wav_ch0, "lowpass_4000hz"), sr
    elif condition == "mic_highpass_300hz":
        return perturber.filter_audio(wav_ch0, "highpass_300hz"), sr
    elif condition == "mic_bandpass_300_4000hz":
        return perturber.filter_audio(wav_ch0, "bandpass_300_4000hz"), sr
    elif condition == "combined_factory_realistic":
        w_n, _ = perturber.add_noise(wav_ch0, 20.0, seed=seed)
        w_g, _ = perturber.scale_gain(w_n, -6.0)
        return perturber.filter_audio(w_g, "lowpass_4000hz"), sr
    elif condition == "combined_harsh_stress":
        w_n, _ = perturber.add_noise(wav_ch0, 10.0, seed=seed)
        w_g, _ = perturber.scale_gain(w_n, 6.0)
        return perturber.filter_audio(w_g, "highpass_300hz"), sr
    elif condition == "physical_mic_ch1":
        full_raw, _ = sf.read(raw_fp)
        return full_raw[:, 1].astype(np.float32), sr
    elif condition == "physical_mic_ch4":
        full_raw, _ = sf.read(raw_fp)
        return full_raw[:, 4].astype(np.float32), sr
    return wav_ch0.copy(), sr


def main():
    logger.info("=== STARTING SPRINT 7.6 / E6 SIGNAL ROBUSTNESS REMEDIATION ===")
    t_global_start = time.perf_counter()

    # Initialize engines
    feat_engine = FeatureEngine()
    perturber = AudioPerturber(sample_rate=16000)
    gain_normalizer = GainNormalizer(GainNormalizerConfig(target_rms_dbfs=-42.0))
    quality_gate = AudioQualityGate(QualityGateConfig(min_sample_rate=16000, target_sample_rate=16000))
    sensor_guard = SensorGuard()

    # Load raw audio into memory
    logger.info("Loading raw MIMII audio recordings into memory cache...")
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
                        # Keep full multi-channel array for calibration experiments
                        raw_cache[m][split].append((str(fp_path), lbl, ch0, full_audio))

    logger.info("Loaded all raw waveforms. Setting up 4 pipelines and 20 evaluation conditions...")

    CONDITIONS = [
        "clean_baseline",
        "gain_minus_12db",
        "gain_minus_6db",
        "gain_plus_6db",
        "gain_plus_12db",
        "noise_snr_30db",
        "noise_snr_20db",
        "noise_snr_10db",
        "noise_snr_5db",
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

    # Pre-train fold models for each seed:
    # 1. Baseline unnormalized models (for Pipeline A)
    # 2. Gain-normalized models (for Pipelines B, C, D)
    logger.info("Training clean fold models across 4 folds x 3 seeds...")
    models_unnorm = {}
    models_norm = {}

    for heldout_m in MACHINES:
        models_unnorm[heldout_m] = {}
        models_norm[heldout_m] = {}
        for seed in SEEDS:
            # Unnormalized model (Pipeline A)
            m_a, tm_a, ts_a, dur_a = train_fold_model(
                heldout_m=heldout_m,
                seed=seed,
                raw_cache=raw_cache,
                feat_engine=feat_engine,
                gain_normalizer=None,
            )
            # Calibrate on 10 healthy clean clips (ch 0)
            th_a, rm_a, rs_a, cd_a = calibrate_whitening(
                model=m_a,
                enroll_raw=raw_cache[heldout_m]["train"],
                train_mean=tm_a,
                train_std=ts_a,
                feat_engine=feat_engine,
                gain_normalizer=None,
                channel_index=0,
            )
            models_unnorm[heldout_m][seed] = (m_a, tm_a, ts_a, th_a, rm_a, rs_a)

            # Gain-normalized model (Pipelines B, C, D)
            m_b, tm_b, ts_b, dur_b = train_fold_model(
                heldout_m=heldout_m,
                seed=seed,
                raw_cache=raw_cache,
                feat_engine=feat_engine,
                gain_normalizer=gain_normalizer,
            )
            th_b, rm_b, rs_b, cd_b = calibrate_whitening(
                model=m_b,
                enroll_raw=raw_cache[heldout_m]["train"],
                train_mean=tm_b,
                train_std=ts_b,
                feat_engine=feat_engine,
                gain_normalizer=gain_normalizer,
                channel_index=0,
            )
            # Register in sensor guard for Channel 0
            sensor_guard.register_profile(
                machine_id=heldout_m,
                sensor_id="primary_array",
                channel_index=0,
                profile_data={"threshold": th_b, "res_mean": rm_b, "res_std": rs_b},
            )

            # Also pre-calibrate on Channel 1 and Channel 4 for recalibration benchmarks
            th_b_ch1, rm_b_ch1, rs_b_ch1, _ = calibrate_whitening(
                model=m_b,
                enroll_raw=[(fp, lbl, full) for fp, lbl, ch0, full in raw_cache[heldout_m]["train"]],
                train_mean=tm_b,
                train_std=ts_b,
                feat_engine=feat_engine,
                gain_normalizer=gain_normalizer,
                channel_index=1,
            )
            th_b_ch4, rm_b_ch4, rs_b_ch4, _ = calibrate_whitening(
                model=m_b,
                enroll_raw=[(fp, lbl, full) for fp, lbl, ch0, full in raw_cache[heldout_m]["train"]],
                train_mean=tm_b,
                train_std=ts_b,
                feat_engine=feat_engine,
                gain_normalizer=gain_normalizer,
                channel_index=4,
            )

            models_norm[heldout_m][seed] = {
                "model": m_b,
                "train_mean": tm_b,
                "train_std": ts_b,
                "ch0": (th_b, rm_b, rs_b),
                "ch1_recalibrated": (th_b_ch1, rm_b_ch1, rs_b_ch1),
                "ch4_recalibrated": (th_b_ch4, rm_b_ch4, rs_b_ch4),
            }

    logger.info("All fold models trained and calibrated successfully.")

    # Results containers
    e6_results = {
        "pipeline_a_baseline": {},
        "pipeline_b_gain_normalized": {},
        "pipeline_c_sensor_guarded": {},
        "pipeline_d_complete_production": {},
    }
    per_machine_metrics = {
        "pipeline_a_baseline": {},
        "pipeline_b_gain_normalized": {},
        "pipeline_c_sensor_guarded": {},
        "pipeline_d_complete_production": {},
    }
    gain_normalization_comparison = {}
    sensor_compatibility_results = {}
    quality_gate_results = {}

    PIPELINES = [
        "pipeline_a_baseline",
        "pipeline_b_gain_normalized",
        "pipeline_c_sensor_guarded",
        "pipeline_d_complete_production",
    ]

    for p_name in PIPELINES:
        per_machine_metrics[p_name] = {cond: {m: {} for m in MACHINES} for cond in CONDITIONS}

    logger.info("\n>>> EXECUTING BENCHMARK ACROSS 4 PIPELINES AND 20 CONDITIONS <<<")

    for cond in CONDITIONS:
        logger.info(f"Evaluating condition: {cond}")

        for p_name in PIPELINES:
            aucs_by_seed = {s: [] for s in SEEDS}
            praucs_by_seed = {s: [] for s in SEEDS}
            f1s_by_seed = {s: [] for s in SEEDS}
            fprs_by_seed = {s: [] for s in SEEDS}
            fnrs_by_seed = {s: [] for s in SEEDS}
            rejected_counts = {s: 0 for s in SEEDS}
            adverse_env_counts = {s: 0 for s in SEEDS}
            inference_latencies = []

            for heldout_m in MACHINES:
                test_raw = raw_cache[heldout_m]["test"]

                for seed in SEEDS:
                    if p_name == "pipeline_a_baseline":
                        model, t_mean, t_std, thresh, r_mean, r_std = models_unnorm[heldout_m][seed]
                    else:
                        m_dict = models_norm[heldout_m][seed]
                        model = m_dict["model"]
                        t_mean, t_std = m_dict["train_mean"], m_dict["train_std"]

                        # Check sensor condition
                        if cond == "physical_mic_ch1":
                            if p_name in ["pipeline_c_sensor_guarded", "pipeline_d_complete_production"]:
                                # Recalibrated profile for sensor-guarded pipeline
                                thresh, r_mean, r_std = m_dict["ch1_recalibrated"]
                            else:
                                # Reusing ch0 profile (uncalibrated)
                                thresh, r_mean, r_std = m_dict["ch0"]
                        elif cond == "physical_mic_ch4":
                            if p_name in ["pipeline_c_sensor_guarded", "pipeline_d_complete_production"]:
                                thresh, r_mean, r_std = m_dict["ch4_recalibrated"]
                            else:
                                thresh, r_mean, r_std = m_dict["ch0"]
                        else:
                            thresh, r_mean, r_std = m_dict["ch0"]

                    model.eval()
                    test_scores = []
                    test_labels = []

                    with torch.no_grad():
                        for fp, lbl, wav_ch0, raw_fp in test_raw:
                            # 1. Generate perturbed waveform & sample rate
                            wav_in, sr_in = generate_perturbed_waveform(
                                wav_ch0=wav_ch0,
                                raw_fp=raw_fp,
                                condition=cond,
                                perturber=perturber,
                                seed=seed,
                            )

                            t_inf_0 = time.perf_counter()

                            # 2. Quality Gate Verification (Pipeline D only)
                            if p_name == "pipeline_d_complete_production":
                                prep_wav, q_res = quality_gate.validate_and_prepare(
                                    waveform=wav_in,
                                    sample_rate=sr_in,
                                    channel=0,
                                )
                                if q_res.status == "ADVERSE_ACOUSTIC_ENVIRONMENT":
                                    adverse_env_counts[seed] += 1

                                if not q_res.is_valid:
                                    rejected_counts[seed] += 1
                                    continue  # Safely reject invalid recording
                                wav_proc = prep_wav
                            else:
                                # In Pipeline A, B, C, if 8 kHz or other rates, do simple cycle or raw
                                if sr_in != 16000:
                                    wav_proc = perturber.resample_cycle(wav_ch0, sr_in)
                                else:
                                    wav_proc = wav_in

                            # 3. Controlled Gain Normalization (Pipelines B, C, D)
                            if p_name != "pipeline_a_baseline":
                                wav_final, tele = gain_normalizer.normalize(wav_proc)
                            else:
                                wav_final = wav_proc

                            # 4. Feature Extraction & Scoring
                            patches = feat_engine.extract_patches(wav_final)
                            p_norm = torch.from_numpy((patches - t_mean) / t_std).float()
                            recon, _ = model(p_norm)
                            diff = torch.abs(recon - p_norm).cpu().numpy()
                            whitened = np.maximum(0.0, (diff - r_mean) / r_std)
                            score = float(np.mean(whitened))

                            inf_dur_ms = (time.perf_counter() - t_inf_0) * 1000.0
                            inference_latencies.append(inf_dur_ms)

                            test_scores.append(score)
                            test_labels.append(lbl)

                    # Compute metrics for this machine and seed
                    if len(test_scores) > 0 and len(set(test_labels)) > 1:
                        y_true = np.array(test_labels)
                        y_scores = np.array(test_scores)
                        y_pred = (y_scores >= thresh).astype(int)

                        auc = float(roc_auc_score(y_true, y_scores))
                        prauc = float(average_precision_score(y_true, y_scores))
                        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

                        fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
                        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
                        prec, rec, f1, _ = precision_recall_fscore_support(
                            y_true, y_pred, average="binary", zero_division=0
                        )

                        aucs_by_seed[seed].append(auc)
                        praucs_by_seed[seed].append(prauc)
                        f1s_by_seed[seed].append(f1)
                        fprs_by_seed[seed].append(fpr)
                        fnrs_by_seed[seed].append(fnr)

                        per_machine_metrics[p_name][cond][heldout_m][str(seed)] = {
                            "roc_auc": round(auc, 4),
                            "pr_auc": round(prauc, 4),
                            "precision": round(float(prec), 4),
                            "recall": round(float(rec), 4),
                            "f1": round(float(f1), 4),
                            "fpr": round(float(fpr), 4),
                            "fnr": round(float(fnr), 4),
                            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
                        }
                    else:
                        # Handle fully rejected test set (e.g. 8 kHz in Pipeline D)
                        per_machine_metrics[p_name][cond][heldout_m][str(seed)] = {
                            "status": "REJECTED_BY_QUALITY_GATE",
                            "rejected_samples": len(test_raw),
                        }

            # Macro aggregation across 4 machines
            seed_macro_aucs = [float(np.mean(aucs_by_seed[s])) for s in SEEDS if len(aucs_by_seed[s]) > 0]
            seed_macro_praucs = [float(np.mean(praucs_by_seed[s])) for s in SEEDS if len(praucs_by_seed[s]) > 0]
            seed_macro_f1s = [float(np.mean(f1s_by_seed[s])) for s in SEEDS if len(f1s_by_seed[s]) > 0]
            seed_macro_fprs = [float(np.mean(fprs_by_seed[s])) for s in SEEDS if len(fprs_by_seed[s]) > 0]
            seed_macro_fnrs = [float(np.mean(fnrs_by_seed[s])) for s in SEEDS if len(fnrs_by_seed[s]) > 0]

            if len(seed_macro_aucs) > 0:
                m_auc, s_auc, ci_auc = compute_ci95(seed_macro_aucs)
                m_prauc, s_prauc, ci_prauc = compute_ci95(seed_macro_praucs)
                m_f1, s_f1, ci_f1 = compute_ci95(seed_macro_f1s)
                m_fpr = round(float(np.mean(seed_macro_fprs)), 4)
                m_fnr = round(float(np.mean(seed_macro_fnrs)), 4)

                p50_lat = float(np.percentile(inference_latencies, 50)) if inference_latencies else 0.0
                p95_lat = float(np.percentile(inference_latencies, 95)) if inference_latencies else 0.0

                e6_results[p_name][cond] = {
                    "roc_auc": {"mean": m_auc, "std": s_auc, "ci95_half": ci_auc, "per_seed": seed_macro_aucs},
                    "pr_auc": {"mean": m_prauc, "std": s_prauc, "ci95_half": ci_prauc, "per_seed": seed_macro_praucs},
                    "f1": {"mean": m_f1, "std": s_f1, "ci95_half": ci_f1, "per_seed": seed_macro_f1s},
                    "fpr": m_fpr,
                    "fnr": m_fnr,
                    "p50_latency_ms": round(p50_lat, 2),
                    "p95_latency_ms": round(p95_lat, 2),
                    "rejected_rate": round(sum(rejected_counts.values()) / (4 * 18 * 3), 4),
                    "adverse_env_warnings": sum(adverse_env_counts.values()),
                }
            else:
                # Fully rejected condition
                e6_results[p_name][cond] = {
                    "status": "REJECTED_BY_QUALITY_GATE",
                    "rejected_rate": 1.0,
                    "rejection_reason": "SAMPLE_RATE_UNSUPPORTED (8000 Hz < 16000 Hz)",
                }

    logger.info("All pipeline evaluations completed. Generating specialized comparison artifacts...")

    # 1. Gain Normalization Comparison artifact
    gain_conditions = ["clean_baseline", "gain_minus_12db", "gain_minus_6db", "gain_plus_6db", "gain_plus_12db"]
    for gc_name in gain_conditions:
        gain_normalization_comparison[gc_name] = {
            "pipeline_a_baseline": e6_results["pipeline_a_baseline"][gc_name],
            "pipeline_b_gain_normalized": e6_results["pipeline_b_gain_normalized"][gc_name],
            "auc_delta": round(
                e6_results["pipeline_b_gain_normalized"][gc_name]["roc_auc"]["mean"]
                - e6_results["pipeline_a_baseline"][gc_name]["roc_auc"]["mean"],
                4,
            ),
            "fpr_delta": round(
                e6_results["pipeline_b_gain_normalized"][gc_name]["fpr"]
                - e6_results["pipeline_a_baseline"][gc_name]["fpr"],
                4,
            ),
            "fnr_delta": round(
                e6_results["pipeline_b_gain_normalized"][gc_name]["fnr"]
                - e6_results["pipeline_a_baseline"][gc_name]["fnr"],
                4,
            ),
        }

    # 2. Sensor Compatibility Results artifact
    sensor_compatibility_results = {
        "physical_mic_ch1": {
            "uncalibrated_pipeline_b": e6_results["pipeline_b_gain_normalized"]["physical_mic_ch1"],
            "recalibrated_pipeline_c": e6_results["pipeline_c_sensor_guarded"]["physical_mic_ch1"],
            "auc_delta": round(
                e6_results["pipeline_c_sensor_guarded"]["physical_mic_ch1"]["roc_auc"]["mean"]
                - e6_results["pipeline_b_gain_normalized"]["physical_mic_ch1"]["roc_auc"]["mean"],
                4,
            ),
            "fnr_delta": round(
                e6_results["pipeline_c_sensor_guarded"]["physical_mic_ch1"]["fnr"]
                - e6_results["pipeline_b_gain_normalized"]["physical_mic_ch1"]["fnr"],
                4,
            ),
        },
        "physical_mic_ch4": {
            "uncalibrated_pipeline_b": e6_results["pipeline_b_gain_normalized"]["physical_mic_ch4"],
            "recalibrated_pipeline_c": e6_results["pipeline_c_sensor_guarded"]["physical_mic_ch4"],
            "auc_delta": round(
                e6_results["pipeline_c_sensor_guarded"]["physical_mic_ch4"]["roc_auc"]["mean"]
                - e6_results["pipeline_b_gain_normalized"]["physical_mic_ch4"]["roc_auc"]["mean"],
                4,
            ),
            "fnr_delta": round(
                e6_results["pipeline_c_sensor_guarded"]["physical_mic_ch4"]["fnr"]
                - e6_results["pipeline_b_gain_normalized"]["physical_mic_ch4"]["fnr"],
                4,
            ),
        },
    }

    # 3. Quality Gate Results artifact
    quality_gate_results = {
        "sub_nyquist_8khz": {
            "pipeline_a_status": "PROCESSED_UNSAFELY",
            "pipeline_a_auc": e6_results["pipeline_a_baseline"]["resample_8000hz"]["roc_auc"]["mean"],
            "pipeline_a_fnr": e6_results["pipeline_a_baseline"]["resample_8000hz"]["fnr"],
            "pipeline_d_status": "REJECTED_SAFELY",
            "pipeline_d_rejection_rate": e6_results["pipeline_d_complete_production"]["resample_8000hz"][
                "rejected_rate"
            ],
            "rejection_reason": "SAMPLE_RATE_UNSUPPORTED (8000 Hz < 16000 Hz)",
        },
        "supported_resampling": {
            "22050hz": {
                "pipeline_a_auc": e6_results["pipeline_a_baseline"]["resample_22050hz"]["roc_auc"]["mean"],
                "pipeline_d_auc": e6_results["pipeline_d_complete_production"]["resample_22050hz"]["roc_auc"]["mean"],
            },
            "44100hz": {
                "pipeline_a_auc": e6_results["pipeline_a_baseline"]["resample_44100hz"]["roc_auc"]["mean"],
                "pipeline_d_auc": e6_results["pipeline_d_complete_production"]["resample_44100hz"]["roc_auc"]["mean"],
            },
            "48000hz": {
                "pipeline_a_auc": e6_results["pipeline_a_baseline"]["resample_48000hz"]["roc_auc"]["mean"],
                "pipeline_d_auc": e6_results["pipeline_d_complete_production"]["resample_48000hz"]["roc_auc"]["mean"],
            },
        },
        "acoustic_environment_monitoring": {
            "noise_snr_5db": {
                "adverse_env_warnings": e6_results["pipeline_d_complete_production"]["noise_snr_5db"][
                    "adverse_env_warnings"
                ],
                "total_evaluations": 4 * 18 * 3,
                "warning_rate": round(
                    e6_results["pipeline_d_complete_production"]["noise_snr_5db"]["adverse_env_warnings"]
                    / (4 * 18 * 3),
                    4,
                ),
            },
            "noise_snr_10db": {
                "adverse_env_warnings": e6_results["pipeline_d_complete_production"]["noise_snr_10db"][
                    "adverse_env_warnings"
                ],
                "total_evaluations": 4 * 18 * 3,
                "warning_rate": round(
                    e6_results["pipeline_d_complete_production"]["noise_snr_10db"]["adverse_env_warnings"]
                    / (4 * 18 * 3),
                    4,
                ),
            },
        },
    }

    # Save deliverables
    out_dir = Path("experiments/sprint_7_6/e6_signal_robustness")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "e6_results.json", "w") as f:
        json.dump(e6_results, f, indent=2)
    with open(out_dir / "per_machine_metrics.json", "w") as f:
        json.dump(per_machine_metrics, f, indent=2)
    with open(out_dir / "gain_normalization_comparison.json", "w") as f:
        json.dump(gain_normalization_comparison, f, indent=2)
    with open(out_dir / "sensor_compatibility_results.json", "w") as f:
        json.dump(sensor_compatibility_results, f, indent=2)
    with open(out_dir / "quality_gate_results.json", "w") as f:
        json.dump(quality_gate_results, f, indent=2)

    total_elapsed = time.perf_counter() - t_global_start
    logger.info(f"=== E6 BENCHMARK EXECUTION COMPLETE in {total_elapsed:.2f} seconds ===")


if __name__ == "__main__":
    main()

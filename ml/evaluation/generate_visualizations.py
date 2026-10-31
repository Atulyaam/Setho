"""Waveform and Spectrogram Visualization Generator for Stetho.

Produces clear acoustic visualization artifacts comparing normal and anomalous
fan recordings:
1. Time-domain waveform comparison (Normal vs. Anomalous)
2. Time-frequency Log-Mel spectrogram comparison (Normal vs. Anomalous)
3. 8-channel microphone array comparison (Spatial acoustic consistency)
"""

import logging
from pathlib import Path
import sys
from typing import Optional

# Ensure repository root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from stetho_audio.channel_policy import ChannelHandler, ChannelPolicy
from ml.preprocessing.log_mel import LogMelConfig, LogMelExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stetho_visualizer")


def generate_visualizations(
    raw_data_dir: Path = Path("data/mimii/raw/fan"),
    output_dir: Path = Path("docs/evaluation/visualizations"),
    machine_id: str = "id_00",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    norm_dir = raw_data_dir / machine_id / "normal"
    abnorm_dir = raw_data_dir / machine_id / "abnormal"

    norm_files = sorted(list(norm_dir.glob("*.wav")))
    abnorm_files = sorted(list(abnorm_dir.glob("*.wav")))

    if not norm_files or not abnorm_files:
        logger.warning(f"Insufficient audio files in {norm_dir} or {abnorm_dir} for visualization")
        return

    norm_path = norm_files[0]
    abnorm_path = abnorm_files[0]

    logger.info(f"Loading normal audio: {norm_path}")
    data_norm, sr = sf.read(str(norm_path), dtype="float32", always_2d=True)
    logger.info(f"Loading abnormal audio: {abnorm_path}")
    data_abnorm, sr_ab = sf.read(str(abnorm_path), dtype="float32", always_2d=True)

    # Channel 0 reference waveforms
    sig_norm = ChannelHandler.apply(data_norm, ChannelPolicy.CHANNEL_0)
    sig_abnorm = ChannelHandler.apply(data_abnorm, ChannelPolicy.CHANNEL_0)

    time_axis = np.linspace(0, len(sig_norm) / sr, len(sig_norm))

    # --- 1. Waveform Comparison Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)
    axes[0].plot(time_axis, sig_norm, color="#1f77b4", linewidth=0.6, alpha=0.9)
    axes[0].set_title(f"Stetho: Normal Machine Sound Waveform (Fan {machine_id} - {norm_path.name})", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(time_axis, sig_abnorm, color="#d62728", linewidth=0.6, alpha=0.9)
    axes[1].set_title(f"Stetho: Anomalous Machine Sound Waveform (Fan {machine_id} - {abnorm_path.name})", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    wav_plot_path = output_dir / "waveform_normal_vs_abnormal.png"
    plt.savefig(wav_plot_path, dpi=180)
    plt.close()
    logger.info(f"Saved waveform comparison: {wav_plot_path}")

    # --- 2. Log-Mel Spectrogram Comparison Plot ---
    mel_extractor = LogMelExtractor(LogMelConfig(sample_rate=sr, n_mels=64, hop_length=256))
    mel_norm, _ = mel_extractor.extract_numpy(sig_norm)
    mel_abnorm, _ = mel_extractor.extract_numpy(sig_abnorm)

    # Shape: (1, 64, frames) -> squeeze to (64, frames)
    mel_norm_2d = mel_norm.squeeze()
    mel_abnorm_2d = mel_abnorm.squeeze()

    vmin = min(float(mel_norm_2d.min()), float(mel_abnorm_2d.min()))
    vmax = max(float(mel_norm_2d.max()), float(mel_abnorm_2d.max()))

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    im1 = axes[0].imshow(
        mel_norm_2d,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title(f"Normal Log-Mel Spectrogram (64 Bins, Fan {machine_id})", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Mel Frequency Bin")
    fig.colorbar(im1, ax=axes[0], format="%+2.0f dB", label="Magnitude (dB)")

    im2 = axes[1].imshow(
        mel_abnorm_2d,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(f"Anomalous Log-Mel Spectrogram (64 Bins, Fan {machine_id})", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Time Frame (16ms step, 50% overlap windowing)")
    axes[1].set_ylabel("Mel Frequency Bin")
    fig.colorbar(im2, ax=axes[1], format="%+2.0f dB", label="Magnitude (dB)")

    plt.tight_layout()
    spec_plot_path = output_dir / "spectrogram_normal_vs_abnormal.png"
    plt.savefig(spec_plot_path, dpi=180)
    plt.close()
    logger.info(f"Saved spectrogram comparison: {spec_plot_path}")

    # --- 3. Multichannel Array Comparison Plot ---
    fig, axes = plt.subplots(4, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    for ch in range(min(8, data_norm.shape[1])):
        ch_sig = data_norm[:, ch]
        rms = np.sqrt(np.mean(ch_sig**2))
        axes[ch].plot(time_axis[: sr * 2], ch_sig[: sr * 2], linewidth=0.5, color="#2ca02c")
        axes[ch].set_title(f"Channel {ch} (RMS: {rms:.4f})", fontsize=9)
        axes[ch].grid(True, linestyle=":", alpha=0.4)
        if ch % 2 == 0:
            axes[ch].set_ylabel("Amp")
        if ch >= 6:
            axes[ch].set_xlabel("Time (s) [0-2s snippet]")

    plt.suptitle(f"MIMII 8-Channel Microphone Array Recording Comparison (Fan {machine_id} Normal)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    ch_plot_path = output_dir / "channel_comparison.png"
    plt.savefig(ch_plot_path, dpi=180)
    plt.close()
    logger.info(f"Saved multichannel comparison: {ch_plot_path}")


if __name__ == "__main__":
    generate_visualizations()

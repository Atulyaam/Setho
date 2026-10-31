"""Dense Autoencoder Anomaly Detector and Trainer for Stetho.

Wraps the PyTorch DenseAutoencoder into the unified BaseAnomalyDetector interface:
- Self-supervised representation learning on normal machine sounds only
- Log-Mel patch extraction and train-only statistical standardization
- Early stopping on normal development loss
- Reconstruction-error based anomaly scoring
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.models.base import BaseAnomalyDetector
from ml.models.baseline_autoencoder.model import DenseAutoencoder
from ml.preprocessing.log_mel import LogMelConfig, LogMelExtractor
from ml.preprocessing.windowing import AudioWindowing, WindowConfig
from stetho_audio.channel_policy import ChannelHandler, ChannelPolicy
from stetho_audio.preprocessing import AudioPreprocessor, PreprocessConfig

logger = logging.getLogger("stetho_autoencoder")


class DenseAutoencoderDetector(BaseAnomalyDetector):
    """Machine-specific Dense Autoencoder anomaly detector."""

    def __init__(
        self,
        machine_id: str = "id_00",
        input_dim: int = 4096,  # 64 * 64
        latent_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 8,
        max_epochs: int = 50,
        early_stopping_patience: int = 8,
        device: str = "auto",
        seed: int = 42,
    ):
        super().__init__(machine_id=machine_id, model_type="dense_autoencoder")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [512, 128]
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Feature processing engines
        self.preprocessor = AudioPreprocessor(
            PreprocessConfig(target_sample_rate=16000, channel_policy=ChannelPolicy.CHANNEL_0)
        )
        self.mel_extractor = LogMelExtractor(
            LogMelConfig(sample_rate=16000, n_mels=64, hop_length=256)
        )
        self.windowing = AudioWindowing(
            WindowConfig(spec_window_frames=64, spec_hop_frames=32)
        )

        # Model instance
        self.model = DenseAutoencoder(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
        ).to(self.device)

        # Normalization parameters (computed strictly on training data)
        self.train_mean: Optional[float] = None
        self.train_std: Optional[float] = None
        self.training_metadata: Dict[str, Any] = {}

    def _extract_patches_from_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """Extract standardized 4096-dim patches from 1D waveform."""
        spec, _ = self.mel_extractor.extract_numpy(waveform)  # (1, 64, frames)
        windows, _ = self.windowing.slice_spectrogram_numpy(spec.squeeze(0))  # (N, 64, 64)
        flat_patches = windows.reshape(windows.shape[0], -1)  # (N, 4096)
        return flat_patches

    def _extract_patches_from_files(self, file_paths: List[Union[str, Path]]) -> np.ndarray:
        """Extract and aggregate all patches across a list of audio files."""
        all_patches = []
        for p in file_paths:
            wav, _ = self.preprocessor.process_file(p)
            patches = self._extract_patches_from_waveform(wav)
            all_patches.append(patches)
        if not all_patches:
            return np.empty((0, self.input_dim), dtype=np.float32)
        return np.concatenate(all_patches, axis=0).astype(np.float32)

    def fit(
        self,
        train_files: List[Union[str, Path]],
        dev_files: Optional[List[Union[str, Path]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train dense autoencoder on normal training patches only."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        start_time = time.perf_counter()

        logger.info(f"Extracting log-Mel patches from {len(train_files)} training files for {self.machine_id}...")
        train_patches = self._extract_patches_from_files(train_files)
        if train_patches.shape[0] == 0:
            raise ValueError("No patches extracted from train files")

        # Compute normalization statistics strictly from training data
        self.train_mean = float(np.mean(train_patches))
        self.train_std = float(np.std(train_patches)) + 1e-6

        # Standardize train patches
        train_norm = (train_patches - self.train_mean) / self.train_std

        # Development data for early stopping
        dev_loader = None
        if dev_files:
            dev_patches = self._extract_patches_from_files(dev_files)
            dev_norm = (dev_patches - self.train_mean) / self.train_std
            dev_dataset = TensorDataset(torch.from_numpy(dev_norm).float())
            dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=False)

        train_dataset = TensorDataset(torch.from_numpy(train_norm).float())
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        criterion = nn.L1Loss()

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        loss_history = []

        self.model.train()
        for epoch in range(1, self.max_epochs + 1):
            epoch_train_losses = []
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon, _ = self.model(batch_x)
                loss = criterion(recon, batch_x)

                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError(f"NaN/Inf loss detected at epoch {epoch}")

                loss.backward()
                optimizer.step()
                epoch_train_losses.append(loss.item())

            train_loss = float(np.mean(epoch_train_losses))

            # Evaluate on development set
            val_loss = train_loss
            if dev_loader is not None:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for (batch_v,) in dev_loader:
                        batch_v = batch_v.to(self.device)
                        recon_v, _ = self.model(batch_v)
                        v_loss = criterion(recon_v, batch_v)
                        val_losses.append(v_loss.item())
                val_loss = float(np.mean(val_losses))
                self.model.train()

            loss_history.append({"epoch": epoch, "train_loss": round(train_loss, 5), "val_loss": round(val_loss, 5)})

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Load best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self.is_fitted = True
        duration = time.perf_counter() - start_time

        self.training_metadata = {
            "machine_id": self.machine_id,
            "model_type": self.model_type,
            "train_sample_count": len(train_files),
            "train_patch_count": train_patches.shape[0],
            "dev_sample_count": len(dev_files) if dev_files else 0,
            "total_epochs": len(loss_history),
            "best_val_loss": round(best_val_loss, 5),
            "final_train_loss": round(loss_history[-1]["train_loss"], 5),
            "training_duration_sec": round(duration, 4),
            "device": str(self.device),
            "train_mean": self.train_mean,
            "train_std": self.train_std,
            "loss_history": loss_history,
        }
        logger.info(f"Autoencoder training complete in {duration:.2f}s (Best Val Loss: {best_val_loss:.5f})")
        return self.training_metadata

    def score_waveform(self, waveform: np.ndarray) -> float:
        """Compute recording-level continuous anomaly score.
        Calculates mean L1 reconstruction error across all 19 temporal patches.
        HIGHER score strictly means MORE ANOMALOUS.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")

        patches = self._extract_patches_from_waveform(waveform)
        if patches.shape[0] == 0:
            return 0.0

        # Standardize using training distribution
        patches_norm = (patches - self.train_mean) / self.train_std
        x_tensor = torch.from_numpy(patches_norm).float().to(self.device)

        self.model.eval()
        with torch.no_grad():
            recon_errors = self.model.compute_reconstruction_error(x_tensor)  # (N,)
            score = float(torch.mean(recon_errors).item())

        return score

    def score_file(self, file_path: Union[str, Path]) -> float:
        """Process audio file and calculate reconstruction anomaly score."""
        wav, _ = self.preprocessor.process_file(file_path)
        return self.score_waveform(wav)

    def save(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """Save model checkpoint, normalization parameters, and metadata."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        ckpt_file = out_path / "model.pt"
        meta_file = out_path / "metadata.json"

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": self.hidden_dims,
            "train_mean": self.train_mean,
            "train_std": self.train_std,
            "threshold": self.threshold,
        }, ckpt_file)

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "machine_id": self.machine_id,
                "model_type": self.model_type,
                "threshold": self.threshold,
                "training_metadata": self.training_metadata,
            }, f, indent=2)

        return {"checkpoint": str(ckpt_file), "metadata": str(meta_file)}

    @classmethod
    def load(cls, model_dir: Union[str, Path], device: str = "auto") -> "DenseAutoencoderDetector":
        """Deserialize autoencoder detector from directory."""
        p = Path(model_dir)
        ckpt_file = p / "model.pt"
        meta_file = p / "metadata.json"

        if not ckpt_file.exists() or not meta_file.exists():
            raise FileNotFoundError(f"Missing artifacts in {model_dir}")

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        ckpt = torch.load(ckpt_file, map_location="cpu")
        inst = cls(
            machine_id=meta["machine_id"],
            input_dim=ckpt["input_dim"],
            latent_dim=ckpt["latent_dim"],
            hidden_dims=ckpt["hidden_dims"],
            device=device,
        )
        inst.model.load_state_dict(ckpt["model_state_dict"])
        inst.model.to(inst.device)
        inst.model.eval()

        inst.train_mean = ckpt["train_mean"]
        inst.train_std = ckpt["train_std"]
        inst.threshold = meta.get("threshold")
        inst.training_metadata = meta.get("training_metadata", {})
        inst.is_fitted = True
        return inst


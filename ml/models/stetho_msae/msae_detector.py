"""Stetho-MSAE High-Level Anomaly Detector Pipeline.

Wraps StethoMSAE neural network with end-to-end audio ingestion,
train-only log-Mel normalization, early stopping, checkpointing,
and flexible multi-aggregation reconstruction anomaly scoring.
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.models.base import BaseAnomalyDetector
from ml.models.stetho_msae.model import StethoMSAE
from ml.preprocessing.log_mel import LogMelConfig, LogMelExtractor
from ml.preprocessing.windowing import AudioWindowing, WindowConfig
from stetho_audio.channel_policy import ChannelPolicy
from stetho_audio.preprocessing import AudioPreprocessor, PreprocessConfig

logger = logging.getLogger("stetho_msae_detector")


class StethoMSAEDetector(BaseAnomalyDetector):
    """Machine-specific acoustic anomaly detector using Stetho-MSAE architecture."""

    def __init__(
        self,
        machine_id: str = "id_00",
        base_channels: int = 32,
        dilations: Tuple[int, ...] = (1, 2, 4),
        latent_dim: int = 64,
        use_depthwise: bool = True,
        learning_rate: float = 0.0005,
        weight_decay: float = 1e-4,
        batch_size: int = 8,
        max_epochs: int = 50,
        early_stopping_patience: int = 8,
        loss_type: str = "l1",
        aggregation: str = "mean",  # "mean", "top_k", "max"
        scoring_aggregation: Optional[str] = None,
        top_k: Optional[int] = None,
        top_k_patches: int = 3,
        device: str = "auto",
        seed: int = 42,
    ):
        super().__init__(machine_id=machine_id, model_type="stetho_msae")
        self.base_channels = base_channels
        self.dilations = tuple(dilations)
        self.latent_dim = latent_dim
        self.use_depthwise = use_depthwise
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.loss_type = loss_type
        self.aggregation = scoring_aggregation if scoring_aggregation is not None else aggregation
        self.top_k_patches = top_k if top_k is not None else top_k_patches
        self.seed = seed

        # Resolve compute device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Preprocessing & Feature Extraction
        self.preprocessor = AudioPreprocessor(
            PreprocessConfig(
                target_sample_rate=16000,
                channel_policy=ChannelPolicy.CHANNEL_0,
                normalization="peak",
            )
        )
        self.mel_config = LogMelConfig(
            sample_rate=16000,
            n_fft=1024,
            hop_length=256,
            n_mels=64,
            f_min=50.0,
            f_max=8000.0,
        )
        self.mel_extractor = LogMelExtractor(self.mel_config)
        self.windowing = AudioWindowing(
            WindowConfig(
                spec_window_frames=64,
                spec_hop_frames=32,
            )
        )

        # Instantiate StethoMSAE model
        self.model = StethoMSAE(
            in_channels=1,
            base_channels=self.base_channels,
            dilations=self.dilations,
            latent_dim=self.latent_dim,
            use_depthwise=self.use_depthwise,
        ).to(self.device)

        # Train-only standardization constants
        self.train_mean: float = 0.0
        self.train_std: float = 1.0
        self.training_metadata: Dict[str, Any] = {}

    def _extract_patches_from_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """Extract (N, 1, 64, 64) 2D patches from raw 1D audio waveform."""
        if waveform.ndim != 1:
            waveform = waveform.squeeze()

        # Extract log-Mel: shape (1, 64, num_frames)
        spec, _ = self.mel_extractor.extract_numpy(waveform)
        # Window into patches: shape (num_patches, 64, 64)
        patches, _ = self.windowing.slice_spectrogram_numpy(spec.squeeze(0))
        # Add channel dimension: (num_patches, 1, 64, 64)
        patches_4d = np.expand_dims(patches, axis=1).astype(np.float32)
        return patches_4d

    def _extract_patches_from_files(self, file_paths: List[Union[str, Path]]) -> np.ndarray:
        """Extract and concatenate 2D patches across multiple audio files."""
        all_patches = []
        for path in file_paths:
            wav, _ = self.preprocessor.process_file(path)
            patches = self._extract_patches_from_waveform(wav)
            all_patches.append(patches)
        if not all_patches:
            return np.empty((0, 1, 64, 64), dtype=np.float32)
        return np.concatenate(all_patches, axis=0)

    def fit(
        self,
        train_files: List[Union[str, Path]],
        dev_files: Optional[List[Union[str, Path]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Fit Stetho-MSAE exclusively on normal training audio."""
        if not train_files:
            raise ValueError("train_files list cannot be empty")

        start_time = time.perf_counter()
        logger.info(
            f"Extracting log-Mel patches from {len(train_files)} training files for {self.machine_id}..."
        )
        train_patches = self._extract_patches_from_files(train_files)
        if train_patches.shape[0] == 0:
            raise ValueError("No patches extracted from train files")

        # Compute train-only normalization statistics
        self.train_mean = float(np.mean(train_patches))
        self.train_std = float(np.std(train_patches)) + 1e-6

        # Standardize train patches
        train_norm = (train_patches - self.train_mean) / self.train_std
        train_dataset = TensorDataset(torch.from_numpy(train_norm).float())
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        # Development set for validation loss & early stopping
        dev_loader = None
        if dev_files:
            dev_patches = self._extract_patches_from_files(dev_files)
            dev_norm = (dev_patches - self.train_mean) / self.train_std
            dev_dataset = TensorDataset(torch.from_numpy(dev_norm).float())
            dev_loader = DataLoader(
                dev_dataset, batch_size=self.batch_size, shuffle=False
            )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=1e-6
        )

        if self.loss_type == "l1":
            criterion = nn.L1Loss()
        elif self.loss_type == "mse":
            criterion = nn.MSELoss()
        elif self.loss_type == "smooth_l1":
            criterion = nn.SmoothL1Loss()
        else:
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
                    raise RuntimeError(f"NaN/Inf loss encountered at epoch {epoch}")

                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_train_losses.append(loss.item())

            scheduler.step()
            train_loss = float(np.mean(epoch_train_losses))

            # Validation loss evaluation
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

            loss_history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "lr": round(optimizer.param_groups[0]["lr"], 7),
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self.is_fitted = True
        duration = time.perf_counter() - start_time

        param_counts = self.model.count_parameters()
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
            "parameter_counts": param_counts,
            "dilations": list(self.dilations),
            "use_depthwise": self.use_depthwise,
            "latent_dim": self.latent_dim,
            "loss_type": self.loss_type,
            "loss_history": loss_history,
        }
        logger.info(
            f"Stetho-MSAE training complete in {duration:.2f}s "
            f"(Best Val Loss: {best_val_loss:.5f}, Params: {param_counts['total_params']:,})"
        )
        return self.training_metadata

    def score_patches(self, patches: np.ndarray) -> float:
        """Compute continuous anomaly score from standardized (N, 1, 64, 64) patches."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring")
        if patches.shape[0] == 0:
            return 0.0

        patches_norm = (patches - self.train_mean) / self.train_std
        x_tensor = torch.from_numpy(patches_norm).float().to(self.device)

        self.model.eval()
        with torch.no_grad():
            patch_errors = self.model.compute_reconstruction_error(x_tensor).cpu().numpy()

        if self.aggregation == "mean":
            score = float(np.mean(patch_errors))
        elif self.aggregation == "top_k":
            k = min(self.top_k_patches, len(patch_errors))
            top_k = np.partition(patch_errors, -k)[-k:]
            score = float(np.mean(top_k))
        elif self.aggregation == "max":
            score = float(np.max(patch_errors))
        else:
            score = float(np.mean(patch_errors))

        return score

    def score_waveform(self, waveform: np.ndarray) -> float:
        """Process 1D waveform and calculate continuous anomaly score."""
        patches = self._extract_patches_from_waveform(waveform)
        return self.score_patches(patches)

    def score_file(self, file_path: Union[str, Path]) -> float:
        """Process audio file and calculate continuous anomaly score."""
        wav, _ = self.preprocessor.process_file(file_path)
        return self.score_waveform(wav)

    def save(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """Serialize detector state and model weights to directory."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted detector")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        model_file = out_path / "model.pt"
        msae_file = out_path / "stetho_msae.pt"
        meta_file = out_path / "metadata.json"
        config_file = out_path / "config.json"
        history_file = out_path / "history.json"

        torch.save(self.model.state_dict(), model_file)
        torch.save(self.model.state_dict(), msae_file)

        metadata = {
            **self.training_metadata,
            "threshold": self.threshold,
            "train_mean": self.train_mean,
            "train_std": self.train_std,
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(self.training_metadata.get("loss_history", []), f, indent=2)

        config_dict = {
            "machine_id": self.machine_id,
            "base_channels": self.base_channels,
            "dilations": list(self.dilations),
            "latent_dim": self.latent_dim,
            "use_depthwise": self.use_depthwise,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "loss_type": self.loss_type,
            "aggregation": self.aggregation,
            "top_k_patches": self.top_k_patches,
            "seed": self.seed,
        }
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        return {
            "model_path": str(model_file),
            "metadata_path": str(meta_file),
            "config_path": str(config_file),
        }

    @classmethod
    def load(cls, artifact_dir: Union[str, Path], device: str = "auto") -> "StethoMSAEDetector":
        """Instantiate and load detector from saved artifact directory."""
        art_path = Path(artifact_dir)
        config_file = art_path / "config.json"
        meta_file = art_path / "metadata.json"
        model_file = art_path / "model.pt"

        if not model_file.exists():
            raise FileNotFoundError(f"Model weights not found: {model_file}")

        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        detector = cls(
            machine_id=cfg["machine_id"],
            base_channels=cfg.get("base_channels", 32),
            dilations=tuple(cfg.get("dilations", [1, 2, 4])),
            latent_dim=cfg.get("latent_dim", 64),
            use_depthwise=cfg.get("use_depthwise", True),
            learning_rate=cfg.get("learning_rate", 0.0005),
            weight_decay=cfg.get("weight_decay", 1e-4),
            batch_size=cfg.get("batch_size", 8),
            max_epochs=cfg.get("max_epochs", 50),
            early_stopping_patience=cfg.get("early_stopping_patience", 8),
            loss_type=cfg.get("loss_type", "l1"),
            aggregation=cfg.get("aggregation", "mean"),
            top_k_patches=cfg.get("top_k_patches", 3),
            device=device,
            seed=cfg.get("seed", 42),
        )

        state_dict = torch.load(model_file, map_location=detector.device)
        detector.model.load_state_dict(state_dict)
        detector.model.eval()

        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            detector.threshold = meta.get("threshold")
            detector.train_mean = meta.get("train_mean", 0.0)
            detector.train_std = meta.get("train_std", 1.0)
            detector.training_metadata = meta

        detector.is_fitted = True
        return detector

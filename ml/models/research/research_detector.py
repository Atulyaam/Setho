"""Unified Research Anomaly Detector Wrapper.

Wraps candidate neural architectures into the BaseAnomalyDetector interface:
- MultiScaleCNNAE
- ResidualCNNAE
- CNNGRUAttentionAE
- TeacherStudentModel

Supports Phase 5 Optimization and Ablation dimensions:
- Loss functions: L1, MSE, Smooth L1, Frequency-Weighted L1
- Optimizers: Adam, AdamW
- Schedulers: Constant with Warmup, Cosine Annealing, ReduceLROnPlateau
- Normalization: GroupNorm, BatchNorm, LayerNorm
- Latent dimensions: 32, 64, 128
- Temporal context / patch frames: 64, 96, 128
- Self-supervised augmentations: time-shift, noise injection, spec_augment
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
from ml.models.research.cnn_gru_attention_ae import CNNGRUAttentionAE
from ml.models.research.multiscale_cnn_ae import MultiScaleCNNAE
from ml.models.research.residual_cnn_ae import ResidualCNNAE
from ml.models.research.teacher_student_cnn import TeacherStudentModel
from ml.preprocessing.feature_suite import UnifiedFeatureSuite

logger = logging.getLogger("stetho_research_detector")


class FrequencyWeightedL1Loss(nn.Module):
    """L1 Loss weighted proportionally towards higher frequency mel bins."""

    def __init__(self, freq_bins: int = 64, slope: float = 1.5):
        super().__init__()
        weights = 1.0 + slope * (torch.arange(freq_bins, dtype=torch.float32) / float(freq_bins))
        self.register_buffer("weights", weights.view(1, 1, freq_bins, 1))

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(recon - target)
        return torch.mean(self.weights * diff)


def _apply_augmentation(batch_x: torch.Tensor, aug_type: Optional[str]) -> torch.Tensor:
    """Apply data augmentation to normalized training patches."""
    if not aug_type or aug_type == "none":
        return batch_x
    if aug_type == "time_shift":
        shift = int(np.random.randint(-4, 5))
        return torch.roll(batch_x, shifts=shift, dims=-1)
    elif aug_type == "noise_injection":
        noise = torch.randn_like(batch_x) * 0.05
        return batch_x + noise
    elif aug_type == "spec_augment":
        b, c, f, t = batch_x.shape
        x_aug = batch_x.clone()
        f_start = np.random.randint(0, max(1, f - 4))
        x_aug[:, :, f_start : f_start + 4, :] = 0.0
        t_start = np.random.randint(0, max(1, t - 4))
        x_aug[:, :, :, t_start : t_start + 4] = 0.0
        return x_aug
    return batch_x


class ResearchNeuralDetector(BaseAnomalyDetector):
    """Unified detector wrapper for research neural architectures."""

    MODEL_CLASSES = {
        "multiscale_cnn_ae": MultiScaleCNNAE,
        "residual_cnn_ae": ResidualCNNAE,
        "cnn_gru_attention_ae": CNNGRUAttentionAE,
        "teacher_student_cnn": TeacherStudentModel,
    }

    def __init__(
        self,
        machine_id: str = "id_00",
        arch_type: str = "multiscale_cnn_ae",
        base_channels: int = 16,
        latent_dim: int = 64,
        norm_type: str = "group",
        loss_type: str = "l1",
        optimizer_type: str = "adamw",
        lr_schedule: str = "constant",
        patch_frames: int = 64,
        augmentation: Optional[str] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 8,
        max_epochs: int = 25,
        early_stopping_patience: int = 6,
        preset: str = "standard_64mel",
        seed: int = 42,
    ):
        super().__init__(machine_id=machine_id, model_type=arch_type)
        self.arch_type = arch_type
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.norm_type = norm_type
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type
        self.lr_schedule = lr_schedule
        self.patch_frames = patch_frames
        self.augmentation = augmentation
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.preset = preset
        self.seed = seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_suite = UnifiedFeatureSuite(preset_name=preset)

        input_size = (64, patch_frames)

        # Instantiate model
        if arch_type == "multiscale_cnn_ae":
            self.model = MultiScaleCNNAE(
                in_channels=1,
                base_channels=base_channels,
                latent_dim=latent_dim,
                norm_type=norm_type,
                input_size=input_size,
            )
        elif arch_type == "residual_cnn_ae":
            self.model = ResidualCNNAE(
                in_channels=1,
                base_channels=base_channels,
                latent_dim=latent_dim,
                norm_type=norm_type,
                input_size=input_size,
            )
        elif arch_type == "cnn_gru_attention_ae":
            self.model = CNNGRUAttentionAE(
                in_channels=1,
                freq_bins=64,
                time_steps=patch_frames,
                cnn_channels=base_channels,
                latent_dim=latent_dim,
            )
        elif arch_type == "teacher_student_cnn":
            self.model = TeacherStudentModel(
                in_channels=1,
                base_channels=base_channels,
                out_dim=latent_dim,
            )
        else:
            raise ValueError(f"Unknown arch_type: {arch_type}")

        self.model.to(self.device)
        self.train_mean: Optional[float] = None
        self.train_std: Optional[float] = None
        self.training_metadata: Dict[str, Any] = {}

    def _extract_patches_from_files(self, file_paths: List[Union[str, Path]]) -> np.ndarray:
        all_patches = []
        hop_frames = max(16, self.patch_frames // 2)
        for p in file_paths:
            patches, _ = self.feature_suite.extract_patches(
                p,
                patch_frames=self.patch_frames,
                hop_frames=hop_frames,
            )
            all_patches.append(patches)
        if not all_patches:
            return np.empty((0, 1, 64, self.patch_frames), dtype=np.float32)
        return np.concatenate(all_patches, axis=0)

    def _get_loss_criterion(self) -> nn.Module:
        if self.loss_type == "mse":
            return nn.MSELoss()
        elif self.loss_type == "smooth_l1":
            return nn.SmoothL1Loss(beta=1.0)
        elif self.loss_type == "freq_weighted_l1":
            return FrequencyWeightedL1Loss(freq_bins=64, slope=1.5).to(self.device)
        else:  # default "l1"
            return nn.L1Loss()

    def _get_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.optimizer_type == "adam":
            return torch.optim.Adam(params, lr=self.learning_rate)
        else:  # "adamw" default
            return torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)

    def fit(self, train_files: List[Union[str, Path]], dev_files: Optional[List[Union[str, Path]]] = None) -> Dict[str, Any]:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        t_start = time.perf_counter()

        train_patches = self._extract_patches_from_files(train_files)
        if len(train_patches) == 0:
            raise ValueError("No patches extracted from train files")

        self.train_mean = float(np.mean(train_patches))
        self.train_std = float(np.std(train_patches)) + 1e-6

        train_norm = (train_patches - self.train_mean) / self.train_std
        train_ds = TensorDataset(torch.from_numpy(train_norm).float())
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        dev_loader = None
        if dev_files:
            dev_patches = self._extract_patches_from_files(dev_files)
            dev_norm = (dev_patches - self.train_mean) / self.train_std
            dev_ds = TensorDataset(torch.from_numpy(dev_norm).float())
            dev_loader = DataLoader(dev_ds, batch_size=self.batch_size, shuffle=False)

        optimizer = self._get_optimizer()
        criterion = self._get_loss_criterion()

        # Scheduler
        scheduler = None
        if self.lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_epochs, eta_min=self.learning_rate * 0.05
            )
        elif self.lr_schedule == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
            )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self.model.train()
        for epoch in range(1, self.max_epochs + 1):
            epoch_losses = []
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()

                # Apply augmentation during training if enabled
                if self.augmentation:
                    batch_x_train = _apply_augmentation(batch_x, self.augmentation)
                else:
                    batch_x_train = batch_x

                if self.arch_type == "teacher_student_cnn":
                    _, _, err = self.model(batch_x_train)
                    loss = err.mean()
                else:
                    recon, _ = self.model(batch_x_train)
                    loss = criterion(recon, batch_x)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            val_loss = float(np.mean(epoch_losses))
            if dev_loader:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for (batch_dev,) in dev_loader:
                        batch_dev = batch_dev.to(self.device)
                        if self.arch_type == "teacher_student_cnn":
                            _, _, err = self.model(batch_dev)
                            val_losses.append(err.mean().item())
                        else:
                            recon, _ = self.model(batch_dev)
                            val_losses.append(criterion(recon, batch_dev).item())
                val_loss = float(np.mean(val_losses)) if val_losses else val_loss
                self.model.train()

            # Step scheduler
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

        self.training_metadata = {
            "machine_id": self.machine_id,
            "arch_type": self.arch_type,
            "base_channels": self.base_channels,
            "latent_dim": self.latent_dim,
            "norm_type": self.norm_type,
            "loss_type": self.loss_type,
            "optimizer_type": self.optimizer_type,
            "lr_schedule": self.lr_schedule,
            "patch_frames": self.patch_frames,
            "augmentation": self.augmentation,
            "train_samples": len(train_files),
            "train_patches": len(train_patches),
            "best_val_loss": round(best_val_loss, 5),
            "epochs_trained": epoch,
            "training_time_s": round(time.perf_counter() - t_start, 2),
            "param_count": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device),
        }
        return self.training_metadata

    def score_patches(self, patches: np.ndarray) -> np.ndarray:
        if self.train_mean is None or self.train_std is None:
            raise RuntimeError("Detector must be fitted or loaded before scoring")

        patches_norm = (patches - self.train_mean) / self.train_std
        x = torch.from_numpy(patches_norm).float().to(self.device)

        self.model.eval()
        with torch.no_grad():
            if self.arch_type == "teacher_student_cnn":
                _, _, err = self.model(x)
                return err.cpu().numpy()
            else:
                recon, _ = self.model(x)
                err = torch.mean(torch.abs(recon - x), dim=(1, 2, 3))
                return err.cpu().numpy()

    def score_file(self, file_path: Union[str, Path]) -> float:
        hop_frames = max(16, self.patch_frames // 2)
        patches, _ = self.feature_suite.extract_patches(
            file_path,
            patch_frames=self.patch_frames,
            hop_frames=hop_frames,
        )
        if len(patches) == 0:
            return 0.0
        scores = self.score_patches(patches)
        return float(np.mean(scores))

    def score_waveform(self, waveform: np.ndarray, sample_rate: int = 16000) -> float:
        if waveform.ndim > 1:
            if waveform.shape[0] > waveform.shape[1]:
                waveform = waveform[:, 0]
            else:
                waveform = waveform[0, :]
        waveform = np.ascontiguousarray(waveform, dtype=np.float32)
        peak = np.max(np.abs(waveform))
        if peak > 0:
            waveform = waveform / peak

        hop_frames = max(16, self.patch_frames // 2)
        patches, _ = self.feature_suite.extract_patches_from_waveform(
            waveform,
            patch_frames=self.patch_frames,
            hop_frames=hop_frames,
        )
        if len(patches) == 0:
            return 0.0
        scores = self.score_patches(patches)
        return float(np.mean(scores))

    def save(self, path: Union[str, Path]) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_type": self.model_type,
            "machine_id": self.machine_id,
            "arch_type": self.arch_type,
            "base_channels": self.base_channels,
            "latent_dim": self.latent_dim,
            "norm_type": self.norm_type,
            "loss_type": self.loss_type,
            "optimizer_type": self.optimizer_type,
            "lr_schedule": self.lr_schedule,
            "patch_frames": self.patch_frames,
            "augmentation": self.augmentation,
            "preset": self.preset,
            "state_dict": self.model.state_dict(),
            "train_mean": self.train_mean,
            "train_std": self.train_std,
            "training_metadata": self.training_metadata,
        }
        torch.save(checkpoint, save_path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ResearchNeuralDetector":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        detector = cls(
            machine_id=checkpoint.get("machine_id", "id_00"),
            arch_type=checkpoint.get("arch_type", "multiscale_cnn_ae"),
            base_channels=checkpoint.get("base_channels", 16),
            latent_dim=checkpoint.get("latent_dim", 64),
            norm_type=checkpoint.get("norm_type", "group"),
            loss_type=checkpoint.get("loss_type", "l1"),
            optimizer_type=checkpoint.get("optimizer_type", "adamw"),
            lr_schedule=checkpoint.get("lr_schedule", "constant"),
            patch_frames=checkpoint.get("patch_frames", 64),
            augmentation=checkpoint.get("augmentation", None),
            preset=checkpoint.get("preset", "standard_64mel"),
        )
        detector.model.load_state_dict(checkpoint["state_dict"])
        detector.train_mean = checkpoint.get("train_mean")
        detector.train_std = checkpoint.get("train_std")
        detector.training_metadata = checkpoint.get("training_metadata", {})
        detector.model.eval()
        return detector

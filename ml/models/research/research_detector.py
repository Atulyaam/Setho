"""Unified Research Anomaly Detector Wrapper.

Wraps candidate neural architectures into the BaseAnomalyDetector interface:
- MultiScaleCNNAE
- ResidualCNNAE
- CNNGRUAttentionAE
- TeacherStudentModel
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
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.preset = preset
        self.seed = seed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_suite = UnifiedFeatureSuite(preset_name=preset)

        # Instantiate model
        if arch_type == "multiscale_cnn_ae":
            self.model = MultiScaleCNNAE(in_channels=1, base_channels=base_channels, latent_dim=latent_dim)
        elif arch_type == "residual_cnn_ae":
            self.model = ResidualCNNAE(in_channels=1, base_channels=base_channels, latent_dim=latent_dim)
        elif arch_type == "cnn_gru_attention_ae":
            self.model = CNNGRUAttentionAE(in_channels=1, freq_bins=64, time_steps=64, cnn_channels=base_channels, latent_dim=latent_dim)
        elif arch_type == "teacher_student_cnn":
            self.model = TeacherStudentModel(in_channels=1, base_channels=base_channels, out_dim=latent_dim)
        else:
            raise ValueError(f"Unknown arch_type: {arch_type}")

        self.model.to(self.device)
        self.train_mean: Optional[float] = None
        self.train_std: Optional[float] = None
        self.training_metadata: Dict[str, Any] = {}

    def _extract_patches_from_files(self, file_paths: List[Union[str, Path]]) -> np.ndarray:
        all_patches = []
        for p in file_paths:
            patches, _ = self.feature_suite.extract_patches(p)
            all_patches.append(patches)
        if not all_patches:
            return np.empty((0, 1, 64, 64), dtype=np.float32)
        return np.concatenate(all_patches, axis=0)

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

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.L1Loss()

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self.model.train()
        for epoch in range(1, self.max_epochs + 1):
            epoch_losses = []
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                if self.arch_type == "teacher_student_cnn":
                    _, _, err = self.model(batch_x)
                    loss = err.mean()
                else:
                    recon, _ = self.model(batch_x)
                    loss = criterion(recon, batch_x)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            val_loss = float(np.mean(epoch_losses))
            if dev_loader:
                self.model.eval()
                v_losses = []
                with torch.no_grad():
                    for (batch_v,) in dev_loader:
                        batch_v = batch_v.to(self.device)
                        if self.arch_type == "teacher_student_cnn":
                            _, _, err = self.model(batch_v)
                            v_losses.append(err.mean().item())
                        else:
                            recon_v, _ = self.model(batch_v)
                            v_losses.append(criterion(recon_v, batch_v).item())
                val_loss = float(np.mean(v_losses))
                self.model.train()

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
        self.is_fitted = True
        duration = time.perf_counter() - t_start

        param_count = sum(p.numel() for p in self.model.parameters())
        self.training_metadata = {
            "machine_id": self.machine_id,
            "arch_type": self.arch_type,
            "duration_sec": round(duration, 2),
            "best_val_loss": round(best_val_loss, 5),
            "parameter_count": param_count,
        }
        return self.training_metadata

    def score_patches(self, patches: np.ndarray) -> float:
        if not self.is_fitted or self.train_mean is None:
            raise RuntimeError("Model must be fitted before scoring.")
        norm_patches = (patches - self.train_mean) / self.train_std

        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(norm_patches).float().to(self.device)
            if self.arch_type == "teacher_student_cnn":
                _, _, err = self.model(tensor_x)
                errors = err.cpu().numpy()
            else:
                recon, _ = self.model(tensor_x)
                diff = torch.abs(recon - tensor_x)
                errors = diff.mean(dim=(1, 2, 3)).cpu().numpy()

        return round(float(np.mean(errors)), 6)

    def score_waveform(self, waveform: np.ndarray) -> float:
        wav = waveform
        if wav.ndim > 1:
            wav = wav[:, 0]  # Channel 0
        peak = np.max(np.abs(wav))
        if peak > 1e-6:
            wav = wav / peak
        spec, _ = self.feature_suite.mel_extractor.extract_numpy(wav)
        patches, _ = self.feature_suite.windowing.slice_spectrogram_numpy(spec.squeeze(0))
        patches_4d = np.expand_dims(patches, axis=1).astype(np.float32)
        return self.score_patches(patches_4d)

    def score_file(self, file_path: Union[str, Path]) -> float:
        patches, _ = self.feature_suite.extract_patches(file_path)
        return self.score_patches(patches)

    def save(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        model_path = p / "model.pt"
        torch.save(self.model.state_dict(), model_path)

        meta = {
            "machine_id": self.machine_id,
            "arch_type": self.arch_type,
            "base_channels": self.base_channels,
            "latent_dim": self.latent_dim,
            "train_mean": self.train_mean,
            "train_std": self.train_std,
            "threshold": self.threshold,
            "preset": self.preset,
            "training_metadata": self.training_metadata,
        }
        meta_path = p / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return {"model_path": str(model_path), "metadata_path": str(meta_path)}

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> "ResearchNeuralDetector":
        p = Path(model_dir)
        with open(p / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(
            machine_id=meta["machine_id"],
            arch_type=meta["arch_type"],
            base_channels=meta["base_channels"],
            latent_dim=meta["latent_dim"],
            preset=meta.get("preset", "standard_64mel"),
        )
        instance.train_mean = meta["train_mean"]
        instance.train_std = meta["train_std"]
        instance.threshold = meta.get("threshold")
        instance.model.load_state_dict(torch.load(p / "model.pt", map_location=instance.device))
        instance.model.eval()
        instance.is_fitted = True
        return instance


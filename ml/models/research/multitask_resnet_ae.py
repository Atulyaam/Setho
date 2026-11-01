"""Multi-Task / Multi-Category Representation Autoencoder for Industrial Acoustics.

Trains a unified shared convolutional residual autoencoder across multiple industrial
categories (Fan, Pump, Valve), establishing a rich generalized acoustic prior.
Supports seamless integration with FewShotMachineCalibrator.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ml.models.base import BaseAnomalyDetector
from ml.models.research.residual_cnn_ae import ResidualCNNAE
from ml.preprocessing.feature_suite import UnifiedFeatureSuite


class MultiTaskResNetDetector(BaseAnomalyDetector):
    """Unified Multi-Category ResNet Autoencoder."""

    def __init__(
        self,
        machine_id: str = "multi_category",
        base_channels: int = 16,
        latent_dim: int = 64,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        max_epochs: int = 15,
        seed: int = 42,
    ):
        super().__init__(machine_id=machine_id, model_type="multitask_resnet_ae")
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.seed = seed
        self.patch_frames = 64
        self.patch_step = 32

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cpu")
        self.model = ResidualCNNAE(
            in_channels=1,
            base_channels=base_channels,
            latent_dim=latent_dim,
            norm_type="group",
            input_size=(64, 64),
        ).to(self.device)

        self.feature_suite = UnifiedFeatureSuite(preset_name="standard_64mel")
        self.threshold = 1.0

    def _extract_patches_from_file(self, file_path: str) -> Optional[torch.Tensor]:
        try:
            patches, _ = self.feature_suite.extract_patches(
                file_path,
                patch_frames=self.patch_frames,
                hop_frames=self.patch_step,
            )
            if len(patches) == 0:
                return None
            return torch.from_numpy(patches).float()
        except Exception:
            return None

    def fit(self, normal_files: List[str]) -> "MultiTaskResNetDetector":
        all_patches = []
        for fp in normal_files:
            p = self._extract_patches_from_file(fp)
            if p is not None:
                all_patches.append(p)

        if not all_patches:
            raise ValueError("No valid training patches extracted.")

        train_x = torch.cat(all_patches, dim=0)
        dataset = TensorDataset(train_x)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.model.train()

        for epoch in range(self.max_epochs):
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon, _ = self.model(batch_x)
                loss = F.l1_loss(recon, batch_x)
                loss.backward()
                optimizer.step()

        self.model.eval()
        return self

    def score_patches(self, patches: np.ndarray) -> np.ndarray:
        self.model.eval()
        t = torch.from_numpy(patches).float().to(self.device)
        with torch.no_grad():
            recon, _ = self.model(t)
            err = torch.abs(recon - t)
            return torch.mean(err, dim=(1, 2, 3)).cpu().numpy()

    def score_file(self, file_path: str) -> float:
        self.model.eval()
        p = self._extract_patches_from_file(file_path)
        if p is None:
            return 0.0

        p = p.to(self.device)
        with torch.no_grad():
            recon, _ = self.model(p)
            err = torch.abs(recon - p)
            patch_scores = torch.mean(err, dim=(1, 2, 3)).cpu().numpy()
        return float(np.mean(patch_scores))

    def predict(self, file_path: str) -> Dict[str, Any]:
        score = self.score_file(file_path)
        is_anomaly = score >= self.threshold
        return {
            "file_path": file_path,
            "anomaly_score": score,
            "threshold": self.threshold,
            "is_anomaly": bool(is_anomaly),
        }

    def score_waveform(self, waveform: np.ndarray, sample_rate: int = 16000) -> float:
        patches, _ = self.feature_suite.extract_patches_from_waveform(
            waveform, patch_frames=self.patch_frames, hop_frames=self.patch_step
        )
        if len(patches) == 0:
            return 0.0
        return float(np.mean(self.score_patches(patches)))

    def save(self, path: Union[str, Path]) -> None:
        torch.save({"state_dict": self.model.state_dict(), "threshold": self.threshold}, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MultiTaskResNetDetector":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        det = cls()
        det.model.load_state_dict(ckpt["state_dict"])
        det.threshold = ckpt.get("threshold", 1.0)
        return det

    def get_latent(self, file_path: str) -> Optional[np.ndarray]:
        self.model.eval()
        p = self._extract_patches_from_file(file_path)
        if p is None:
            return None
        p = p.to(self.device)
        with torch.no_grad():
            z = self.model.encode(p)
        return z.cpu().numpy()

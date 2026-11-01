"""Variational Residual CNN Autoencoder (VAE-ResNet) for Acoustic Anomaly Detection.

Introduces probabilistic latent regularization:
- Reparameterized latent distribution z ~ N(mu, sigma^2)
- Analytical KL-divergence loss pushing latent distribution toward N(0, I)
- Bottleneck dropout for robust representation learning
- Smooth, continuous latent space for enhanced out-of-distribution calibration
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
from ml.models.research.residual_cnn_ae import ConvResBlock
from ml.preprocessing.feature_suite import UnifiedFeatureSuite


class VariationalResNetAE(nn.Module):
    """Residual CNN Variational Autoencoder."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        latent_dim: int = 64,
        norm_type: str = "group",
        input_size: Tuple[int, int] = (64, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.norm_type = norm_type
        self.input_size = input_size

        freq_down = input_size[0] // 8
        time_down = input_size[1] // 8
        self.spatial_dim = (freq_down, time_down)

        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.c4 = c4

        # Encoder
        self.in_conv = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.res1 = ConvResBlock(c1, norm_type=norm_type)

        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.res2 = ConvResBlock(c2, norm_type=norm_type)

        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.res3 = ConvResBlock(c3, norm_type=norm_type)

        self.down3 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1)
        self.res4 = ConvResBlock(c4, norm_type=norm_type)

        flattened_dim = c4 * freq_down * time_down
        self.fc_mu = nn.Linear(flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(flattened_dim, latent_dim)
        self.drop = nn.Dropout(p=dropout)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, flattened_dim)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=4, stride=2, padding=1)
        self.dec_res3 = ConvResBlock(c3, norm_type=norm_type)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1)
        self.dec_res2 = ConvResBlock(c2, norm_type=norm_type)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1)
        self.dec_res1 = ConvResBlock(c1, norm_type=norm_type)

        self.out_conv = nn.Conv2d(c1, in_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.res1(F.leaky_relu(self.in_conv(x), 0.2))
        h = self.res2(F.leaky_relu(self.down1(h), 0.2))
        h = self.res3(F.leaky_relu(self.down2(h), 0.2))
        h = self.res4(F.leaky_relu(self.down3(h), 0.2))
        flat = h.view(x.shape[0], -1)
        mu = self.fc_mu(flat)
        logvar = torch.clamp(self.fc_logvar(flat), min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.drop(z)
        h = F.leaky_relu(self.fc_dec(z), 0.2).view(
            -1, self.c4, self.spatial_dim[0], self.spatial_dim[1]
        )
        h = self.dec_res3(F.leaky_relu(self.up3(h), 0.2))
        h = self.dec_res2(F.leaky_relu(self.up2(h), 0.2))
        h = self.dec_res1(F.leaky_relu(self.up1(h), 0.2))
        return self.out_conv(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class VariationalResNetDetector(BaseAnomalyDetector):
    """Anomaly Detector wrapper for VariationalResNetAE."""

    def __init__(
        self,
        machine_id: str = "id_00",
        base_channels: int = 16,
        latent_dim: int = 64,
        beta_kl: float = 0.001,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 16,
        max_epochs: int = 15,
        seed: int = 42,
    ):
        super().__init__(machine_id=machine_id, model_type="vae_resnet_ae")
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.beta_kl = beta_kl
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.seed = seed
        self.patch_frames = 64
        self.patch_step = 32

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cpu")
        self.model = VariationalResNetAE(
            in_channels=1,
            base_channels=base_channels,
            latent_dim=latent_dim,
            norm_type="group",
            input_size=(64, 64),
            dropout=dropout,
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

    def fit(self, normal_files: List[str]) -> "VariationalResNetDetector":
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
                recon, mu, logvar = self.model(batch_x)
                recon_loss = F.l1_loss(recon, batch_x)
                kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                loss = recon_loss + self.beta_kl * kl_loss
                loss.backward()
                optimizer.step()

        self.model.eval()
        return self

    def score_file(self, file_path: str) -> float:
        self.model.eval()
        p = self._extract_patches_from_file(file_path)
        if p is None:
            return 0.0

        p = p.to(self.device)
        with torch.no_grad():
            recon, mu, _ = self.model(p)
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
        feats = self.feature_suite.extract(waveform)
        mel = feats["log_mel"]
        if mel.shape[1] < self.patch_frames:
            mel = np.pad(mel, ((0, 0), (0, self.patch_frames - mel.shape[1])), mode="reflect")
        patches = [mel[:, i : i + self.patch_frames] for i in range(0, mel.shape[1] - self.patch_frames + 1, self.patch_step)]
        if not patches:
            return 0.0
        t = torch.tensor(np.array(patches), dtype=torch.float32).unsqueeze(1).to(self.device)
        with torch.no_grad():
            recon, mu, _ = self.model(t)
            err = torch.abs(recon - t)
            scores = torch.mean(err, dim=(1, 2, 3)).cpu().numpy()
        return float(np.mean(scores))

    def save(self, path: Union[str, Path]) -> None:
        torch.save({"state_dict": self.model.state_dict(), "threshold": self.threshold}, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "VariationalResNetDetector":
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
            mu, _ = self.model.encode(p)
        return mu.cpu().numpy()

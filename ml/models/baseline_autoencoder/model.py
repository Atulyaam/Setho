"""Dense Autoencoder Neural Baseline Model for Stetho.

A lightweight multi-layer perceptron autoencoder operating on flattened
time-frequency log-Mel spectrogram patches (64 bins x 64 frames = 4096 dims).
Provides a reference neural baseline for reconstruction-based acoustic anomaly detection.
"""

from typing import List, Tuple
import torch
import torch.nn as nn


class DenseAutoencoder(nn.Module):
    """Dense bottleneck autoencoder for spectrogram patch reconstruction."""

    def __init__(
        self,
        input_dim: int = 4096,  # 64 Mel bins * 64 time frames
        hidden_dims: List[int] = None,
        latent_dim: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 128]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # Encoder
        enc_layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            enc_layers.append(nn.Linear(curr_dim, h_dim))
            enc_layers.append(nn.BatchNorm1d(h_dim))
            enc_layers.append(nn.LeakyReLU(0.1))
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        enc_layers.append(nn.Linear(curr_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        curr_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers.append(nn.Linear(curr_dim, h_dim))
            dec_layers.append(nn.BatchNorm1d(h_dim))
            dec_layers.append(nn.LeakyReLU(0.1))
            if dropout > 0:
                dec_layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        dec_layers.append(nn.Linear(curr_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Project input patches to latent representation."""
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct input patches from latent representation."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (reconstructed_x, latent_z)."""
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def compute_reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute sample-wise L1 reconstruction error: |x - x_recon|.

        Args:
            x: Input tensor of shape (batch, 4096) or (batch, 1, 64, 64).
            reduction: "mean" gives scalar error per patch; "none" gives (batch, 4096).

        Returns:
            Per-sample reconstruction error tensor of shape (batch,).
        """
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        x_recon, _ = self.forward(x)
        diff = torch.abs(x - x_recon)
        if reduction == "mean":
            return torch.mean(diff, dim=-1)
        return diff


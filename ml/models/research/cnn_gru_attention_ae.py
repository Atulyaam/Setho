"""CNN-GRU-Attention Autoencoder for Industrial Acoustic Dynamics.

Features:
- Hybrid spatial-temporal architecture
- 2D CNN frontend extracting spectral representations per time step
- Bidirectional GRU modeling sequential mechanical cycles and transitions
- Temporal multi-head or scaled dot-product attention
- GRU + Transposed CNN decoder reconstructing time-frequency spectrogram
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSelfAttention(nn.Module):
    """Scaled dot-product temporal attention over time steps."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = 1.0 / (hidden_dim ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_weights = F.softmax(torch.bmm(q, k.transpose(1, 2)) * self.scale, dim=-1)
        out = torch.bmm(attn_weights, v)
        return out + x


class CNNGRUAttentionAE(nn.Module):
    """CNN-GRU with Temporal Attention Autoencoder."""

    def __init__(
        self,
        in_channels: int = 1,
        freq_bins: int = 64,
        time_steps: int = 64,
        cnn_channels: int = 32,
        gru_hidden_dim: int = 64,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.freq_bins = freq_bins
        self.time_steps = time_steps
        self.latent_dim = latent_dim

        # 1. 2D CNN Frontend: (B, 1, F, T) -> reduces F to compact feature per time step
        # Downsample frequency from 64 -> 32 -> 16 -> 8
        self.frontend = nn.Sequential(
            nn.Conv2d(in_channels, cnn_channels, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cnn_channels, cnn_channels * 2, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cnn_channels * 2, cnn_channels * 2, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Spatial F is now 64 / 8 = 8. Flattening F * channels = 8 * 64 = 512
        conv_out_dim = (freq_bins // 8) * (cnn_channels * 2)

        # 2. Bidirectional GRU Encoder
        self.gru_enc = nn.GRU(
            input_size=conv_out_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # 3. Temporal Self Attention
        self.attention = TemporalSelfAttention(gru_hidden_dim * 2)

        # 4. Latent projection: (B, T, 2*H) -> (B, latent_dim)
        self.to_latent = nn.Linear(gru_hidden_dim * 2, latent_dim)
        self.from_latent = nn.Linear(latent_dim, gru_hidden_dim * 2)

        # 5. GRU Decoder
        self.gru_dec = nn.GRU(
            input_size=gru_hidden_dim * 2,
            hidden_size=conv_out_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # 6. CNN Backend (upsample F back from 8 -> 16 -> 32 -> 64)
        self.backend = nn.Sequential(
            nn.ConvTranspose2d(cnn_channels * 2, cnn_channels * 2, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(cnn_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(cnn_channels * 2, cnn_channels, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(cnn_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(cnn_channels, in_channels, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 1, F, T)
        b, c, f, t = x.shape
        feat = self.frontend(x)  # (B, C_out, F', T) where F'=8
        feat = feat.permute(0, 3, 1, 2).contiguous().view(b, t, -1)  # (B, T, D)
        gru_out, _ = self.gru_enc(feat)  # (B, T, 2*H)
        attn_out = self.attention(gru_out)  # (B, T, 2*H)
        # Global pooled latent vector
        z = self.to_latent(attn_out.mean(dim=1))  # (B, latent_dim)
        return z, attn_out

    def decode(self, z: torch.Tensor, t_len: int = 64) -> torch.Tensor:
        b = z.shape[0]
        h_expanded = self.from_latent(z).unsqueeze(1).repeat(1, t_len, 1)  # (B, T, 2*H)
        dec_gru_out, _ = self.gru_dec(h_expanded)  # (B, T, conv_out_dim)
        # Reshape to (B, C, F', T)
        c_feat = 64
        f_feat = 8
        reshaped = dec_gru_out.view(b, t_len, c_feat, f_feat).permute(0, 2, 3, 1).contiguous()
        recon = self.backend(reshaped)
        return recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z, _ = self.encode(x)
        recon = self.decode(z, t_len=x.shape[3])
        return recon, z


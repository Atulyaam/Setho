"""Residual CNN Autoencoder (ResNet-AE) for Industrial Acoustic Anomaly Detection.

Features:
- Symmetric Encoder-Decoder with pure residual convolutional blocks
- Pre-activation or post-activation GroupNorm + LeakyReLU
- Progressive channel doubling (24 -> 48 -> 96 -> 192)
- Compact latent projection (64 dimensions)
- Direct residual shortcuts within each scale stage
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvResBlock(nn.Module):
    """Standard 2D Convolutional Residual Block with GroupNorm."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=min(4, channels), num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=min(4, channels), num_channels=channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + res)


class ResidualCNNAE(nn.Module):
    """Deep Residual CNN Autoencoder."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.latent_dim = latent_dim

        # Encoder: 64x64 -> 32x32 -> 16x16 -> 8x8 -> 1x1
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.in_conv = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.res1 = ConvResBlock(c1)

        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.res2 = ConvResBlock(c2)

        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.res3 = ConvResBlock(c3)

        self.down3 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1)
        self.res4 = ConvResBlock(c4)

        # Bottleneck: 8x8x192 -> latent_dim
        self.to_latent = nn.Conv2d(c4, latent_dim, kernel_size=8)
        self.from_latent = nn.ConvTranspose2d(latent_dim, c4, kernel_size=8)

        # Decoder: 8x8 -> 16x16 -> 32x32 -> 64x64
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=4, stride=2, padding=1)
        self.dec_res3 = ConvResBlock(c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1)
        self.dec_res2 = ConvResBlock(c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1)
        self.dec_res1 = ConvResBlock(c1)

        self.out_conv = nn.Conv2d(c1, in_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.res1(F.leaky_relu(self.in_conv(x), 0.2))
        h = self.res2(F.leaky_relu(self.down1(h), 0.2))
        h = self.res3(F.leaky_relu(self.down2(h), 0.2))
        h = self.res4(F.leaky_relu(self.down3(h), 0.2))
        z = self.to_latent(h).squeeze(-1).squeeze(-1)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z.unsqueeze(-1).unsqueeze(-1))
        h = self.dec_res3(F.leaky_relu(self.up3(h), 0.2))
        h = self.dec_res2(F.leaky_relu(self.up2(h), 0.2))
        h = self.dec_res1(F.leaky_relu(self.up1(h), 0.2))
        return self.out_conv(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

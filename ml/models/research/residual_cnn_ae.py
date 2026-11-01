"""Residual CNN Autoencoder (ResNet-AE) for Industrial Acoustic Anomaly Detection.

Features:
- Symmetric Encoder-Decoder with pure residual convolutional blocks
- Configurable normalization: GroupNorm, BatchNorm, or LayerNorm
- Progressive channel doubling (16 -> 32 -> 64 -> 128)
- Configurable latent projection (32, 64, 128 dimensions)
- Supports arbitrary temporal patch widths (64, 96, 128 frames)
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_norm_layer(norm_type: str, channels: int) -> nn.Module:
    """Helper to return specified normalization layer."""
    if norm_type == "batch":
        return nn.BatchNorm2d(channels)
    elif norm_type == "layer":
        return nn.GroupNorm(num_groups=1, num_channels=channels)
    else:  # "group" default
        return nn.GroupNorm(num_groups=min(4, channels), num_channels=channels)


class ConvResBlock(nn.Module):
    """Standard 2D Convolutional Residual Block with configurable normalization."""

    def __init__(self, channels: int, norm_type: str = "group"):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = _get_norm_layer(norm_type, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _get_norm_layer(norm_type, channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + res)


class ResidualCNNAE(nn.Module):
    """Deep Residual CNN Autoencoder with configurable components."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        latent_dim: int = 64,
        norm_type: str = "group",
        input_size: Tuple[int, int] = (64, 64),
    ):
        super().__init__()
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.norm_type = norm_type
        self.input_size = input_size

        freq_down = input_size[0] // 8
        time_down = input_size[1] // 8

        # Encoder: HxW -> (H/2)x(W/2) -> (H/4)x(W/4) -> (H/8)x(W/8)
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.in_conv = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.res1 = ConvResBlock(c1, norm_type=norm_type)

        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.res2 = ConvResBlock(c2, norm_type=norm_type)

        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.res3 = ConvResBlock(c3, norm_type=norm_type)

        self.down3 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1)
        self.res4 = ConvResBlock(c4, norm_type=norm_type)

        # Bottleneck: (c4, freq_down, time_down) -> latent_dim
        if freq_down == 8 and time_down == 8:
            self.to_latent = nn.Conv2d(c4, latent_dim, kernel_size=8)
            self.from_latent = nn.ConvTranspose2d(latent_dim, c4, kernel_size=8)
            self.use_linear_bottleneck = False
        else:
            self.fc_enc = nn.Linear(c4 * freq_down * time_down, latent_dim)
            self.fc_dec = nn.Linear(latent_dim, c4 * freq_down * time_down)
            self.use_linear_bottleneck = True

        # Decoder: (H/8)x(W/8) -> (H/4)x(W/4) -> (H/2)x(W/2) -> HxW
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=4, stride=2, padding=1)
        self.dec_res3 = ConvResBlock(c3, norm_type=norm_type)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1)
        self.dec_res2 = ConvResBlock(c2, norm_type=norm_type)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1)
        self.dec_res1 = ConvResBlock(c1, norm_type=norm_type)

        self.out_conv = nn.Conv2d(c1, in_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.res1(F.leaky_relu(self.in_conv(x), 0.2))
        h = self.res2(F.leaky_relu(self.down1(h), 0.2))
        h = self.res3(F.leaky_relu(self.down2(h), 0.2))
        h = self.res4(F.leaky_relu(self.down3(h), 0.2))
        if not self.use_linear_bottleneck:
            z = self.to_latent(h).squeeze(-1).squeeze(-1)
        else:
            z = self.fc_enc(h.view(x.shape[0], -1))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        c4 = self.base_channels * 8
        if not self.use_linear_bottleneck:
            h = self.from_latent(z.unsqueeze(-1).unsqueeze(-1))
        else:
            freq_down = self.input_size[0] // 8
            time_down = self.input_size[1] // 8
            h = F.leaky_relu(self.fc_dec(z), 0.2).view(-1, c4, freq_down, time_down)

        h = self.dec_res3(F.leaky_relu(self.up3(h), 0.2))
        h = self.dec_res2(F.leaky_relu(self.up2(h), 0.2))
        h = self.dec_res1(F.leaky_relu(self.up1(h), 0.2))
        return self.out_conv(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

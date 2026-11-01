"""Multi-Scale Multi-Kernel Depthwise-Separable CNN Autoencoder for Industrial Acoustic Anomaly Detection.

Features:
- Multi-scale branches with complementary kernel geometries:
  - 3x3: isotropic local acoustic texture
  - 3x5: elongated temporal kernel capturing sustained motor harmonics and blade pass
  - 5x3: elongated spectral kernel capturing broadband mechanical transients and impact bursts
- Depthwise-separable convolutions for compact parameter footprint
- Residual skip connections with GroupNorm stabilization
- Latent bottleneck projection (64-128 dim)
- Lightweight mirror decoder
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiKernelDepthwiseConv(nn.Module):
    """Parallel depthwise-separable conv branches with 3x3, 3x5, and 5x3 kernels."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        branch_channels = out_channels // 3
        rem = out_channels - (branch_channels * 2)

        # Branch 1: 3x3
        self.b1_dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.b1_pw = nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False)

        # Branch 2: 3x5 (freq=3, time=5)
        self.b2_dw = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 5), padding=(1, 2), groups=in_channels, bias=False)
        self.b2_pw = nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False)

        # Branch 3: 5x3 (freq=5, time=3)
        self.b3_dw = nn.Conv2d(in_channels, in_channels, kernel_size=(5, 3), padding=(2, 1), groups=in_channels, bias=False)
        self.b3_pw = nn.Conv2d(in_channels, rem, kernel_size=1, bias=False)

        self.norm = nn.GroupNorm(num_groups=min(4, out_channels), num_channels=out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.b1_pw(self.b1_dw(x))
        out2 = self.b2_pw(self.b2_dw(x))
        out3 = self.b3_pw(self.b3_dw(x))
        merged = torch.cat([out1, out2, out3], dim=1)
        return self.act(self.norm(merged))


class MultiScaleResBlock(nn.Module):
    """Residual block combining MultiKernelDepthwiseConv with shortcut."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = MultiKernelDepthwiseConv(channels, channels)
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(4, channels), num_channels=channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.conv1(x)
        out = self.act(self.norm(self.pw(self.dw(out))))
        return out + res


class MultiScaleCNNAE(nn.Module):
    """Multi-Scale Multi-Kernel CNN Autoencoder with smooth Upsampling."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        latent_dim: int = 64,
        input_size: Tuple[int, int] = (64, 64),
    ):
        super().__init__()
        self.input_size = input_size
        self.base_channels = base_channels
        self.latent_dim = latent_dim

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        # Stem: (B, 1, 64, 64) -> (B, c1, 64, 64)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(4, c1), c1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 1: 64x64 -> 32x32
        self.stage1 = MultiScaleResBlock(c1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Stage 2: 32x32 -> 16x16
        self.proj1_2 = nn.Conv2d(c1, c2, kernel_size=1, bias=False)
        self.stage2 = MultiScaleResBlock(c2)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Stage 3: 16x16 -> 8x8
        self.proj2_3 = nn.Conv2d(c2, c3, kernel_size=1, bias=False)
        self.stage3 = MultiScaleResBlock(c3)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bottleneck: 8x8xc3 -> latent_dim
        self.fc_enc = nn.Linear(c3 * 8 * 8, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, c3 * 8 * 8)

        # Decoder Stage 3: 8x8 -> 16x16
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_stage3 = MultiScaleResBlock(c3)
        self.dec_proj3_2 = nn.Conv2d(c3, c2, kernel_size=1, bias=False)

        # Decoder Stage 2: 16x16 -> 32x32
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_stage2 = MultiScaleResBlock(c2)
        self.dec_proj2_1 = nn.Conv2d(c2, c1, kernel_size=1, bias=False)

        # Decoder Stage 1: 32x32 -> 64x64
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_stage1 = MultiScaleResBlock(c1)

        # Head: (B, c1, 64, 64) -> (B, 1, 64, 64)
        self.head = nn.Conv2d(c1, in_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.pool1(self.stage1(h))
        h = self.pool2(self.stage2(self.proj1_2(h)))
        h = self.pool3(self.stage3(self.proj2_3(h)))
        h_flat = h.view(x.shape[0], -1)
        z = self.fc_enc(h_flat)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        c3 = self.base_channels * 4
        h = F.leaky_relu(self.fc_dec(z), 0.2).view(-1, c3, 8, 8)
        h = self.dec_proj3_2(self.dec_stage3(self.up3(h)))
        h = self.dec_proj2_1(self.dec_stage2(self.up2(h)))
        h = self.dec_stage1(self.up1(h))
        recon = self.head(h)
        return recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


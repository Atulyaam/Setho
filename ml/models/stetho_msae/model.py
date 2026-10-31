"""Stetho-MSAE: Shared Multi-Scale Depthwise-Separable CNN Autoencoder.

Custom neural architecture for machine-specific acoustic anomaly detection:
- Single log-Mel frontend representation (64 x 64 patches)
- Depthwise-separable convolutions for compact representation & low parameter footprint
- Multi-scale temporal dilations (1, 2, 4) in residual blocks to capture micro-transients and periodic motor cycles
- Compact bottleneck latent space (64 or 128 dimensions)
- Lightweight CNN decoder reconstructing clean normal acoustic envelopes
- Reconstruction-based anomaly discrimination via sample-wise L1 error
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise-Separable 2D Convolution with dilation and normalization.
    
    Splits spatial/temporal filtering (depthwise) from channel combination (pointwise)
    to achieve 8-9x reduction in convolution parameters while preserving receptive field.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        use_depthwise: bool = True,
        num_groups: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.use_depthwise = use_depthwise

        # Effective padding to maintain spatial dimensions when stride=1
        if stride == 1 and padding == "same":
            pad_val = (kernel_size - 1) * dilation // 2
        else:
            pad_val = padding

        if use_depthwise:
            # Depthwise convolution: filters each input channel independently
            self.depthwise = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad_val,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            )
            # Pointwise convolution: 1x1 projection across channels
            self.pointwise = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        else:
            # Standard convolution for ablation comparison
            self.depthwise = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad_val,
                dilation=dilation,
                bias=False,
            )
            self.pointwise = nn.Identity()

        # GroupNorm is robust to small batch sizes (e.g. batch_size = 4, 8)
        groups = min(num_groups, out_channels)
        while out_channels % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.norm(out)
        out = self.act(out)
        return out


class ResidualMSAEBlock(nn.Module):
    """Multi-Scale Residual Block with dilated depthwise-separable convolutions.
    
    Implements:
    x_res = Conv_d(x) -> Conv_d(x_res)
    output = LeakyReLU(Shortcut(x) + x_res)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
        use_depthwise: bool = True,
        num_groups: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation

        # Padding matches dilation to keep spatial dimensions identical
        pad = dilation

        self.conv1 = DepthwiseSeparableConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=pad,
            dilation=dilation,
            use_depthwise=use_depthwise,
            num_groups=num_groups,
        )
        self.conv2 = DepthwiseSeparableConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=pad,
            dilation=dilation,
            use_depthwise=use_depthwise,
            num_groups=num_groups,
        )

        # 1x1 projection shortcut if channel dimensions change
        if in_channels != out_channels:
            groups = min(num_groups, out_channels)
            while out_channels % groups != 0 and groups > 1:
                groups -= 1
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(groups, out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.out_act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv1(x)
        res = self.conv2(res)
        shortcut = self.shortcut(x)
        return self.out_act(shortcut + res)


class MSAEEncoder(nn.Module):
    """Multi-Scale Dilated Depthwise-Separable Encoder.
    
    Input: (B, 1, 64, 64)
    Output: Latent vector z (B, latent_dim)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        dilations: Tuple[int, ...] = (1, 2, 4),
        latent_dim: int = 64,
        use_depthwise: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.dilations = dilations

        # Initial projection from single audio channel to base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(min(4, base_channels), base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Multi-scale residual stages with progressive temporal dilation
        # Stage 1: dilation[0] (default: 1) -> 64x64 to 32x32
        self.stage1 = ResidualMSAEBlock(
            base_channels, base_channels, dilation=dilations[0], use_depthwise=use_depthwise
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # 64x64 -> 32x32

        # Stage 2: dilation[1] (default: 2) -> 32x32 to 16x16
        c2 = base_channels * 2  # 64
        self.stage2 = ResidualMSAEBlock(
            base_channels, c2, dilation=dilations[1], use_depthwise=use_depthwise
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16

        # Stage 3: dilation[2] (default: 4) -> 16x16 to 8x8
        c3 = base_channels * 2  # 64
        self.stage3 = ResidualMSAEBlock(
            c2, c3, dilation=dilations[2], use_depthwise=use_depthwise
        )
        self.pool3 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8

        # Latent projection: (c3 * 8 * 8) -> latent_dim
        self.flatten_dim = c3 * 8 * 8  # 64 * 8 * 8 = 4096
        self.fc_latent = nn.Linear(self.flatten_dim, latent_dim)
        self.latent_norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h = self.pool1(self.stage1(h))  # (B, 32, 32, 32)
        h = self.pool2(self.stage2(h))  # (B, 64, 16, 16)
        h = self.pool3(self.stage3(h))  # (B, 64, 8, 8)
        
        flat = torch.flatten(h, 1)
        z = self.latent_norm(self.fc_latent(flat))
        return z, h


class MSAEDecoder(nn.Module):
    """Lightweight CNN Decoder reconstructing 2D log-Mel spectrogram patches.
    
    Input: Latent vector z (B, latent_dim)
    Output: Reconstructed patch (B, 1, 64, 64)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        base_channels: int = 32,
        out_channels: int = 1,
        use_depthwise: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.c3 = base_channels * 2  # 64

        # Project latent vector back to spatial feature volume (64 x 8 x 8)
        self.fc_unflat = nn.Linear(latent_dim, self.c3 * 8 * 8)
        self.act_unflat = nn.LeakyReLU(0.2, inplace=True)

        # Stage 1: 8x8 -> 16x16
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = DepthwiseSeparableConv2d(
            self.c3, self.c3, kernel_size=3, padding=1, use_depthwise=use_depthwise
        )

        # Stage 2: 16x16 -> 32x32
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = DepthwiseSeparableConv2d(
            self.c3, base_channels, kernel_size=3, padding=1, use_depthwise=use_depthwise
        )

        # Stage 3: 32x32 -> 64x64
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3 = DepthwiseSeparableConv2d(
            base_channels, base_channels, kernel_size=3, padding=1, use_depthwise=use_depthwise
        )

        # Final reconstruction to single-channel log-Mel
        self.recon_head = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.act_unflat(self.fc_unflat(z))
        h = h.view(-1, self.c3, 8, 8)
        h = self.dec1(self.up1(h))  # (B, 64, 16, 16)
        h = self.dec2(self.up2(h))  # (B, 32, 32, 32)
        h = self.dec3(self.up3(h))  # (B, 32, 64, 64)
        recon = self.recon_head(h)   # (B, 1, 64, 64)
        return recon


class StethoMSAE(nn.Module):
    """Complete Stetho-MSAE Autoencoder Model.
    
    Supports controlled ablation variants:
    - use_depthwise=True: Depthwise-separable convolutions
    - use_depthwise=False: Standard full 2D convolutions (Ablation B)
    - dilations=(1, 2, 4): Multi-scale temporal dilation (Full Stetho-MSAE)
    - dilations=(1, 1, 1): No multi-scale dilation (Ablation B & C)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        dilations: Tuple[int, ...] = (1, 2, 4),
        latent_dim: int = 64,
        use_depthwise: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.dilations = tuple(dilations)
        self.latent_dim = latent_dim
        self.use_depthwise = use_depthwise

        self.encoder = MSAEEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            dilations=self.dilations,
            latent_dim=latent_dim,
            use_depthwise=use_depthwise,
        )
        self.decoder = MSAEDecoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            out_channels=in_channels,
            use_depthwise=use_depthwise,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.encoder(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: returns (reconstructed_patch, latent_z)."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def compute_reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute sample-wise L1 reconstruction error across spatial dimensions.
        
        Args:
            x: Input tensor (B, 1, 64, 64) or (B, 64, 64)
            reduction: 'mean' or 'none'
            
        Returns:
            Tensor of shape (B,) representing L1 error per patch.
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)
        recon, _ = self.forward(x)
        # L1 error per patch: mean over (channel, freq, time)
        diff = torch.abs(recon - x)
        sample_errors = torch.mean(diff, dim=(1, 2, 3))
        return sample_errors

    def count_parameters(self) -> Dict[str, int]:
        """Compute trainable and total parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        return {
            "total_params": total,
            "trainable_params": trainable,
            "encoder_params": encoder_params,
            "decoder_params": decoder_params,
        }


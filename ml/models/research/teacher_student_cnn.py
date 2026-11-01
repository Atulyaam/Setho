"""Teacher-Student CNN Anomaly Detector for Industrial Acoustic Anomaly Detection.

Features:
- Fixed random/pretrained feature teacher network T
- Trainable student network S learning to predict teacher representations on normal sound
- During inference, abnormal patterns cause high predictive error between Teacher and Student
- Highly complementary to reconstruction-based autoencoders
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CompactFeatureNet(nn.Module):
    """3-layer convolutional feature network for Teacher and Student."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16, out_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(min(4, base_channels), base_channels)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(min(4, base_channels * 2), base_channels * 2)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.GroupNorm(min(4, base_channels * 4), base_channels * 4)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
        h = F.leaky_relu(self.norm2(self.conv2(h)), 0.2)
        h = F.leaky_relu(self.norm3(self.conv3(h)), 0.2)
        h = self.pool(h).squeeze(-1).squeeze(-1)
        return self.fc(h)


class TeacherStudentModel(nn.Module):
    """Teacher-Student discrepancy model."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16, out_dim: int = 64):
        super().__init__()
        self.teacher = CompactFeatureNet(in_channels, base_channels, out_dim)
        self.student = CompactFeatureNet(in_channels, base_channels, out_dim)

        # Freeze teacher weights
        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            t_feat = self.teacher(x)
        s_feat = self.student(x)
        error = F.mse_loss(s_feat, t_feat, reduction="none").mean(dim=-1)
        return s_feat, t_feat, error


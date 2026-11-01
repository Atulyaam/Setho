"""Automated verification tests for Phase 7 Distillation & Margin Cascade."""

from pathlib import Path
import numpy as np
import pytest
import torch

from ml.models.base import BaseAnomalyDetector
from ml.models.research.distilled_student import CompactStudentAE, DistillationLoss
from ml.models.research.margin_cascade import MarginGatedCascade


class MockDetector(BaseAnomalyDetector):
    """Simple mock detector for cascade policy testing."""

    def __init__(self, machine_id: str, fixed_score: float):
        super().__init__(machine_id=machine_id, model_type="mock")
        self.fixed_score = fixed_score

    def fit(self, train_files, dev_files=None):
        return {}

    def score_file(self, file_path) -> float:
        return self.fixed_score

    def score_waveform(self, waveform: np.ndarray, sample_rate: int = 16000) -> float:
        return self.fixed_score

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        return cls("id_00", 0.0)


def test_compact_student_forward_backward():
    """Verify CompactStudentAE shape, compactness, and gradients."""
    model = CompactStudentAE(in_channels=1, base_channels=8, latent_dim=32)
    x = torch.randn(2, 1, 64, 64)

    recon, z = model(x)
    assert recon.shape == (2, 1, 64, 64)
    assert z.shape == (2, 32)

    loss = torch.nn.functional.l1_loss(recon, x)
    loss.backward()

    # Verify gradients
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Gradient missing for {name}"

    param_count = sum(p.numel() for p in model.parameters())
    assert param_count < 150_000, f"Student parameter count {param_count} too large"


def test_distillation_loss():
    """Verify DistillationLoss multi-target computation."""
    criterion = DistillationLoss(teacher_latent_dim=64, student_latent_dim=32)

    s_recon = torch.randn(2, 1, 64, 64, requires_grad=True)
    t_recon = torch.randn(2, 1, 64, 64)
    s_z = torch.randn(2, 32, requires_grad=True)
    t_z = torch.randn(2, 64)
    target = torch.randn(2, 1, 64, 64)

    loss, meta = criterion(s_recon, t_recon, s_z, t_z, target)
    assert loss.item() > 0.0
    assert "loss_recon" in meta
    assert "loss_mimic" in meta
    assert "loss_latent" in meta

    loss.backward()
    assert s_recon.grad is not None
    assert s_z.grad is not None


def test_margin_cascade_routing():
    """Verify MarginGatedCascade early exit vs escalation."""
    student = MockDetector("id_00", fixed_score=0.50)
    teacher = MockDetector("id_00", fixed_score=0.95)

    cascade = MarginGatedCascade(
        machine_id="id_00",
        student_detector=student,
        teacher_detector=teacher,
        margin=0.10,
    )
    cascade.student_threshold = 0.50  # score = 0.50, dist = 0.0 < margin 0.10 -> Ambiguous!
    cascade.teacher_threshold = 0.60

    # 1. Ambiguous case -> should escalate to teacher
    score, info = cascade.score_file_with_telemetry("ambiguous.wav")
    assert info["escalated"] is True
    assert info["stage"] == 2
    assert score == 0.95

    # 2. Confident case -> should early exit
    student.fixed_score = 0.20  # dist = |0.20 - 0.50| = 0.30 >= margin 0.10 -> Confident!
    score_conf, info_conf = cascade.score_file_with_telemetry("confident.wav")
    assert info_conf["escalated"] is False
    assert info_conf["stage"] == 1
    assert score_conf == 0.20


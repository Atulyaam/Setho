"""Adaptive Routing and Escalation Policy for Unified Acoustic Inference.

Implements Stage 3, Stage 4, Stage 5, and Stage 6 of Sprint 8.1:
- Input quality verification
- Fast Student first-pass inference
- Confidence margin gating
- Category-aware specialist model selection
- Machine-specific residual whitening
- Uncertainty-aware decision zone evaluation
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("stetho_adaptive_router")


class RoutingPath(str, Enum):
    FAST_STUDENT_EXIT = "FAST_STUDENT_EXIT"
    SPECIALIST_ESCALATED = "SPECIALIST_ESCALATED"
    RESIDUAL_WHITENED = "RESIDUAL_WHITENED"
    QUALITY_REJECTED = "QUALITY_REJECTED"


class DecisionStatus(str, Enum):
    NORMAL = "NORMAL"
    ANOMALY = "ANOMALY"
    UNCERTAIN = "UNCERTAIN"


class DecisionConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class RoutingPolicyConfig:
    """Configurable margins and thresholds for adaptive routing."""
    cascade_margin: float = 0.05  # Margin gamma for student fast exit
    uncertainty_relative_margin: float = 0.06  # +/- 6% around threshold is UNCERTAIN
    patch_variance_uncertainty_threshold: float = 0.40  # High patch dispersion triggers UNCERTAIN
    student_threshold: float = 0.37356
    fan_specialist_threshold: float = 0.35208
    multitask_threshold: float = 0.35208
    enable_direct_calibrated_routing: bool = False  # If True, calibrated assets bypass student


@dataclass
class RoutingResult:
    """Outcome of adaptive routing and decision evaluation."""
    path: RoutingPath
    status: DecisionStatus
    confidence: DecisionConfidence
    primary_score: float
    operating_threshold: float
    models_executed: int
    models_used: List[str]
    escalation_reason: Optional[str] = None
    uncertainty_reason: Optional[str] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)


class AdaptiveRouter:
    """Evaluates multi-stage acoustic inference and routes dynamically."""

    def __init__(self, config: Optional[RoutingPolicyConfig] = None):
        self.config = config or RoutingPolicyConfig()

    def evaluate_student_stage(
        self,
        student_score: float,
        has_calibration_profile: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """Determine whether the sample can early-exit at the fast student stage.
        
        Returns:
            (can_exit: bool, reason: Optional[str])
        """
        # If machine has calibrated residual whitening, always escalate to specialist to apply whitening
        if has_calibration_profile:
            return False, "MACHINE_CALIBRATION_PROFILE_ACTIVE"

        delta = abs(student_score - self.config.student_threshold)
        if delta >= self.config.cascade_margin:
            return True, f"HIGH_CONFIDENCE_STUDENT_EXIT (delta={delta:.4f} >= gamma={self.config.cascade_margin})"
        
        return False, f"AMBIGUOUS_STUDENT_MARGIN (delta={delta:.4f} < gamma={self.config.cascade_margin})"

    def select_specialist_model_type(self, machine_category: str) -> str:
        """Select appropriate specialist model based on physical machinery category."""
        clean_cat = (machine_category or "universal").lower().strip()
        if clean_cat == "fan":
            return "resnet_ae_specialist"
        elif clean_cat in ["pump", "valve"]:
            return "multitask_resnet_ae"
        else:
            return "multitask_resnet_ae"

    def determine_decision_and_confidence(
        self,
        score: float,
        threshold: float,
        patch_scores: np.ndarray,
        route_path: RoutingPath,
        quality_warnings: List[str],
    ) -> Tuple[DecisionStatus, DecisionConfidence, Optional[str]]:
        """Compute three-zone status (NORMAL, ANOMALY, UNCERTAIN) without forcing binary choices."""
        dist = abs(score - threshold)
        rel_dist = dist / max(threshold, 1e-6)

        # Compute spatial-temporal patch variance / dispersion
        patch_std = float(np.std(patch_scores)) if len(patch_scores) > 1 else 0.0
        patch_mean = float(np.mean(patch_scores)) if len(patch_scores) > 1 else 1.0
        cv = patch_std / max(patch_mean, 1e-6)  # Coefficient of variation

        # 1. Quality Warning Penalty
        if quality_warnings:
            if rel_dist < self.config.uncertainty_relative_margin * 2.0:
                return (
                    DecisionStatus.UNCERTAIN,
                    DecisionConfidence.LOW,
                    f"ADVERSE_ENVIRONMENT_BORDERLINE: {'; '.join(quality_warnings)}",
                )

        # 2. Borderline Threshold Uncertainty Zone
        if rel_dist < self.config.uncertainty_relative_margin:
            return (
                DecisionStatus.UNCERTAIN,
                DecisionConfidence.LOW,
                f"BORDERLINE_SCORE: Score {score:.4f} is within {rel_dist*100:.1f}% of threshold {threshold:.4f}",
            )

        # 3. High Patch Dispersion / Acoustic Flutter Warning
        if cv > self.config.patch_variance_uncertainty_threshold and rel_dist < 0.15:
            return (
                DecisionStatus.UNCERTAIN,
                DecisionConfidence.MEDIUM,
                f"HIGH_PATCH_DISPERSION: Coefficient of variation {cv:.2f} indicates transient acoustic bursts",
            )

        # 4. Confident Classifications
        is_anomaly = score >= threshold
        if rel_dist >= 0.20:
            confidence = DecisionConfidence.HIGH
        elif rel_dist >= 0.10:
            confidence = DecisionConfidence.MEDIUM
        else:
            confidence = DecisionConfidence.LOW

        status = DecisionStatus.ANOMALY if is_anomaly else DecisionStatus.NORMAL
        return status, confidence, None

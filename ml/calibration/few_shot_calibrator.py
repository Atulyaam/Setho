"""Few-Shot Machine Calibration Engine for Industrial Acoustic Anomaly Detection.

Implements lightweight, rapid, leakage-safe adaptation for previously unseen physical machines:
1. Uncalibrated Baseline
2. Mean-Variance Residual Whitening (Spatial-Spectral Bias Subtraction)
3. Mahalanobis Latent Centering & Covariance Distance
4. Normal-Reference Latent Memory Bank (k-NN Distance)

Supports sample sweeps with N in {2, 5, 10} normal baseline recordings.
Never accesses abnormal or test-set evaluation recordings during calibration.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ml.models.base import BaseAnomalyDetector
from ml.preprocessing.feature_suite import UnifiedFeatureSuite

logger = logging.getLogger("stetho_few_shot_calibration")


@dataclass
class FewShotCalibrationResult:
    """Telemetry and metrics from few-shot machine calibration."""
    machine_id: str
    method: str
    n_shots: int
    threshold: float
    calibration_time_s: float
    calibrated_scores: List[float]
    metadata: Dict[str, Any]


class FewShotMachineCalibrator:
    """Per-machine rapid statistical calibrator for unseen physical assets."""

    CALIBRATION_METHODS = [
        "uncalibrated",
        "mean_var_residual",
        "mahalanobis_latent",
        "memory_bank_knn",
    ]

    def __init__(
        self,
        base_detector: BaseAnomalyDetector,
        method: str = "mean_var_residual",
        n_shots: int = 5,
        target_quantile: float = 0.95,
        feature_preset: str = "standard_64mel",
        device: Optional[torch.device] = None,
    ):
        self.detector = base_detector
        self.method = method if method in self.CALIBRATION_METHODS else "mean_var_residual"
        self.n_shots = n_shots
        self.target_quantile = target_quantile
        self.feature_suite = UnifiedFeatureSuite(preset_name=feature_preset)
        self.device = device or torch.device("cpu")

        # Calibrated parameters
        self.is_calibrated: bool = False
        self.operating_threshold: float = getattr(base_detector, "threshold", 0.35)
        self.baseline_residual_mean: Optional[np.ndarray] = None
        self.baseline_residual_std: Optional[np.ndarray] = None
        self.latent_centroid: Optional[np.ndarray] = None
        self.latent_precision: Optional[np.ndarray] = None
        self.memory_bank: Optional[np.ndarray] = None  # (K, D)
        self.calibration_meta: Dict[str, Any] = {}

    def _extract_patches_for_files(self, file_paths: List[Union[str, Path]]) -> np.ndarray:
        """Extract all 64x64 patches across the given audio files."""
        all_patches = []
        for p in file_paths:
            patches, _ = self.feature_suite.extract_patches(
                p, patch_frames=64, hop_frames=32
            )
            if len(patches) > 0:
                all_patches.append(patches)
        if not all_patches:
            raise ValueError("No patches could be extracted from calibration files.")
        return np.concatenate(all_patches, axis=0)

    def calibrate(self, normal_file_paths: List[Union[str, Path]]) -> FewShotCalibrationResult:
        """Calibrate the model on N normal recordings from the target machine."""
        t_start = time.perf_counter()
        
        # Select N shots
        cal_paths = normal_file_paths[:self.n_shots]
        actual_shots = len(cal_paths)
        if actual_shots == 0:
            raise ValueError("Calibration requires at least 1 normal recording.")

        if self.method == "uncalibrated":
            # Just compute default scores and default threshold
            scores = [self.detector.score_file(p) for p in cal_paths]
            self.operating_threshold = float(np.percentile(scores, self.target_quantile * 100))
            self.is_calibrated = True
            cal_time = time.perf_counter() - t_start
            return FewShotCalibrationResult(
                machine_id=self.detector.machine_id,
                method="uncalibrated",
                n_shots=actual_shots,
                threshold=round(self.operating_threshold, 5),
                calibration_time_s=round(cal_time, 4),
                calibrated_scores=[round(s, 5) for s in scores],
                metadata={"status": "uncalibrated_baseline"},
            )

        # Extract normal calibration patches
        patches = self._extract_patches_for_files(cal_paths)
        t_mean = getattr(self.detector, "train_mean", None)
        if t_mean is None:
            t_mean = -42.5
        t_std = getattr(self.detector, "train_std", None)
        if t_std is None or t_std == 0.0:
            t_std = 18.2
        norm_patches = (patches - t_mean) / t_std
        x_t = torch.from_numpy(norm_patches).float().to(self.device)

        model = getattr(self.detector, "model", None)
        if model is None:
            raise RuntimeError("Base detector model not found.")
        model.to(self.device).eval()

        with torch.no_grad():
            out = model(x_t)
            if isinstance(out, (tuple, list)):
                recon = out[0]
                latent = out[1]
            else:
                recon = out
                latent = getattr(model, "last_latent", None)
            recon_np = recon.cpu().numpy()
            x_np = x_t.cpu().numpy()
            latent_np = latent.cpu().numpy() if latent is not None else np.zeros((x_np.shape[0], 64))

        # Element-wise absolute reconstruction error per patch: (B, 1, 64, 64)
        residuals = np.abs(recon_np - x_np)

        if self.method == "mean_var_residual":
            # Compute spatial-spectral residual bias per time-frequency bin across normal clips
            # Shape: (1, 1, 64, 64)
            self.baseline_residual_mean = np.mean(residuals, axis=0, keepdims=True)
            self.baseline_residual_std = np.std(residuals, axis=0, keepdims=True) + 1e-4

        elif self.method == "mahalanobis_latent":
            # Latent representations: (B, D)
            # Flatten if latent is spatial
            if latent_np.ndim > 2:
                latent_np = latent_np.reshape(latent_np.shape[0], -1)
            self.latent_centroid = np.mean(latent_np, axis=0)  # (D,)
            cov = np.cov(latent_np, rowvar=False)
            dim = cov.shape[0]
            # Regularize covariance to avoid singularity on small N
            reg_cov = cov + (1e-3 * np.eye(dim))
            try:
                self.latent_precision = np.linalg.pinv(reg_cov)
            except Exception as e:
                logger.warning(f"Covariance inversion failed, using identity: {e}")
                self.latent_precision = np.eye(dim)

        elif self.method == "memory_bank_knn":
            # Subsample normal latent representations into a reference bank
            if latent_np.ndim > 2:
                latent_np = latent_np.reshape(latent_np.shape[0], -1)
            # Cap memory bank size at 500 representative patches
            max_bank = min(500, len(latent_np))
            indices = np.linspace(0, len(latent_np) - 1, max_bank, dtype=int)
            self.memory_bank = latent_np[indices]

        self.is_calibrated = True

        # Now compute calibrated scores on the calibration set to derive the operating threshold
        cal_scores = [self.score_file(p) for p in cal_paths]
        thresh = float(np.percentile(cal_scores, self.target_quantile * 100))
        self.operating_threshold = max(1e-4, thresh)
        cal_time = time.perf_counter() - t_start

        return FewShotCalibrationResult(
            machine_id=self.detector.machine_id,
            method=self.method,
            n_shots=actual_shots,
            threshold=round(self.operating_threshold, 5),
            calibration_time_s=round(cal_time, 4),
            calibrated_scores=[round(s, 5) for s in cal_scores],
            metadata={
                "cal_patches_count": len(patches),
                "threshold_quantile": self.target_quantile,
            },
        )

    fit = calibrate

    def score_patches(self, patches: np.ndarray) -> np.ndarray:
        """Score patches using the calibrated method."""
        if not self.is_calibrated:
            # Fallback to detector uncalibrated scoring
            return self.detector.score_patches(patches)

        t_mean = getattr(self.detector, "train_mean", None)
        if t_mean is None:
            t_mean = -42.5
        t_std = getattr(self.detector, "train_std", None)
        if t_std is None or t_std == 0.0:
            t_std = 18.2
        norm_patches = (patches - t_mean) / t_std
        x_t = torch.from_numpy(norm_patches).float().to(self.device)

        model = self.detector.model
        model.eval()
        with torch.no_grad():
            out = model(x_t)
            if isinstance(out, (tuple, list)):
                recon = out[0]
                latent = out[1]
            else:
                recon = out
                latent = getattr(model, "last_latent", None)
            recon_np = recon.cpu().numpy()
            x_np = x_t.cpu().numpy()
            latent_np = latent.cpu().numpy() if latent is not None else np.zeros((x_np.shape[0], 64))

        residuals = np.abs(recon_np - x_np)

        if self.method == "uncalibrated":
            return np.mean(residuals, axis=(1, 2, 3))

        elif self.method == "mean_var_residual":
            # Subtract baseline machine resonance and normalize residual
            # Whitened residual: (B, 1, 64, 64)
            whitened = np.maximum(0.0, (residuals - self.baseline_residual_mean) / self.baseline_residual_std)
            return np.mean(whitened, axis=(1, 2, 3))

        elif self.method == "mahalanobis_latent":
            if latent_np.ndim > 2:
                latent_np = latent_np.reshape(latent_np.shape[0], -1)
            diff = latent_np - self.latent_centroid
            # D_M^2 = sum_j (diff * P * diff)
            d_sq = np.sum((diff @ self.latent_precision) * diff, axis=1)
            d_m = np.sqrt(np.maximum(0.0, d_sq))
            recon_err = np.mean(residuals, axis=(1, 2, 3))
            # Hybrid combination
            return 0.5 * d_m + 0.5 * recon_err

        elif self.method == "memory_bank_knn":
            if latent_np.ndim > 2:
                latent_np = latent_np.reshape(latent_np.shape[0], -1)
            # Compute pairwise Euclidean distances to memory bank (B, K)
            diff = latent_np[:, np.newaxis, :] - self.memory_bank[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=-1)  # (B, K)
            if dists.shape[1] >= 2:
                sorted_dists = np.sort(dists, axis=1)
                return sorted_dists[:, 1]
            return np.min(dists, axis=1)

        return np.mean(residuals, axis=(1, 2, 3))

    def score_file(self, file_path: Union[str, Path]) -> float:
        """Score an audio file using calibrated scoring."""
        patches, _ = self.feature_suite.extract_patches(
            file_path, patch_frames=64, hop_frames=32
        )
        if len(patches) == 0:
            return 0.0
        patch_scores = self.score_patches(patches)
        # Use top-10% mean to capture localized anomaly bursts
        k = max(1, int(len(patch_scores) * 0.10))
        top_scores = np.sort(patch_scores)[-k:]
        return float(np.mean(top_scores))

    def predict_file(self, file_path: Union[str, Path]) -> Tuple[int, float]:
        """Predict binary label (0: normal, 1: anomaly) and anomaly score."""
        score = self.score_file(file_path)
        is_anomaly = 1 if score >= self.operating_threshold else 0
        return is_anomaly, score

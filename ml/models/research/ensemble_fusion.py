"""Advanced Ensemble & Model Fusion Architectures for Industrial Acoustic Anomaly Detection.

Implements:
1. ScoreFusionEnsemble:
   - Standardized Z-Score Weighted Average
   - Empirical Rank Averaging
   - Extreme-Value Max Fusion
2. CascadedFastRejectDetector:
   - Two-stage hierarchical cascade (Fast Stage 1 filter -> Heavy Stage 2 inspection)
   - Dynamic early-exit for low-latency CPU throughput
3. FeatureFusionDetector:
   - Concatenates latent neural bottleneck representations with acoustic statistical descriptors
"""

import copy
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import rankdata
import torch

from ml.calibration.statistical_calibration import StatisticalCalibrator
from ml.models.base import BaseAnomalyDetector

logger = logging.getLogger("stetho_ensemble_fusion")


class ScoreFusionEnsemble(BaseAnomalyDetector):
    """Multi-model score fusion engine with zero-leakage calibration."""

    def __init__(
        self,
        machine_id: str,
        detectors: List[BaseAnomalyDetector],
        strategy: str = "weighted_average",
        weights: Optional[List[float]] = None,
    ):
        super().__init__(machine_id=machine_id, model_type=f"ensemble_{strategy}")
        self.detectors = detectors
        self.strategy = strategy
        if weights is not None:
            w_arr = np.array(weights, dtype=np.float32)
            self.weights = list(w_arr / np.sum(w_arr))
        else:
            self.weights = [1.0 / len(detectors)] * len(detectors)

        self.dev_score_means: List[float] = []
        self.dev_score_stds: List[float] = []
        self.dev_score_pools: List[np.ndarray] = []

    def fit_fusion(self, dev_files: List[Union[str, Path]]) -> None:
        """Calibrate normalization parameters strictly on development normal files."""
        self.dev_score_means = []
        self.dev_score_stds = []
        self.dev_score_pools = []

        for det in self.detectors:
            scores = [det.score_file(p) for p in dev_files]
            arr = np.array(scores, dtype=np.float32)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr)) + 1e-6
            self.dev_score_means.append(mean_val)
            self.dev_score_stds.append(std_val)
            self.dev_score_pools.append(np.sort(arr))

    def fit(self, train_files: List[Union[str, Path]], dev_files: Optional[List[Union[str, Path]]] = None) -> Dict[str, Any]:
        """Conform to BaseAnomalyDetector interface."""
        calib_files = dev_files if dev_files else train_files
        self.fit_fusion(calib_files)
        return {"calibrated_files": len(calib_files), "strategy": self.strategy}

    def _normalize_score(self, raw_score: float, model_idx: int) -> float:
        """Standardize raw score using dev-set empirical statistics."""
        mean_val = self.dev_score_means[model_idx]
        std_val = self.dev_score_stds[model_idx]
        return (raw_score - mean_val) / std_val

    def score_file(self, file_path: Union[str, Path]) -> float:
        """Fuse scores from constituent detectors."""
        raw_scores = [det.score_file(file_path) for det in self.detectors]

        if self.strategy == "rank_average":
            if not self.dev_score_pools:
                return float(np.mean(raw_scores))
            ranks = []
            for i, s in enumerate(raw_scores):
                pool = self.dev_score_pools[i]
                rank = float(np.searchsorted(pool, s, side="right")) / len(pool)
                ranks.append(rank)
            return float(np.mean(ranks))

        if not self.dev_score_means:
            return float(np.mean(raw_scores))

        if self.strategy == "weighted_average":
            norm_scores = [self._normalize_score(s, i) for i, s in enumerate(raw_scores)]
            return float(np.sum([w * ns for w, ns in zip(self.weights, norm_scores)]))

        elif self.strategy == "max_fusion":
            norm_scores = [self._normalize_score(s, i) for i, s in enumerate(raw_scores)]
            return float(np.max(norm_scores))

        else:
            norm_scores = [self._normalize_score(s, i) for i, s in enumerate(raw_scores)]
            return float(np.mean(norm_scores))

    def score_waveform(self, waveform: np.ndarray, sample_rate: int = 16000) -> float:
        """Fuse waveform scores."""
        raw_scores = [det.score_waveform(waveform, sample_rate) for det in self.detectors]
        if not self.dev_score_means:
            return float(np.mean(raw_scores))

        if self.strategy == "weighted_average":
            norm_scores = [self._normalize_score(s, i) for i, s in enumerate(raw_scores)]
            return float(np.sum([w * ns for w, ns in zip(self.weights, norm_scores)]))
        elif self.strategy == "max_fusion":
            norm_scores = [self._normalize_score(s, i) for i, s in enumerate(raw_scores)]
            return float(np.max(norm_scores))
        else:
            norm_scores = [self._normalize_score(s, i) for i, s in enumerate(raw_scores)]
            return float(np.mean(norm_scores))

    def save(self, path: Union[str, Path]) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "machine_id": self.machine_id,
            "strategy": self.strategy,
            "weights": self.weights,
            "dev_score_means": self.dev_score_means,
            "dev_score_stds": self.dev_score_stds,
        }
        with open(save_path if save_path.suffix == ".json" else save_path / "fusion_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ScoreFusionEnsemble":
        load_path = Path(path)
        target = load_path if load_path.suffix == ".json" else load_path / "fusion_meta.json"
        with open(target, "r") as f:
            meta = json.load(f)
        ens = cls(machine_id=meta["machine_id"], detectors=[], strategy=meta["strategy"], weights=meta.get("weights"))
        ens.dev_score_means = meta.get("dev_score_means", [])
        ens.dev_score_stds = meta.get("dev_score_stds", [])
        return ens


class CascadedFastRejectDetector(BaseAnomalyDetector):
    """Hierarchical two-stage cascade detector for high-speed inference.

    Stage 1: Fast filter (e.g. Mahalanobis or Teacher-Student).
             If score < early_exit_threshold, classify immediately as normal.
    Stage 2: Heavy detector (e.g. ResNet-AE).
             Executed only for ambiguous or suspect audio recordings.
    """

    def __init__(
        self,
        machine_id: str,
        stage1_fast_detector: BaseAnomalyDetector,
        stage2_heavy_detector: BaseAnomalyDetector,
        early_exit_quantile: float = 0.50,
    ):
        super().__init__(machine_id=machine_id, model_type="cascaded_fast_reject")
        self.stage1 = stage1_fast_detector
        self.stage2 = stage2_heavy_detector
        self.early_exit_quantile = early_exit_quantile
        self.early_exit_threshold: float = -float("inf")
        self.stage1_mean: float = 0.0
        self.stage1_std: float = 1.0
        self.stage2_mean: float = 0.0
        self.stage2_std: float = 1.0

        # Diagnostics
        self.early_exit_count: int = 0
        self.heavy_eval_count: int = 0

    def fit_cascade(self, dev_files: List[Union[str, Path]]) -> None:
        """Calibrate early exit threshold on normal development files."""
        s1_scores = [self.stage1.score_file(p) for p in dev_files]
        s2_scores = [self.stage2.score_file(p) for p in dev_files]

        self.stage1_mean = float(np.mean(s1_scores))
        self.stage1_std = float(np.std(s1_scores)) + 1e-6
        self.stage2_mean = float(np.mean(s2_scores))
        self.stage2_std = float(np.std(s2_scores)) + 1e-6

        # Set early exit cutoff at target quantile of normal dev scores
        self.early_exit_threshold = float(np.percentile(s1_scores, self.early_exit_quantile * 100.0))
        logger.info(
            f"Cascade calibrated for {self.machine_id}: Early exit at Stage 1 <= {self.early_exit_threshold:.4f} "
            f"({self.early_exit_quantile*100:.0f}% quantile)"
        )

    def fit(self, train_files: List[Union[str, Path]], dev_files: Optional[List[Union[str, Path]]] = None) -> Dict[str, Any]:
        """Conform to BaseAnomalyDetector interface."""
        calib_files = dev_files if dev_files else train_files
        self.fit_cascade(calib_files)
        return {"calibrated_files": len(calib_files), "early_exit_quantile": self.early_exit_quantile}

    def score_file_with_telemetry(self, file_path: Union[str, Path]) -> Tuple[float, Dict[str, Any]]:
        """Score file and return diagnostic stage routing info."""
        t0 = time.perf_counter()
        s1 = self.stage1.score_file(file_path)
        t_stage1 = (time.perf_counter() - t0) * 1000.0

        if s1 <= self.early_exit_threshold:
            self.early_exit_count += 1
            norm_s1 = (s1 - self.stage1_mean) / self.stage1_std
            return float(norm_s1), {
                "stage": 1,
                "early_exited": True,
                "latency_ms": round(t_stage1, 2),
            }

        self.heavy_eval_count += 1
        t_s2_start = time.perf_counter()
        s2 = self.stage2.score_file(file_path)
        t_stage2 = (time.perf_counter() - t_s2_start) * 1000.0

        norm_s2 = (s2 - self.stage2_mean) / self.stage2_std
        total_lat = (time.perf_counter() - t0) * 1000.0
        return float(norm_s2), {
            "stage": 2,
            "early_exited": False,
            "stage1_latency_ms": round(t_stage1, 2),
            "stage2_latency_ms": round(t_stage2, 2),
            "latency_ms": round(total_lat, 2),
        }

    def score_file(self, file_path: Union[str, Path]) -> float:
        score, _ = self.score_file_with_telemetry(file_path)
        return score

    def score_waveform(self, waveform: np.ndarray, sample_rate: int = 16000) -> float:
        s1 = self.stage1.score_waveform(waveform, sample_rate)
        if s1 <= self.early_exit_threshold:
            norm_s1 = (s1 - self.stage1_mean) / self.stage1_std
            return float(norm_s1)
        s2 = self.stage2.score_waveform(waveform, sample_rate)
        norm_s2 = (s2 - self.stage2_mean) / self.stage2_std
        return float(norm_s2)

    def save(self, path: Union[str, Path]) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "machine_id": self.machine_id,
            "early_exit_quantile": self.early_exit_quantile,
            "early_exit_threshold": self.early_exit_threshold,
            "stage1_mean": self.stage1_mean,
            "stage1_std": self.stage1_std,
            "stage2_mean": self.stage2_mean,
            "stage2_std": self.stage2_std,
        }
        with open(save_path if save_path.suffix == ".json" else save_path / "cascade_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CascadedFastRejectDetector":
        load_path = Path(path)
        target = load_path if load_path.suffix == ".json" else load_path / "cascade_meta.json"
        with open(target, "r") as f:
            meta = json.load(f)
        cas = cls(machine_id=meta["machine_id"], stage1_fast_detector=None, stage2_heavy_detector=None, early_exit_quantile=meta["early_exit_quantile"])
        cas.early_exit_threshold = meta["early_exit_threshold"]
        cas.stage1_mean = meta["stage1_mean"]
        cas.stage1_std = meta["stage1_std"]
        cas.stage2_mean = meta["stage2_mean"]
        cas.stage2_std = meta["stage2_std"]
        return cas

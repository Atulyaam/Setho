"""Machine-Specific Statistical Calibration Engine for Stetho.

Establishes operating anomaly decision thresholds using normal development recordings only.
Strictly isolates calibration from final-test evaluation to prevent data leakage.
Records calibration sample count and explicitly marks small-sample thresholds as provisional.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ml.models.base import BaseAnomalyDetector

logger = logging.getLogger("stetho_calibration")


@dataclass
class CalibrationProfile:
    """Statistical calibration metadata and calibrated threshold."""
    machine_id: str
    model_type: str
    target_quantile: float
    operating_threshold: float
    dev_sample_count: int
    dev_scores: List[float]
    dev_mean: float
    dev_std: float
    dev_min: float
    dev_max: float
    dev_p90: float
    dev_p95: float
    dev_p99: float
    is_provisional: bool
    limitation_notice: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "CalibrationProfile":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class StatisticalCalibrator:
    """Calibrates machine-specific detection threshold on development normal audio."""

    def __init__(self, target_quantile: float = 0.95):
        self.target_quantile = target_quantile

    def calibrate(
        self,
        detector: BaseAnomalyDetector,
        dev_files: List[Union[str, Path]],
    ) -> CalibrationProfile:
        """Score normal development files and compute quantile threshold."""
        if not dev_files:
            raise ValueError("Development files list cannot be empty for calibration")

        logger.info(
            f"Calibrating {detector.model_type} for machine {detector.machine_id} "
            f"on {len(dev_files)} normal development files..."
        )

        dev_scores = []
        for f in dev_files:
            score = detector.score_file(f)
            dev_scores.append(float(score))

        scores_arr = np.array(dev_scores, dtype=np.float64)
        threshold_val = float(np.percentile(scores_arr, self.target_quantile * 100))

        # Flag small sample sizes as provisional
        is_provisional = len(dev_files) < 10
        limitation_notice = (
            f"Provisional empirical quantile ({self.target_quantile * 100:.0f}th percentile). "
            f"Calibration sample size N={len(dev_files)} is small. "
            "Does not guarantee an exact real-world False Positive Rate."
            if is_provisional
            else f"Calibrated {self.target_quantile * 100:.0f}th percentile operating threshold."
        )

        profile = CalibrationProfile(
            machine_id=detector.machine_id,
            model_type=detector.model_type,
            target_quantile=self.target_quantile,
            operating_threshold=round(threshold_val, 6),
            dev_sample_count=len(dev_files),
            dev_scores=[round(s, 6) for s in dev_scores],
            dev_mean=round(float(np.mean(scores_arr)), 6),
            dev_std=round(float(np.std(scores_arr)), 6),
            dev_min=round(float(np.min(scores_arr)), 6),
            dev_max=round(float(np.max(scores_arr)), 6),
            dev_p90=round(float(np.percentile(scores_arr, 90)), 6),
            dev_p95=round(float(np.percentile(scores_arr, 95)), 6),
            dev_p99=round(float(np.percentile(scores_arr, 99)), 6),
            is_provisional=is_provisional,
            limitation_notice=limitation_notice,
        )

        # Update detector threshold
        detector.threshold = profile.operating_threshold
        logger.info(
            f"Calibration complete for {detector.machine_id}: threshold={profile.operating_threshold:.5f} "
            f"(Provisional={is_provisional})"
        )
        return profile


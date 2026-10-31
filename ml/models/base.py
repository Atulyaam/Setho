"""Unified Anomaly Detector Interface for Stetho."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class BaseAnomalyDetector(ABC):
    """Abstract base class for all machine-specific anomaly detectors."""

    def __init__(self, machine_id: str, model_type: str):
        self.machine_id = machine_id
        self.model_type = model_type
        self.threshold: Optional[float] = None
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, train_files: List[Union[str, Path]], **kwargs) -> Dict[str, Any]:
        """Fit model on normal training recordings only. Returns training metadata."""
        pass

    @abstractmethod
    def score_waveform(self, waveform: np.ndarray) -> float:
        """Compute continuous anomaly score for a 1D 16 kHz audio waveform.
        HIGHER score MUST strictly mean MORE ANOMALOUS.
        """
        pass

    @abstractmethod
    def score_file(self, file_path: Union[str, Path]) -> float:
        """Compute continuous anomaly score for an audio file on disk."""
        pass

    def predict(self, score: float) -> bool:
        """Binary decision: True = anomalous, False = normal.
        Requires threshold to be calibrated.
        """
        if self.threshold is None:
            raise RuntimeError("Threshold must be calibrated before calling predict()")
        return score > self.threshold

    @abstractmethod
    def save(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """Serialize model, preprocessors, and metadata to directory."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, model_dir: Union[str, Path]) -> "BaseAnomalyDetector":
        """Deserialize detector from directory."""
        pass


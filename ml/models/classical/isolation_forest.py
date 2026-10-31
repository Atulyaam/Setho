"""Isolation Forest Classical Baseline Anomaly Detector for Stetho.

Trains an unsupervised ensemble of isolation trees on compact statistical
acoustic features extracted from normal machine recordings only.
Higher anomaly score strictly denotes greater deviation from the normal acoustic envelope.
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from ml.features.acoustic_features import AcousticFeatureConfig, AcousticFeatureExtractor, AcousticFeatureScaler
from ml.models.base import BaseAnomalyDetector

logger = logging.getLogger("stetho_isolation_forest")


class IsolationForestDetector(BaseAnomalyDetector):
    """Machine-specific Isolation Forest anomaly detector."""

    def __init__(
        self,
        machine_id: str = "id_00",
        n_estimators: int = 200,
        contamination: Union[str, float] = "auto",
        max_features: float = 1.0,
        random_state: int = 42,
        feature_config: Optional[AcousticFeatureConfig] = None,
    ):
        super().__init__(machine_id=machine_id, model_type="isolation_forest")
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_features = max_features
        self.random_state = random_state

        self.feature_extractor = AcousticFeatureExtractor(feature_config)
        self.scaler = AcousticFeatureScaler()
        self.model: Optional[IsolationForest] = None
        self.training_metadata: Dict[str, Any] = {}

    def fit(self, train_files: List[Union[str, Path]], **kwargs) -> Dict[str, Any]:
        """Fit Isolation Forest on normal training recordings only."""
        if not train_files:
            raise ValueError("train_files list cannot be empty")

        start_time = time.perf_counter()
        logger.info(f"Extracting acoustic features from {len(train_files)} training files for {self.machine_id}...")

        features_list = []
        for f in train_files:
            feat = self.feature_extractor.extract_from_file(f)
            features_list.append(feat)

        X_train = np.stack(features_list, axis=0)  # Shape (N, 38)

        # Fit scaler exclusively on training data
        X_train_scaled = self.scaler.fit_transform(X_train)

        logger.info(f"Training Isolation Forest ({self.n_estimators} estimators) on {X_train_scaled.shape}...")
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train_scaled)
        self.is_fitted = True

        duration = time.perf_counter() - start_time
        self.training_metadata = {
            "machine_id": self.machine_id,
            "model_type": self.model_type,
            "train_sample_count": len(train_files),
            "feature_dim": X_train.shape[1],
            "training_duration_sec": round(duration, 4),
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
        }
        logger.info(f"Training complete in {duration:.2f}s")
        return self.training_metadata

    def score_waveform(self, waveform: np.ndarray) -> float:
        """Calculate continuous anomaly score for 1D 16 kHz waveform.
        Negates score_samples so that higher score strictly equals more anomalous.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before scoring")

        raw_feat = self.feature_extractor.extract_from_waveform(waveform)
        scaled_feat = self.scaler.transform(raw_feat).reshape(1, -1)

        # IsolationForest.score_samples() returns opposite of anomaly score (e.g. -0.6 is anomalous, -0.4 is normal)
        # We negate it so higher score = more anomalous
        score = -float(self.model.score_samples(scaled_feat)[0])
        return score

    def score_file(self, file_path: Union[str, Path]) -> float:
        """Process file and calculate continuous anomaly score."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before scoring")

        raw_feat = self.feature_extractor.extract_from_file(file_path)
        scaled_feat = self.scaler.transform(raw_feat).reshape(1, -1)
        score = -float(self.model.score_samples(scaled_feat)[0])
        return score

    def save(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """Serialize detector artifacts to directory."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save unfitted detector")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        model_file = out_path / "isolation_forest.joblib"
        scaler_file = out_path / "scaler.joblib"
        meta_file = out_path / "metadata.json"

        joblib.dump(self.model, model_file)
        self.scaler.save(scaler_file)

        config_data = {
            "machine_id": self.machine_id,
            "model_type": self.model_type,
            "n_estimators": self.n_estimators,
            "contamination": str(self.contamination),
            "max_features": self.max_features,
            "random_state": self.random_state,
            "threshold": self.threshold,
            "training_metadata": self.training_metadata,
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        return {
            "model": str(model_file),
            "scaler": str(scaler_file),
            "metadata": str(meta_file),
        }

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> "IsolationForestDetector":
        """Load detector from saved directory."""
        p = Path(model_dir)
        meta_file = p / "metadata.json"
        model_file = p / "isolation_forest.joblib"
        scaler_file = p / "scaler.joblib"

        if not meta_file.exists() or not model_file.exists() or not scaler_file.exists():
            raise FileNotFoundError(f"Missing required artifact in {model_dir}")

        with open(meta_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        inst = cls(
            machine_id=cfg["machine_id"],
            n_estimators=cfg["n_estimators"],
            contamination=cfg["contamination"] if cfg["contamination"] != "auto" else "auto",
            max_features=cfg["max_features"],
            random_state=cfg["random_state"],
        )
        inst.model = joblib.load(model_file)
        inst.scaler = AcousticFeatureScaler.load(scaler_file)
        inst.threshold = cfg.get("threshold")
        inst.training_metadata = cfg.get("training_metadata", {})
        inst.is_fitted = True
        return inst


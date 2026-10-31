"""Experiment Configuration and Reproducibility System for Stetho Sprint 2.

Defines unified configuration schemas, hash-based data provenance,
system metadata collection, and deterministic execution parameters.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional, Union
import yaml

import torch


def compute_file_hash(path: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file for data provenance tracking."""
    p = Path(path)
    if not p.exists():
        return "missing"
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def collect_system_metadata() -> Dict[str, Any]:
    """Capture runtime software and hardware environment metadata."""
    cuda_avail = bool(torch.cuda.is_available())
    device_name = str(torch.cuda.get_device_name(0)) if cuda_avail else "CPU"
    return {
        "python_version": str(sys.version.split()[0]),
        "torch_version": str(torch.__version__),
        "platform": str(platform.platform()),
        "processor": str(platform.processor()),
        "cuda_available": cuda_avail,
        "device_name": device_name,
    }


@dataclass
class FeatureConfig:
    """Feature extraction parameters."""
    feature_type: str = "log_mel"  # "log_mel" or "statistical"
    sample_rate: int = 16000
    channel_policy: str = "channel_0"
    channel_index: int = 0
    # Log-Mel specific parameters
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 64
    f_min: float = 50.0
    f_max: float = 8000.0
    spec_window_frames: int = 64
    spec_hop_frames: int = 32
    # Statistical / Classical specific parameters
    n_mfcc: int = 13
    include_centroid: bool = True
    include_bandwidth: bool = True
    include_rolloff: bool = True
    include_zcr: bool = True
    include_rms: bool = True


@dataclass
class ModelHyperparameters:
    """Hyperparameters for baseline models."""
    model_type: str = "isolation_forest"  # "isolation_forest" or "dense_autoencoder"
    # Isolation Forest
    n_estimators: int = 200
    contamination: Union[str, float] = "auto"
    max_features: float = 1.0
    # Dense Autoencoder
    latent_dim: int = 32
    hidden_dims: List[int] = field(default_factory=lambda: [512, 128])
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 8
    max_epochs: int = 50
    early_stopping_patience: int = 8
    loss: str = "l1"
    optimizer: str = "AdamW"


@dataclass
class ExperimentConfig:
    """Complete, self-contained experiment configuration."""
    experiment_id: str
    machine_id: str
    model: ModelHyperparameters
    features: FeatureConfig
    seed: int = 42
    train_split: str = "ml/data/splits/machine_specific/{machine_id}/train_normal.csv"
    dev_split: str = "ml/data/splits/machine_specific/{machine_id}/dev_normal.csv"
    test_split: str = "ml/data/splits/machine_specific/{machine_id}/test.csv"
    dataset_manifest_hash: str = ""
    artifact_dir: str = "ml/artifacts/{model_type}/{machine_id}"
    device: str = "auto"  # "auto", "cpu", "cuda"
    system_metadata: Dict[str, Any] = field(default_factory=collect_system_metadata)

    def __post_init__(self):
        # Resolve path templates
        self.train_split = self.train_split.format(machine_id=self.machine_id)
        self.dev_split = self.dev_split.format(machine_id=self.machine_id)
        self.test_split = self.test_split.format(machine_id=self.machine_id)
        self.artifact_dir = self.artifact_dir.format(
            model_type=self.model.model_type, machine_id=self.machine_id
        )

        if not self.dataset_manifest_hash:
            manifest_path = Path("ml/data/manifests/audio_metadata.csv")
            self.dataset_manifest_hash = compute_file_hash(manifest_path)

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def set_machine_id(self, machine_id: str) -> None:
        """Update machine_id and recalculate path templates."""
        self.machine_id = machine_id
        self.experiment_id = f"baseline_{'if' if self.model.model_type == 'isolation_forest' else 'dae'}_{machine_id}"
        self.train_split = f"ml/data/splits/machine_specific/{machine_id}/train_normal.csv"
        self.dev_split = f"ml/data/splits/machine_specific/{machine_id}/dev_normal.csv"
        self.test_split = f"ml/data/splits/machine_specific/{machine_id}/test.csv"
        self.artifact_dir = f"ml/artifacts/{self.model.model_type}/{machine_id}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        model_params = ModelHyperparameters(**raw.pop("model"))
        feature_params = FeatureConfig(**raw.pop("features"))
        return cls(model=model_params, features=feature_params, **raw)

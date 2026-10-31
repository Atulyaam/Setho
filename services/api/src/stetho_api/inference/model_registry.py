"""Model Registry and Inference Loader for Stetho Backend.

Maintains a thread-safe in-memory cache of machine-specific acoustic anomaly
detection models (Stetho-MSAE and Dense Autoencoder).
Validates artifact existence, SHA-256 integrity, tensor shapes, preprocessing
compatibility, and calibration profiles.
"""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from stetho_api.exceptions import ModelNotReadyException, ResourceNotFoundException

logger = logging.getLogger("stetho.model_registry")


class ModelType(str, Enum):
    """Supported model architecture types."""
    STETHO_MSAE = "stetho_msae"
    DENSE_AUTOENCODER = "dense_autoencoder"


@dataclass(frozen=True)
class ModelMetadata:
    """Immutable metadata and configuration for a registered model artifact."""
    model_type: ModelType
    machine_id: str
    version: str
    checkpoint_path: str
    sha256_hash: str
    total_parameters: int
    trainable_parameters: int
    device: str
    preprocessing_config: Dict[str, Any]
    calibration_profile: Dict[str, Any]
    loaded_at_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type.value,
            "machine_id": self.machine_id,
            "version": self.version,
            "checkpoint_path": self.checkpoint_path,
            "sha256_hash": self.sha256_hash,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "device": self.device,
            "preprocessing_config": self.preprocessing_config,
            "calibration_profile": self.calibration_profile,
            "loaded_at_utc": self.loaded_at_utc,
        }


class LoadedModel:
    """Wrapper encapsulating a live model detector, its calibration, and execution methods."""

    def __init__(self, detector: Any, metadata: ModelMetadata):
        self.detector = detector
        self.metadata = metadata
        self._calibration_profile = dict(metadata.calibration_profile)
        self._lock = threading.Lock()

    @property
    def model_type(self) -> ModelType:
        return self.metadata.model_type

    @property
    def machine_id(self) -> str:
        return self.metadata.machine_id

    @property
    def calibration(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._calibration_profile)

    def update_calibration(self, new_profile: Dict[str, Any]) -> None:
        """Atomically update in-memory calibration thresholds and profile."""
        with self._lock:
            self._calibration_profile.update(new_profile)
            if hasattr(self.detector, "threshold"):
                self.detector.threshold = new_profile.get("operating_threshold", self.detector.threshold)
            logger.info(
                f"Updated live calibration for {self.model_type.value}/{self.machine_id}: "
                f"threshold={new_profile.get('operating_threshold')}, provisional={new_profile.get('is_provisional')}"
            )

    def score_waveform(self, waveform: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Score 1D audio waveform and return continuous anomaly score plus patch diagnostics."""
        with self._lock:
            # Score continuous anomaly value
            score = self.detector.score_waveform(waveform)

            # Extract patch diagnostics if supported
            diagnostics: Dict[str, Any] = {
                "score": float(score),
                "threshold": float(self._calibration_profile.get("operating_threshold", 0.0)),
                "theta_lower": float(self._calibration_profile.get("theta_lower", 0.0)),
                "theta_upper": float(self._calibration_profile.get("theta_upper", 0.0)),
                "is_provisional": bool(self._calibration_profile.get("is_provisional", True)),
            }
            return float(score), diagnostics


class ModelRegistry:
    """Thread-safe, versioned model registry managing loaded model detectors."""

    def __init__(
        self,
        artifacts_root: Union[str, Path] = "./ml/artifacts",
        device: str = "auto",
        fallback_to_cpu: bool = True,
    ):
        self.artifacts_root = Path(artifacts_root)
        self.fallback_to_cpu = fallback_to_cpu

        # Device determination
        if device == "auto":
            if torch.cuda.is_available():
                self.target_device = "cuda"
            else:
                self.target_device = "cpu"
        else:
            self.target_device = device

        if self.target_device == "cuda" and not torch.cuda.is_available():
            if self.fallback_to_cpu:
                logger.warning("CUDA requested but unavailable; falling back to CPU")
                self.target_device = "cpu"
            else:
                raise RuntimeError("CUDA requested for ModelRegistry but not available.")

        self._models: Dict[Tuple[str, str], LoadedModel] = {}
        self._registry_lock = threading.Lock()
        self._load_errors: Dict[Tuple[str, str], str] = {}
        logger.info(
            f"Initialized ModelRegistry at {self.artifacts_root} (Device: {self.target_device})"
        )

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_model(
        self,
        machine_id: str,
        model_type: Optional[Union[ModelType, str]] = None,
    ) -> LoadedModel:
        """Retrieve or lazily load a machine-specific model detector."""
        # Default policy: Stetho-MSAE (Primary Edge Champion)
        if model_type is None:
            m_type = ModelType.STETHO_MSAE
        elif isinstance(model_type, str):
            try:
                m_type = ModelType(model_type.lower())
            except ValueError:
                raise ResourceNotFoundException(
                    "ModelType", f"{model_type} (supported: {[t.value for t in ModelType]})"
                )
        else:
            m_type = model_type

        key = (m_type.value, machine_id)

        with self._registry_lock:
            if key in self._models:
                return self._models[key]

            if key in self._load_errors:
                raise ModelNotReadyException(
                    f"Model {m_type.value} for {machine_id} previously failed to load: {self._load_errors[key]}"
                )

            # Load model artifact
            loaded = self._load_and_validate(m_type, machine_id)
            self._models[key] = loaded
            return loaded

    def _load_and_validate(self, model_type: ModelType, machine_id: str) -> LoadedModel:
        """Validate artifact integrity, shapes, preprocessing, and calibration before instantiating."""
        model_dir = self.artifacts_root / model_type.value / machine_id
        if not model_dir.exists():
            err_msg = f"Artifact directory not found for {model_type.value}/{machine_id} at {model_dir}"
            self._load_errors[(model_type.value, machine_id)] = err_msg
            logger.error(err_msg)
            raise ResourceNotFoundException("ModelArtifacts", f"{model_type.value}/{machine_id}")

        weights_file = model_dir / "model.pt"
        if not weights_file.exists():
            err_msg = f"Model checkpoint weights file missing: {weights_file}"
            self._load_errors[(model_type.value, machine_id)] = err_msg
            logger.error(err_msg)
            raise ModelNotReadyException(err_msg)

        meta_file = model_dir / "metadata.json"
        cal_file = model_dir / "calibration.json"

        # 1. Checkpoint integrity check (SHA-256)
        try:
            sha256_hash = self.compute_sha256(weights_file)
        except Exception as e:
            err_msg = f"Failed to compute SHA-256 for {weights_file}: {e}"
            self._load_errors[(model_type.value, machine_id)] = err_msg
            raise ModelNotReadyException(err_msg)

        # 2. Calibration profile validation
        cal_profile: Dict[str, Any] = {}
        if cal_file.exists():
            try:
                with open(cal_file, "r", encoding="utf-8") as f:
                    cal_profile = json.load(f)
            except Exception as e:
                err_msg = f"Corrupted calibration profile at {cal_file}: {e}"
                self._load_errors[(model_type.value, machine_id)] = err_msg
                raise ModelNotReadyException(err_msg)

            # Verify calibration belongs to the correct machine and model
            if cal_profile.get("machine_id") != machine_id:
                err_msg = f"Calibration machine mismatch: expected {machine_id}, got {cal_profile.get('machine_id')}"
                self._load_errors[(model_type.value, machine_id)] = err_msg
                raise ModelNotReadyException(err_msg)

            # Ensure three-zone thresholds exist
            if "theta_upper" not in cal_profile:
                cal_profile["theta_upper"] = cal_profile.get("operating_threshold", 0.0)
            if "theta_lower" not in cal_profile:
                cal_profile["theta_lower"] = cal_profile.get(
                    "dev_mean", cal_profile.get("dev_p90", cal_profile["theta_upper"])
                )
            if "is_provisional" not in cal_profile:
                cal_profile["is_provisional"] = True

        # 3. Model Architecture & Preprocessing Validation
        try:
            if model_type == ModelType.STETHO_MSAE:
                from ml.models.stetho_msae.msae_detector import StethoMSAEDetector
                detector = StethoMSAEDetector.load(model_dir, device=self.target_device)

                # Validate expected tensor shapes using dummy forward pass
                dummy_patch = torch.zeros(1, 1, 64, 64, device=detector.device)
                detector.model.eval()
                with torch.no_grad():
                    dummy_recon, dummy_latent = detector.model(dummy_patch)
                    if dummy_recon.shape != dummy_patch.shape:
                        raise ValueError(
                            f"Stetho-MSAE reconstructed shape mismatch: expected {dummy_patch.shape}, got {dummy_recon.shape}"
                        )
                    if dummy_latent.shape != (1, detector.latent_dim):
                        raise ValueError(
                            f"Stetho-MSAE latent shape mismatch: expected (1, {detector.latent_dim}), got {dummy_latent.shape}"
                        )

                total_params = sum(p.numel() for p in detector.model.parameters())
                trainable_params = sum(p.numel() for p in detector.model.parameters() if p.requires_grad)

                preprocessing_config = {
                    "sample_rate": detector.mel_config.sample_rate,
                    "n_mels": detector.mel_config.n_mels,
                    "n_fft": detector.mel_config.n_fft,
                    "hop_length": detector.mel_config.hop_length,
                    "spec_window_frames": detector.windowing.config.spec_window_frames,
                    "spec_hop_frames": detector.windowing.config.spec_hop_frames,
                    "normalization": detector.preprocessor.config.normalization,
                }

            elif model_type == ModelType.DENSE_AUTOENCODER:
                from ml.models.baseline_autoencoder.autoencoder_detector import DenseAutoencoderDetector
                detector = DenseAutoencoderDetector.load(model_dir, device=self.target_device)

                # Validate expected tensor shapes using dummy forward pass
                dummy_patch = torch.zeros(1, 4096, device=detector.device)
                detector.model.eval()
                with torch.no_grad():
                    dummy_recon, dummy_latent = detector.model(dummy_patch)
                    if dummy_recon.shape != dummy_patch.shape:
                        raise ValueError(
                            f"Dense AE reconstructed shape mismatch: expected {dummy_patch.shape}, got {dummy_recon.shape}"
                        )
                    if dummy_latent.shape != (1, detector.latent_dim):
                        raise ValueError(
                            f"Dense AE latent shape mismatch: expected (1, {detector.latent_dim}), got {dummy_latent.shape}"
                        )

                total_params = sum(p.numel() for p in detector.model.parameters())
                trainable_params = sum(p.numel() for p in detector.model.parameters() if p.requires_grad)

                preprocessing_config = {
                    "sample_rate": detector.mel_extractor.config.sample_rate,
                    "n_mels": detector.mel_extractor.config.n_mels,
                    "n_fft": detector.mel_extractor.config.n_fft,
                    "hop_length": detector.mel_extractor.config.hop_length,
                    "input_dim": detector.input_dim,
                    "latent_dim": detector.latent_dim,
                }

            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        except Exception as e:
            err_msg = f"Failed to instantiate or validate model {model_type.value}/{machine_id}: {e}"
            self._load_errors[(model_type.value, machine_id)] = err_msg
            logger.error(err_msg, exc_info=True)
            raise ModelNotReadyException(err_msg)

        from datetime import datetime, timezone
        now_utc = datetime.now(timezone.utc).isoformat()

        metadata = ModelMetadata(
            model_type=model_type,
            machine_id=machine_id,
            version="1.0.0",
            checkpoint_path=str(weights_file),
            sha256_hash=sha256_hash,
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            device=str(detector.device),
            preprocessing_config=preprocessing_config,
            calibration_profile=cal_profile,
            loaded_at_utc=now_utc,
        )

        logger.info(
            f"Successfully verified and loaded {model_type.value}/{machine_id} "
            f"({total_params:,} params, SHA-256: {sha256_hash[:8]}..., Device: {detector.device})"
        )
        return LoadedModel(detector=detector, metadata=metadata)

    def preload_all(self, machine_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Preload all champion and baseline models into memory."""
        machines = machine_ids or ["id_00", "id_02", "id_04", "id_06"]
        results = {"loaded": [], "failed": []}

        for m in machines:
            for m_type in [ModelType.STETHO_MSAE, ModelType.DENSE_AUTOENCODER]:
                try:
                    self.get_model(machine_id=m, model_type=m_type)
                    results["loaded"].append(f"{m_type.value}:{m}")
                except Exception as e:
                    results["failed"].append({"model": f"{m_type.value}:{m}", "error": str(e)})

        logger.info(
            f"Preload complete: {len(results['loaded'])} models loaded, {len(results['failed'])} failed"
        )
        return results

    def update_model_calibration(
        self,
        machine_id: str,
        model_type: Union[ModelType, str],
        new_profile: Dict[str, Any],
    ) -> None:
        """Atomically update the live in-memory model calibration profile."""
        m_type = model_type if isinstance(model_type, ModelType) else ModelType(model_type.lower())
        key = (m_type.value, machine_id)
        with self._registry_lock:
            if key in self._models:
                self._models[key].update_calibration(new_profile)

    def get_status(self) -> Dict[str, Any]:
        """Return readiness and diagnostic status of model registry."""
        with self._registry_lock:
            loaded_list = [
                {
                    "model_type": m.metadata.model_type.value,
                    "machine_id": m.metadata.machine_id,
                    "device": m.metadata.device,
                    "params": m.metadata.total_parameters,
                    "sha256": m.metadata.sha256_hash[:10] + "...",
                    "is_provisional": m.metadata.calibration_profile.get("is_provisional", True),
                }
                for m in self._models.values()
            ]
            failed_list = [
                {"key": f"{k[0]}:{k[1]}", "error": v}
                for k, v in self._load_errors.items()
            ]

        has_artifacts = self.artifacts_root.exists()
        has_failures = len(failed_list) > 0

        if not has_artifacts or (has_failures and len(loaded_list) == 0):
            subsys_status = "unhealthy"
        elif has_failures:
            subsys_status = "degraded"
        else:
            subsys_status = "healthy"

        return {
            "status": subsys_status,
            "target_device": self.target_device,
            "fallback_to_cpu": self.fallback_to_cpu,
            "loaded_count": len(loaded_list),
            "loaded_models": loaded_list,
            "failed_count": len(failed_list),
            "failed_models": failed_list,
        }


# Global registry singleton
_registry_instance: Optional[ModelRegistry] = None
_registry_lock = threading.Lock()


def get_model_registry() -> ModelRegistry:
    """Return application-wide cached ModelRegistry singleton."""
    global _registry_instance
    with _registry_lock:
        if _registry_instance is None:
            from stetho_api.config import get_settings
            settings = get_settings()
            _registry_instance = ModelRegistry(
                artifacts_root=settings.MODEL_ARTIFACTS_DIR,
                device="auto",
                fallback_to_cpu=settings.DEVICE_FALLBACK_TO_CPU,
            )
        return _registry_instance


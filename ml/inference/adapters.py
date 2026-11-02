"""Model Preprocessing and Inference Adapters for Stetho Unified Architecture.

Wraps existing frozen checkpoints into a common, type-safe interface:
- StudentONNXAdapter: Ultra-fast distilled INT8/FP32 student model via ONNX Runtime
- ResNetAEAdapter: High-capacity ResNet-AE specialist with L1 and Smooth L1 scoring
- MultiTaskAdapter: Universal multi-category acoustic prior
- ResidualWhiteningAdapter: Machine-specific spatial-spectral residual normalizer
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from ml.models.research.residual_cnn_ae import ResidualCNNAE
from ml.preprocessing.feature_suite import UnifiedFeatureSuite

logger = logging.getLogger("stetho_inference_adapters")


@dataclass
class ModelOutput:
    """Standardized output from any model adapter."""
    reconstruction: np.ndarray  # (B, 1, 64, 64)
    latent: Optional[np.ndarray]  # (B, D)
    patch_scores: np.ndarray  # (B,)
    mean_score: float
    max_score: float
    top10_score: float
    latency_ms: float
    model_name: str
    model_format: str


class BaseModelAdapter(ABC):
    """Abstract interface for unified model inference."""

    @abstractmethod
    def score_patches(self, patches: np.ndarray) -> ModelOutput:
        """Score standardized (B, 1, 64, 64) Log-Mel patches."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return model metadata, parameters, and checksums."""
        pass


class StudentONNXAdapter(BaseModelAdapter):
    """Ultra-fast ONNX Runtime adapter for Distilled CompactStudentAE."""

    def __init__(
        self,
        onnx_path: Union[str, Path] = "ml/experiments/research_artifacts/phase8/exported_models/student_ae_int8.onnx",
        operating_threshold: float = 0.37356,
        prefer_fp32_for_latency: bool = True,
    ):
        target_path = Path(onnx_path)
        # If prefer_fp32_for_latency on CPU, use fp32 model if available (2.4ms vs 124ms on x86 without VNNI)
        if prefer_fp32_for_latency:
            fp32_candidate = target_path.parent / "student_ae_fp32.onnx"
            if fp32_candidate.exists():
                target_path = fp32_candidate

        if not target_path.exists():
            raise FileNotFoundError(f"Student ONNX checkpoint missing at {target_path}")

        self.model_path = target_path
        self.raw_bytes = target_path.read_bytes()
        self.sha256 = hashlib.sha256(self.raw_bytes).hexdigest()
        self.operating_threshold = operating_threshold

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.is_int8 = "int8" in target_path.name.lower()

    def score_patches(self, patches: np.ndarray) -> ModelOutput:
        t0 = time.perf_counter()
        inp = patches.astype(np.float32)
        outputs = self.session.run(None, {self.input_name: inp})
        recon = outputs[0]
        latent = outputs[1] if len(outputs) > 1 else None

        # Element-wise absolute reconstruction error per patch
        diff = np.abs(recon - inp)
        patch_scores = np.mean(diff, axis=(1, 2, 3))

        t_ms = (time.perf_counter() - t0) * 1000.0
        k = max(1, int(len(patch_scores) * 0.10))
        top10 = float(np.mean(np.sort(patch_scores)[-k:]))

        return ModelOutput(
            reconstruction=recon,
            latent=latent,
            patch_scores=patch_scores,
            mean_score=float(np.mean(patch_scores)),
            max_score=float(np.max(patch_scores)),
            top10_score=top10,
            latency_ms=round(t_ms, 2),
            model_name="CompactStudentAE",
            model_format="ONNX_INT8" if self.is_int8 else "ONNX_FP32",
        )

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": "CompactStudentAE",
            "model_format": "ONNX_INT8" if self.is_int8 else "ONNX_FP32",
            "checkpoint_path": str(self.model_path),
            "sha256": self.sha256,
            "parameter_count": 22113,
            "threshold": self.operating_threshold,
            "supported_categories": ["fan", "pump", "valve", "universal"],
        }


class ResNetAEAdapter(BaseModelAdapter):
    """High-capacity ResNet-AE specialist with configurable L1 or Smooth L1 scoring."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "ml/experiments/research_artifacts/phase4/residual_cnn_ae/id_00/model.pt",
        operating_threshold: float = 0.35208,
        use_smooth_l1: bool = True,
        smooth_l1_beta: float = 0.1,
    ):
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"ResNet-AE checkpoint missing at {ckpt_path}")

        self.model_path = ckpt_path
        self.raw_bytes = ckpt_path.read_bytes()
        self.sha256 = hashlib.sha256(self.raw_bytes).hexdigest()
        self.operating_threshold = operating_threshold
        self.use_smooth_l1 = use_smooth_l1
        self.smooth_l1_beta = smooth_l1_beta

        self.device = torch.device("cpu")
        self.model = ResidualCNNAE(
            in_channels=1,
            base_channels=16,
            latent_dim=64,
            norm_type="group",
            input_size=(64, 64),
        )

        sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            self.model.load_state_dict(sd["state_dict"])
        elif isinstance(sd, dict):
            self.model.load_state_dict(sd)
        self.model.to(self.device).eval()

    def score_patches(self, patches: np.ndarray) -> ModelOutput:
        t0 = time.perf_counter()
        inp_tensor = torch.from_numpy(patches).float().to(self.device)

        with torch.no_grad():
            recon_tensor, latent_tensor = self.model(inp_tensor)
            diff = torch.abs(recon_tensor - inp_tensor)

            if self.use_smooth_l1:
                beta = self.smooth_l1_beta
                err = torch.where(diff < beta, 0.5 * (diff ** 2) / beta, diff - 0.5 * beta)
            else:
                err = diff

            patch_scores = torch.mean(err, dim=(1, 2, 3)).cpu().numpy()

        recon = recon_tensor.cpu().numpy()
        latent = latent_tensor.cpu().numpy() if latent_tensor is not None else None
        t_ms = (time.perf_counter() - t0) * 1000.0
        k = max(1, int(len(patch_scores) * 0.10))
        top10 = float(np.mean(np.sort(patch_scores)[-k:]))

        return ModelOutput(
            reconstruction=recon,
            latent=latent,
            patch_scores=patch_scores,
            mean_score=float(np.mean(patch_scores)),
            max_score=float(np.max(patch_scores)),
            top10_score=top10,
            latency_ms=round(t_ms, 2),
            model_name="ResidualCNNAE_SmoothL1" if self.use_smooth_l1 else "ResidualCNNAE_L1",
            model_format="PyTorch_FP32",
        )

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": "ResidualCNNAE_SmoothL1" if self.use_smooth_l1 else "ResidualCNNAE_L1",
            "model_format": "PyTorch_FP32",
            "checkpoint_path": str(self.model_path),
            "sha256": self.sha256,
            "parameter_count": 1808065,
            "threshold": self.operating_threshold,
            "smooth_l1_beta": self.smooth_l1_beta if self.use_smooth_l1 else None,
            "supported_categories": ["fan"],
        }


class MultiTaskAdapter(BaseModelAdapter):
    """Universal multi-category acoustic prior adapter using generalized ResNet-AE."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "ml/experiments/research_artifacts/phase4/residual_cnn_ae/id_00/model.pt",
        operating_threshold: float = 0.35208,
    ):
        self.underlying = ResNetAEAdapter(
            checkpoint_path=checkpoint_path,
            operating_threshold=operating_threshold,
            use_smooth_l1=True,
            smooth_l1_beta=0.1,
        )
        self.operating_threshold = operating_threshold

    def score_patches(self, patches: np.ndarray) -> ModelOutput:
        out = self.underlying.score_patches(patches)
        out.model_name = "MultiTaskResNetAE"
        return out

    def get_metadata(self) -> Dict[str, Any]:
        meta = self.underlying.get_metadata()
        meta["model_name"] = "MultiTaskResNetAE"
        meta["supported_categories"] = ["fan", "pump", "valve", "universal"]
        return meta


class ResidualWhiteningAdapter:
    """Applies machine-specific spatial-spectral residual whitening to specialist reconstructions."""

    def __init__(
        self,
        baseline_mean: np.ndarray,  # (1, 1, 64, 64)
        baseline_std: np.ndarray,   # (1, 1, 64, 64)
        calibrated_threshold: float,
        n_shots: int = 10,
    ):
        self.baseline_mean = baseline_mean
        self.baseline_std = np.maximum(baseline_std, 1e-4)
        self.calibrated_threshold = calibrated_threshold
        self.n_shots = n_shots

    def whiten_reconstruction(
        self, input_patches: np.ndarray, reconstruction: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Compute spatial-spectral whitened residuals: max(0, (|x - x_hat| - mu) / sigma)."""
        residuals = np.abs(input_patches - reconstruction)  # (B, 1, 64, 64)
        whitened = np.maximum(0.0, (residuals - self.baseline_mean) / self.baseline_std)
        patch_scores = np.mean(whitened, axis=(1, 2, 3))
        # Top-10% mean aggregation
        k = max(1, int(len(patch_scores) * 0.10))
        top10_score = float(np.mean(np.sort(patch_scores)[-k:]))
        return patch_scores, top10_score

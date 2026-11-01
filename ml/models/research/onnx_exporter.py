"""ONNX Export, Graph Optimization, INT8 Dynamic Quantization, and ORT Inference Engine.

Features:
- export_to_onnx: Exports PyTorch models to optimized ONNX graphs with dynamic batch support
- verify_numerical_parity: Validates max absolute deviation < 1e-4
- quantize_onnx_dynamic: Quantizes ONNX model to INT8
- ORTAnomalyDetector: Direct high-performance ONNX Runtime inference wrapper conforming to BaseAnomalyDetector
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import soundfile as sf
import torch
import torch.nn as nn

from ml.models.base import BaseAnomalyDetector
from ml.preprocessing.feature_suite import UnifiedFeatureSuite

logger = logging.getLogger("stetho_onnx_exporter")


def export_to_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: Union[str, Path],
    opset_version: int = 17,
) -> Path:
    """Export a PyTorch autoencoder model to ONNX with dynamic batch size."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to("cpu")
    dummy_input = dummy_input.to("cpu")

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_patch"],
        output_names=["reconstruction", "latent"],
        dynamic_axes={
            "input_patch": {0: "batch_size"},
            "reconstruction": {0: "batch_size"},
            "latent": {0: "batch_size"},
        },
        dynamo=False,
    )

    # Verify ONNX model integrity
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    logger.info(f"Successfully exported and verified ONNX model: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return out_path


def verify_numerical_parity(
    torch_model: nn.Module,
    onnx_path: Union[str, Path],
    sample_input: torch.Tensor,
    tol: float = 1e-4,
) -> Tuple[bool, float]:
    """Verify numerical parity between PyTorch and ONNX Runtime predictions."""
    torch_model.eval()
    torch_model.to("cpu")
    sample_cpu = sample_input.to("cpu")

    with torch.no_grad():
        pt_recon, _ = torch_model(sample_cpu)
    pt_recon_np = pt_recon.numpy()

    # ONNX Runtime inference
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    ort_outputs = session.run(None, {input_name: sample_cpu.numpy()})
    ort_recon_np = ort_outputs[0]

    max_diff = float(np.max(np.abs(pt_recon_np - ort_recon_np)))
    is_parity = max_diff < tol
    logger.info(f"Numerical parity check on {onnx_path}: max_diff={max_diff:.6e} (tolerance={tol}, passed={is_parity})")
    return is_parity, max_diff


def quantize_to_int8(
    onnx_input_path: Union[str, Path],
    onnx_output_path: Union[str, Path],
) -> Path:
    """Apply dynamic INT8 quantization to an ONNX graph."""
    in_p = Path(onnx_input_path)
    out_p = Path(onnx_output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(in_p),
        model_output=str(out_p),
        weight_type=QuantType.QInt8,
    )
    logger.info(
        f"Quantized model to INT8: {out_p} ({out_p.stat().st_size / 1024:.1f} KB vs "
        f"{in_p.stat().st_size / 1024:.1f} KB, -{(1 - out_p.stat().st_size / in_p.stat().st_size)*100:.1f}%)"
    )
    return out_p


class ORTAnomalyDetector(BaseAnomalyDetector):
    """High-speed ONNX Runtime Inference Anomaly Detector."""

    def __init__(
        self,
        machine_id: str,
        onnx_model_path: Union[str, Path],
        train_mean: float,
        train_std: float,
        threshold: float = 0.5,
        patch_frames: int = 64,
        preset: str = "standard_64mel",
        intra_op_threads: int = 2,
    ):
        super().__init__(machine_id=machine_id, model_type="ort_optimized_ae")
        self.onnx_path = Path(onnx_model_path)
        self.train_mean = train_mean
        self.train_std = train_std
        self.threshold = threshold
        self.patch_frames = patch_frames
        self.preset = preset

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = intra_op_threads
        self.session = ort.InferenceSession(str(self.onnx_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.feature_suite = UnifiedFeatureSuite(preset_name=preset)

    def fit(self, train_files, dev_files=None):
        return {}

    def score_patches(self, patches: np.ndarray) -> np.ndarray:
        norm_patches = (patches - self.train_mean) / self.train_std
        outputs = self.session.run(None, {self.input_name: norm_patches})
        recon = outputs[0]
        err = np.mean(np.abs(recon - norm_patches), axis=(1, 2, 3))
        return err

    def score_file(self, file_path: Union[str, Path]) -> float:
        hop = max(16, self.patch_frames // 2)
        patches, _ = self.feature_suite.extract_patches(file_path, patch_frames=self.patch_frames, hop_frames=hop)
        if len(patches) == 0:
            return 0.0
        scores = self.score_patches(patches)
        return float(np.mean(scores))

    def score_waveform(self, waveform: np.ndarray, sample_rate: int = 16000) -> float:
        if waveform.ndim > 1:
            waveform = waveform[0]
        peak = np.max(np.abs(waveform))
        if peak > 0:
            waveform = waveform / peak
        hop = max(16, self.patch_frames // 2)
        patches, _ = self.feature_suite.extract_patches_from_waveform(waveform, patch_frames=self.patch_frames, hop_frames=hop)
        if len(patches) == 0:
            return 0.0
        scores = self.score_patches(patches)
        return float(np.mean(scores))

    def save(self, path: Union[str, Path]) -> None:
        pass

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ORTAnomalyDetector":
        raise NotImplementedError("Direct instantiation with onnx path preferred")


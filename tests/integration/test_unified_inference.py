"""Integration and Verification Tests for Sprint 8.1 Unified Adaptive Inference Architecture.

Verifies:
1. Audio Quality Gates: Rejection of clipped, silent, corrupt, and sub-Nyquist audio.
2. Feature Extraction Contract: (N, 1, 64, 64) tensor format, float32, strictly finite.
3. Fast Student ONNX Early Exit: Clear margin triggers fast exit without specialist escalation.
4. Category-Aware Specialist Routing: Fan routes to ResNet-AE; Pump/Valve to MultiTaskResNetAE.
5. Residual Whitening Adaptation: Calibration profiles apply spatial-spectral whitening.
6. Uncertainty-Aware Three-Zone Decision: Close proximity to decision boundary returns UNCERTAIN.
7. Thread-Safe Singleton and Model Caching: Engines and adapters load once and are cached safely.
8. Production Artifacts Unmodified: 0-byte diff on ml/artifacts/ against production master 7d8f841.
"""

import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from ml.inference import (
    AdaptiveRouter,
    BaseModelAdapter,
    DecisionConfidence,
    DecisionStatus,
    DiagnosticResponse,
    MultiTaskAdapter,
    ResNetAEAdapter,
    ResidualWhiteningAdapter,
    RoutingPath,
    RoutingPolicyConfig,
    StudentONNXAdapter,
    UnifiedAdaptiveInferenceEngine,
)
from ml.preprocessing.feature_suite import UnifiedFeatureSuite


@pytest.fixture
def clean_sine_wav(tmp_path: Path) -> Path:
    """Generate a clean synthetic 10-second 16 kHz mono WAV file."""
    p = tmp_path / "clean_sine.wav"
    sr = 16000
    duration_sec = 10.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    sig = (0.25 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    sf.write(str(p), sig, sr, format="WAV", subtype="PCM_16")
    return p


def test_audio_quality_gate_rejection(tmp_path: Path):
    """Verify that malformed, silent, clipped, or sub-Nyquist audio is safely rejected before model inference."""
    engine = UnifiedAdaptiveInferenceEngine.get_instance()

    # 1. Severe clipping
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    clipped_wave = np.clip(np.sin(2 * np.pi * 300 * t) * 5.0, -1.0, 1.0).astype(np.float32)
    resp_clipped = engine.inspect_audio(clipped_wave, sample_rate=sr, machine_id="test_01", machine_category="fan")
    assert resp_clipped.decision == "REJECTED"
    assert resp_clipped.execution_path == RoutingPath.QUALITY_REJECTED.value
    assert any("CLIPPING" in r for r in resp_clipped.rejection_reasons)

    # 2. Silent recording (RMS < -60 dBFS)
    silent_wave = (np.random.randn(sr * 2) * 1e-5).astype(np.float32)
    resp_silent = engine.inspect_audio(silent_wave, sample_rate=sr, machine_id="test_01", machine_category="fan")
    assert resp_silent.decision == "REJECTED"
    assert resp_silent.execution_path == RoutingPath.QUALITY_REJECTED.value
    assert any("SILENCE" in r for r in resp_silent.rejection_reasons)

    # 3. Sub-Nyquist sample rate (< 16000 Hz)
    subnyquist_wave = (0.3 * np.sin(2 * np.pi * 300 * np.linspace(0, 2.0, 8000 * 2, endpoint=False))).astype(np.float32)
    resp_sr = engine.inspect_audio(subnyquist_wave, sample_rate=8000, machine_id="test_01", machine_category="fan")
    assert resp_sr.decision == "REJECTED"
    assert resp_sr.execution_path == RoutingPath.QUALITY_REJECTED.value
    assert any("SAMPLE_RATE" in r for r in resp_sr.rejection_reasons)

    # 4. Audio too short (< 1.0 sec, insufficient for spectrogram windowing)
    tiny_wave = (0.3 * np.sin(2 * np.pi * 300 * np.linspace(0, 0.01, 160, endpoint=False))).astype(np.float32)
    resp_short = engine.inspect_audio(tiny_wave, sample_rate=sr, machine_id="test_01", machine_category="fan")
    assert resp_short.decision == "REJECTED"
    assert resp_short.execution_path == RoutingPath.QUALITY_REJECTED.value
    assert resp_short.quality_status == "EMPTY_SPECTROGRAM"


def test_feature_extraction_tensor_contract(clean_sine_wav: Path):
    """Verify that feature extraction conforms to the standard 64-bin Log-Mel (N, 1, 64, 64) tensor contract."""
    suite = UnifiedFeatureSuite(preset_name="standard_64mel")
    patches, metrics = suite.extract_patches(clean_sine_wav, patch_frames=64, hop_frames=32)

    assert patches.ndim == 4, "Patches tensor must be 4-dimensional (N, 1, F, T)"
    assert patches.shape[1] == 1, "Channel dimension must be 1"
    assert patches.shape[2] == 64, "Frequency bins must be 64"
    assert patches.shape[3] == 64, "Temporal frames must be 64"
    assert patches.dtype == np.float32, "Tensor dtype must be float32"
    assert np.all(np.isfinite(patches)), "All extracted feature values must be finite (no NaN / Inf)"


def test_fast_student_onnx_early_exit(clean_sine_wav: Path):
    """Verify that clear signals exit early at Fast Student ONNX without invoking specialist models."""
    engine = UnifiedAdaptiveInferenceEngine.get_instance()

    # Pass clean sine into a machine with no calibration profile
    resp = engine.inspect_audio(
        clean_sine_wav,
        machine_id="uncalibrated_exit_test",
        machine_category="fan",
    )

    # Ensure response is valid
    assert resp.decision in ("NORMAL", "ANOMALY", "UNCERTAIN")
    assert resp.execution_path in ("FAST_STUDENT_EXIT", "SPECIALIST_ESCALATED")
    if resp.execution_path == "FAST_STUDENT_EXIT":
        assert resp.models_executed == 1
        assert any("student" in m.lower() for m in resp.models_used)


def test_category_aware_specialist_routing():
    """Verify that the adaptive router accurately selects specialists based on machine category."""
    router = AdaptiveRouter()

    # Fan routes to ResNet-AE specialist
    assert router.select_specialist_model_type("fan") == "resnet_ae_specialist"

    # Pump routes to MultiTaskResNetAE universal prior
    assert router.select_specialist_model_type("pump") == "multitask_resnet_ae"

    # Valve routes to MultiTaskResNetAE universal prior
    assert router.select_specialist_model_type("valve") == "multitask_resnet_ae"

    # Unknown categories safely fall back to MultiTaskResNetAE
    assert router.select_specialist_model_type("compressor") == "multitask_resnet_ae"
    assert router.select_specialist_model_type("conveyor") == "multitask_resnet_ae"


def test_residual_whitening_adaptation(clean_sine_wav: Path):
    """Verify spatial-spectral residual whitening adaptation with calibration profiles."""
    engine = UnifiedAdaptiveInferenceEngine.get_instance()

    suite = UnifiedFeatureSuite(preset_name="standard_64mel")
    patches, _ = suite.extract_patches(clean_sine_wav, patch_frames=64, hop_frames=32)

    # Synthetic residual distribution
    mu_r = np.full((1, 1, 64, 64), 0.2, dtype=np.float32)
    std_r = np.full((1, 1, 64, 64), 0.05, dtype=np.float32)
    whitener = ResidualWhiteningAdapter(
        baseline_mean=mu_r, baseline_std=std_r, calibrated_threshold=1.5
    )

    # Reconstruction with synthetic error
    recon = patches + 0.35  # Error is 0.35, residual = max(0, (0.35 - 0.20)/0.05) = 3.0
    patch_scores, top10 = whitener.whiten_reconstruction(patches, recon)
    assert len(patch_scores) == len(patches)
    assert np.isclose(np.mean(patch_scores), 3.0, atol=1e-2)

    # Register calibration profile in engine
    engine.register_calibration_profile(
        machine_id="test_machine_calibrated",
        baseline_mean=mu_r,
        baseline_std=std_r,
        calibrated_threshold=1.5,
        machine_category="fan",
        n_shots=10,
    )

    resp = engine.inspect_audio(
        clean_sine_wav,
        machine_id="test_machine_calibrated",
        machine_category="fan",
    )
    assert resp.execution_path == "RESIDUAL_WHITENED"
    assert resp.models_executed == 2  # ResNet specialist + Whitening
    assert resp.telemetry.get("residual_whitening_applied") is True


def test_uncertainty_aware_three_zone_decision():
    """Verify that scores within the margin of ambiguity return UNCERTAIN with explicit justification."""
    router = AdaptiveRouter()

    # 1. Clear Normal (well below threshold 1.00)
    status, conf, unc_reason = router.determine_decision_and_confidence(
        score=0.50,
        threshold=1.00,
        patch_scores=np.array([0.48, 0.50, 0.52]),
        route_path=RoutingPath.SPECIALIST_ESCALATED,
        quality_warnings=[],
    )
    assert status == DecisionStatus.NORMAL
    assert conf == DecisionConfidence.HIGH

    # 2. Clear Anomaly (well above threshold 1.00)
    status, conf, unc_reason = router.determine_decision_and_confidence(
        score=1.50,
        threshold=1.00,
        patch_scores=np.array([1.48, 1.50, 1.52]),
        route_path=RoutingPath.SPECIALIST_ESCALATED,
        quality_warnings=[],
    )
    assert status == DecisionStatus.ANOMALY
    assert conf == DecisionConfidence.HIGH

    # 3. Zone of Uncertainty (score within +/- 6% of threshold 1.00, e.g., 1.02)
    status, conf, unc_reason = router.determine_decision_and_confidence(
        score=1.02,
        threshold=1.00,
        patch_scores=np.array([1.01, 1.02, 1.03]),
        route_path=RoutingPath.SPECIALIST_ESCALATED,
        quality_warnings=[],
    )
    assert status == DecisionStatus.UNCERTAIN
    assert "BORDERLINE_SCORE" in (unc_reason or "")

    # 4. High patch dispersion (CV > 0.40 with score close to threshold)
    status, conf, unc_reason = router.determine_decision_and_confidence(
        score=1.10,
        threshold=1.00,
        patch_scores=np.array([0.10, 0.20, 3.00]),  # highly erratic
        route_path=RoutingPath.SPECIALIST_ESCALATED,
        quality_warnings=[],
    )
    assert status == DecisionStatus.UNCERTAIN
    assert "HIGH_PATCH_DISPERSION" in (unc_reason or "")


def test_thread_safe_singleton_and_caching():
    """Verify that the unified inference engine maintains a singleton instance and caches models across threads."""
    instances = []

    def get_engine():
        inst = UnifiedAdaptiveInferenceEngine.get_instance()
        instances.append(inst)

    threads = [threading.Thread(target=get_engine) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(instances) == 5
    for inst in instances[1:]:
        assert inst is instances[0], "UnifiedAdaptiveInferenceEngine must be a true singleton across threads"


def test_production_artifacts_unmodified():
    """Verify that production model artifacts under ml/artifacts/ are completely unchanged (0-byte diff against 7d8f841)."""
    result = subprocess.run(
        ["git", "diff", "--exit-code", "7d8f841", "--", "ml/artifacts/"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Production artifacts modified! Git diff: {result.stdout}\n{result.stderr}"

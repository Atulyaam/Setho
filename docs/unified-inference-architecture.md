# Stetho Unified Existing Model Inference Architecture

**Document Reference:** ARCH-STETHO-8.1-UNIFIED-001  
**Sprint / Phase:** Sprint 8.1 Unified Existing Model Inference Architecture  
**Status:** IMPLEMENTED & VERIFIED  
**Branch:** `feature/field-data-readiness`  
**Production Master:** `7d8f841` (Untouched, 0-byte diff on `ml/artifacts/`)  
**Core Directive:** **Zero model retraining, zero new neural networks, zero weight modifications.**  

---

## 1. Architectural Vision & Core Principles

Previous research sprints developed several specialized acoustic detection components:
- High-accuracy residual autoencoders (`ResidualCNNAE` / `ResNet-AE` with 1.81M parameters).
- Robust loss formulations (`Smooth L1` with $\beta=0.1$ from Sprint 7.6 / E2).
- Generalized multi-category acoustic priors (`MultiTaskResNetAE` for fans, pumps, valves).
- Ultra-compact, low-latency edge models (`CompactStudentAE` with 22.1k parameters, exported as INT8/FP32 ONNX).
- Adaptive confidence filtering (`MarginGatedCascade`).
- Few-shot casing resonance cancellation (`Residual Whitening`).
- Audio signal verification gates (`AudioQualityGate`, `SensorGuard`).

Historically, these components operated as fragmented research prototypes with heterogeneous input conventions and distinct evaluation runners. **Sprint 8.1 unifies all existing components into a single, cohesive, cloud-compatible, adaptive inference architecture** without modifying existing model weights or retraining any neural network.

```
+---------------------------------------------------------------------------------------------------------+
|                              STETHO UNIFIED ADAPTIVE INFERENCE ENGINE                                   |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  [ Ingested Acoustic WAV Recording ] (Local File, Raw Bytes, or Streaming Ingestion)                   |
|         |                                                                                               |
|         v                                                                                               |
|  [ STAGE 1: Shared Audio Validation Gate ]                                                              |
|  - Sample rate verification (fs >= 16 kHz; resample 44.1k/48k -> 16k)                                  |
|  - Clipping ratio rejection (<= 0.1% ceiling samples)                                                   |
|  - Dynamic range & silence check (RMS > -60 dBFS)                                                       |
|  - Adverse environment screening (spectral flatness, HF energy ratio)                                  |
|         |                                                                                               |
|         | Passed -> Standardized 16 kHz Mono Waveform                                                   |
|         v                                                                                               |
|  [ STAGE 2: Standardized Feature Extraction ]                                                           |
|  - 64-bin Log-Mel Spectrogram (N_fft=1024, hop=256, 64 mel filterbanks)                                 |
|  - Sliding-window patch extraction: (B, 1, 64, 64) with hop=32 frames                                   |
|         |                                                                                               |
|         v                                                                                               |
|  [ STAGE 3: Fast Student ONNX First-Pass ] (CompactStudentAE INT8 / FP32)                               |
|  - CPU Inference Latency: 2.4 ms (FP32) / 124 ms (INT8)                                                 |
|  - Compute initial reconstruction error: s_student = mean(|x - x_hat_student|)                          |
|  - Evaluate confidence margin: Delta = |s_student - theta_student|                                      |
|         |                                                                                               |
|         +---> [ High Confidence Exit ] (Delta >= gamma AND no calibrated profile active)                |
|         |     -> FAST_STUDENT_EXIT (1 model executed; < 5ms CPU)                                        |
|         |                                                                                               |
|         v (Delta < gamma OR Calibrated Profile Active OR Multi-Category Asset)                          |
|  [ STAGE 4: Category-Aware Specialist Selection ]                                                       |
|  - If machine_category == "fan": Select ResNet-AE Specialist (Smooth L1, beta=0.1)                     |
|  - If machine_category in ["pump", "valve"]: Select MultiTaskResNetAE Prior                            |
|         |                                                                                               |
|         v                                                                                               |
|  [ STAGE 5: Machine-Specific Residual Whitening ]                                                       |
|  - If Asset has Calibrated Profile (N=10 healthy enrollment clips):                                    |
|      Whitened Residual: r_whitened = max(0, (|x - x_hat_specialist| - mu_r) / sigma_r)                  |
|      Anomaly Score: Top-10% mean of whitened patch residuals                                           |
|  - If Uncalibrated Asset:                                                                               |
|      Specialist Smooth L1 Reconstruction Error: L_beta(|x - x_hat|)                                     |
|         |                                                                                               |
|         v                                                                                               |
|  [ STAGE 6: Uncertainty-Aware Decision Zone ]                                                           |
|  - Boundary Proximity Check: If score is within +/- 6% of threshold -> UNCERTAIN                        |
|  - Transient Acoustic Burst Check: High patch coefficient of variation -> UNCERTAIN                    |
|  - Confident Classifications: NORMAL or ANOMALY with HIGH/MEDIUM/LOW confidence                        |
|         |                                                                                               |
|         v                                                                                               |
|  [ STAGE 7: Unified Diagnostic Response ]                                                               |
|  - Structured JSON response containing decisions, granular telemetry, models executed, and audit trail  |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Common Model Interface Contract

To eliminate ad-hoc model invocations, all models implement the abstract `BaseModelAdapter` contract defined in [`ml/inference/adapters.py`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/inference/adapters.py):

```python
class BaseModelAdapter(ABC):
    @abstractmethod
    def score_patches(self, patches: np.ndarray) -> ModelOutput:
        """Score standardized (B, 1, 64, 64) Log-Mel patches."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return model metadata, parameters, and checksums."""
        pass
```

### Standardized `ModelOutput` Dataclass
Every model adapter produces a uniform output structure regardless of execution engine (ONNX Runtime or PyTorch):
- `reconstruction`: Reconstructed spectrogram patches of shape `(B, 1, 64, 64)`.
- `latent`: Latent bottleneck vectors of shape `(B, D)` ($D=32$ for Student, $D=64$ for ResNet-AE).
- `patch_scores`: 1D array of per-patch anomaly scores of shape `(B,)`.
- `mean_score`: Unweighted arithmetic mean across all patches.
- `max_score`: Peak patch reconstruction error.
- `top10_score`: Mean of the highest 10% scoring patches (captures localized, transient burst anomalies).
- `latency_ms`: Measured forward pass wall-clock time in milliseconds.
- `model_name` & `model_format`: Precise identifier and engine (e.g. `CompactStudentAE` / `ONNX_FP32`).

---

## 3. Seven Stages of the Unified Inference Engine

### Stage 1: Shared Audio Validation Gate
Before any acoustic features are computed, raw audio is screened by [`AudioQualityGate`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/preprocessing/audio_quality_gate.py):
1. **Sampling Rate Enforcer:** Minimum $16,000\text{ Hz}$. High-rate recordings ($44.1\text{k}, 48\text{k}$) are polyphase-resampled to exactly $16\text{ kHz}$. Sub-Nyquist streams ($< 16\text{ kHz}$) are immediately rejected.
2. **Clipping Guard:** Digital clipping ratio must be $\le 0.1\%$.
3. **Signal Energy Guard:** Waveform RMS must be $> -60\text{ dBFS}$ (rejects silent or disconnected sensor channels).
4. **Adverse Acoustic Environment:** Identifies ambient white noise (spectral flatness $> 0.12$) or air-jet hiss (high-frequency power ratio $> -14\text{ dBFS}$), passing warnings to downstream decision stages.

### Stage 2: Standardized Feature Extraction
Audio is converted into time-frequency representations by [`UnifiedFeatureSuite`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/preprocessing/feature_suite.py):
- Filterbank: 64 mel bins spanning $50\text{ Hz}$ to $8000\text{ Hz}$.
- Windowing: $N_{\text{fft}} = 1024$, hop size $= 256$ samples ($16\text{ ms}$ step, 75% overlap).
- Slicing: Non-overlapping or 50%-overlapping patches of shape $(64, 64)$ (representing $1.024\text{ seconds}$ per patch). For a standard 10-second clip, this generates $B=19$ patches.

### Stage 3: Fast Student ONNX First-Pass
The $(B, 1, 64, 64)$ patches are fed to [`StudentONNXAdapter`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/inference/adapters.py), which executes the distilled `CompactStudentAE` model via ONNX Runtime CPU:
- Input tensor: `['input_patch']` normalized by training statistics.
- Forward pass time: **$2.43\text{ ms}$** on CPU.
- Decision margin calculation:
  $$\Delta_{\text{student}} = |s_{\text{student}} - \theta_{\text{student}}|$$
  where $\theta_{\text{student}} = 0.37356$.
- **Fast-Exit Condition:** If $\Delta_{\text{student}} \ge \gamma$ ($\gamma = 0.05$) AND the asset does not require machine-specific residual whitening, the inference engine early-exits immediately. Only **1 model** is executed, completing end-to-end inspection in under $20\text{ ms}$.

### Stage 4: Category-Aware Specialist Model Selection
If $\Delta_{\text{student}} < \gamma$ (ambiguous boundary) or if the asset has a specialized calibration profile:
- **Category `fan`:** Routes to [`ResNetAEAdapter`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/inference/adapters.py) loading the machine-specific or generalized `ResidualCNNAE` weights with Smooth L1 scoring.
- **Category `pump` or `valve`:** Routes to [`MultiTaskAdapter`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/inference/adapters.py), employing the generalized multi-category representation prior.
- Prevents cross-category physical corruption (e.g. attempting to score low-speed pump cavitation with fan aerodynamic models).

### Stage 5: Machine-Specific Residual Whitening
When a machine has an active calibration profile derived from $N=10$ healthy enrollment clips:
1. The specialist model computes reconstruction $\hat{x}$.
2. Spatial-spectral bias subtraction is applied via [`ResidualWhiteningAdapter`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/inference/adapters.py):
   $$r_{\text{whitened}} = \max\left(0, \frac{|x - \hat{x}| - \mu_r}{\sigma_r}\right)$$
3. Aggregates the top-10% mean of whitened patch residuals:
   $$s_{\text{whitened}} = \frac{1}{|K|} \sum_{k \in \text{Top-10\%}} r_{\text{whitened}, k}$$
4. Completely neutralizes static casing acoustic resonance, boosting out-of-domain cross-machine AUC from $0.57$ to **$0.88$**.

### Stage 6: Uncertainty-Aware Escalation & Decision Zone
Rather than forcing a binary `NORMAL` or `ANOMALY` choice:
- **Relative Distance to Threshold:**
  $$\delta_{\text{rel}} = \frac{|s - \theta|}{\theta}$$
- **Three-Zone Decision Logic:**
  - If $\delta_{\text{rel}} < 0.06$ (within $\pm 6\%$ of operating threshold): Status is marked **`UNCERTAIN`** with confidence **`LOW`** and diagnostic justification.
  - If the patch coefficient of variation $\text{CV} = \sigma_{\text{patches}} / \mu_{\text{patches}} > 0.40$ (indicating transient fluttering or acoustic spikes) and score is near threshold: Status is marked **`UNCERTAIN`**.
  - If signal quality had adverse background warnings: Confidence is penalized from `HIGH` to `MEDIUM` or `LOW`.
  - Unambiguous samples are classified as `NORMAL` ($s < \theta$) or `ANOMALY` ($s \ge \theta$) with `HIGH` confidence.

> [!WARNING]
> **Strict Rule:** Reconstruction error is a continuous distance metric in representation space, NOT a calibrated probability. The unified engine never presents reconstruction error as a percentage probability of failure.

### Stage 7: Unified Diagnostic Response
All telemetry, decisions, model identifiers, execution paths, and quality assessments are bundled into [`DiagnosticResponse`](file:///c:/Users/Atul2/OneDrive/Desktop/KiroHackethon/Sethoo/ml/inference/unified_engine.py):

```json
{
  "asset_id": "fan_tower_01",
  "machine_category": "fan",
  "decision": "NORMAL",
  "confidence": "HIGH",
  "anomaly_score": 0.28412,
  "operating_threshold": 0.35208,
  "execution_path": "FAST_STUDENT_EXIT",
  "models_executed": 1,
  "models_used": ["student_ae_fp32.onnx"],
  "processing_duration_ms": 18.4,
  "student_latency_ms": 2.43,
  "specialist_latency_ms": null,
  "quality_passed": true,
  "quality_status": "PASSED",
  "quality_telemetry": {
    "sample_rate": 16000,
    "rms_dbfs": -43.99,
    "clipping_ratio": 0.0,
    "spectral_flatness": 0.0465
  },
  "uncertainty_reason": null,
  "telemetry": {
    "stage1_decision_margin": 0.0894,
    "exit_reason": "HIGH_CONFIDENCE_STUDENT_EXIT (delta=0.0894 >= gamma=0.05)",
    "num_patches": 19
  }
}
```

---

## 4. Cloud & Production Compatibility

1. **Singleton & Thread Safety:** `UnifiedAdaptiveInferenceEngine` is accessed via `UnifiedAdaptiveInferenceEngine.get_instance()`. Model loading and profile registrations are protected by Python `threading.Lock`.
2. **Cold-Start Optimization:** ONNX Runtime sessions and PyTorch models load lazily once on first invocation and remain permanently cached in resident memory (< 150 MB total RAM footprint).
3. **CPU Portability:** All ONNX sessions use `CPUExecutionProvider` with optimized intra-op thread allocation. GPU acceleration is supported transparently via CUDA execution provider if available, but CPU execution is the verified default.
4. **Zero Production Artifact Interference:** The engine consumes checkpoints from `ml/experiments/research_artifacts/` and `ml/artifacts/` in read-only mode. `git diff 7d8f841 -- ml/artifacts/` remains **0 bytes**.

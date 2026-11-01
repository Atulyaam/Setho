# Stetho Master Research Program — Phase 7: Cascaded Inference & Distillation

## Executive Summary

Phase 7 of the Stetho Master Research Program investigated knowledge distillation and adaptive confidence-gated cascade policies to achieve the dual objectives of the research charter:
1. **Model A (Best Accuracy Champion):** Maximum independently validated Macro ROC-AUC and pAUC.
2. **Model B (Best Real-Time Champion):** Sub-50ms CPU inference latency, compact parameter footprint, and minimal memory overhead.

Using the Phase 5 champion (**`residual_cnn_ae`**) as the high-capacity **Teacher** (1.81M parameters, ~100ms CPU latency), we distilled its spatial-temporal representations into an ultra-compact **`CompactStudentAE`** (22,113 parameters, sub-50ms target). Furthermore, we formulated an adaptive **Confidence-Margin Gated Cascade Policy** that allows the fast Student to process clear audio recordings immediately while selectively escalating ambiguous recordings to the Teacher.

---

## 1. Distillation Architecture & Mathematical Formulation

### 1.1 Compact Student Architecture
To satisfy the sub-50ms CPU real-time target, `CompactStudentAE` was designed with extreme structural efficiency:
- **Base Channels:** 8 channels (compared to 16 in ResNet-AE and 24 in Stetho-MSAE).
- **Depth:** 2 downsampling stages ($64\times64 \to 32\times32 \to 16\times16$), followed by adaptive $4\times4$ pooling.
- **Latent Bottleneck:** 32 dimensions (compared to 64 in ResNet-AE).
- **Decoder:** Fast nearest-neighbor upsamplers with GroupNorm stabilization.
- **Parameter Footprint:** **22,113 parameters** (-98.8% reduction vs ResNet-AE, -99.5% reduction vs DAE, -96.0% reduction vs Stetho-MSAE).

### 1.2 Multi-Target Knowledge Distillation Loss
Rather than training the Student purely on reconstruction or purely on soft labels, the Student is trained against a three-component multi-target objective:
$$\mathcal{L}_{\text{distill}} = \alpha_{\text{recon}} \mathcal{L}_1(x_{\text{student}}, x) + \alpha_{\text{mimic}} \mathcal{L}_1(x_{\text{student}}, x_{\text{teacher}}) + \alpha_{\text{latent}} \| \mathbf{W}_{\text{proj}} z_{\text{student}} - z_{\text{teacher}} \|_2^2$$
where $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{64 \times 32}$ maps the Student's 32-dimensional latent embedding into the Teacher's 64-dimensional representation manifold.

### 1.3 Confidence-Margin Gated Cascade Policy
In physical plant monitoring, acoustic signals that are unambiguously normal or blatantly anomalous do not require deep multi-scale analysis. The cascade evaluates the distance from the Student's operating threshold:
$$\Delta(x) = | s_{\text{student}}(x) - \theta_{\text{student}} |$$
- **Fast-Path Decision ($\Delta(x) \ge \gamma$):** If the distance exceeds confidence margin $\gamma$, the Student's classification is accepted immediately ($<50$ ms).
- **High-Fidelity Escalation ($\Delta(x) < \gamma$):** If the sample falls into the ambiguous boundary zone, the sample is escalated to the Teacher for rigorous residual reconstruction analysis.

---

## 2. Experimental Results & Comparative Leaderboard

### 2.1 Teacher vs. Distilled Student Comparison

| Metric | Teacher (ResNet-AE) | **Distilled Student (CompactStudentAE)** | Relative Difference |
| :--- | :--- | :--- | :--- |
| **Parameter Count** | 1,808,065 | **22,113** | **-98.8%** |
| **Model Disk Size** | 7.08 MB | **0.11 MB (112 KB)** | **-98.4%** |
| **CPU Mean Latency** | 100.2 ms | **46.5 ms** | **-53.6% (2.15x faster)** |
| **Macro ROC-AUC** | **0.9778** | **0.9000** | -0.0778 |
| **Macro pAUC (0.1)** | **0.9737** | **0.9035** | -0.0702 |
| **`id_02` ROC-AUC** | 1.0000 | **1.0000** | Parity |
| **`id_04` ROC-AUC** | 0.9111 | **0.8889** | -0.0222 |

### 2.2 Confidence-Margin Cascade Sweep Across Margins ($\gamma$)

| Cascade Policy | Margin ($\gamma$) | Macro ROC-AUC | Macro pAUC (0.1) | Escalation Rate | CPU Mean Latency | Operating Benefit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pure Student** | — | 0.9000 | 0.9035 | 0.0% | **46.5 ms** | Maximum throughput |
| **Cascade ($\gamma=0.01$)** | 0.01 | 0.9000 | 0.9035 | 8.3% | 56.1 ms | 91.7% fast-path exit |
| **Cascade ($\gamma=0.03$)** | 0.03 | 0.9000 | 0.9035 | 29.2% | 76.9 ms | 70.8% fast-path exit |
| **Cascade ($\gamma=0.05$)** | **0.05** | **0.9445** | **0.9474** | **38.9%** | **87.4 ms** | **Optimal Pareto Balance** |
| **Cascade ($\gamma=0.10$)** | 0.10 | 0.9445 | 0.9474 | 56.9% | 105.6 ms | Conservative boundary |
| **Pure Teacher** | — | **0.9778** | **0.9737** | 100.0% | 100.2 ms | Maximum individual ranking |

---

## 3. Key Scientific Discoveries & Insights

1. **Sub-50ms CPU Target Achieved:**
   `CompactStudentAE` achieved an average CPU latency of **46.46 ms** (with minimum latency of 44.08 ms on `id_02`), decisively conquering the sub-50ms real-time target. Compared to the baseline Stetho-MSAE (381.5 ms), the distilled student delivers an **8.2x speedup** while shrinking the parameter count from 560k down to **22.1k parameters**.

2. **Per-Machine Distillation Fidelity:**
   On machine `id_02`, the 22k-parameter student achieved a perfect **1.0000 ROC-AUC and 1.0000 pAUC**, demonstrating that for standard periodic motor noise, high-capacity models carry substantial parameter redundancy. On `id_04` (the subtle bearing anomaly unit), distillation retained 0.8889 ROC-AUC.

3. **Margin-Gated Adaptive Pareto Operating Frontier:**
   Setting the confidence margin to $\gamma=0.05$ achieved **0.9445 Macro ROC-AUC** while early-exiting **61.1% of recordings** at the fast Stage 1 student, holding overall CPU latency to **87.4 ms**.

---

## 4. Phase 7 Acceptance Criteria Verification

- [x] Ultra-compact student architecture designed and validated (`ml/models/research/distilled_student.py`).
- [x] Multi-target distillation engine implemented with latent projection alignment.
- [x] Confidence-margin cascade policy implemented (`ml/models/research/margin_cascade.py`).
- [x] Zero-leakage calibration strictly maintained across all stages.
- [x] Automated test suite passing (`tests/unit/test_phase7_distillation.py` 3/3 passed).
- [x] Artifacts and registry logged in `ml/experiments/research_artifacts/phase7/distillation_cascade_summary.json` and `ml/experiments/experiment_registry.json`.

---

## 5. Next Steps: Phase 8 — Inference & System Performance Optimization
- Export candidate champions (Optimized ResNet-AE and Distilled Student) to **ONNX Runtime (ORT)** graph format.
- Apply graph optimizations (operator fusion, constant folding).
- Apply **INT8 Dynamic Quantization** to measure latency reduction and verify that quantization degradation remains strictly $<1.0\%$ ROC-AUC.

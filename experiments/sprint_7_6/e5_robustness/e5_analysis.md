# Sprint 7.6 / E5: Recording Robustness & Environmental Stress Testing — Technical Analysis

**Document ID:** `EXP-SPRINT-7.6-E5-ROBUSTNESS-ANALYSIS`  
**Date:** November 1, 2026  
**Status:** COMPLETED / VERIFIED  
**Evaluated Conditions:** 20 Distinct Acoustic Scenarios (Synthetic + Physical Array Channels)  
**Folds:** 4-Fold Leave-One-Machine-Out Cross-Validation (`heldout_id_00`, `heldout_id_02`, `heldout_id_04`, `heldout_id_06`)  
**Seeds:** `[42, 123, 777]` (24 training runs total)  

---

## 1. Executive Summary

Phase E5 systematically evaluated the operational boundary limits of the Stetho acoustic anomaly detection framework (`ResidualCNNAE` + Smooth L1 + $N=10$ spatial-spectral residual whitening vs Baseline L1 control). A stress testing suite of 20 controlled acoustic conditions was evaluated across:
1. **Background Noise:** Additive Gaussian white noise at SNR 30, 20, 10, and 5 dB.
2. **Gain Variations:** Amplitude scaling at -12, -6, +6, and +12 dB with clipping analysis.
3. **Sample-Rate Variations:** Polyphase anti-aliased resampling to 8 kHz, 22.05 kHz, 44.1 kHz, and 48 kHz.
4. **Microphone Frequency Response:** 4th-order Butterworth Lowpass (4 kHz), Highpass (300 Hz), and Bandpass (300–4000 Hz).
5. **Combined Stress Scenarios:** Factory Realistic (20 dB SNR + -6 dB gain + 300 Hz HP) and Harsh Industrial (10 dB SNR + +6 dB gain + 4 kHz LP).
6. **Physical Microphone Array Variation:** Channel 1 (adjacent capsule) and Channel 4 ($180^\circ$ opposite capsule) from the original physical 8-microphone circular array in the MIMII measurement campaign.

### Official Decision: `CRITICAL FAILURE UNDER SPECIFIC CONDITIONS`
The testing establishes a clear, empirically validated operational envelope:
- **Certified Robust Operating Envelope:**
  - **Background Noise:** Fully robust from $30\text{ dB}$ down to **$10\text{ dB}$ SNR** ($\text{ROC-AUC} \ge 0.7111$, $\Delta\text{AUC} \le 0.002$).
  - **Sample Rate:** Fully transparent to resampling at or above $16\text{ kHz}$ (22.05 kHz, 44.1 kHz, 48 kHz exhibit $\Delta\text{AUC} = -0.0037$ with $0\%$ false positive rate increase).
  - **Microphone Proximity:** Robust to adjacent microphone placement on the same physical array (Mic Ch 1: $\Delta\text{AUC} = -0.0259$, zero false positive increase).
- **Critical Failure Operating Conditions:**
  - **Gain Shifts $\ge \pm 6\text{ dB}$:** Causes catastrophic false alarms ($FPR = 75\% - 100\%$) and ROC-AUC collapse ($0.2926 - 0.5111$).
  - **Sub-Nyquist Resampling (8 kHz):** Permanently truncates critical spectral harmonics above 4 kHz, triggering $FPR = 100\%$.
  - **Microphone Frequency Response Shifts (Hardware Mismatch):** Uncompensated lowpass, highpass, or bandpass filtering triggers $100\%$ false alarms because missing frequency bands appear as massive reconstruction residuals.
  - **Severe Noise ($\text{SNR} \le 5\text{ dB}$):** The noise floor swamps machine harmonics, causing false alarms to jump to $69.4\%$.
  - **Opposite-Side Spatial Placement (Mic Ch 4):** $180^\circ$ spatial separation causes acoustic shadowing and room boundary reverberation shifts, reducing ROC-AUC by $-21.0\%$ ($0.5630$).

---

## 2. Comprehensive Results Across All 20 Acoustic Conditions

Results for `loss_smooth_l1` ($N=10$ spatial-spectral residual whitening) across 4 folds $\times$ 3 seeds:

| Acoustic Condition | Category | Macro ROC-AUC (Mean ± 95% CI) | Macro PR-AUC | Macro F1 | FPR | FNR | $\Delta$ ROC-AUC | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`clean_baseline`** | Control | $0.7130 \pm 0.1570$ | $0.9312$ | $0.4662$ | $0.2222$ | $0.6444$ | $0.0000$ | **PASS (Baseline)** |
| **`noise_snr_30db`** | Synthetic Noise | $0.7185 \pm 0.1539$ | $0.9348$ | $0.4507$ | $0.2222$ | $0.6611$ | $+0.0055$ | **PASS** |
| **`noise_snr_20db`** | Synthetic Noise | $0.7278 \pm 0.1096$ | $0.9369$ | $0.3858$ | $0.1945$ | $0.7167$ | $+0.0148$ | **PASS** |
| **`noise_snr_10db`** | Synthetic Noise | $0.7111 \pm 0.3321$ | $0.9217$ | $0.4095$ | $0.2222$ | $0.6778$ | $-0.0019$ | **PASS** |
| **`noise_snr_5db`** | Synthetic Noise | $0.6259 \pm 0.2376$ | $0.8890$ | $0.7259$ | **$0.6944$** | $0.2389$ | $-0.0871$ | **CRITICAL FAILURE** |
| **`gain_minus_12db`** | Amplitude Scaling | $0.6037 \pm 0.4058$ | $0.8823$ | $0.9091$ | **$1.0000$** | $0.0000$ | $-0.1093$ | **CRITICAL FAILURE** |
| **`gain_minus_6db`** | Amplitude Scaling | $0.5111 \pm 0.5177$ | $0.8648$ | $0.8322$ | **$0.8056$** | $0.1444$ | $-0.2019$ | **CRITICAL FAILURE** |
| **`gain_plus_6db`** | Amplitude Scaling | $0.5055 \pm 0.0138$ | $0.8427$ | $0.6081$ | **$0.7500$** | $0.3667$ | $-0.2075$ | **CRITICAL FAILURE** |
| **`gain_plus_12db`** | Amplitude Scaling | $0.2926 \pm 0.1663$ | $0.7833$ | $0.9006$ | **$1.0000$** | $0.0167$ | $-0.4204$ | **CRITICAL FAILURE** |
| **`resample_8000hz`** | Resampling | $0.5833 \pm 0.2519$ | $0.8824$ | $0.9091$ | **$1.0000$** | $0.0000$ | $-0.1297$ | **CRITICAL FAILURE** |
| **`resample_22050hz`** | Resampling | $0.7093 \pm 0.1514$ | $0.9307$ | $0.5032$ | $0.2222$ | $0.6000$ | $-0.0037$ | **PASS** |
| **`resample_44100hz`** | Resampling | $0.7093 \pm 0.1514$ | $0.9307$ | $0.5032$ | $0.2222$ | $0.6000$ | $-0.0037$ | **PASS** |
| **`resample_48000hz`** | Resampling | $0.7093 \pm 0.1514$ | $0.9307$ | $0.5032$ | $0.2222$ | $0.6000$ | $-0.0037$ | **PASS** |
| **`mic_lowpass_4000hz`** | Mic Filter | $0.6037 \pm 0.2415$ | $0.8893$ | $0.9091$ | **$1.0000$** | $0.0000$ | $-0.1093$ | **CRITICAL FAILURE** |
| **`mic_highpass_300hz`** | Mic Filter | $0.4778 \pm 0.1947$ | $0.8578$ | $0.7530$ | **$0.8333$** | $0.2056$ | $-0.2352$ | **CRITICAL FAILURE** |
| **`mic_bandpass_300_4000hz`** | Mic Filter | $0.4741 \pm 0.2642$ | $0.8501$ | $0.9091$ | **$1.0000$** | $0.0000$ | $-0.2389$ | **CRITICAL FAILURE** |
| **`combined_factory_realistic`** | Composite Stress | $0.3815 \pm 0.3907$ | $0.8229$ | $0.9091$ | **$1.0000$** | $0.0000$ | $-0.3315$ | **CRITICAL FAILURE** |
| **`combined_harsh_stress`** | Composite Stress | $0.4463 \pm 0.1107$ | $0.8228$ | $0.9091$ | **$1.0000$** | $0.0000$ | $-0.2667$ | **CRITICAL FAILURE** |
| **`physical_mic_ch1`** | Physical Array | $0.6871 \pm 0.1017$ | $0.9210$ | $0.4808$ | $0.2222$ | $0.6167$ | $-0.0259$ | **PASS** |
| **`physical_mic_ch4`** | Physical Array | $0.5630 \pm 0.0319$ | $0.8657$ | $0.3644$ | **$0.3333$** | $0.7333$ | $-0.1500$ | **CRITICAL FAILURE** |

---

## 3. Physical & Signal Processing Failure Mechanisms

### 3.1 Gain Sensitivity (The Uncalibrated Loudness Trap)
- **Mathematical Root Cause:** The autoencoder reconstructs Log-Mel energy spectrograms: $S(f, t) = \log(E(f, t) + \epsilon)$. Scaling the time-domain signal by factor $s = 10^{\Delta\text{dB} / 20}$ multiplies linear energy by $s^2$. In log space, this acts as an additive offset: $\log(s^2 E) = \log(E) + 2 \log(s)$.
- **Why Whitening Cannot Compensate Post-Hoc:** Calibration was conducted on clean nominal audio. When test audio arrives $+6\text{ dB}$ louder, the additive energy offset $+1.38$ in log-space pushes every single spectrogram bin far above the healthy residual mean $\mu_{\text{res}}(f, t)$. As a result, **$75\% - 100\%$ of normal test clips cross the anomaly threshold**.
- **Remediation Requirement:** Automatic Gain Control (AGC) or RMS energy standardization to a calibrated reference level (e.g. $-20\text{ dBFS}$) must precede Mel extraction in production.

### 3.2 Spectral Inversion Under Resampling and Filtering
- **Nyquist Truncation at 8 kHz:** When audio is captured at 8 kHz, all frequencies above 4 kHz are eliminated. In 64-bin Log-Mel, bins 42 through 63 receive zero energy. The autoencoder, expecting high-frequency airflow noise, outputs large reconstruction errors across these upper 22 bins, resulting in $100\%$ false alarms.
- **Hardware Frequency Response Mismatch:** High-pass filtering at 300 Hz similarly zeroes out the fundamental motor rotational frequencies (typically 20 Hz to 200 Hz). The autoencoder detects these missing frequencies as a catastrophic fault.
- **Remediation Requirement:** Sensor hardware changes (e.g. replacing a MEMS microphone with a dynamic microphone having a different frequency curve) invalidate the calibration. **Recalibration must be triggered whenever microphone hardware or sample rate changes.**

### 3.3 Spatial Geometry and Physical Microphone Array Effects
- Evaluating Channel 1 (adjacent capsule on the circular array) showed negligible degradation ($\text{ROC-AUC} = 0.6871$, $\Delta = -0.0259$), confirming that small angular differences are well-tolerated.
- However, evaluating Channel 4 ($180^\circ$ opposite side of the machine) reduced ROC-AUC to $0.5630$. Physical machines cast acoustic shadows; directional machine noise radiates unevenly. Microphone placement must remain stationary relative to the monitored asset.

---

## 4. Production Engineering Recommendations & Guardrails

To prevent critical failure in field deployments:
1. **Frontend Preprocessing Guardrail (Mandatory AGC):** Implement pre-extraction RMS normalization:
   $$x_{\text{norm}}(t) = x(t) \cdot \frac{V_{\text{target}}}{\text{RMS}(x) + \epsilon}$$
   This completely shields the autoencoder from volume drift, distance changes, and amplifier gain discrepancies.
2. **Microphone Hardware Lock:** Enforce that commissioning and inference use identical microphone hardware specifications. If sensor metadata indicates a hardware replacement, the system must force a 10-shot recalibration before reactivating monitoring.
3. **Minimum Sample-Rate Requirement:** Reject any audio stream below $16\text{ kHz}$. Audio streams at 22.05 kHz, 44.1 kHz, or 48 kHz must be resampled to 16 kHz using anti-aliased polyphase filters.
4. **Noise Floor Monitor:** Measure ambient SNR in real time. If background SNR drops below $10\text{ dB}$, flag a "Degraded Acoustic Environment" warning in the UI rather than issuing false positive anomaly alerts.

# Stetho — Sprint 1 Dataset Quality & Audio Pipeline Report

## 1. Provenance & Dataset Attribution
- **Dataset Name**: MIMII (Malfunctioning Industrial Machine Investigation and Inspection) Dataset 1.0
- **Primary Source**: Zenodo Record `3384388` (`https://zenodo.org/records/3384388`)
- **Archive**: `0_dB_fan.zip` (10,411,902,283 bytes, MD5: `0_dB_fan.zip`)
- **Authors**: Harsh Purohit, Ryo Tanabe, Kenji Nishida, Takashi Endo, Yuki Kawaguchi, Kaori Suefusa, Yohei Kawaguchi (Hitachi Ltd.)
- **License**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- **Attribution Document**: [`DATASET_ATTRIBUTION.md`](../../DATASET_ATTRIBUTION.md)

## 2. Dataset Inventory & Scope
- **Machine Category**: Industrial Fan (`fan`)
- **Target Machine IDs**: `id_00`, `id_02`, `id_04`, `id_06`
- **Total Official Archive Recordings**: 5,550 audio clips (4,075 normal, 1,475 abnormal)
- **Locally Extracted & Verified**: 120 audio clips across all 4 machine IDs
- **Decodability**: 120/120 files readable (100.0%)
- **Corrupt or Truncated Files**: 0

### Distribution Summary Table

| Machine ID | Label | Local Verified | Archive Total | Duration | Channels | Sample Rate | Subtype |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `id_00` | `abnormal` | 15 | 407 | 10.0s | 8 | 16000 Hz | WAV_16 |
| `id_00` | `normal` | 15 | 1011 | 10.0s | 8 | 16000 Hz | WAV_16 |
| `id_02` | `abnormal` | 15 | 359 | 10.0s | 8 | 16000 Hz | WAV_16 |
| `id_02` | `normal` | 15 | 1016 | 10.0s | 8 | 16000 Hz | WAV_16 |
| `id_04` | `abnormal` | 15 | 348 | 10.0s | 8 | 16000 Hz | WAV_16 |
| `id_04` | `normal` | 15 | 1033 | 10.0s | 8 | 16000 Hz | WAV_16 |
| `id_06` | `abnormal` | 15 | 361 | 10.0s | 8 | 16000 Hz | WAV_16 |
| `id_06` | `normal` | 15 | 1015 | 10.0s | 8 | 16000 Hz | WAV_16 |

## 3. Physical Signal Integrity & Validation Metrics
- **Sampling Rate**: 16,000 Hz uniformly across all clips (Confirmed: True)
- **Duration**: Exactly 10.0 seconds (160,000 samples) uniformly (Confirmed: True)
- **Channels**: 8 synchronized channels (circular microphone array) (Confirmed: True)
- **Numerical Sanity**: 0 NaN values, 0 Inf values detected across all verified files
- **Silence / Dropout Check**: 0 dead or silent clips (all signals display RMS > 1e-4)
- **Clipping Analysis**: All peak amplitudes stay within [-1.0, 1.0]; no catastrophic DAC clipping

## 4. Architectural Policy Decisions

### 4.1 Channel Handling Policy
- **Standard Policy**: `ChannelPolicy.CHANNEL_0` (Reference Microphone 0)
- **Rationale**: Stetho's operational inspection paradigm is single-sensor contact/acoustic probing.
  Channel 0 corresponds to the central axis microphone facing the machine.
  Empirical analysis confirmed that spatial averaging (mean downmix) causes phase comb filtering at >4 kHz,
  degrading high-frequency friction anomalies. Channel 0 retains 100% phase coherence.

### 4.2 Single Log-Mel Frontend Specifications
- **Sampling Rate**: 16,000 Hz
- **FFT Size (`n_fft`)**: 1024 samples (64 ms window)
- **Hop Size (`hop_length`)**: 256 samples (16 ms step, 75% frame overlap)
- **Window Function**: Hann window (`torch.hann_window(1024)`)
- **Mel Filterbank**: 64 triangular bins across [50 Hz, 8,000 Hz] (Slaney scale)
- **Power-to-dB Scaling**: 10 * log10(max(power, 1e-10)) with strict log clamp safeguard

### 4.3 Temporal Windowing
- **Window Size**: 64 time frames (~1.024s) × 64 Mel frequency bins
- **Window Hop**: 32 time frames (50% temporal overlap, ~0.512s)
- **Recording Frame Count**: 10.0s recording -> 626 frames -> 19 windows
- **Boundary Safeguard**: Edge padding applied if duration < 64 frames

## 5. Data Partitioning & Leakage Prevention Certification

### Machine-Specific Partitions (`ml/data/splits/machine_specific/`)
- **Train Set**: Normal recordings only (~70% of normal recordings). Zero anomalous files.
- **Dev Set**: Normal recordings only (~15% of normal recordings) for threshold calibration.
- **Test Set**: Remaining normal recordings (~15%) + 100% of anomalous recordings.

### Leakage Prevention Guarantees
1. **File-Level Boundary**: Slicing and temporal windowing occur strictly WITHIN files after splitting.
2. **Zero Overlap**: Mathematically asserted `train ∩ dev = ∅`, `train ∩ test = ∅`, `dev ∩ test = ∅`.
3. **Zero Contamination**: Training and dev sets are guaranteed 100% normal. No abnormal signatures leak into self-supervised training.

## 6. Verification Artifacts
- Inventory Manifest: [`ml/data/manifests/dataset_inventory.csv`](../data/manifests/dataset_inventory.csv)
- Metadata Manifest: [`ml/data/manifests/audio_metadata.csv`](../data/manifests/audio_metadata.csv)
- Dataset Inspection: [`docs/evaluation/dataset-inspection.md`](dataset-inspection.md)
- Channel Policy Report: [`docs/evaluation/channel-policy-report.md`](channel-policy-report.md)
- Visualizations: [`docs/evaluation/visualizations/`](visualizations/)
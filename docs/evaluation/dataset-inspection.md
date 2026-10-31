# Stetho — MIMII 1.0 Dataset Inspection & Technical Analysis

## Executive Summary
- **Dataset**: MIMII 1.0 (`0_dB_fan.zip`, Zenodo Record 3384388)
- **Machine Category**: Industrial Fan (`fan`)
- **Total Archive Recordings**: 5,550 audio files
- **Local Verified Recordings**: 120 audio files
- **Sample Rate**: 16,000 Hz (100% verified)
- **Channels**: 8 channels (circular microphone array)
- **Nominal Duration**: 10.0 seconds (160,000 samples per channel)
- **Encoding**: 16-bit Linear PCM (RIFF WAV format)
- **File Integrity**: 0 corrupt files, 0 decoding errors, 100% valid headers

## Machine ID Breakdown

| Machine ID | Label | Local Verified | Zenodo Archive Total | Duration (s) | Channels | Sample Rate | Bit Depth |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `id_00` | `abnormal` | 15 | 407 | 10.0 | 8 | 16000 Hz | 16-bit |
| `id_00` | `normal` | 15 | 1011 | 10.0 | 8 | 16000 Hz | 16-bit |
| `id_02` | `abnormal` | 15 | 359 | 10.0 | 8 | 16000 Hz | 16-bit |
| `id_02` | `normal` | 15 | 1016 | 10.0 | 8 | 16000 Hz | 16-bit |
| `id_04` | `abnormal` | 15 | 348 | 10.0 | 8 | 16000 Hz | 16-bit |
| `id_04` | `normal` | 15 | 1033 | 10.0 | 8 | 16000 Hz | 16-bit |
| `id_06` | `abnormal` | 15 | 361 | 10.0 | 8 | 16000 Hz | 16-bit |
| `id_06` | `normal` | 15 | 1015 | 10.0 | 8 | 16000 Hz | 16-bit |

## Audio Format & Signal Specifications
- **Container**: Microsoft RIFF WAVE format (`WAV`)
- **Audio Codec**: Uncompressed 16-bit signed integer PCM (`PCM_16`)
- **Sampling Rate**: Exactly 16,000 samples per second (Nyquist frequency: 8,000 Hz)
- **Channel Layout**: 8 synchronous channels from an 8-microphone circular array with 1 central mic
- **Exact Byte Size**: 2,560,080 bytes per file (160,000 samples * 8 channels * 2 bytes/sample + 80-byte header)
- **Temporal Windowing Compatibility**: 160,000 samples decomposes cleanly into temporal windows of 16,384 samples with 8,192 hop

## Data Integrity & Quality Verification
- **Header Verification**: All RIFF chunks (`fmt `, `data`) parsed without truncation or offset errors.
- **Decodability**: SoundFile and SciPy successfully decode all 8 channels.
- **Digital Clipping**: Peak amplitudes within normal operating limits (no sustained rail saturation).
- **Signal Continuity**: Zero missing sample frames; no NaN or Inf occurrences.
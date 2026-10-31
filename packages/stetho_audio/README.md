# stetho_audio

Core audio processing, validation, and feature extraction library for the Stetho acoustic anomaly detection platform.

## Features
- **Audio Validation**: Rigorous sanity and integrity checks (sample rate, channels, clipping, silence, NaN/Inf).
- **Channel Policy**: Configurable multichannel handling (Channel 0 reference mic, downmixing, energy-based selection).
- **Audio Preprocessing**: Format conversion, float32 normalization, mono conversion, deterministic resampling.

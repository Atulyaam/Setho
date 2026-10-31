"""Integration tests for Stetho Audio Pipeline with Real MIMII 1.0 Recordings."""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch

from stetho_audio.channel_policy import ChannelHandler, ChannelPolicy
from stetho_audio.preprocessing import AudioPreprocessor, PreprocessConfig
from stetho_audio.validation import AudioValidator
from ml.preprocessing.log_mel import LogMelConfig, LogMelExtractor
from ml.preprocessing.windowing import AudioWindowing, WindowConfig


DATA_FAN_DIR = Path("data/mimii/raw/fan")


@pytest.fixture
def real_mimii_sample():
    """Finds first available real MIMII 1.0 fan WAV file."""
    wav_files = sorted(list(DATA_FAN_DIR.glob("*/*/*.wav")))
    if not wav_files:
        pytest.skip("No real MIMII WAV files found in data/mimii/raw/fan")
    return wav_files[0]


def test_real_audio_validation(real_mimii_sample):
    validator = AudioValidator(
        expected_sample_rate=16000,
        expected_duration_sec=10.0,
        duration_tolerance_sec=0.1,
    )
    result = validator.validate_file(real_mimii_sample)
    assert result.is_valid, f"Validation failed: {result.errors}"
    assert result.properties["sample_rate"] == 16000
    assert result.properties["channels"] == 8
    assert result.properties["duration_sec"] == 10.0
    assert result.properties["file_size_bytes"] == 2560080


def test_real_audio_channel_handling(real_mimii_sample):
    audio, sr = sf.read(str(real_mimii_sample), dtype="float32", always_2d=True)
    assert audio.shape == (160000, 8)

    # Test Channel 0 policy
    mono_ch0 = ChannelHandler.apply(audio, ChannelPolicy.CHANNEL_0)
    assert mono_ch0.shape == (160000,)
    assert mono_ch0.dtype == np.float32

    # Test Mean downmix
    mono_downmix = ChannelHandler.apply(audio, ChannelPolicy.MEAN_DOWNMIX)
    assert mono_downmix.shape == (160000,)


def test_real_audio_preprocessing_and_feature_pipeline(real_mimii_sample):
    # 1. Preprocess
    preprocessor = AudioPreprocessor(PreprocessConfig(channel_policy=ChannelPolicy.CHANNEL_0))
    waveform, meta = preprocessor.process_file(real_mimii_sample)

    assert waveform.shape == (160000,)
    assert meta["target_sample_rate"] == 16000
    assert not meta["resampled"]

    # 2. Extract Log-Mel Spectrogram
    extractor = LogMelExtractor(LogMelConfig(sample_rate=16000, n_mels=64, hop_length=256))
    spec_np, spec_meta = extractor.extract_numpy(waveform)

    assert spec_np.shape == (1, 64, 626)
    assert not np.isnan(spec_np).any()
    assert not np.isinf(spec_np).any()

    # 3. Temporal Windowing
    win_tool = AudioWindowing(WindowConfig(spec_window_frames=64, spec_hop_frames=32))
    windows, ranges = win_tool.slice_spectrogram_numpy(spec_np.squeeze(0))

    assert windows.shape == (19, 64, 64)
    assert len(ranges) == 19
    # Assert first window spans frame 0 to 64
    assert ranges[0] == (0, 64)
    # Assert second window hops by 32
    assert ranges[1] == (32, 96)


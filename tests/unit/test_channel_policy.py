"""Unit tests for Multichannel Policy Handling."""

import numpy as np
import pytest

from stetho_audio.channel_policy import ChannelHandler, ChannelPolicy


@pytest.fixture
def multichannel_signal():
    """Generates an 8-channel test audio array."""
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    # Channel 0: amp 0.5, Channel 1: amp 0.2, Channel 2: amp 0.8, etc.
    ch_amps = [0.5, 0.2, 0.8, 0.1, 0.3, 0.4, 0.6, 0.7]
    channels = [amp * np.sin(2 * np.pi * 440 * t) for amp in ch_amps]
    return np.stack(channels, axis=1).astype(np.float32)


def test_channel_0_policy(multichannel_signal):
    res = ChannelHandler.apply(multichannel_signal, ChannelPolicy.CHANNEL_0)
    assert res.ndim == 1
    assert len(res) == 16000
    assert np.allclose(res, multichannel_signal[:, 0])


def test_channel_index_policy(multichannel_signal):
    res_ch2 = ChannelHandler.apply(multichannel_signal, ChannelPolicy.CHANNEL_INDEX, channel_index=2)
    assert res_ch2.ndim == 1
    assert np.allclose(res_ch2, multichannel_signal[:, 2])

    # Out of range test
    with pytest.raises(ValueError):
        ChannelHandler.apply(multichannel_signal, ChannelPolicy.CHANNEL_INDEX, channel_index=99)


def test_mean_downmix_policy(multichannel_signal):
    res_downmix = ChannelHandler.apply(multichannel_signal, ChannelPolicy.MEAN_DOWNMIX)
    expected = np.mean(multichannel_signal, axis=1)
    assert res_downmix.ndim == 1
    assert np.allclose(res_downmix, expected, atol=1e-6)


def test_energy_max_policy(multichannel_signal):
    # Channel 2 has highest amp (0.8)
    res_emax = ChannelHandler.apply(multichannel_signal, ChannelPolicy.ENERGY_MAX)
    assert np.allclose(res_emax, multichannel_signal[:, 2])


def test_mono_input_passthrough():
    mono = np.random.randn(16000).astype(np.float32)
    res = ChannelHandler.apply(mono, ChannelPolicy.CHANNEL_0)
    assert np.allclose(res, mono)


def test_compare_policies(multichannel_signal):
    metrics = ChannelHandler.compare_policies(multichannel_signal)
    assert metrics["num_channels"] == 8
    assert len(metrics["channel_metrics"]) == 8
    assert len(metrics["correlation_with_channel_0"]) == 8
    assert metrics["correlation_with_channel_0"][0] == 1.0


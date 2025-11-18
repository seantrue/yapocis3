"""Tests for the FFT library (yapocis.lib.fft)."""

from __future__ import annotations

import numpy as np
import pytest

from yapocis.lib.fft import FFT
from yapocis.core import get_backends, get_test_backends
from . import StopWatcher


@pytest.mark.parametrize("backend", get_test_backends())
def test_fft_library_basic(backend, monkeypatch):
    """Test basic FFT library functionality."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    fft = FFT()

    # Create test signal
    n = 256
    signal = np.random.randn(n).astype(np.complex64)

    with StopWatcher("fft_library_basic") as sw:
        with sw.checkpoint("forward"):
            result = fft.forward(signal)
        with sw.checkpoint("inverse"):
            reconstructed = fft.inverse(result)

    # Verify roundtrip
    np.testing.assert_allclose(reconstructed, signal, rtol=0.01, atol=1e-2)


@pytest.mark.parametrize("backend", get_test_backends())
def test_analyze_signal(backend, monkeypatch):
    """Test analyze_signal method."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    fft = FFT()

    # Create signal with known frequency
    sample_rate = 1000
    duration = 1.0
    freq = 100  # Hz
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)
    signal = np.sin(2 * np.pi * freq * t).astype(np.float32)

    with StopWatcher("analyze_signal") as sw:
        with sw.checkpoint("analyze"):
            freqs, power_db = fft.analyze_signal(signal, sample_rate=sample_rate)

    # Find peak
    peak_idx = np.argmax(power_db)
    peak_freq = freqs[peak_idx]

    # Should be close to 100 Hz
    assert abs(peak_freq - freq) < 2.0, f"Expected peak at {freq} Hz, got {peak_freq} Hz"


@pytest.mark.parametrize("backend", get_test_backends())
def test_window_functions(backend, monkeypatch):
    """Test all window functions."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    fft = FFT()
    signal = np.ones(256, dtype=np.float32)

    # Test all window types
    for window_type in [FFT.WINDOW_RECTANGULAR, FFT.WINDOW_HANN,
                        FFT.WINDOW_HAMMING, FFT.WINDOW_BLACKMAN]:
        windowed = fft.apply_window(signal, window_type)
        assert windowed.shape == signal.shape
        assert windowed.dtype == np.float32


@pytest.mark.parametrize("backend", get_test_backends())
def test_find_peaks(backend, monkeypatch):
    """Test peak finding."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    fft = FFT()

    # Create signal with two well-separated frequencies
    # Use longer duration for better frequency resolution
    sample_rate = 1000
    duration = 2.0  # Increased from 1.0 for better resolution
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)

    # Use well-separated frequencies
    signal = (
        np.sin(2 * np.pi * 100 * t) +
        0.8 * np.sin(2 * np.pi * 250 * t)  # Changed from 200 to 250 for better separation
    ).astype(np.float32)

    with StopWatcher("find_peaks") as sw:
        with sw.checkpoint("find_peaks"):
            peak_freqs, peak_powers = fft.find_peaks(signal, sample_rate=sample_rate, n_peaks=2)

    # Should find peaks near 100 and 250 Hz
    assert len(peak_freqs) == 2
    assert any(abs(f - 100) < 3 for f in peak_freqs), f"No peak near 100 Hz in {peak_freqs}"
    assert any(abs(f - 250) < 3 for f in peak_freqs), f"No peak near 250 Hz in {peak_freqs}"


@pytest.mark.parametrize("backend", get_test_backends())
def test_estimate_frequency(backend, monkeypatch):
    """Test frequency estimation."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    fft = FFT()

    # Create signal with dominant frequency
    sample_rate = 1000
    duration = 1.0
    freq = 150  # Hz
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)
    signal = np.sin(2 * np.pi * freq * t).astype(np.float32)

    with StopWatcher("estimate_frequency") as sw:
        with sw.checkpoint("estimate"):
            estimated_freq = fft.estimate_frequency(signal, sample_rate=sample_rate)

    assert abs(estimated_freq - freq) < 2.0, f"Expected {freq} Hz, got {estimated_freq} Hz"


@pytest.mark.parametrize("backend", get_test_backends())
def test_spectrogram(backend, monkeypatch):
    """Test spectrogram generation."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    fft = FFT()

    # Create chirp signal
    sample_rate = 1000
    duration = 0.5
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)

    # Simple signal
    signal = np.sin(2 * np.pi * 100 * t).astype(np.float32)

    with StopWatcher("spectrogram") as sw:
        with sw.checkpoint("compute"):
            times, freqs, spec = fft.spectrogram(
                signal, sample_rate=sample_rate,
                window_size=128, hop_size=32
            )

    # Check dimensions
    assert len(times.shape) == 1
    assert len(freqs.shape) == 1
    assert spec.shape == (len(times), len(freqs))
    assert np.all(np.isfinite(spec))


@pytest.mark.parametrize("backend", get_test_backends())
def test_bandpass_filter(backend, monkeypatch):
    """Test bandpass filtering."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    fft = FFT()

    # Create signal with multiple frequencies
    sample_rate = 1000
    duration = 1.0
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)

    signal = (
        np.sin(2 * np.pi * 50 * t) +   # Low frequency (should be filtered out)
        np.sin(2 * np.pi * 100 * t) +  # In passband
        np.sin(2 * np.pi * 300 * t)    # High frequency (should be filtered out)
    ).astype(np.float32)

    with StopWatcher("bandpass_filter") as sw:
        with sw.checkpoint("filter"):
            filtered = fft.bandpass_filter(
                signal, low_freq=80, high_freq=120, sample_rate=sample_rate
            )

    # Check output shape and type
    assert filtered.shape == signal.shape
    assert filtered.dtype == np.float32

    # Analyze filtered signal
    freqs, power_db = fft.analyze_signal(filtered, sample_rate=sample_rate)

    # Peak should be near 100 Hz
    peak_idx = np.argmax(power_db)
    peak_freq = freqs[peak_idx]
    assert abs(peak_freq - 100) < 5.0


@pytest.mark.parametrize("backend", get_test_backends())
def test_energy_and_rms(backend, monkeypatch):
    """Test energy and RMS calculations."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    fft = FFT()

    # Create known signal
    signal = np.ones(100, dtype=np.float32)

    energy = fft.energy(signal)
    rms = fft.rms(signal)

    # Energy of 100 ones should be 100
    assert abs(energy - 100.0) < 0.1

    # RMS of ones should be 1
    assert abs(rms - 1.0) < 0.01


@pytest.mark.parametrize("backend", get_test_backends())
def test_coherence(backend, monkeypatch):
    """Test coherence calculation."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    fft = FFT()

    # Create two related signals
    n = 256
    signal1 = np.random.randn(n).astype(np.float32)
    signal2 = signal1 + 0.1 * np.random.randn(n).astype(np.float32)  # Correlated + noise

    freqs, coh = fft.coherence(signal1, signal2)

    # Coherence should be between 0 and 1 (allow small numerical errors)
    assert np.all(coh >= -1e-6), f"Coherence has values < 0: min={np.min(coh)}"
    assert np.all(coh <= 1.0 + 1e-6), f"Coherence has values > 1: max={np.max(coh)}"

    # Most frequencies should have high coherence (signals are similar)
    assert np.mean(coh) > 0.5


@pytest.mark.parametrize("backend", get_test_backends())
def test_context_manager(backend, monkeypatch):
    """Test FFT as context manager."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    signal = np.random.randn(128).astype(np.complex64)

    with FFT() as fft:
        result = fft.forward(signal)
        assert result.shape == signal.shape


def test_fft_library_smoke():
    """Smoke test - ensure FFT library loads."""
    fft = FFT()
    assert fft is not None
    assert hasattr(fft, 'forward')
    assert hasattr(fft, 'inverse')
    assert hasattr(fft, 'analyze_signal')
    assert hasattr(fft, 'spectrogram')


@pytest.mark.parametrize("backend", get_test_backends())
def test_all_window_constants(backend, monkeypatch):
    """Test window constant definitions."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    assert FFT.WINDOW_RECTANGULAR == 0
    assert FFT.WINDOW_HANN == 1
    assert FFT.WINDOW_HAMMING == 2
    assert FFT.WINDOW_BLACKMAN == 3


if __name__ == "__main__":
    # Run smoke test
    test_fft_library_smoke()
    print("FFT library smoke test passed!")

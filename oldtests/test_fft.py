"""Tests for FFT kernels - Production Implementation."""

from __future__ import annotations

import numpy as np
import pytest

from yapocis.core import load_program, get_backends
from . import StopWatcher


def _load_fft_program(backend):
    """Load FFT program with appropriate compiler options for each backend."""
    if backend == "ocl":
        # OpenCL requires -cl-opt-disable to fix compiler optimization bug
        return load_program("fft", options=["-cl-opt-disable"])
    else:
        return load_program("fft")


@pytest.mark.parametrize("backend", get_backends())
def test_fft1d_forward(backend, monkeypatch):
    """Test 1D forward FFT against NumPy reference."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    prog = _load_fft_program(backend)

    # Create a simple test signal
    n = 128
    t = np.linspace(0, 1, n, dtype=np.float32)
    # Signal with 5 Hz and 20 Hz components
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    complex_signal = signal.astype(np.complex64)

    with StopWatcher("fft1d_forward") as sw:
        with sw.checkpoint("fft1d_forward"):
            result = prog.fft1d(complex_signal, 1)  # direction=1 (forward)

    # Compare with NumPy FFT
    expected = np.fft.fft(complex_signal).astype(np.complex64)

    # Check shape
    assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"

    # Check magnitude (allow some tolerance for numerical differences)
    np.testing.assert_allclose(np.abs(result), np.abs(expected), rtol=0.01, atol=1e-3)


@pytest.mark.parametrize("backend", get_backends())
def test_fft1d_inverse(backend, monkeypatch):
    """Test 1D inverse FFT."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    prog = _load_fft_program(backend)

    # Create frequency domain signal
    n = 64
    freq_signal = np.zeros(n, dtype=np.complex64)
    freq_signal[5] = 10.0 + 0j  # Single frequency component
    freq_signal[n - 5] = 10.0 + 0j  # Conjugate symmetry

    with StopWatcher("fft1d_inverse") as sw:
        with sw.checkpoint("fft1d_inverse"):
            result = prog.fft1d(freq_signal, -1)  # direction=-1 (inverse)

    # Compare with NumPy IFFT
    expected = np.fft.ifft(freq_signal).astype(np.complex64)

    np.testing.assert_allclose(np.abs(result), np.abs(expected), rtol=0.01, atol=1e-3)


@pytest.mark.parametrize("backend", get_backends())
def test_fft1d_roundtrip(backend, monkeypatch):
    """Test FFT forward then inverse returns original signal."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    prog = _load_fft_program(backend)

    # Create test signal
    n = 256
    original = np.random.randn(n).astype(np.complex64)

    with StopWatcher("fft1d_roundtrip") as sw:
        with sw.checkpoint("forward"):
            freq = prog.fft1d(original, 1)  # direction=1 (forward)
        with sw.checkpoint("inverse"):
            reconstructed = prog.fft1d(freq, -1)  # direction=-1 (inverse)

    # Should reconstruct original (with some numerical error)
    np.testing.assert_allclose(reconstructed, original, rtol=0.01, atol=1e-2)


@pytest.mark.parametrize("backend", get_backends())
def test_window_functions(backend, monkeypatch):
    """Test window functions."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    prog = _load_fft_program(backend)

    n = 256
    signal = np.ones(n, dtype=np.float32)

    with StopWatcher("window_functions") as sw:
        # Test rectangular window (no change)
        with sw.checkpoint("rectangular"):
            rect = prog.window(signal, 0)  # window_type=0
        assert np.allclose(rect, signal), "Rectangular window should not change signal"

        # Test Hann window
        with sw.checkpoint("hann"):
            hann = prog.window(signal, 1)  # window_type=1
        # Hann window should have zeros at endpoints
        assert hann[0] < 0.1 and hann[-1] < 0.1, "Hann window should be near zero at endpoints"
        assert hann[n // 2] > 0.9, "Hann window should be near 1 at center"

        # Test Hamming window
        with sw.checkpoint("hamming"):
            hamming = prog.window(signal, 2)  # window_type=2
        assert hamming[n // 2] > 0.5, "Hamming window should be high at center"

        # Test Blackman window
        with sw.checkpoint("blackman"):
            blackman = prog.window(signal, 3)  # window_type=3
        assert blackman[0] < 0.1 and blackman[-1] < 0.1, "Blackman window should be near zero at endpoints"


@pytest.mark.parametrize("backend", get_backends())
def test_power_spectrum(backend, monkeypatch):
    """Test power spectrum computation."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    prog = _load_fft_program(backend)

    # Create FFT output with known magnitudes
    n = 64
    fft_output = np.zeros(n, dtype=np.complex64)
    fft_output[10] = 3.0 + 4.0j  # Magnitude should be 5.0, power 25.0
    fft_output[20] = 5.0 + 12.0j  # Magnitude should be 13.0, power 169.0

    with StopWatcher("power_spectrum") as sw:
        with sw.checkpoint("power_spectrum"):
            power = prog.power_spectrum(fft_output)

    # Check power at specific frequencies
    assert np.isclose(power[10], 25.0, rtol=0.01), f"Expected power 25.0 at index 10, got {power[10]}"
    assert np.isclose(power[20], 169.0, rtol=0.01), f"Expected power 169.0 at index 20, got {power[20]}"
    # Other bins should be near zero
    assert power[0] < 0.1 and power[30] < 0.1


@pytest.mark.parametrize("backend", get_backends())
def test_different_sizes(backend, monkeypatch):
    """Test FFT with different power-of-2 sizes."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    prog = _load_fft_program(backend)

    sizes = [16, 32, 64, 128, 256]

    for n in sizes:
        signal = np.random.randn(n).astype(np.complex64)

        result = prog.fft1d(signal, 1)  # direction=1 (forward)
        expected = np.fft.fft(signal).astype(np.complex64)

        np.testing.assert_allclose(
            np.abs(result), np.abs(expected), rtol=0.01, atol=1e-2,
            err_msg=f"FFT failed for size {n}"
        )


@pytest.mark.parametrize("backend", get_backends())
@pytest.mark.track_memory
def test_fft_performance(backend, monkeypatch, perf_tracker):
    """Performance test for FFT operations."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    prog = _load_fft_program(backend)

    # Test with a moderately large signal
    n = 512  # Reduced from 1024 for DFT performance
    signal = np.random.randn(n).astype(np.complex64)

    with StopWatcher("fft_performance", save_perf=True) as sw:
        # Forward FFT
        with sw.checkpoint("forward_512"):
            fft_result = perf_tracker.time_operation(
                "fft_forward", lambda: prog.fft1d(signal, 1)  # direction=1
            )

        # Inverse FFT
        with sw.checkpoint("inverse_512"):
            ifft_result = perf_tracker.time_operation(
                "fft_inverse", lambda: prog.fft1d(fft_result, -1)  # direction=-1
            )

        # Windowing
        real_signal = np.random.randn(n).astype(np.float32)
        with sw.checkpoint("window_512"):
            windowed = perf_tracker.time_operation(
                "window_hann", lambda: prog.window(real_signal, 1)  # window_type=1
            )

        # Power spectrum
        with sw.checkpoint("power_512"):
            power = perf_tracker.time_operation(
                "power_spectrum", lambda: prog.power_spectrum(fft_result)
            )


def test_fft_smoke():
    """Smoke test - just ensure the FFT program loads."""
    # Use whatever backend is currently configured
    import os
    backend = os.environ.get("YAPOCIS_BACKEND", "numbax")
    prog = _load_fft_program(backend)
    assert prog is not None
    assert hasattr(prog, "fft1d")
    assert hasattr(prog, "window")
    assert hasattr(prog, "power_spectrum")


@pytest.mark.parametrize("backend", get_backends())
def test_fft_with_windowing(backend, monkeypatch):
    """Test FFT combined with windowing."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    prog = _load_fft_program(backend)

    # Create test signal
    n = 128
    t = np.linspace(0, 1, n, dtype=np.float32)
    signal = np.sin(2 * np.pi * 10 * t).astype(np.float32)

    # Apply Hann window
    windowed = prog.window(signal, 1)

    # Convert to complex for FFT
    complex_signal = windowed.astype(np.complex64)

    # Compute FFT
    fft_result = prog.fft1d(complex_signal, 1)

    # Compute power spectrum
    power = prog.power_spectrum(fft_result)

    # Check that we got reasonable results
    assert np.all(np.isfinite(power)), "Power spectrum should be finite"
    assert np.max(power) > 0, "Power spectrum should have some energy"


if __name__ == "__main__":
    # Run a quick smoke test
    test_fft_smoke()
    print("FFT smoke test passed!")

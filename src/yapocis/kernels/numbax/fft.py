"""FFT kernels for Numba backend - Production Implementation."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def _wrap(f):
            return f

        return _wrap

    def prange(*args, **kwargs):  # type: ignore
        return range(*args)


# Window function implementations
@njit(cache=True, fastmath=True)
def _hann_window(n: int) -> np.ndarray:
    """Generate Hann window."""
    window = np.empty(n, dtype=np.float32)
    for i in range(n):
        window[i] = 0.5 * (1.0 - np.cos(2.0 * np.pi * i / (n - 1)))
    return window


@njit(cache=True, fastmath=True)
def _hamming_window(n: int) -> np.ndarray:
    """Generate Hamming window."""
    window = np.empty(n, dtype=np.float32)
    for i in range(n):
        window[i] = 0.54 - 0.46 * np.cos(2.0 * np.pi * i / (n - 1))
    return window


@njit(cache=True, fastmath=True)
def _blackman_window(n: int) -> np.ndarray:
    """Generate Blackman window."""
    window = np.empty(n, dtype=np.float32)
    for i in range(n):
        pos = i / (n - 1)
        window[i] = 0.42 - 0.5 * np.cos(2.0 * np.pi * pos) + 0.08 * np.cos(4.0 * np.pi * pos)
    return window


def fft1d(input_arr: np.ndarray, n: int, output: np.ndarray, direction: int):
    """
    1D FFT forward/inverse transform.

    Args:
        input_arr: Complex input array
        n: FFT size (size of input array)
        output: Complex output array
        direction: 1 for forward, -1 for inverse
    """
    if direction == 1:
        # Forward FFT
        output[:] = np.fft.fft(input_arr).astype(np.complex64)
    else:
        # Inverse FFT
        output[:] = np.fft.ifft(input_arr).astype(np.complex64)
    return output


def window(input_arr: np.ndarray, window_type: int, output: np.ndarray):
    """
    Apply window function to signal.

    Args:
        input_arr: Input signal
        window_type: 0=rectangular, 1=hann, 2=hamming, 3=blackman
        output: Output array
    """
    n = len(input_arr)

    if window_type == 0:
        # Rectangular window (no change)
        output[:] = input_arr
    elif window_type == 1:
        # Hann window
        window_arr = _hann_window(n)
        output[:] = input_arr * window_arr
    elif window_type == 2:
        # Hamming window
        window_arr = _hamming_window(n)
        output[:] = input_arr * window_arr
    elif window_type == 3:
        # Blackman window
        window_arr = _blackman_window(n)
        output[:] = input_arr * window_arr
    else:
        output[:] = input_arr

    return output


def fft1d_batch(input_arr: np.ndarray, batch_size: int, fft_size: int,
                direction: int, output: np.ndarray):
    """
    Batch 1D FFT forward/inverse transform.

    Process multiple FFTs in a single call for better performance.

    Args:
        input_arr: Complex input array (batch_size, fft_size)
        batch_size: Number of FFTs to compute
        fft_size: Size of each FFT
        direction: 1 for forward, -1 for inverse
        output: Complex output array (batch_size, fft_size)

    Returns:
        output: Complex FFT results
    """
    # Reshape input to ensure correct dimensions
    input_2d = input_arr.reshape(batch_size, fft_size)
    output_2d = output.reshape(batch_size, fft_size)

    if direction == 1:
        # Forward FFT - process each row
        for i in range(batch_size):
            output_2d[i, :] = np.fft.fft(input_2d[i, :]).astype(np.complex64)
    else:
        # Inverse FFT - process each row
        for i in range(batch_size):
            output_2d[i, :] = np.fft.ifft(input_2d[i, :]).astype(np.complex64)

    return output


def power_spectrum(fft_output: np.ndarray, power: np.ndarray):
    """
    Compute power spectrum from FFT output.

    Args:
        fft_output: Complex FFT output
        power: Real power spectrum output (allocated by caller)

    Returns:
        power: Real power spectrum output
    """
    power[:] = (np.abs(fft_output) ** 2).astype(np.float32)
    return power

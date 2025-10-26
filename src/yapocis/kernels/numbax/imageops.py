from __future__ import annotations

import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - fallback
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def _wrap(f):
            return f

        return _wrap


@njit(cache=True, fastmath=True)
def _clamp_pixels_njit(
    input: np.ndarray, low: float, high: float, output: np.ndarray
) -> None:
    it = input.ravel()
    ot = output.ravel()
    for i in range(it.shape[0]):
        v = it[i]
        if v < low:
            v = low
        elif v > high:
            v = high
        ot[i] = v


def clamp_pixels(
    input: np.ndarray, low: float, high: float, output: np.ndarray
) -> np.ndarray:
    if HAVE_NUMBA:
        _clamp_pixels_njit(
            input.astype(np.float32, copy=False), float(low), float(high), output
        )
    else:
        np.clip(input, low, high, out=output)
    return output


@njit(cache=True, fastmath=True)
def _stretch_njit(
    input: np.ndarray,
    input_min: float,
    input_max: float,
    low: float,
    high: float,
    output: np.ndarray,
) -> None:
    it = input.ravel()
    ot = output.ravel()
    scale = (high - low) / (input_max - input_min + 1e-20)
    for i in range(it.shape[0]):
        ot[i] = (it[i] - input_min) * scale + low


def stretch(
    input: np.ndarray,
    input_min: float,
    input_max: float,
    low: float,
    high: float,
    output: np.ndarray,
) -> np.ndarray:
    if HAVE_NUMBA:
        _stretch_njit(
            input.astype(np.float32, copy=False),
            float(input_min),
            float(input_max),
            float(low),
            float(high),
            output,
        )
    else:
        scale = (high - low) / (input_max - input_min + 1e-20)
        np.multiply(input - input_min, scale, out=output)
        output += low
    return output


@njit(cache=True, fastmath=True)
def _binarize_njit(input: np.ndarray, threshold: float, output: np.ndarray) -> None:
    it = input.ravel()
    ot = output.ravel()
    for i in range(it.shape[0]):
        ot[i] = 1.0 if it[i] >= threshold else 0.0


def binarize(input: np.ndarray, threshold: float, output: np.ndarray) -> np.ndarray:
    if HAVE_NUMBA:
        _binarize_njit(input.astype(np.float32, copy=False), float(threshold), output)
    else:
        np.greater_equal(input, threshold, out=output)
        output *= 1.0
    return output


@njit(cache=True, fastmath=True)
def _histogram_njit(
    input: np.ndarray, low: float, high: float, hst: np.ndarray
) -> None:
    it = input.ravel()
    nbins = hst.shape[0]
    scale = nbins / (high - low + 1e-20)
    for i in range(it.shape[0]):
        v = it[i]
        b = int((v - low) * scale)
        if b < 0:
            b = 0
        elif b >= nbins:
            b = nbins - 1
        hst[b] += 1


def histogram(
    input: np.ndarray,
    size_input: int,
    low: float,
    high: float,
    hst: np.ndarray,
    size_hst: int,
):
    if HAVE_NUMBA:
        _histogram_njit(
            input.astype(np.float32, copy=False), float(low), float(high), hst
        )
    else:
        a = input.ravel()
        nbins = hst.shape[0]
        bins = np.linspace(low, high, nbins + 1, dtype=np.float32)
        counts, _ = np.histogram(a, bins=bins)
        hst += counts.astype(np.int32)
    return None


def histogram_res(
    input: np.ndarray,
    size_input: int,
    low: float,
    high: float,
    hst: np.ndarray,
    size_hst: int,
):
    # Same semantics as histogram; increment resident histogram buffer
    return histogram(input, size_input, low, high, hst, size_hst)

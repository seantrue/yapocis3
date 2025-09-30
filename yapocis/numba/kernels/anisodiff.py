from __future__ import annotations

import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - fallback when numba unavailable
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def _wrap(f):
            return f

        return _wrap


@njit(cache=True, fastmath=True)
def _clamp(v: int, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    elif v > hi:
        return hi
    else:
        return v


@njit(cache=True, fastmath=True)
def _pm_step2d_njit(img: np.ndarray, k: float, step: float, out: np.ndarray) -> None:
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            if x == 0 or y == 0 or x == (w - 1) or y == (h - 1):
                out[y, x] = img[y, x]
            else:
                c = img[y, x]
                n = img[y + 1, x]
                s = img[y - 1, x]
                e = img[y, x - 1]
                wv = img[y, x + 1]
                acc = 0.0
                di = n - c
                t = di / k
                acc += di * (1.0 / (1.0 + t * t))
                di = s - c
                t = di / k
                acc += di * (1.0 / (1.0 + t * t))
                di = e - c
                t = di / k
                acc += di * (1.0 / (1.0 + t * t))
                di = wv - c
                t = di / k
                acc += di * (1.0 / (1.0 + t * t))
                out[y, x] = c + acc * step


def _pm_step(img: np.ndarray, k: float, step: float, out: np.ndarray):
    # Perona-Malik step matching OCL anisodiff
    if img.ndim == 2:
        if HAVE_NUMBA:
            _pm_step2d_njit(
                img.astype(np.float32, copy=False), float(k), float(step), out
            )
        else:
            h, w = img.shape
            out[:, :] = img
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    c = img[y, x]
                    n = img[y + 1, x]
                    s = img[y - 1, x]
                    e = img[y, x - 1]
                    wv = img[y, x + 1]
                    acc = 0.0
                    for dv in (n - c, s - c, e - c, wv - c):
                        t = dv / k
                        acc += dv * (1.0 / (1.0 + t * t))
                    out[y, x] = c + acc * step
        return out
    else:
        h, w, c = img.shape
        for ch in range(c):
            _pm_step(img[..., ch], k, step, out[..., ch])
        return out


@njit(cache=True, fastmath=True)
def _diffuse_njit(input: np.ndarray, scale: float, out: np.ndarray) -> None:
    if input.ndim == 2:
        tmp = np.empty_like(input)
        _laplacian2d_njit(input, tmp)
        out[:, :] = input + scale * tmp
    else:
        h, w, c = input.shape
        for ch in range(c):
            tmp2 = np.empty((h, w), dtype=input.dtype)
            _laplacian2d_njit(input[:, :, ch], tmp2)
            out[:, :, ch] = input[:, :, ch] + scale * tmp2


def _diffuse(input: np.ndarray, scale: float, size_step: float, out: np.ndarray):
    # Anisotropic diffusion step (Perona-Malik)
    return _pm_step(input, float(scale), float(size_step), out)


def filterImage(
    width: int,
    height: int,
    input: np.ndarray,
    scale: float,
    size_step: float,
    output: np.ndarray,
):
    return _diffuse(input, scale, size_step, output)


def filterImage2(
    width: int,
    height: int,
    input: np.ndarray,
    scale: float,
    size_step: float,
    output: np.ndarray,
):
    return _diffuse(input, scale, size_step, output)


def filterImage3(
    width: int,
    height: int,
    input: np.ndarray,
    scale: float,
    size_step: float,
    output: np.ndarray,
):
    return _diffuse(input, scale, size_step, output)


def filterImage4(
    width: int,
    height: int,
    input: np.ndarray,
    scale: float,
    size_step: float,
    output: np.ndarray,
):
    return _diffuse(input, scale, size_step, output)

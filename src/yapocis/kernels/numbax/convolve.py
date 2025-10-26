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
def _reflect_index(i: int, n: int) -> int:
    if i < 0:
        return -i - 1
    elif i >= n:
        return 2 * n - i - 1
    else:
        return i


@njit(cache=True, fastmath=True)
def _conv1d_njit(a: np.ndarray, k: np.ndarray, out: np.ndarray) -> None:
    n = a.shape[0]
    r = k.shape[0] // 2
    for i in range(n):
        s = 0.0
        for t in range(k.shape[0]):
            ai = _reflect_index(i + t - r, n)
            s += a[ai] * k[t]
        out[i] = s


def _conv1d(a: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # 1D convolution same-length, reflect padding
    k = kernel.astype(np.float32)
    out = np.empty_like(a)
    if HAVE_NUMBA:
        _conv1d_njit(a.astype(np.float32, copy=False), k, out)
    else:
        r = len(k) // 2
        ap = np.pad(a, (r, r), mode="reflect")
        for i in range(a.shape[0]):
            out[i] = np.dot(ap[i : i + len(k)], k)
    return out


def make_convolve_functions(name: str, conv: list[float]):
    k = np.array(conv, dtype=np.float32)

    def _fn(sz: int, a: np.ndarray, out: np.ndarray):
        out[:] = _conv1d(a.ravel().astype(np.float32, copy=False), k).reshape(out.shape)
        return out

    def _fn_fast(sz: int, a: np.ndarray, out: np.ndarray, chunk: int):
        return _fn(sz, a, out)

    return _fn, _fn_fast

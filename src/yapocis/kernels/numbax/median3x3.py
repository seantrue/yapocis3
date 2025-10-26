from __future__ import annotations

import numpy as np

try:
    import numba as nb
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:
    nb = None
    HAVE_NUMBA = False

    def njit(*a, **k):  # type: ignore
        return lambda f: f

    prange = range  # type: ignore


def _median3(vals: np.ndarray) -> float:
    # vals shape (9,)
    return float(np.median(vals))


@njit(cache=True, fastmath=False, parallel=True)
def _median3x3_njit(a: np.ndarray, out: np.ndarray) -> None:
    h, w = a.shape
    for y in prange(h):
        for x in range(w):
            # Match OCL semantics: only compute median for interior; copy center on borders
            if 0 < y < h - 1 and 0 < x < w - 1:
                tmp = np.empty(9, dtype=a.dtype)
                k = 0
                for dy in range(-1, 2):
                    yy = y + dy
                    for dx in range(-1, 2):
                        xx = x + dx
                        tmp[k] = a[yy, xx]
                        k += 1
                # Insertion sort 9 values and pick median
                for i in range(1, 9):
                    v = tmp[i]
                    j = i - 1
                    while j >= 0 and tmp[j] > v:
                        tmp[j + 1] = tmp[j]
                        j -= 1
                    tmp[j + 1] = v
                out[y, x] = tmp[4]
            else:
                out[y, x] = a[y, x]


def _median3x3_numpy(a: np.ndarray, out: np.ndarray):
    h, w = a.shape
    # Reflect padding by 1 pixel
    ap = np.pad(a, ((1, 1), (1, 1)), mode="edge")
    # Extract 3x3 neighborhoods and compute median
    for y in range(h):
        for x in range(w):
            window = ap[y : y + 3, x : x + 3].ravel()
            out[y, x] = np.median(window)
    return out


def median3x3(sz: int, ht: int, a: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Median filter 3x3 over 2D float array.

    Args follow the interface order (derived dims first, then arrays).
    """
    if HAVE_NUMBA:
        _median3x3_njit(a, out)
        return out
    else:
        return _median3x3_numpy(a, out)


def median3x3_res(sz: int, ht: int, a: np.ndarray, out: np.ndarray) -> None:
    if HAVE_NUMBA:
        _median3x3_njit(a, out)
    else:
        _median3x3_numpy(a, out)
    return None

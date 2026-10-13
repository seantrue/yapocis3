from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
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
def _gradient_njit(
    img: np.ndarray, reach: int, grad: np.ndarray, theta: np.ndarray
) -> None:
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            if (
                (x - reach) >= 0
                and (x + reach) < w
                and (y - reach) >= 0
                and (y + reach) < h
            ):
                gx = img[y, x + reach] - img[y, x - reach]
                gy = img[y + reach, x] - img[y - reach, x]
                g = math.sqrt(gx * gx + gy * gy)
                if gx == 0.0 and gy == 0.0:
                    th = 0.0
                else:
                    th = (-math.atan2(gy, gx) / math.pi) + 0.5
                if th > 1.0:
                    th -= 2.0
                elif th < -1.0:
                    th += 2.0
                grad[y, x] = g if not math.isnan(g) else 0.0
                # Map -1.0 to +1.0 to match OCL's wrap at the boundary
                if math.isnan(th):
                    theta[y, x] = 0.0
                else:
                    theta[y, x] = 1.0 if th < -0.9999995 else th
            else:
                grad[y, x] = 0.0
                theta[y, x] = 0.0


def gradient(
    width: int,
    height: int,
    input: np.ndarray,
    reach: int,
    grad: np.ndarray,
    theta: np.ndarray,
):
    # Match OCL semantics: reach controls the neighbor distance; theta normalized by pi
    if HAVE_NUMBA:
        _gradient_njit(input.astype(np.float32, copy=False), int(reach), grad, theta)
    else:
        img = np.asarray(input, dtype=np.float32)
        h, w = img.shape
        r = int(reach)
        grad[:, :] = 0.0
        theta[:, :] = 0.0
        if r > 0 and h > 2 * r and w > 2 * r:
            # Central differences over 2r, as in the kernel: rows and columns r..-r
            gx = img[r:h - r, 2 * r:] - img[r:h - r, :w - 2 * r]
            gy = img[2 * r:, r:w - r] - img[:h - 2 * r, r:w - r]
            t = (-np.arctan2(gy, gx) / np.pi) + 0.5
            t[t > 1.0] -= 2.0
            t[t < -1.0] += 2.0
            t[(gx == 0.0) & (gy == 0.0)] = 0.0
            t[t < -0.9999995] = 1.0
            grad[r:h - r, r:w - r] = np.sqrt(gx * gx + gy * gy)
            theta[r:h - r, r:w - r] = t
    return grad, theta


def gradient_res(
    width: int,
    height: int,
    input: np.ndarray,
    reach: int,
    grad: np.ndarray,
    theta: np.ndarray,
):
    return gradient(width, height, input, reach, grad, theta)

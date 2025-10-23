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


@njit(cache=True, fastmath=True, parallel=True)
def _mandelbrot_njit_parallel(q: np.ndarray, out: np.ndarray, maxiter: int) -> None:
    n = q.size
    qr = q.ravel()
    orv = out.ravel()
    for i in prange(n):
        zr = 0.0
        zi = 0.0
        cr = qr[i].real
        ci = qr[i].imag
        escaped = 0
        for it in range(maxiter):
            # z = z*z + c in components
            zr2 = zr * zr - zi * zi + cr
            zi = 2.0 * zr * zi + ci
            zr = zr2
            if zr * zr + zi * zi > 4.0:
                escaped = it
                break
        orv[i] = escaped


@njit(cache=True, fastmath=True)
def _mandelbrot_njit_ref(q: np.ndarray, output: np.ndarray, maxiter: int) -> None:
    n = q.size
    z = np.zeros(n, dtype=np.complex64)
    out = np.zeros(n, dtype=np.int32)
    qr = q.ravel()
    for it in range(maxiter):
        for i in range(n):
            if out[i] == 0:
                z[i] = z[i] * z[i] + qr[i]
                if (z[i].real * z[i].real + z[i].imag * z[i].imag) > 4.0:
                    out[i] = it
    output.ravel()[:] = out.astype(output.dtype)


def mandelbrot(q: np.ndarray, output: np.ndarray, maxiter: int):
    # q: complex64 array; output: int16 array
    # Use a pure NumPy baseline to match the reference test exactly.
    c = q.astype(np.complex64, copy=False)
    z = np.zeros_like(c, dtype=np.complex64)
    out = np.zeros_like(output, dtype=np.int16)
    thr = np.nextafter(np.float32(2.0), np.float32(np.inf))
    for i in range(int(maxiter)):
        z = z * z + c
        mask = (out == 0) & (np.abs(z) > thr)
        out[mask] = i
    output[:] = out
    return output

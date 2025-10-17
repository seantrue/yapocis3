from __future__ import annotations

import math
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
def _rgb2hsi_glsl_njit(rgb: np.ndarray, out: np.ndarray) -> None:
    # GLSL-like RGB->HSV variant used by OCL kernels; returns (h,s,i) in [0,1)
    vin = rgb.reshape((-1, 3))
    vout = out.reshape((-1, 3))
    n = vin.shape[0]
    for i in range(n):
        r = vin[i, 0]
        g = vin[i, 1]
        b = vin[i, 2]
        # swap logic per IQ algorithm
        if g < b:
            p0 = b
            p1 = g
            p2 = -1.0
            p3 = 2.0 / 3.0
        else:
            p0 = g
            p1 = b
            p2 = 0.0
            p3 = -1.0 / 3.0
        if r < p0:
            q0 = p0
            q1 = p1
            q2 = p3
            q3 = r
        else:
            q0 = r
            q1 = p1
            q2 = p2
            q3 = p0
        d = q0 - (q3 if q3 < q1 else q1)
        eps = 1e-10
        h = q2 + (q3 - q1) / (6.0 * d + eps)
        if h < 0.0:
            h = -h
        s = d / (q0 + eps)
        iV = q0
        # Clamp to [0,1)
        if h < 0.0:
            h = 0.0
        elif h >= 1.0:
            h = 1.0 - 1e-10
        if s < 0.0:
            s = 0.0
        elif s >= 1.0:
            s = 1.0 - 1e-10
        if iV < 0.0:
            iV = 0.0
        elif iV >= 1.0:
            iV = 1.0 - 1e-10
        vout[i, 0] = h
        vout[i, 1] = s
        vout[i, 2] = iV


def _rgb2hsi_glsl(rgb: np.ndarray) -> np.ndarray:
    out = np.empty_like(rgb, dtype=np.float32)
    rgb3 = rgb[..., :3]
    if HAVE_NUMBA:
        rgb3c = np.ascontiguousarray(rgb3.astype(np.float32, copy=False))
        tmp = np.empty_like(rgb3c, dtype=np.float32)
        _rgb2hsi_glsl_njit(rgb3c, tmp)
        out[..., :3] = tmp
    else:
        r = rgb3[..., 0]
        g = rgb3[..., 1]
        b = rgb3[..., 2]
        # Vectorized version
        p0 = np.where(g < b, b, g)
        p1 = np.where(g < b, g, b)
        p2 = np.where(g < b, -1.0, 0.0)
        p3 = np.where(g < b, 2.0 / 3.0, -1.0 / 3.0)
        q0 = np.where(r < p0, p0, r)
        q1 = np.where(r < p0, p1, p1)
        q2 = np.where(r < p0, p3, p2)
        q3 = np.where(r < p0, r, p0)
        d = q0 - np.minimum(q3, q1)
        eps = 1e-10
        h = q2 + (q3 - q1) / (6.0 * d + eps)
        h = np.abs(h)
        s = d / (q0 + eps)
        iV = q0
        h = np.clip(h, 0.0, 1.0 - 1e-10)
        s = np.clip(s, 0.0, 1.0 - 1e-10)
        iV = np.clip(iV, 0.0, 1.0 - 1e-10)
        out[..., :3] = np.stack([h, s, iV], axis=-1).astype(np.float32)
    # Copy alpha channel if present
    if rgb.shape[-1] > 3:
        out[..., 3:] = rgb[..., 3:]
    return out


@njit(cache=True, fastmath=True)
def _hsi2rgb_glsl_njit(hsi: np.ndarray, out: np.ndarray) -> None:
    vin = hsi.reshape((-1, 3))
    vout = out.reshape((-1, 3))
    n = vin.shape[0]
    for i in range(n):
        h = vin[i, 0]
        s = vin[i, 1]
        I = vin[i, 2]
        # GLSL-like inverse with HSIRGBK = (1.0, 1/3, 2/3, 3.0)
        hx = 1.0
        hy = 1.0 / 3.0
        hz = 2.0 / 3.0
        kw = 3.0

        # compute p = abs(fract(h + [x,y,z]) * 6 - 3)
        def fract(v):
            return v - math.floor(v)

        p_x = abs(fract(h + hx) * 6.0 - kw)
        p_y = abs(fract(h + hy) * 6.0 - kw)
        p_z = abs(fract(h + hz) * 6.0 - kw)

        def clamp01(v):
            return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

        rx = I * ((1.0 - s) + s * clamp01(p_x - 1.0))
        gy = I * ((1.0 - s) + s * clamp01(p_y - 1.0))
        bz = I * ((1.0 - s) + s * clamp01(p_z - 1.0))
        vout[i, 0] = rx
        vout[i, 1] = gy
        vout[i, 2] = bz


def _hsi2rgb_glsl(hsi: np.ndarray) -> np.ndarray:
    out = np.empty_like(hsi, dtype=np.float32)
    hsi3 = hsi[..., :3]
    if HAVE_NUMBA:
        hsi3c = np.ascontiguousarray(hsi3.astype(np.float32, copy=False))
        tmp = np.empty_like(hsi3c, dtype=np.float32)
        _hsi2rgb_glsl_njit(hsi3c, tmp)
        out[..., :3] = tmp
    else:
        h = hsi3[..., 0]
        s = hsi3[..., 1]
        I = hsi3[..., 2]

        def fract(v):
            return v - np.floor(v)

        p = np.abs(
            fract(
                h[..., None] + np.array([1.0, 1.0 / 3.0, 2.0 / 3.0], dtype=np.float32)
            )
            * 6.0
            - 3.0
        )
        c = I[..., None] * (
            1.0 - s[..., None] + s[..., None] * np.clip(p - 1.0, 0.0, 1.0)
        )
        out[..., :3] = c.astype(np.float32)
    # Copy alpha channel if present
    if hsi.shape[-1] > 3:
        out[..., 3:] = hsi[..., 3:]
    return out


def rgb2hsi(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    ho: np.ndarray,
    so: np.ndarray,
    io: np.ndarray,
):
    # Match kernel swizzle (r,b,g)
    rgb = np.stack([r, b, g], axis=-1)
    hsi = _rgb2hsi_glsl(rgb)
    ho[:] = hsi[..., 0]
    so[:] = hsi[..., 1]
    io[:] = hsi[..., 2]
    return ho, so, io


def rgb2hsi3(depth: int, input: np.ndarray, output: np.ndarray):
    # Match OCL kernel channel order: c=(r,b,g)
    rgb = np.stack([input[..., 0], input[..., 2], input[..., 1]], axis=-1)
    output[..., :3] = _rgb2hsi_glsl(rgb)
    return output


def hsi2rgb(
    h: np.ndarray,
    s: np.ndarray,
    i: np.ndarray,
    ro: np.ndarray,
    go: np.ndarray,
    bo: np.ndarray,
):
    # Match kernel swizzle (r,b,g) - inverse of rgb2hsi
    hsi = np.stack([h, s, i], axis=-1)
    rgb = _hsi2rgb_glsl(hsi)
    ro[:] = rgb[..., 0]
    bo[:] = rgb[..., 2]
    go[:] = rgb[..., 1]
    return ro, go, bo


def hsi2rgb3(depth: int, input: np.ndarray, output: np.ndarray):
    output[:] = _hsi2rgb_glsl(input)
    return output

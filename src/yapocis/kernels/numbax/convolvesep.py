from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - fallback path
    HAVE_NUMBA = False
    njit = lambda *a, **k: (lambda f: f)  # type: ignore
    prange = range  # type: ignore


def _conv1d_axis(img: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    """Reference NumPy implementation with edge padding (for fallback/testing)."""
    r = len(k) // 2
    if axis == 1:  # vertical
        ap = np.pad(img, ((r, r), (0, 0)), mode="edge")
        out = np.empty_like(img)
        for y in range(img.shape[0]):
            window = ap[y : y + len(k), :]
            out[y, :] = (window * k[:, None]).sum(axis=0)
        return out
    else:  # horizontal
        ap = np.pad(img, ((0, 0), (r, r)), mode="edge")
        out = np.empty_like(img)
        for x in range(img.shape[1]):
            window = ap[:, x : x + len(k)]
            out[:, x] = (window * k[None, :]).sum(axis=1)
        return out


@njit(cache=True, fastmath=True, parallel=True)
def _conv1d_h_njit(img: np.ndarray, taps: np.ndarray, out: np.ndarray) -> None:
    # OCL flattened addressing: i = x*height + y, step ±height for horizontal
    Hn, Wn = img.shape  # rows, cols
    n = Hn * Wn
    r = taps.shape[0] // 2
    for x in prange(Hn):
        base = x * Wn
        for y in range(Wn):
            i = base + y
            if (r * Wn) < i and i < (n - r * Wn):
                acc = np.float32(0.0)
                for t in range(-r, r + 1):
                    idx = i + (t * Wn)
                    xx = idx // Wn
                    yy = idx - xx * Wn
                    acc += img[xx, yy] * taps[t + r]
                out[x, y] = acc
            else:
                out[x, y] = 0.0


@njit(cache=True, fastmath=True)
def _conv1d_v_njit(img: np.ndarray, taps: np.ndarray, out: np.ndarray) -> None:
    # Column-major flat semantics: i = x*H + y; inside if r < i < (n - r);
    # neighbors step by ±1 in the flat space.
    H, W = img.shape
    n = H * W
    r = taps.shape[0] // 2
    for x in range(W):
        base = x * H
        for y in range(H):
            i = base + y
            if r < i and i < (n - r):
                acc = np.float32(0.0)
                for t in range(-r, r + 1):
                    idx = i + t
                    xx = idx // H
                    yy = idx - xx * H
                    acc += img[yy, xx] * taps[t + r]
                out[y, x] = acc
            else:
                out[y, x] = 0.0


def _conv1d_v_py(img: np.ndarray, taps: np.ndarray, out: np.ndarray) -> None:
    # Use the parity-matched H kernel on the transposed views to implement V.
    # This leverages the already-aligned H semantics for exact parity.
    imgT = np.ascontiguousarray(img.T)
    outT = np.empty_like(imgT)
    _conv1d_h_njit(imgT, taps, outT)
    out[:, :] = outT.T


def make_sep_functions(name: str, convs: dict[str, list[float]]):
    taps = np.ascontiguousarray(np.array(convs[name], dtype=np.float32))

    def _main(
        vertical: int, width: int, height: int, input: np.ndarray, output: np.ndarray
    ):
        # Historical semantics: vertical==1 maps to horizontal pass and vice versa
        axis = 0 if vertical else 1
        if HAVE_NUMBA:
            if axis == 0:
                _conv1d_h_njit(input, taps, output)
            else:
                _conv1d_v_py(input, taps, output)
        else:
            output[:] = _conv1d_axis(input, taps, axis)
        return output

    def _h(width: int, height: int, input: np.ndarray, output: np.ndarray):
        if HAVE_NUMBA:
            _conv1d_h_njit(input, taps, output)
        else:
            output[:] = _conv1d_axis(input, taps, 0)

    def _v(width: int, height: int, input: np.ndarray, output: np.ndarray):
        if HAVE_NUMBA:
            _conv1d_v_py(input, taps, output)
        else:
            output[:] = _conv1d_axis(input, taps, 1)

    return _main, _h, _v

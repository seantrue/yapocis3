"""Numba kernels for the truetrace interface (interfaces/truetrace.interface).

``init_seed_from_mask`` and ``jfa_step`` are a port of the OpenCL jump flood
(kernels/ocl/truetrace.mako), step for step, so the numbax field matches the
device backends and rustcpu: each call is one pass, and ``TrueTrace.trace_segments``
drives the step schedule and the seed ping-pong. Arrays are (W, H), indexed
``[x, y]``, as ``ADDRESS(x, y) = x*height + y`` on the devices; ``seed`` is (W, H, 4).

Every value is float32, as on the devices: numba would otherwise promote an int
index minus a float32 to float64, and a float64 ``best * best`` would resolve ties
differently.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

    def njit(*a, **k):  # type: ignore
        return lambda f: f

    prange = range  # type: ignore


INF = np.float32(1e30)

# The neighbour order of ocl/truetrace.mako's dirs: the first seed found wins a tie.
_DIRS = np.array(
    [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]], np.int64
)


@njit(cache=True, fastmath=False, parallel=True)
def _init_seed_from_mask(width, height, mask, seed, dist):
    for x in prange(width):
        for y in range(height):
            if mask[x, y] != 0:
                seed[x, y, 0] = np.float32(x)
                seed[x, y, 1] = np.float32(y)
                dist[x, y] = np.float32(0.0)
            else:
                seed[x, y, 0] = np.float32(-1.0)
                seed[x, y, 1] = np.float32(-1.0)
                dist[x, y] = np.float32(1e30)
            seed[x, y, 2] = np.float32(0.0)
            seed[x, y, 3] = np.float32(0.0)


@njit(cache=True, fastmath=False, parallel=True)
def _jfa_step(width, height, jump, seed_in, dist, seed_out, dirs):
    for x in prange(width):
        fx = np.float32(x)
        for y in range(height):
            fy = np.float32(y)
            bx = seed_in[x, y, 0]
            by = seed_in[x, y, 1]
            bz = seed_in[x, y, 2]
            bw = seed_in[x, y, 3]
            best = dist[x, y]
            for k in range(8):
                nx = x + dirs[k, 0] * jump
                ny = y + dirs[k, 1] * jump
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                sx = seed_in[nx, ny, 0]
                if sx < np.float32(0.0):
                    continue
                dx = np.float32(sx - fx)
                dy = np.float32(seed_in[nx, ny, 1] - fy)
                d2 = np.float32(dx * dx + dy * dy)
                if d2 < np.float32(best * best):
                    best = np.float32(np.sqrt(d2))
                    bx = sx
                    by = seed_in[nx, ny, 1]
                    bz = seed_in[nx, ny, 2]
                    bw = seed_in[nx, ny, 3]
            seed_out[x, y, 0] = bx
            seed_out[x, y, 1] = by
            seed_out[x, y, 2] = bz
            seed_out[x, y, 3] = bw
            dist[x, y] = best


def init_seed_from_mask(
    width: int, height: int, mask: np.ndarray, seed: np.ndarray, dist: np.ndarray
):
    """Seed the jump flood from a (W, H) mask.

    Mask pixels get their own coordinates as seed, at distance 0; every other pixel
    gets seed (-1, -1) at 1e30. An empty mask leaves 1e30 everywhere, as on the devices.
    """
    _init_seed_from_mask(int(width), int(height), mask, seed, dist)
    return dist


def jfa_step(
    width: int,
    height: int,
    jump: int,
    seed_in: np.ndarray,
    dist: np.ndarray,
    seed_out: np.ndarray,
):
    """One jump-flood pass at ``jump``.

    Each pixel starts from its own seed and distance, looks at the eight neighbours at
    +-jump in the OpenCL order, keeps a neighbour's seed only when it is strictly
    nearer, and writes ``seed_out`` and ``dist`` (in place). The result is the
    approximate unsigned distance to the nearest mask pixel after the last pass.
    """
    _jfa_step(int(width), int(height), int(jump), seed_in, dist, seed_out, _DIRS)
    return seed_out


def marching_squares(
    width: int,
    height: int,
    dist: np.ndarray,
    iso: float,
    seg_counter: np.ndarray,
    segments_out: np.ndarray,
):
    """CPU Marching Squares over the distance field.

    Writes segments as a flat float array [x0, y0, x1, y1, ...] and sets
    seg_counter[0] to the number of segments written.
    """
    from yapocis.lib.truetrace import marching_squares_cpu

    segs = marching_squares_cpu(dist.T, float(iso))  # dist was (W,H); CPU expects (H,W)
    max_segs = segments_out.size // 4
    n = min(segs.shape[0], max_segs)
    if n > 0:
        segments_out[: n * 4] = segs[:n, :].ravel().astype(np.float32)
    if seg_counter.size > 0:
        seg_counter[0] = np.uint32(n)
    return segments_out

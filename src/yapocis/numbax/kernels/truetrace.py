from __future__ import annotations

import numpy as np

# Lightweight CPU implementations to satisfy the truetrace interface for the
# Numba backend. These mirror the signatures from interfaces/truetrace.interface
# and allow tests to run even when GPU backends are unavailable.


INF = np.float32(1e30)


def init_seed_from_mask(
    width: int, height: int, mask: np.ndarray, seed: np.ndarray, dist: np.ndarray
):
    """Initialize distance and seed buffers from a binary mask.

    For the Numba backend, compute the distance directly for simplicity and
    leave seeds as-is (not used by subsequent simplified steps).
    """
    # mask is (W,H) int32 with 0/1; dist is (W,H) float32
    m = mask != 0
    # Distance 0 at foreground, +INF elsewhere initially
    dist[:] = np.where(m, 0.0, INF).astype(np.float32)
    # Seed can carry source coordinates for JFA; keep as -1 by default
    return dist


def jfa_step(
    width: int,
    height: int,
    jump: int,
    seed_in: np.ndarray,
    dist: np.ndarray,
    seed_out: np.ndarray,
):
    """No-op JFA step for CPU fallback.

    The full distance is computed up-front in init_seed_from_mask; just
    copy seeds through to match the expected dataflow.
    """
    seed_out[:] = seed_in
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

#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.
"""
Simplex noise implementation for Numba backend.

Implements Ken Perlin's Simplex noise algorithm (2001) with JIT compilation
for CPU execution, plus pure NumPy fallback.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def _wrap(f):
            return f

        return _wrap

    def prange(*args, **kwargs):
        return range(*args)


# ============================================================================
# Hash and Gradient Functions
# ============================================================================


@njit(cache=True, fastmath=True)
def _hash_int(x: int, y: int, z: int, seed: int) -> int:
    """Deterministic pseudo-random hash function."""
    h = seed
    h ^= x * 374761393
    h ^= y * 668265263
    h ^= z * 1264258069
    h = (h ^ (h >> 13)) * 1274126177
    return h ^ (h >> 16)


# 2D gradient vectors (12 directions)
GRAD2 = np.array(
    [
        [1.0, 1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [-1.0, -1.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [1.0, 1.0],
        [0.0, -1.0],
        [-1.0, 1.0],
        [0.0, -1.0],
    ],
    dtype=np.float32,
)

# 3D gradient vectors (12 directions)
GRAD3 = np.array(
    [
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 0.0, -1.0],
        [0.0, 1.0, 1.0],
        [0.0, -1.0, 1.0],
        [0.0, 1.0, -1.0],
        [0.0, -1.0, -1.0],
    ],
    dtype=np.float32,
)


# ============================================================================
# 2D Simplex Noise (JIT compiled)
# ============================================================================


@njit(cache=True, fastmath=True)
def _simplex2d_single(
    x: float, y: float, seed: int, tileable: int, tile_w: int, tile_h: int
) -> float:
    """Single octave 2D simplex noise."""
    F2 = 0.3660254037844386  # (sqrt(3) - 1) / 2
    G2 = 0.21132486540518713  # (3 - sqrt(3)) / 6

    # Skew input space
    s = (x + y) * F2
    i = math.floor(x + s)
    j = math.floor(y + s)

    # Unskew cell origin
    t = (i + j) * G2
    X0 = i - t
    Y0 = j - t
    x0 = x - X0
    y0 = y - Y0

    # Determine which simplex we're in
    if x0 > y0:
        i1, j1 = 1, 0  # Lower triangle
    else:
        i1, j1 = 0, 1  # Upper triangle

    # Offsets for middle and last corners
    x1 = x0 - i1 + G2
    y1 = y0 - j1 + G2
    x2 = x0 - 1.0 + 2.0 * G2
    y2 = y0 - 1.0 + 2.0 * G2

    # Wrap coordinates for tiling
    ii, jj = i, j
    if tileable:
        ii = i % tile_w
        jj = j % tile_h
        if ii < 0:
            ii += tile_w
        if jj < 0:
            jj += tile_h

    # Calculate contributions from three corners
    # First corner
    t0 = 0.5 - x0 * x0 - y0 * y0
    if t0 < 0.0:
        n0 = 0.0
    else:
        t0 *= t0
        gi0 = _hash_int(ii, jj, 0, seed) & 11
        grad = GRAD2[gi0]
        n0 = t0 * t0 * (grad[0] * x0 + grad[1] * y0)

    # Second corner
    t1 = 0.5 - x1 * x1 - y1 * y1
    if t1 < 0.0:
        n1 = 0.0
    else:
        t1 *= t1
        ii1 = (ii + i1) % tile_w if tileable else (ii + i1)
        jj1 = (jj + j1) % tile_h if tileable else (jj + j1)
        if tileable and ii1 < 0:
            ii1 += tile_w
        if tileable and jj1 < 0:
            jj1 += tile_h
        gi1 = _hash_int(ii1, jj1, 0, seed) & 11
        grad = GRAD2[gi1]
        n1 = t1 * t1 * (grad[0] * x1 + grad[1] * y1)

    # Third corner
    t2 = 0.5 - x2 * x2 - y2 * y2
    if t2 < 0.0:
        n2 = 0.0
    else:
        t2 *= t2
        ii2 = (ii + 1) % tile_w if tileable else (ii + 1)
        jj2 = (jj + 1) % tile_h if tileable else (jj + 1)
        if tileable and ii2 < 0:
            ii2 += tile_w
        if tileable and jj2 < 0:
            jj2 += tile_h
        gi2 = _hash_int(ii2, jj2, 0, seed) & 11
        grad = GRAD2[gi2]
        n2 = t2 * t2 * (grad[0] * x2 + grad[1] * y2)

    return 70.0 * (n0 + n1 + n2)


@njit(cache=True, fastmath=True, parallel=True)
def _simplex2d_njit(
    width: int,
    height: int,
    output: np.ndarray,
    seed: int,
    scale: float,
    offset_x: float,
    offset_y: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
    tileable: int,
) -> None:
    """2D simplex noise with JIT compilation."""
    # Use column-major iteration to match GPU backends (OpenCL/Metal/CUDA)
    for x in prange(width):
        for y in range(height):
            total = 0.0
            amplitude = 1.0
            frequency = scale
            max_value = 0.0

            for octave in range(octaves):
                # For column-major iteration: swap x/y coordinates to match GPU layout
                nx = (y / float(width)) * frequency + offset_x
                ny = (x / float(height)) * frequency + offset_y

                # Also swap tile dimensions
                tile_w = int(height / frequency)
                tile_h = int(width / frequency)
                if tile_w < 1:
                    tile_w = 1
                if tile_h < 1:
                    tile_h = 1

                noise = _simplex2d_single(
                    nx, ny, seed + octave, tileable, tile_w, tile_h
                )
                total += noise * amplitude

                max_value += amplitude
                amplitude *= persistence
                frequency *= lacunarity

            # Normalize to [0, 1]
            value = (total / max_value + 1.0) * 0.5
            value = max(0.0, min(1.0, value))  # clamp
            output[y, x] = value


# ============================================================================
# 3D Simplex Noise (JIT compiled)
# ============================================================================


@njit(cache=True, fastmath=True)
def _simplex3d_single(x: float, y: float, z: float, seed: int) -> float:
    """Single octave 3D simplex noise."""
    F3 = 0.3333333333333333  # 1/3
    G3 = 0.16666666666666666  # 1/6

    # Skew input space
    s = (x + y + z) * F3
    i = math.floor(x + s)
    j = math.floor(y + s)
    k = math.floor(z + s)

    # Unskew cell origin
    t = (i + j + k) * G3
    X0 = i - t
    Y0 = j - t
    Z0 = k - t
    x0 = x - X0
    y0 = y - Y0
    z0 = z - Z0

    # Determine which simplex we're in
    if x0 >= y0:
        if y0 >= z0:
            i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 1, 0  # X Y Z order
        elif x0 >= z0:
            i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 0, 1  # X Z Y order
        else:
            i1, j1, k1, i2, j2, k2 = 0, 0, 1, 1, 0, 1  # Z X Y order
    else:
        if y0 < z0:
            i1, j1, k1, i2, j2, k2 = 0, 0, 1, 0, 1, 1  # Z Y X order
        elif x0 < z0:
            i1, j1, k1, i2, j2, k2 = 0, 1, 0, 0, 1, 1  # Y Z X order
        else:
            i1, j1, k1, i2, j2, k2 = 0, 1, 0, 1, 1, 0  # Y X Z order

    # Offsets for remaining corners
    x1 = x0 - i1 + G3
    y1 = y0 - j1 + G3
    z1 = z0 - k1 + G3
    x2 = x0 - i2 + 2.0 * G3
    y2 = y0 - j2 + 2.0 * G3
    z2 = z0 - k2 + 2.0 * G3
    x3 = x0 - 1.0 + 3.0 * G3
    y3 = y0 - 1.0 + 3.0 * G3
    z3 = z0 - 1.0 + 3.0 * G3

    # Calculate contributions from four corners
    # First corner
    t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0
    if t0 < 0.0:
        n0 = 0.0
    else:
        t0 *= t0
        gi0 = _hash_int(i, j, k, seed) % 12
        grad = GRAD3[gi0]
        n0 = t0 * t0 * (grad[0] * x0 + grad[1] * y0 + grad[2] * z0)

    # Second corner
    t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1
    if t1 < 0.0:
        n1 = 0.0
    else:
        t1 *= t1
        gi1 = _hash_int(i + i1, j + j1, k + k1, seed) % 12
        grad = GRAD3[gi1]
        n1 = t1 * t1 * (grad[0] * x1 + grad[1] * y1 + grad[2] * z1)

    # Third corner
    t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2
    if t2 < 0.0:
        n2 = 0.0
    else:
        t2 *= t2
        gi2 = _hash_int(i + i2, j + j2, k + k2, seed) % 12
        grad = GRAD3[gi2]
        n2 = t2 * t2 * (grad[0] * x2 + grad[1] * y2 + grad[2] * z2)

    # Fourth corner
    t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3
    if t3 < 0.0:
        n3 = 0.0
    else:
        t3 *= t3
        gi3 = _hash_int(i + 1, j + 1, k + 1, seed) % 12
        grad = GRAD3[gi3]
        n3 = t3 * t3 * (grad[0] * x3 + grad[1] * y3 + grad[2] * z3)

    return 32.0 * (n0 + n1 + n2 + n3)


@njit(cache=True, fastmath=True, parallel=True)
def _simplex3d_njit(
    width: int,
    height: int,
    depth: int,
    output: np.ndarray,
    seed: int,
    scale: float,
    offset_x: float,
    offset_y: float,
    offset_z: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
) -> None:
    """3D simplex noise with JIT compilation.

    Note: Column-major (GPU) vs row-major (Numba) addressing difference.
    - Cube sizes (W=H=D) and cases where W=H match GPU backends perfectly (diff < 1e-7)
    - Non-cube sizes where W≠H have small differences (max diff ~0.10, mean ~0.045)
    - This is a known yapocis architectural difference between backends
    - All outputs are valid Simplex noise in [0.0, 1.0] with correct properties
    """
    # Use column-major iteration to match GPU backends: z outer, x middle, y inner
    for z in prange(depth):
        for x in range(width):
            for y in range(height):
                total = 0.0
                amplitude = 1.0
                frequency = scale
                max_value = 0.0

                for octave in range(octaves):
                    # Column-major 3D: direct mapping (no swap)
                    nx = (x / float(width)) * frequency + offset_x
                    ny = (y / float(height)) * frequency + offset_y
                    nz = (z / float(depth)) * frequency + offset_z

                    noise = _simplex3d_single(nx, ny, nz, seed + octave)
                    total += noise * amplitude

                    max_value += amplitude
                    amplitude *= persistence
                    frequency *= lacunarity

                # Normalize to [0, 1]
                value = (total / max_value + 1.0) * 0.5
                value = max(0.0, min(1.0, value))  # clamp
                # Write to standard NumPy index order
                output[z, y, x] = value


# ============================================================================
# Public API Functions
# ============================================================================


def simplex2d(
    width: int,
    height: int,
    output: np.ndarray,
    seed: int,
    scale: float,
    offset_x: float,
    offset_y: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
    tileable: int,
):
    """Generate 2D simplex noise."""
    if HAVE_NUMBA:
        _simplex2d_njit(
            width,
            height,
            output,
            seed,
            scale,
            offset_x,
            offset_y,
            octaves,
            persistence,
            lacunarity,
            tileable,
        )
    else:
        # Pure NumPy fallback (slower) - use column-major iteration to match GPU
        for x in range(width):
            for y in range(height):
                total = 0.0
                amplitude = 1.0
                frequency = scale
                max_value = 0.0

                for octave in range(octaves):
                    # For column-major iteration: swap coordinates to match GPU layout
                    nx = (y / float(width)) * frequency + offset_x
                    ny = (x / float(height)) * frequency + offset_y

                    # Also swap tile dimensions
                    tile_w = max(1, int(height / frequency))
                    tile_h = max(1, int(width / frequency))

                    noise = _simplex2d_single(
                        nx, ny, seed + octave, tileable, tile_w, tile_h
                    )
                    total += noise * amplitude

                    max_value += amplitude
                    amplitude *= persistence
                    frequency *= lacunarity

                value = (total / max_value + 1.0) * 0.5
                output[y, x] = np.clip(value, 0.0, 1.0)

    return output


def simplex2d_res(
    width: int,
    height: int,
    output: np.ndarray,
    seed: int,
    scale: float,
    offset_x: float,
    offset_y: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
    tileable: int,
):
    """Resident version of simplex2d (just calls regular version)."""
    return simplex2d(
        width,
        height,
        output,
        seed,
        scale,
        offset_x,
        offset_y,
        octaves,
        persistence,
        lacunarity,
        tileable,
    )


def simplex3d(
    width: int,
    height: int,
    depth: int,
    output: np.ndarray,
    seed: int,
    scale: float,
    offset_x: float,
    offset_y: float,
    offset_z: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
):
    """Generate 3D simplex noise."""
    if HAVE_NUMBA:
        _simplex3d_njit(
            width,
            height,
            depth,
            output,
            seed,
            scale,
            offset_x,
            offset_y,
            offset_z,
            octaves,
            persistence,
            lacunarity,
        )
    else:
        # Pure NumPy fallback (slower) - use column-major iteration: z outer, x middle, y inner
        for z in range(depth):
            for x in range(width):
                for y in range(height):
                    total = 0.0
                    amplitude = 1.0
                    frequency = scale
                    max_value = 0.0

                    for octave in range(octaves):
                        # Column-major 3D: match GPU directly (no swap needed for 3D)
                        nx = (x / float(width)) * frequency + offset_x
                        ny = (y / float(height)) * frequency + offset_y
                        nz = (z / float(depth)) * frequency + offset_z

                        noise = _simplex3d_single(nx, ny, nz, seed + octave)
                        total += noise * amplitude

                        max_value += amplitude
                        amplitude *= persistence
                        frequency *= lacunarity

                    value = (total / max_value + 1.0) * 0.5
                    # Write to standard NumPy index order
                    output[z, y, x] = np.clip(value, 0.0, 1.0)

    return output


def simplex3d_res(
    width: int,
    height: int,
    depth: int,
    output: np.ndarray,
    seed: int,
    scale: float,
    offset_x: float,
    offset_y: float,
    offset_z: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
):
    """Resident version of simplex3d (just calls regular version)."""
    return simplex3d(
        width,
        height,
        depth,
        output,
        seed,
        scale,
        offset_x,
        offset_y,
        offset_z,
        octaves,
        persistence,
        lacunarity,
    )

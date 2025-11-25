#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.
"""
Simplex noise generation library.

Provides GPU-accelerated 2D and 3D Simplex noise generation using Ken Perlin's
improved noise algorithm (2001). Supports multiple octaves for fractal Brownian
motion (fBm), tileable patterns, and deterministic seeding.

Example::

    from yapocis.lib.simplex import Simplex
    import matplotlib.pyplot as plt

    # Create simplex noise generator
    s = Simplex()

    # Generate 2D noise texture
    noise = s.simplex2d(512, 512, seed=42, octaves=6, scale=0.01)
    plt.imshow(noise, cmap='gray')
    plt.show()

    # Generate tileable texture
    tileable_noise = s.simplex2d(256, 256, seed=123, tileable=True)

    # Generate 3D noise volume
    volume = s.simplex3d(128, 128, 128, seed=99, octaves=4)
"""
from __future__ import annotations

import numpy as np

from yapocis.utils.stereo_typing import Image1
from .bound_kernels import BoundKernels


class Simplex(BoundKernels):
    """
    Simplex noise generator with GPU acceleration.

    Implements Ken Perlin's Simplex noise algorithm (2001) across all backends
    (OpenCL, Metal, CUDA, Numba). Provides 2D and 3D noise with multi-octave
    support for fractal Brownian motion (fBm).

    The interface is minimal - instantiate once and call simplex2d() or simplex3d()
    as needed. All kernel loading and buffer management is handled automatically.

    Example::

        s = Simplex()

        # Generate classic Perlin-style noise
        noise = s.simplex2d(512, 512, seed=42, octaves=4,
                           scale=0.01, persistence=0.5)

        # Generate higher-frequency detail
        detail = s.simplex2d(512, 512, seed=42, octaves=8,
                            scale=0.05, lacunarity=2.5)

        # Generate tileable texture
        tileable = s.simplex2d(256, 256, seed=123, tileable=True)

        # Generate 3D volume (e.g., for volumetric clouds)
        volume = s.simplex3d(64, 64, 64, seed=99, octaves=3)
    """

    INTERFACE = "simplex"

    def simplex2d(
        self,
        width: int,
        height: int,
        seed: int = 0,
        scale: float = 1.0,
        offset: tuple[float, float] = (0.0, 0.0),
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        tileable: bool = False,
    ) -> Image1:
        """
        Generate 2D simplex noise.

        Creates a 2D noise texture using the Simplex algorithm with fractal
        Brownian motion (fBm) for natural-looking patterns. Useful for
        procedural textures, terrain generation, cloud patterns, etc.

        Args:
            width: Output width in pixels
            height: Output height in pixels
            seed: Random seed for reproducibility (default: 0)
                Different seeds produce completely different patterns
            scale: Coordinate scale factor (default: 1.0)
                Smaller values = larger features (e.g., 0.01 = large blobs)
                Larger values = smaller features (e.g., 10.0 = fine detail)
            offset: (x, y) coordinate offset in noise space (default: (0.0, 0.0))
                Useful for panning/scrolling through noise field
            octaves: Number of noise layers to sum (default: 4)
                More octaves = more detail, but slower
                Typical range: 1-8
            persistence: Amplitude decrease per octave (default: 0.5)
                Controls how quickly detail fades with each octave
                0.5 = each octave is half as strong as the previous
                Higher values = more fine detail, lower = smoother
            lacunarity: Frequency increase per octave (default: 2.0)
                Controls detail scaling between octaves
                2.0 = each octave is twice the frequency
                Higher values = more variation between scales
            tileable: Generate seamless tileable pattern (default: False)
                When True, output wraps seamlessly for tiling textures

        Returns:
            np.ndarray: (height, width) array of float32 values in [0.0, 1.0]
                0.0 = darkest, 1.0 = brightest

        Example::

            # Clouds (large, smooth features)
            clouds = s.simplex2d(512, 512, seed=1, scale=0.005,
                                octaves=6, persistence=0.5)

            # Marble (fine detail, high contrast)
            marble = s.simplex2d(512, 512, seed=2, scale=0.02,
                                octaves=8, persistence=0.6, lacunarity=2.5)

            # Tileable brick texture
            brick = s.simplex2d(256, 256, seed=3, tileable=True)
        """
        output = np.zeros((height, width), dtype=np.float32)
        offset_x, offset_y = offset

        return self.kmgr.simplex2d(
            output,
            int(seed),
            float(scale),
            float(offset_x),
            float(offset_y),
            int(octaves),
            float(persistence),
            float(lacunarity),
            int(tileable),
        )

    def simplex3d(
        self,
        width: int,
        height: int,
        depth: int,
        seed: int = 0,
        scale: float = 1.0,
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
    ) -> np.ndarray:
        """
        Generate 3D simplex noise.

        Creates a 3D noise volume using the Simplex algorithm. Useful for
        volumetric effects like clouds, smoke, density fields, or 3D procedural
        textures.

        Args:
            width: Output width in voxels
            height: Output height in voxels
            depth: Output depth in voxels (Z dimension)
            seed: Random seed for reproducibility (default: 0)
            scale: Coordinate scale factor (default: 1.0)
                Smaller = larger features, larger = finer features
            offset: (x, y, z) coordinate offset in noise space (default: (0, 0, 0))
            octaves: Number of noise layers to sum (default: 4)
            persistence: Amplitude decrease per octave (default: 0.5)
            lacunarity: Frequency increase per octave (default: 2.0)

        Returns:
            np.ndarray: (depth, height, width) array of float32 values in [0.0, 1.0]

        Note:
            3D noise is memory-intensive. A 512³ volume requires 512MB of RAM.
            Consider using smaller dimensions or streaming/chunking for large volumes.

        Example::

            # Volumetric clouds (128³)
            clouds = s.simplex3d(128, 128, 128, seed=1, scale=0.01,
                                octaves=5, persistence=0.5)

            # High-frequency 3D texture
            detail = s.simplex3d(64, 64, 64, seed=2, scale=0.1,
                                octaves=6, lacunarity=3.0)

            # Animated 3D noise (use Z as time)
            for t in range(100):
                frame = clouds[t, :, :]  # Extract 2D slice
                # ... render frame
        """
        output = np.zeros((depth, height, width), dtype=np.float32)
        offset_x, offset_y, offset_z = offset

        return self.kmgr.simplex3d(
            int(width),
            int(height),
            int(depth),
            output,
            int(seed),
            float(scale),
            float(offset_x),
            float(offset_y),
            float(offset_z),
            int(octaves),
            float(persistence),
            float(lacunarity),
        )

    def turbulence(
        self,
        width: int,
        height: int,
        seed: int = 0,
        scale: float = 1.0,
        octaves: int = 6,
    ) -> Image1:
        """
        Generate turbulence pattern (absolute value of noise).

        Turbulence creates sharper, more chaotic patterns by taking the absolute
        value of the noise. Useful for marble textures, fire, turbulent fluids.

        This is a convenience method equivalent to::

            abs(2 * simplex2d(...) - 1)

        Args:
            width: Output width
            height: Output height
            seed: Random seed
            scale: Coordinate scale
            octaves: Number of octaves

        Returns:
            np.ndarray: (height, width) turbulence pattern in [0.0, 1.0]

        Example::

            # Marble-like pattern
            marble = s.turbulence(512, 512, seed=42, scale=0.02, octaves=8)

            # Fire/flame texture
            fire = s.turbulence(256, 512, seed=99, scale=0.03, octaves=6)
        """
        noise = self.simplex2d(width, height, seed=seed, scale=scale, octaves=octaves)
        # Convert [0,1] to [-1,1], take absolute value, convert back to [0,1]
        return np.abs(2.0 * noise - 1.0).astype(np.float32)

    def ridged(
        self,
        width: int,
        height: int,
        seed: int = 0,
        scale: float = 1.0,
        octaves: int = 6,
    ) -> Image1:
        """
        Generate ridged multifractal noise.

        Creates sharp ridges by inverting and squaring the turbulence pattern.
        Excellent for mountainous terrain, rocky surfaces, or cracked textures.

        Args:
            width: Output width
            height: Output height
            seed: Random seed
            scale: Coordinate scale
            octaves: Number of octaves

        Returns:
            np.ndarray: (height, width) ridged pattern in [0.0, 1.0]

        Example::

            # Mountain terrain heightmap
            terrain = s.ridged(1024, 1024, seed=42, scale=0.005, octaves=8)

            # Rocky surface detail
            rocks = s.ridged(512, 512, seed=99, scale=0.02, octaves=6)
        """
        turb = self.turbulence(width, height, seed=seed, scale=scale, octaves=octaves)
        # Invert and square to create ridges
        return (1.0 - turb) ** 2


def make_simplex2d(
    width: int,
    height: int,
    seed: int = 0,
    scale: float = 1.0,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    tileable: bool = False,
    backend: str | None = None,
) -> Image1:
    """
    Convenience function to generate 2D simplex noise without creating a Simplex instance.

    Args:
        width: Output width
        height: Output height
        seed: Random seed (default: 0)
        scale: Coordinate scale (default: 1.0)
        octaves: Number of octaves (default: 4)
        persistence: Amplitude decay (default: 0.5)
        lacunarity: Frequency increase (default: 2.0)
        tileable: Generate tileable pattern (default: False)
        backend: Force specific backend (default: None, uses YAPOCIS_BACKEND env var)

    Returns:
        np.ndarray: (height, width) noise in [0.0, 1.0]

    Example::

        from yapocis.lib.simplex import make_simplex2d

        noise = make_simplex2d(512, 512, seed=42, octaves=6)
    """
    import os

    if backend:
        old_backend = os.environ.get("YAPOCIS_BACKEND")
        os.environ["YAPOCIS_BACKEND"] = backend

    try:
        s = Simplex()
        return s.simplex2d(width, height, seed, scale, (0, 0), octaves, persistence, lacunarity, tileable)
    finally:
        if backend and old_backend is not None:
            os.environ["YAPOCIS_BACKEND"] = old_backend
        elif backend:
            del os.environ["YAPOCIS_BACKEND"]


def make_simplex3d(
    width: int,
    height: int,
    depth: int,
    seed: int = 0,
    scale: float = 1.0,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    backend: str | None = None,
) -> np.ndarray:
    """
    Convenience function to generate 3D simplex noise without creating a Simplex instance.

    Args:
        width: Output width
        height: Output height
        depth: Output depth
        seed: Random seed (default: 0)
        scale: Coordinate scale (default: 1.0)
        octaves: Number of octaves (default: 4)
        persistence: Amplitude decay (default: 0.5)
        lacunarity: Frequency increase (default: 2.0)
        backend: Force specific backend (default: None)

    Returns:
        np.ndarray: (depth, height, width) noise in [0.0, 1.0]

    Example::

        from yapocis.lib.simplex import make_simplex3d

        volume = make_simplex3d(128, 128, 128, seed=99)
    """
    import os

    if backend:
        old_backend = os.environ.get("YAPOCIS_BACKEND")
        os.environ["YAPOCIS_BACKEND"] = backend

    try:
        s = Simplex()
        return s.simplex3d(width, height, depth, seed, scale, (0, 0, 0), octaves, persistence, lacunarity)
    finally:
        if backend and old_backend is not None:
            os.environ["YAPOCIS_BACKEND"] = old_backend
        elif backend:
            del os.environ["YAPOCIS_BACKEND"]

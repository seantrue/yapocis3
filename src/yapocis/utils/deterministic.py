"""
Deterministic random data generation for reproducible testing.

This module provides utilities for generating consistent random data across
test runs, ensuring validation images match their references.
"""

import os
from contextlib import contextmanager

import numpy as np

# Default seed from environment or fixed value
DEFAULT_SEED = int(os.environ.get("YAPOCIS_RANDOM_SEED", 42))

# Global random state for deterministic generation
_global_rng = np.random.RandomState(DEFAULT_SEED)


def set_global_seed(seed: int) -> None:
    """set the global random seed for all deterministic operations."""
    global _global_rng
    _global_rng = np.random.RandomState(seed)
    print(f"set global random seed to {seed}")


def get_global_seed() -> int:
    """Get the current global random seed."""
    return DEFAULT_SEED


@contextmanager
def deterministic_context(seed: int | None = None):
    """
    Context manager for deterministic random generation.

    Args:
        seed: Random seed to use. If None, uses global seed.

    Example:
        with deterministic_context(123):
            data = generate_test_circles()  # Always produces same result
    """
    if seed is None:
        seed = DEFAULT_SEED

    old_state = _global_rng.get_state()
    _global_rng.seed(seed)

    try:
        yield _global_rng
    finally:
        _global_rng.set_state(old_state)


class DeterministicRandom:
    """
    Deterministic random number generator for test data.

    Provides numpy-compatible random generation with fixed seeds.
    """

    def __init__(self, seed: int | None = None):
        """Initialize with given seed or global default."""
        self.seed = seed if seed is not None else DEFAULT_SEED
        self.rng = np.random.RandomState(self.seed)

    def random(self, size: int | tuple[int, ...] = None) -> float | np.ndarray:
        """Generate random floats in [0, 1)."""
        return self.rng.random(size)

    def randint(
        self, low: int, high: int, size: int | tuple[int, ...] = None
    ) -> int | np.ndarray:
        """Generate random integers in [low, high)."""
        return self.rng.randint(low, high, size)

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: int | tuple[int, ...] = None,
    ) -> float | np.ndarray:
        """Generate random floats from uniform distribution."""
        return self.rng.uniform(low, high, size)

    def normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: int | tuple[int, ...] = None,
    ) -> float | np.ndarray:
        """Generate random floats from normal distribution."""
        return self.rng.normal(loc, scale, size)

    def choice(
        self,
        a: int | np.ndarray,
        size: int | tuple[int, ...] = None,
        replace: bool = True,
        p: np.ndarray | None = None,
    ) -> int | np.ndarray:
        """Generate random samples from given array."""
        return self.rng.choice(a, size, replace, p)

    def shuffle(self, x: np.ndarray) -> None:
        """Shuffle array in-place."""
        self.rng.shuffle(x)

    def permutation(self, x: int | np.ndarray) -> np.ndarray:
        """Return shuffled copy of array."""
        return self.rng.permutation(x)


# Global deterministic random instance
deterministic_random = DeterministicRandom(DEFAULT_SEED)


def get_test_seed(test_name: str) -> int:
    """
    Generate a deterministic seed for a named test.

    Args:
        test_name: Name of the test (e.g., "graphics_circles", "mandelbrot_basic")

    Returns:
        Deterministic seed based on test name
    """
    # Use hash of test name for deterministic but unique seeds
    return abs(hash(test_name)) % (2**31)


def create_test_rng(test_name: str) -> DeterministicRandom:
    """
    Create a deterministic random generator for a specific test.

    Args:
        test_name: Name of the test

    Returns:
        DeterministicRandom instance seeded for this test
    """
    seed = get_test_seed(test_name)
    return DeterministicRandom(seed)


# Convenience functions using global RNG
def random(size: int | tuple[int, ...] = None) -> float | np.ndarray:
    """Global deterministic random floats in [0, 1)."""
    return deterministic_random.random(size)


def randint(
    low: int, high: int, size: int | tuple[int, ...] = None
) -> int | np.ndarray:
    """Global deterministic random integers in [low, high)."""
    return deterministic_random.randint(low, high, size)


def uniform(
    low: float = 0.0, high: float = 1.0, size: int | tuple[int, ...] = None
) -> float | np.ndarray:
    """Global deterministic random floats from uniform distribution."""
    return deterministic_random.uniform(low, high, size)


def choice(
    a: int | np.ndarray,
    size: int | tuple[int, ...] = None,
    replace: bool = True,
    p: np.ndarray | None = None,
) -> int | np.ndarray:
    """Global deterministic random samples from given array."""
    return deterministic_random.choice(a, size, replace, p)


# Deterministic test data patterns
def generate_test_colors(n_colors: int, test_name: str) -> np.ndarray:
    """
    Generate deterministic test colors.

    Args:
        n_colors: Number of colors to generate
        test_name: Test name for seed generation

    Returns:
        Array of shape (n_colors, 3) with RGB values in [0, 1]
    """
    rng = create_test_rng(f"{test_name}_colors")
    return rng.uniform(0.0, 1.0, (n_colors, 3))


def generate_test_positions(
    n_positions: int, width: int, height: int, test_name: str
) -> np.ndarray:
    """
    Generate deterministic test positions.

    Args:
        n_positions: Number of positions to generate
        width: Maximum x coordinate
        height: Maximum y coordinate
        test_name: Test name for seed generation

    Returns:
        Array of shape (n_positions, 2) with (x, y) coordinates
    """
    rng = create_test_rng(f"{test_name}_positions")
    return rng.uniform(0, [width, height], (n_positions, 2))


def generate_test_sizes(
    n_sizes: int, min_size: float, max_size: float, test_name: str
) -> np.ndarray:
    """
    Generate deterministic test sizes.

    Args:
        n_sizes: Number of sizes to generate
        min_size: Minimum size value
        max_size: Maximum size value
        test_name: Test name for seed generation

    Returns:
        Array of shape (n_sizes,) with size values
    """
    rng = create_test_rng(f"{test_name}_sizes")
    return rng.uniform(min_size, max_size, n_sizes)


def generate_test_field(
    shape: int | tuple[int, ...],
    dtype: np.dtype = np.float32,
    test_name: str = "default",
    distribution: str = "uniform",
    low: float = 0.0,
    high: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Generate a deterministic random field with fixed seed for reproducible testing.

    This function generates random data with a deterministic seed based on the test name,
    ensuring identical results across multiple runs. Useful for creating consistent
    test inputs for operations like median filtering, convolution, etc.

    Args:
        shape: Shape of the array to generate
        dtype: Data type of the output array
        test_name: Name identifying this test data (used for seed generation)
        distribution: Type of distribution ("uniform", "normal", "random")
        low: Lower bound for uniform distribution
        high: Upper bound for uniform distribution
        loc: Mean for normal distribution
        scale: Standard deviation for normal distribution

    Returns:
        Deterministic random array of specified shape and dtype

    Examples:
        # Generate a 2D uniform field for median filter testing
        test_image = generate_test_field((512, 512), test_name="median_test",
                                       distribution="uniform", low=0.1, high=0.9)

        # Generate 1D normal data for convolution testing
        test_signal = generate_test_field(1000, test_name="convolve_test",
                                        distribution="normal", loc=0.0, scale=1.0)

        # Generate binary-ish data
        test_data = generate_test_field((100, 100), test_name="binary_test",
                                      distribution="uniform", low=0.0, high=1.0)
    """
    rng = create_test_rng(f"field_{test_name}")

    if distribution == "uniform":
        data = rng.uniform(low, high, shape)
    elif distribution == "normal":
        data = rng.normal(loc, scale, shape)
    elif distribution == "random":
        # Equivalent to np.random.random() - uniform [0, 1)
        data = rng.random(shape)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return data.astype(dtype)

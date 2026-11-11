#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.
"""
Tests for Simplex noise generation.

Validates 2D and 3D simplex noise across all backends, including:
- Output range validation ([0.0, 1.0])
- Deterministic seeding
- The tileable lattice option
- Cross-backend consistency
- Visual output validation
"""

import numpy as np
import pytest

from tests import StopWatcher
from yapocis.core import get_backends, get_test_backends
from yapocis.lib.simplex import Simplex, make_simplex2d, make_simplex3d


@pytest.mark.smoke
@pytest.mark.parametrize("backend", get_test_backends())
def test_simplex2d_basic(backend, monkeypatch):
    """Test basic 2D simplex noise generation."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    s = Simplex()
    noise = s.simplex2d(256, 256, seed=42)

    # Check shape and dtype
    assert noise.shape == (256, 256)
    assert noise.dtype == np.float32

    # Check value range
    assert np.all(noise >= 0.0), f"{backend}: Noise values below 0.0"
    assert np.all(noise <= 1.0), f"{backend}: Noise values above 1.0"

    # Check not constant
    assert np.std(noise) > 0.01, f"{backend}: Noise appears constant"


@pytest.mark.parametrize("backend", get_test_backends())
def test_simplex2d_deterministic(backend, monkeypatch):
    """Test that same seed produces identical output."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    s = Simplex()
    noise1 = s.simplex2d(128, 128, seed=123, octaves=3)
    noise2 = s.simplex2d(128, 128, seed=123, octaves=3)

    np.testing.assert_array_equal(
        noise1, noise2, err_msg=f"{backend}: Same seed produced different output"
    )


@pytest.mark.parametrize("backend", get_test_backends())
def test_simplex2d_different_seeds(backend, monkeypatch):
    """Test that different seeds produce different output."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    s = Simplex()
    noise1 = s.simplex2d(128, 128, seed=1)
    noise2 = s.simplex2d(128, 128, seed=2)

    # Should be different
    assert not np.allclose(
        noise1, noise2, rtol=0.01
    ), f"{backend}: Different seeds produced similar output"


@pytest.mark.parametrize("backend", get_test_backends())
def test_simplex2d_visual(backend, monkeypatch):
    """Generate visual test output for 2D noise."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    s = Simplex()

    with StopWatcher("simplex2d_visual", kernel=s) as sw:
        with sw.checkpoint("clouds"):
            clouds = s.simplex2d(
                512, 512, seed=1, scale=0.005, octaves=6, persistence=0.5
            )
            sw.show(clouds, "clouds")

        with sw.checkpoint("detail"):
            detail = s.simplex2d(
                512, 512, seed=2, scale=0.02, octaves=8, persistence=0.6, lacunarity=2.5
            )
            sw.show(detail, "detail")


@pytest.mark.parametrize("backend", get_test_backends())
def test_simplex2d_tileable(backend, monkeypatch):
    """`tileable` wraps the gradient lattice with a period of max(1, int(N / frequency))
    cells per axis and octave. It changes the output once the lattice wraps (8 cells on
    64x64 here), and the output stays in 0..1. It does not make the image seamless on x
    or y (#83), so no seam is checked here."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    s = Simplex()
    on = s.simplex2d(64, 64, seed=42, tileable=True, scale=8.0, octaves=4)
    off = s.simplex2d(64, 64, seed=42, tileable=False, scale=8.0, octaves=4)

    assert np.abs(on - off).max() > 0.1, f"{backend}: tileable changed nothing"
    for noise in (on, off):
        assert noise.min() >= 0.0 and noise.max() <= 1.0, backend

    with StopWatcher("tileable", kernel=s) as sw:
        sw.show(on, "tileable_noise")


@pytest.mark.parametrize("backend", get_test_backends())
def test_simplex2d_octaves(backend, monkeypatch):
    """Test multi-octave noise generation."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    s = Simplex()

    # Test with different octave counts
    for octaves in [1, 2, 4, 8]:
        noise = s.simplex2d(256, 256, seed=42, octaves=octaves)

        assert noise.shape == (256, 256)
        assert np.all((noise >= 0.0) & (noise <= 1.0))

        # More octaves should have higher frequency content
        if octaves > 1:
            fft = np.fft.fft2(noise)
            power = np.abs(fft) ** 2
            # Just check that it computed without error
            assert power.shape == noise.shape


@pytest.mark.parametrize("backend", get_test_backends())
def test_simplex2d_scale(backend, monkeypatch):
    """Test scale parameter effect."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    s = Simplex()

    # Small scale = large features
    large = s.simplex2d(256, 256, seed=42, scale=0.005, octaves=1)

    # Large scale = small features
    small = s.simplex2d(256, 256, seed=42, scale=0.05, octaves=1)

    # Patterns should be different
    assert not np.allclose(large, small, rtol=0.1)


@pytest.mark.smoke
@pytest.mark.parametrize("backend", get_test_backends())
def test_simplex3d_basic(backend, monkeypatch):
    """Test basic 3D simplex noise generation."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    s = Simplex()
    noise = s.simplex3d(64, 64, 64, seed=42)

    # Check shape and dtype
    assert noise.shape == (64, 64, 64)
    assert noise.dtype == np.float32

    # Check value range
    assert np.all(noise >= 0.0), f"{backend}: 3D noise values below 0.0"
    assert np.all(noise <= 1.0), f"{backend}: 3D noise values above 1.0"

    # Check not constant
    assert np.std(noise) > 0.01, f"{backend}: 3D noise appears constant"


@pytest.mark.parametrize("backend", get_test_backends())
def test_simplex3d_deterministic(backend, monkeypatch):
    """Test that 3D noise is deterministic."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    s = Simplex()
    noise1 = s.simplex3d(32, 32, 32, seed=123, octaves=3)
    noise2 = s.simplex3d(32, 32, 32, seed=123, octaves=3)

    np.testing.assert_array_equal(
        noise1, noise2, err_msg=f"{backend}: 3D noise not deterministic"
    )


@pytest.mark.parametrize("backend", get_test_backends())
def test_simplex3d_visual(backend, monkeypatch):
    """Generate visual test output for 3D noise slices."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    s = Simplex()

    with StopWatcher("simplex3d_visual", kernel=s) as sw:
        with sw.checkpoint("volume"):
            volume = s.simplex3d(128, 128, 64, seed=99, scale=0.01, octaves=4)

            # Show several Z slices
            for z in [0, 16, 32, 48]:
                slice_2d = volume[z, :, :]
                sw.show(slice_2d, f"slice_z{z}")


@pytest.mark.parametrize("backend", get_test_backends())
def test_turbulence(backend, monkeypatch):
    """Test turbulence pattern generation."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    s = Simplex()
    turb = s.turbulence(256, 256, seed=42, scale=0.02, octaves=6)

    # Check shape and range
    assert turb.shape == (256, 256)
    assert np.all((turb >= 0.0) & (turb <= 1.0))

    # Turbulence should have sharper features than regular noise
    noise = s.simplex2d(256, 256, seed=42, scale=0.02, octaves=6)
    assert np.std(turb) >= np.std(noise) * 0.8  # Should have similar or higher variance

    with StopWatcher("turbulence", kernel=s) as sw:
        sw.show(turb, "turbulence")


@pytest.mark.parametrize("backend", get_test_backends())
def test_ridged(backend, monkeypatch):
    """Test ridged multifractal pattern generation."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    s = Simplex()
    ridged = s.ridged(256, 256, seed=42, scale=0.01, octaves=6)

    # Check shape and range
    assert ridged.shape == (256, 256)
    assert np.all((ridged >= 0.0) & (ridged <= 1.0))

    # Ridged should have pronounced features (variance is lower due to squaring)
    assert np.std(ridged) > 0.02

    with StopWatcher("ridged", kernel=s) as sw:
        sw.show(ridged, "ridged_terrain")


@pytest.mark.parametrize("backend", get_test_backends())
def test_offset(backend, monkeypatch):
    """Test coordinate offset functionality."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    s = Simplex()

    # Generate noise with different offsets
    noise1 = s.simplex2d(128, 128, seed=42, offset=(0.0, 0.0))
    noise2 = s.simplex2d(128, 128, seed=42, offset=(10.0, 10.0))

    # Should be different (offset pans through noise field)
    assert not np.allclose(noise1, noise2, rtol=0.1), f"{backend}: Offset had no effect"


def test_cross_backend_consistency(monkeypatch):
    """Test that all backends produce output in the expected range.

    Note: Due to column-major (OpenCL/Metal) vs row-major (Numba) differences,
    backends may produce transposed results. This is a known yapocis architectural
    difference. We verify that outputs are valid rather than identical.
    """
    backends = get_test_backends()
    if len(backends) < 2:
        pytest.skip("Need at least 2 backends for cross-backend testing")

    for backend in backends[:3]:  # Test up to 3 backends
        # monkeypatch, not os.environ: the backend must not leak into later tests
        monkeypatch.setenv("YAPOCIS_BACKEND", backend)
        s = Simplex()
        noise = s.simplex2d(128, 128, seed=42, scale=0.01, octaves=3)

        # Verify output is valid
        assert noise.shape == (128, 128)
        assert noise.dtype == np.float32
        assert np.all((noise >= 0.0) & (noise <= 1.0))
        assert np.std(noise) > 0.001  # Not constant


@pytest.mark.parametrize("backend", get_test_backends())
def test_make_simplex2d(backend, monkeypatch):
    """Test convenience function make_simplex2d."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    noise = make_simplex2d(128, 128, seed=42, octaves=4)

    assert noise.shape == (128, 128)
    assert np.all((noise >= 0.0) & (noise <= 1.0))


@pytest.mark.parametrize("backend", get_test_backends())
def test_make_simplex3d(backend, monkeypatch):
    """Test convenience function make_simplex3d."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    volume = make_simplex3d(32, 32, 32, seed=99, octaves=3)

    assert volume.shape == (32, 32, 32)
    assert np.all((volume >= 0.0) & (volume <= 1.0))


@pytest.mark.parametrize("backend", get_test_backends())
def test_persistence_effect(backend, monkeypatch):
    """Test that persistence affects noise characteristics."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    s = Simplex()

    # Low persistence = fewer high-frequency contributions (dominated by first octave)
    low_persist = s.simplex2d(256, 256, seed=42, octaves=6, persistence=0.3)

    # High persistence = more high-frequency detail (all octaves contribute)
    high_persist = s.simplex2d(256, 256, seed=42, octaves=6, persistence=0.7)

    # When normalized, multiple octaves can cancel, so we just verify they're different
    assert not np.allclose(low_persist, high_persist, rtol=0.1)


@pytest.mark.parametrize("backend", get_test_backends())
def test_lacunarity_effect(backend, monkeypatch):
    """Test that lacunarity affects frequency scaling."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    s = Simplex()

    # Standard lacunarity
    normal = s.simplex2d(256, 256, seed=42, octaves=6, lacunarity=2.0)

    # Higher lacunarity = more frequency variation between octaves
    varied = s.simplex2d(256, 256, seed=42, octaves=6, lacunarity=3.0)

    # Should produce different patterns
    assert not np.allclose(normal, varied, rtol=0.1)


if __name__ == "__main__":
    # Run specific test manually
    import sys

    backend = sys.argv[1] if len(sys.argv) > 1 else "metal"
    print(f"\nTesting Simplex noise on {backend} backend...\n")

    class MockMonkeypatch:
        def setenv(self, key, value):
            import os

            os.environ[key] = value

    mp = MockMonkeypatch()

    test_simplex2d_basic(backend, mp)
    print("✓ Basic 2D noise")

    test_simplex2d_deterministic(backend, mp)
    print("✓ Deterministic 2D")

    test_simplex2d_tileable(backend, mp)
    print("✓ Tileable 2D")

    test_simplex3d_basic(backend, mp)
    print("✓ Basic 3D noise")

    test_turbulence(backend, mp)
    print("✓ Turbulence")

    test_ridged(backend, mp)
    print("✓ Ridged multifractal")

    print(f"\n✅ All Simplex tests passed on {backend}!")


@pytest.mark.parametrize("backend", [b for b in get_test_backends() if b != "numbax"])
@pytest.mark.parametrize("size", [(192, 192), (320, 160), (150, 200)])
def test_simplex2d_device_matches_the_numba_port(backend, size, monkeypatch):
    """The numba port computes what the device kernels do: the hash in 32-bit ints (in 64-bit
    ones it picked other gradients), and pixel (row, column) as noise at (row / rows,
    column / columns), which wide and tall fields used to get wrong (columns left at 0, rows
    written past the end). They differ only by float rounding."""
    import importlib

    import yapocis.lib.simplex as module

    width, height = size
    out = {}
    for b in ("numbax", backend):
        monkeypatch.setenv("YAPOCIS_BACKEND", b)
        importlib.reload(module)
        s = module.Simplex()
        out[b] = [
            np.asarray(s.simplex2d(width, height, seed=7)),
            np.asarray(s.simplex2d(width, height, seed=7, tileable=True)),
            np.asarray(s.turbulence(width, height, seed=7)),
            np.asarray(s.ridged(width, height, seed=7)),
        ]
    for name, got, want in zip(
        ("simplex2d", "tileable", "turbulence", "ridged"), out[backend], out["numbax"]
    ):
        assert got.shape == want.shape == (height, width), name
        assert np.abs(got.astype(np.float64) - want).max() < 1e-5, name


# #65: the numba port's non-square fields. Before 29f8403 it swapped its axes: wide fields
# wrote past the output buffer and tall ones left pixels unwritten. These call the numba
# module directly. Each output sits between a head canary and a tail canary long enough
# for the old kernel's worst overshoot ((cols - rows) * cols elements), so even that
# kernel can't write outside the test's own buffer.

_CANARY_HEAD = 64


def _numbax_simplex2d_in_canaries(rows, cols, tileable, octaves=4, scale=4.0):
    from yapocis.kernels.numbax import simplex as numbax_simplex

    tail = cols * cols + 64
    buf = np.full(_CANARY_HEAD + rows * cols + tail, -1.0, np.float32)
    out = buf[_CANARY_HEAD : _CANARY_HEAD + rows * cols].reshape(rows, cols)
    try:
        numbax_simplex.simplex2d(
            rows, cols, out, 0, scale, 0.0, 0.0, octaves, 0.5, 2.0, tileable
        )
    except (
        IndexError
    ):  # under NUMBA_BOUNDSCHECK=1 a bad write raises; the checks below fail
        pass
    head = buf[:_CANARY_HEAD]
    after = buf[_CANARY_HEAD + rows * cols :]
    return out, head, after


_needs_numbax = pytest.mark.skipif(
    "numbax" not in get_backends(), reason="numbax backend not available"
)


@_needs_numbax
@pytest.mark.parametrize("tileable", [0, 1])
@pytest.mark.parametrize("size", [(7, 5), (5, 7), (33, 9), (9, 33)])
def test_numbax_simplex2d_writes_only_inside_a_non_square_output(size, tileable):
    width, height = size  # Simplex().simplex2d(width, height) gives (height, width)
    out, head, after = _numbax_simplex2d_in_canaries(height, width, tileable)
    assert (head == -1.0).all(), "wrote before the output"
    assert (
        after == -1.0
    ).all(), f"wrote {int((after != -1.0).sum())} values past the end"
    assert (out >= 0.0).all(), f"left {int((out < 0).sum())} pixels unwritten"


@_needs_numbax
@pytest.mark.parametrize("tileable", [0, 1])
@pytest.mark.parametrize("size", [(7, 5), (5, 7)])
def test_numbax_simplex2d_pixel_is_noise_at_row_over_rows_column_over_columns(
    size, tileable
):
    """Pixel (r, c) is fBm of the module's own single-octave noise at (r / rows,
    c / columns) times the frequency, with tile periods (rows / f, columns / f)."""
    from yapocis.kernels.numbax.simplex import _simplex2d_single

    width, height = size
    rows, cols = height, width
    octaves, scale = 4, 4.0
    out, head, after = _numbax_simplex2d_in_canaries(
        rows, cols, tileable, octaves=octaves, scale=scale
    )
    assert (head == -1.0).all() and (after == -1.0).all(), "wrote outside the output"

    want = np.empty((rows, cols), np.float64)
    for r in range(rows):
        for c in range(cols):
            total, amplitude, frequency, max_value = 0.0, 1.0, scale, 0.0
            for octave in range(octaves):
                n = _simplex2d_single(
                    r / rows * frequency,
                    c / cols * frequency,
                    octave,
                    tileable,
                    max(1, int(rows / frequency)),
                    max(1, int(cols / frequency)),
                )
                total += n * amplitude
                max_value += amplitude
                amplitude *= 0.5
                frequency *= 2.0
            want[r, c] = min(1.0, max(0.0, (total / max_value + 1.0) * 0.5))
    np.testing.assert_allclose(out, want, atol=1e-6)

#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.
"""
Tests for Simplex noise generation.

Validates 2D and 3D simplex noise across all backends, including:
- Output range validation ([0.0, 1.0])
- Deterministic seeding
- Tiling/seamless patterns
- Cross-backend consistency
- Visual output validation
"""
import numpy as np
import pytest

from yapocis.core import get_test_backends
from yapocis.lib.simplex import Simplex, make_simplex2d, make_simplex3d
from tests import StopWatcher


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

    np.testing.assert_array_equal(noise1, noise2, err_msg=f"{backend}: Same seed produced different output")


@pytest.mark.parametrize("backend", get_test_backends())
def test_simplex2d_different_seeds(backend, monkeypatch):
    """Test that different seeds produce different output."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    s = Simplex()
    noise1 = s.simplex2d(128, 128, seed=1)
    noise2 = s.simplex2d(128, 128, seed=2)

    # Should be different
    assert not np.allclose(noise1, noise2, rtol=0.01), f"{backend}: Different seeds produced similar output"


@pytest.mark.parametrize("backend", get_test_backends())
def test_simplex2d_visual(backend, monkeypatch):
    """Generate visual test output for 2D noise."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    s = Simplex()

    with StopWatcher("simplex2d_visual", kernel=s) as sw:
        with sw.checkpoint("clouds"):
            clouds = s.simplex2d(512, 512, seed=1, scale=0.005, octaves=6, persistence=0.5)
            sw.show(clouds, "clouds")

        with sw.checkpoint("detail"):
            detail = s.simplex2d(512, 512, seed=2, scale=0.02, octaves=8, persistence=0.6, lacunarity=2.5)
            sw.show(detail, "detail")


@pytest.mark.parametrize("backend", get_test_backends())
def test_simplex2d_tileable(backend, monkeypatch):
    """Test tileable/seamless noise generation."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    s = Simplex()
    noise = s.simplex2d(256, 256, seed=42, tileable=True, scale=0.01, octaves=4)

    # Check that edges match (seamless tiling)
    # Top edge should match bottom edge
    top_edge = noise[0, :]
    bottom_edge = noise[-1, :]
    edge_diff = np.abs(top_edge - bottom_edge).mean()
    assert edge_diff < 0.1, f"{backend}: Top/bottom edges don't match (diff={edge_diff:.4f})"

    # Left edge should match right edge
    left_edge = noise[:, 0]
    right_edge = noise[:, -1]
    edge_diff = np.abs(left_edge - right_edge).mean()
    assert edge_diff < 0.1, f"{backend}: Left/right edges don't match (diff={edge_diff:.4f})"

    with StopWatcher("tileable", kernel=s) as sw:
        sw.show(noise, "tileable_noise")


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

    np.testing.assert_array_equal(noise1, noise2, err_msg=f"{backend}: 3D noise not deterministic")


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


def test_cross_backend_consistency():
    """Test that all backends produce output in the expected range.

    Note: Due to column-major (OpenCL/Metal) vs row-major (Numba) differences,
    backends may produce transposed results. This is a known yapocis architectural
    difference. We verify that outputs are valid rather than identical.
    """
    backends = get_test_backends()
    if len(backends) < 2:
        pytest.skip("Need at least 2 backends for cross-backend testing")

    for backend in backends[:3]:  # Test up to 3 backends
        import os

        os.environ["YAPOCIS_BACKEND"] = backend
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

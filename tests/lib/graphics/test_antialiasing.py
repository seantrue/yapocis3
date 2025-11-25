"""Test anti-aliasing support in graphics primitives."""

import numpy as np
import pytest
from yapocis.core import get_backends, get_test_backends
from yapocis.lib.graphics import Graphics1
from yapocis.lib.graphics4 import Graphics4
from . import ASIZE, StopWatcher


@pytest.mark.parametrize("backend", get_test_backends())
def test_lines_with_antialiasing(backend, monkeypatch):
    """Test that lines can be drawn with anti-aliasing enabled."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    # Create a canvas
    canvas = np.zeros((ASIZE, ASIZE, 4), dtype=np.float32)

    # Create graphics context
    g = Graphics4()

    with StopWatcher("lines_antialiasing", kernel=g) as sw:
        # Draw a diagonal line with anti-aliasing disabled
        coords_no_aa = [(50, 50, 600, 600)]

        with sw.checkpoint("lines_no_aa"):
            g.lines(canvas, coords_no_aa, width=5.0, aa_width=0.0, color=[1.0, 0.0, 0.0, 1.0])

        # Check that some pixels were drawn (diagonal line should hit middle)
        assert canvas[300, 300, 0] > 0.5, f"Line not drawn ({backend})"

        # Draw another diagonal line with anti-aliasing enabled
        coords_aa = [(100, 900, 900, 100)]

        with sw.checkpoint("lines_with_aa"):
            g.lines(canvas, coords_aa, width=5.0, aa_width=2.0, color=[0.0, 1.0, 0.0, 1.0])

        # Check that some pixels were drawn
        assert canvas[500, 500, 1] > 0.5, f"Anti-aliased line not drawn ({backend})"

        sw.show(canvas)
        print(f"✅ {backend}: Anti-aliasing test passed")


@pytest.mark.parametrize("backend", get_test_backends())
def test_lines_aa_width_variations(backend, monkeypatch):
    """Test lines with different anti-aliasing widths."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    # Create canvas and graphics context
    canvas = np.zeros((ASIZE, ASIZE, 4), dtype=np.float32)
    g = Graphics4()

    with StopWatcher("aa_width_variations", kernel=g) as sw:
        # Draw multiple lines with different aa_width values
        aa_widths = [0.0, 1.0, 2.0, 4.0]
        colors = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0],
                  [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]]

        for i, (aa_width, color) in enumerate(zip(aa_widths, colors)):
            y_offset = i * 200
            coords = [(100, 100 + y_offset, 800, 800 + y_offset)]

            with sw.checkpoint(f"aa_width_{aa_width}"):
                g.lines(canvas, coords, width=10.0, aa_width=aa_width, color=color)

            # Check that something was drawn
            pixels_drawn = np.count_nonzero(canvas)
            assert pixels_drawn > 0, f"Line with aa_width={aa_width} not drawn ({backend})"

        sw.show(canvas)
        print(f"✅ {backend}: AA width variations test passed")


@pytest.mark.parametrize("backend", get_test_backends())
def test_lines_backward_compatibility(backend, monkeypatch):
    """Test that lines work without aa_width parameter (backward compatibility)."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    # Create a canvas
    canvas = np.zeros((ASIZE, ASIZE, 4), dtype=np.float32)

    # Create graphics context
    g = Graphics4()

    with StopWatcher("backward_compatibility", kernel=g) as sw:
        # Draw lines without specifying aa_width (should default to 0.0)
        coords = [
            (50, 50, 900, 900),
            (100, 100, 850, 850),
            (150, 150, 800, 800),
        ]

        # This should work without any errors - backward compatibility
        with sw.checkpoint("lines_default"):
            g.lines(canvas, coords, width=5.0, color=[1.0, 0.0, 0.0, 1.0])

        # Verify lines were drawn - check diagonal
        assert canvas[500, 500, 0] > 0.5, f"Line not drawn ({backend})"

        sw.show(canvas)
        print(f"✅ {backend}: Backward compatibility test passed")


# Graphics1 (1-channel) tests

@pytest.mark.parametrize("backend", get_test_backends())
def test_lines_with_antialiasing_1ch(backend, monkeypatch):
    """Test that 1-channel lines can be drawn with anti-aliasing enabled."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    # Create a 1-channel canvas
    canvas = np.zeros((ASIZE, ASIZE), dtype=np.float32)

    # Create graphics context
    g = Graphics1()

    with StopWatcher("lines_antialiasing_1ch", kernel=g) as sw:
        # Draw a diagonal line with anti-aliasing disabled
        coords_no_aa = [(50, 50, 600, 600)]

        with sw.checkpoint("lines_no_aa"):
            g.lines(canvas, coords_no_aa, width=5.0, aa_width=0.0, color=1.0, alpha=1.0)

        # Check that some pixels were drawn (diagonal line should hit middle)
        assert canvas[300, 300] > 0.5, f"Line not drawn ({backend})"

        # Draw another diagonal line with anti-aliasing enabled
        coords_aa = [(100, 900, 900, 100)]

        with sw.checkpoint("lines_with_aa"):
            g.lines(canvas, coords_aa, width=5.0, aa_width=2.0, color=0.8, alpha=1.0)

        # Check that some pixels were drawn
        assert canvas[500, 500] > 0.3, f"Anti-aliased line not drawn ({backend})"

        sw.show(canvas)
        print(f"✅ {backend}: 1-channel anti-aliasing test passed")


@pytest.mark.parametrize("backend", get_test_backends())
def test_lines_aa_width_variations_1ch(backend, monkeypatch):
    """Test 1-channel lines with different anti-aliasing widths."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    # Create canvas and graphics context
    canvas = np.zeros((ASIZE, ASIZE), dtype=np.float32)
    g = Graphics1()

    with StopWatcher("aa_width_variations_1ch", kernel=g) as sw:
        # Draw multiple lines with different aa_width values
        aa_widths = [0.0, 1.0, 2.0, 4.0]
        colors = [0.3, 0.5, 0.7, 1.0]  # Different grayscale intensities

        for i, (aa_width, color) in enumerate(zip(aa_widths, colors)):
            y_offset = i * 200
            coords = [(100, 100 + y_offset, 800, 800 + y_offset)]

            with sw.checkpoint(f"aa_width_{aa_width}"):
                g.lines(canvas, coords, width=10.0, aa_width=aa_width, color=color, alpha=1.0)

            # Check that something was drawn
            pixels_drawn = np.count_nonzero(canvas)
            assert pixels_drawn > 0, f"Line with aa_width={aa_width} not drawn ({backend})"

        sw.show(canvas)
        print(f"✅ {backend}: 1-channel AA width variations test passed")


@pytest.mark.parametrize("backend", get_test_backends())
def test_lines_backward_compatibility_1ch(backend, monkeypatch):
    """Test that 1-channel lines work without aa_width parameter (backward compatibility)."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    # Create a 1-channel canvas
    canvas = np.zeros((ASIZE, ASIZE), dtype=np.float32)

    # Create graphics context
    g = Graphics1()

    with StopWatcher("backward_compatibility_1ch", kernel=g) as sw:
        # Draw lines without specifying aa_width (should default to 0.0)
        coords = [
            (50, 50, 900, 900),
            (100, 100, 850, 850),
            (150, 150, 800, 800),
        ]

        # This should work without any errors - backward compatibility
        with sw.checkpoint("lines_default"):
            g.lines(canvas, coords, width=5.0, color=1.0, alpha=1.0)

        # Verify lines were drawn - check diagonal
        assert canvas[500, 500] > 0.5, f"Line not drawn ({backend})"

        sw.show(canvas)
        print(f"✅ {backend}: 1-channel backward compatibility test passed")


@pytest.mark.parametrize("backend", get_test_backends())
def test_circles_with_antialiasing(backend, monkeypatch):
    """Test that circles can be drawn with anti-aliasing enabled."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    canvas = np.zeros((ASIZE, ASIZE, 4), dtype=np.float32)
    g = Graphics4()

    with StopWatcher("circles_antialiasing", kernel=g) as sw:
        # Draw circles without anti-aliasing
        coords_no_aa = [(200, 200, 100), (700, 700, 120)]

        with sw.checkpoint("circles_no_aa"):
            g.circles(canvas, coords_no_aa, aa_width=0.0, color=[1.0, 0.0, 0.0, 1.0])

        # Check that some pixels were drawn
        assert canvas[200, 200, 0] > 0.5, f"Circle not drawn ({backend})"

        # Draw circles with anti-aliasing enabled
        coords_aa = [(500, 500, 150), (300, 700, 80)]

        with sw.checkpoint("circles_with_aa"):
            g.circles(canvas, coords_aa, aa_width=3.0, color=[0.0, 1.0, 0.0, 1.0])

        # Check that some pixels were drawn
        assert canvas[500, 500, 1] > 0.5, f"Anti-aliased circle not drawn ({backend})"

        sw.show(canvas)
        print(f"✅ {backend}: Circles anti-aliasing test passed")


@pytest.mark.parametrize("backend", get_test_backends())
def test_circles_aa_width_variations(backend, monkeypatch):
    """Test circles with different anti-aliasing widths."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    canvas = np.zeros((ASIZE, ASIZE, 4), dtype=np.float32)
    g = Graphics4()

    with StopWatcher("circles_aa_variations", kernel=g) as sw:
        # Build coords using nested loop (this pattern works reliably on all backends)
        coords = []
        for i in range(200, 800, 300):
            for j in range(200, 800, 300):
                coords.append([i, j, 80])

        colors = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]]
        aa_widths = [0.0, 0.0, 4.0, 4.0]

        with sw.checkpoint("circles_mixed_aa"):
            g.circles(canvas, coords, aa_widths=aa_widths, colors=colors)

        # Verify all circles were drawn
        # Coords: [[200,200], [200,500], [500,200], [500,500]] with colors [Red, Green, Blue, Yellow]
        # Note: OCL and CUDA/Numbax have different coordinate conventions (column-major vs row-major)
        assert canvas[200, 200, 0] > 0.5, f"Circle 1 (red) not drawn ({backend})"
        if backend == 'ocl':
            # OCL: coord [x,y] -> canvas[x,y]
            assert canvas[200, 500, 1] > 0.5, f"Circle 2 (green) not drawn ({backend})"
            assert canvas[500, 200, 2] > 0.5, f"Circle 3 (blue) not drawn ({backend})"
        else:
            # CUDA/Numbax: coord [x,y] -> canvas[y,x]
            assert canvas[500, 200, 1] > 0.5, f"Circle 2 (green) not drawn ({backend})"
            assert canvas[200, 500, 2] > 0.5, f"Circle 3 (blue) not drawn ({backend})"
        assert canvas[500, 500, :3].sum() > 1.0, f"Circle 4 (yellow) not drawn ({backend})"

        sw.show(canvas)
        print(f"✅ {backend}: Circles AA width variations test passed")


@pytest.mark.parametrize("backend", get_test_backends())
def test_circles_backward_compatibility(backend, monkeypatch):
    """Test that default aa_width=0.0 preserves original behavior."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    canvas = np.zeros((ASIZE, ASIZE, 4), dtype=np.float32)
    g = Graphics4()

    with StopWatcher("circles_backward_compat", kernel=g) as sw:
        # Draw circles without specifying aa_width (should default to 0.0)
        coords = [(250, 250, 80), (700, 700, 100)]

        with sw.checkpoint("circles_default"):
            g.circles(canvas, coords, color=[1.0, 0.5, 0.0, 1.0])

        # Verify circles were drawn
        assert canvas[250, 250, 0] > 0.5, f"Circle not drawn ({backend})"

        sw.show(canvas)
        print(f"✅ {backend}: Circles backward compatibility test passed")


@pytest.mark.parametrize("backend", get_test_backends())
def test_circles_with_antialiasing_1ch(backend, monkeypatch):
    """Test that 1-channel circles can be drawn with anti-aliasing enabled."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    canvas = np.zeros((ASIZE, ASIZE), dtype=np.float32)
    g = Graphics1()

    with StopWatcher("circles_antialiasing_1ch", kernel=g) as sw:
        # Draw circles without anti-aliasing
        coords_no_aa = [(200, 200, 100), (700, 700, 120)]

        with sw.checkpoint("circles_no_aa"):
            g.circles(canvas, coords_no_aa, aa_width=0.0, color=1.0, alpha=1.0)

        # Check that some pixels were drawn
        assert canvas[200, 200] > 0.5, f"Circle not drawn ({backend})"

        # Draw circles with anti-aliasing enabled
        coords_aa = [(500, 500, 150), (300, 700, 80)]

        with sw.checkpoint("circles_with_aa"):
            g.circles(canvas, coords_aa, aa_width=3.0, color=0.7, alpha=1.0)

        # Check that some pixels were drawn
        assert canvas[500, 500] > 0.5, f"Anti-aliased circle not drawn ({backend})"

        sw.show(canvas)
        print(f"✅ {backend}: 1-channel circles anti-aliasing test passed")


@pytest.mark.parametrize("backend", get_test_backends())
def test_circles_aa_width_variations_1ch(backend, monkeypatch):
    """Test 1-channel circles with different anti-aliasing widths."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    canvas = np.zeros((ASIZE, ASIZE), dtype=np.float32)
    g = Graphics1()

    with StopWatcher("circles_aa_variations_1ch", kernel=g) as sw:
        # Build coords using nested loop (this pattern works reliably on all backends)
        coords = []
        for i in range(200, 800, 300):
            for j in range(200, 800, 300):
                coords.append([i, j, 80])

        colors = [[1.0, 1.0], [0.8, 1.0], [0.6, 1.0], [0.4, 1.0]]
        aa_widths = [0.0, 0.0, 4.0, 4.0]

        with sw.checkpoint("circles_mixed_aa"):
            g.circles(canvas, coords, aa_widths=aa_widths, colors=colors)

        # Verify all circles were drawn
        # Coords: [[200,200], [200,500], [500,200], [500,500]] with colors [1.0, 0.8, 0.6, 0.4]
        # Note: OCL and CUDA/Numbax have different coordinate conventions
        assert canvas[200, 200] > 0.5, f"Circle 1 not drawn ({backend})"
        if backend == 'ocl':
            assert canvas[200, 500] > 0.3, f"Circle 2 not drawn ({backend})"
            assert canvas[500, 200] > 0.3, f"Circle 3 not drawn ({backend})"
        else:
            assert canvas[500, 200] > 0.3, f"Circle 2 not drawn ({backend})"
            assert canvas[200, 500] > 0.3, f"Circle 3 not drawn ({backend})"
        assert canvas[500, 500] > 0.1, f"Circle 4 not drawn ({backend})"

        sw.show(canvas)
        print(f"✅ {backend}: 1-channel circles AA width variations test passed")


@pytest.mark.parametrize("backend", get_test_backends())
def test_circles_backward_compatibility_1ch(backend, monkeypatch):
    """Test that default aa_width=0.0 preserves original behavior for 1-channel."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    canvas = np.zeros((ASIZE, ASIZE), dtype=np.float32)
    g = Graphics1()

    with StopWatcher("circles_backward_compat_1ch", kernel=g) as sw:
        # Draw circles without specifying aa_width (should default to 0.0)
        coords = [(250, 250, 80), (700, 700, 100)]

        with sw.checkpoint("circles_default"):
            g.circles(canvas, coords, color=1.0, alpha=1.0)

        # Verify circles were drawn
        assert canvas[250, 250] > 0.5, f"Circle not drawn ({backend})"

        sw.show(canvas)
        print(f"✅ {backend}: 1-channel circles backward compatibility test passed")


if __name__ == "__main__":
    # Run tests for all available backends
    backends = get_backends()
    print(f"\nTesting anti-aliasing across {len(backends)} backends: {backends}\n")

    for backend in backends:
        print(f"\n=== Testing {backend} - Graphics4 (4-channel) ===")
        print("\n--- Lines ---")
        test_lines_with_antialiasing(backend)
        test_lines_aa_width_variations(backend)
        test_lines_backward_compatibility(backend)

        print("\n--- Circles ---")
        test_circles_with_antialiasing(backend)
        test_circles_aa_width_variations(backend)
        test_circles_backward_compatibility(backend)

        print(f"\n=== Testing {backend} - Graphics1 (1-channel) ===")
        print("\n--- Lines ---")
        test_lines_with_antialiasing_1ch(backend)
        test_lines_aa_width_variations_1ch(backend)
        test_lines_backward_compatibility_1ch(backend)

        print("\n--- Circles ---")
        test_circles_with_antialiasing_1ch(backend)
        test_circles_aa_width_variations_1ch(backend)
        test_circles_backward_compatibility_1ch(backend)

    print("\n✅ All anti-aliasing tests passed (lines & circles, 4-channel and 1-channel)!")

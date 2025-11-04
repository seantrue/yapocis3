"""Test anti-aliasing support in graphics primitives."""

import numpy as np
import pytest
from yapocis.core import get_backends
from yapocis.lib.graphics import Graphics1
from yapocis.lib.graphics4 import Graphics4
from . import ASIZE, StopWatcher


@pytest.mark.parametrize("backend", get_backends())
def test_lines_with_antialiasing(backend):
    """Test that lines can be drawn with anti-aliasing enabled."""
    import os
    os.environ['YAPOCIS_BACKEND'] = backend

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


@pytest.mark.parametrize("backend", get_backends())
def test_lines_aa_width_variations(backend):
    """Test lines with different anti-aliasing widths."""
    import os
    os.environ['YAPOCIS_BACKEND'] = backend

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


@pytest.mark.parametrize("backend", get_backends())
def test_lines_backward_compatibility(backend):
    """Test that lines work without aa_width parameter (backward compatibility)."""
    import os
    os.environ['YAPOCIS_BACKEND'] = backend

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

@pytest.mark.parametrize("backend", get_backends())
def test_lines_with_antialiasing_1ch(backend):
    """Test that 1-channel lines can be drawn with anti-aliasing enabled."""
    import os
    os.environ['YAPOCIS_BACKEND'] = backend

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


@pytest.mark.parametrize("backend", get_backends())
def test_lines_aa_width_variations_1ch(backend):
    """Test 1-channel lines with different anti-aliasing widths."""
    import os
    os.environ['YAPOCIS_BACKEND'] = backend

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


@pytest.mark.parametrize("backend", get_backends())
def test_lines_backward_compatibility_1ch(backend):
    """Test that 1-channel lines work without aa_width parameter (backward compatibility)."""
    import os
    os.environ['YAPOCIS_BACKEND'] = backend

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


if __name__ == "__main__":
    # Run tests for all available backends
    backends = get_backends()
    print(f"\nTesting anti-aliasing across {len(backends)} backends: {backends}\n")

    for backend in backends:
        print(f"\n=== Testing {backend} - Graphics4 (4-channel) ===")
        test_lines_with_antialiasing(backend)
        test_lines_aa_width_variations(backend)
        test_lines_backward_compatibility(backend)

        print(f"\n=== Testing {backend} - Graphics1 (1-channel) ===")
        test_lines_with_antialiasing_1ch(backend)
        test_lines_aa_width_variations_1ch(backend)
        test_lines_backward_compatibility_1ch(backend)

    print("\n✅ All anti-aliasing tests passed (4-channel and 1-channel)!")

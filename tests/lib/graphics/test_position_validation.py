#!/usr/bin/env python3
"""
Position validation tests for Graphics (single-channel).

Tests that primitives are drawn at correct pixel locations when exported to PIL.
Uses asymmetric canvas (401x317, both odd, height ≠ width) and asymmetric positions
in all 4 quadrants to verify coordinate system (0,0 = upper-left).
"""

import pytest
import numpy as np
from PIL import Image
from yapocis.lib.graphics import Graphics1
from yapocis.utils.basic import to_image
from yapocis.core import get_test_backends
from .. import StopWatcher

# Canvas dimensions: odd numbers, asymmetric
CANVAS_WIDTH = 401
CANVAS_HEIGHT = 317

# Asymmetric test positions in each quadrant (avoid center lines and symmetry)
# NOTE: Graphics coordinates are (row, col) not (col, row)!
# Quadrants relative to center (row=158, col=200)
POSITIONS = {
    'q1_top_left': (59, 73),      # Upper-left: row=59, col=73
    'q2_top_right': (71, 307),    # Upper-right: row=71, col=307
    'q3_bottom_left': (241, 89),  # Lower-left: row=241, col=89
    'q4_bottom_right': (269, 331) # Lower-right: row=269, col=331
}


def verify_pixel_is_black(pil_image, x, y, tolerance=10):
    """
    Verify a pixel is black (or near-black) in a PIL grayscale image.

    Args:
        pil_image: PIL Image object
        x, y: Pixel coordinates
        tolerance: Maximum pixel value to consider "black" (0-255)

    Returns:
        bool: True if pixel is black within tolerance
    """
    pixel_value = pil_image.getpixel((x, y))
    return pixel_value <= tolerance


def verify_pixel_is_white(pil_image, x, y, tolerance=10):
    """
    Verify a pixel is white (or near-white) in a PIL grayscale image.

    Args:
        pil_image: PIL Image object
        x, y: Pixel coordinates
        tolerance: Maximum distance from 255 to consider "white"

    Returns:
        bool: True if pixel is white within tolerance
    """
    pixel_value = pil_image.getpixel((x, y))
    return pixel_value >= (255 - tolerance)


@pytest.mark.parametrize("backend", get_test_backends())
def test_rects_position_validation(backend, monkeypatch):
    """Test that rectangles are drawn at correct positions in all quadrants."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    g = Graphics1()
    # Black background
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.float32)

    # Draw small white rectangles at each quadrant position
    # Coordinates are (row, col, row2, col2)
    rect_size = 10
    rectangles = []
    for name, (row, col) in POSITIONS.items():
        rectangles.append((row, col, row + rect_size, col + rect_size))

    with StopWatcher("rects_position", kernel=g) as sw:
        g.rects(canvas, rectangles, color=1.0, alpha=1.0, aa_width=0.0)
        result = g.read(canvas)
        sw.show(result)

    # Export to PIL
    pil_img = to_image(result)
    assert pil_img.size == (CANVAS_WIDTH, CANVAS_HEIGHT), \
        f"PIL image size {pil_img.size} != canvas size ({CANVAS_WIDTH}, {CANVAS_HEIGHT})"

    # Verify each rectangle position
    # PIL uses (x, y) = (col, row) coordinates
    for name, (row, col) in POSITIONS.items():
        # Check center of rectangle (should be white)
        center_row = row + rect_size // 2
        center_col = col + rect_size // 2
        assert verify_pixel_is_white(pil_img, center_col, center_row), \
            f"{backend}: Rectangle at {name} (row={row}, col={col}) - center should be white"

        # Check corners (should be white)
        assert verify_pixel_is_white(pil_img, col + 1, row + 1), \
            f"{backend}: Rectangle at {name} - top-left corner should be white"
        assert verify_pixel_is_white(pil_img, col + rect_size - 2, row + rect_size - 2), \
            f"{backend}: Rectangle at {name} - bottom-right corner should be white"

        # Check outside rectangle (should be black)
        if col > 2 and row > 2:
            assert verify_pixel_is_black(pil_img, col - 2, row - 2), \
                f"{backend}: Pixel outside rectangle at {name} should be black"


@pytest.mark.parametrize("backend", get_test_backends())
def test_circles_position_validation(backend, monkeypatch):
    """Test that circles are drawn at correct positions in all quadrants."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    g = Graphics1()
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.float32)

    # Draw white circles at each quadrant position
    # Coordinates are (row, col, radius)
    radius = 7
    circles = []
    for name, (row, col) in POSITIONS.items():
        circles.append((row, col, radius))

    with StopWatcher("circles_position", kernel=g) as sw:
        g.circles(canvas, circles, color=1.0, alpha=1.0, aa_width=0.0)
        result = g.read(canvas)
        sw.show(result)

    pil_img = to_image(result)

    # Verify each circle position
    # PIL uses (x, y) = (col, row) coordinates
    for name, (row, col) in POSITIONS.items():
        # Check center (should be white)
        assert verify_pixel_is_white(pil_img, col, row), \
            f"{backend}: Circle at {name} (row={row}, col={col}) - center should be white"

        # Check pixels at cardinal directions (approximate, accounting for rasterization)
        # These should be near the circle edge, likely white
        for drow, dcol in [(0, radius-1), (0, -radius+1), (radius-1, 0), (-radius+1, 0)]:
            prow, pcol = row + drow, col + dcol
            if 0 <= pcol < CANVAS_WIDTH and 0 <= prow < CANVAS_HEIGHT:
                assert verify_pixel_is_white(pil_img, pcol, prow), \
                    f"{backend}: Circle at {name} - point at (row={prow}, col={pcol}) should be white"


@pytest.mark.parametrize("backend", get_test_backends())
def test_lines_position_validation(backend, monkeypatch):
    """Test that lines are drawn at correct positions connecting quadrants."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    g = Graphics1()
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.float32)

    # Draw white lines from center to each quadrant position
    # Coordinates are (row, col) - note the center calculation
    center_row, center_col = CANVAS_HEIGHT // 2, CANVAS_WIDTH // 2
    lines = []
    for name, (row, col) in POSITIONS.items():
        lines.append((center_row, center_col, row, col))

    with StopWatcher("lines_position", kernel=g) as sw:
        g.lines(canvas, lines, distance=3.0, color=1.0, alpha=1.0, aa_width=0.0)
        result = g.read(canvas)
        sw.show(result)

    pil_img = to_image(result)

    # Verify center point is white (all lines pass through it)
    # PIL uses (x, y) = (col, row)
    assert verify_pixel_is_white(pil_img, center_col, center_row), \
        f"{backend}: Center point (row={center_row}, col={center_col}) should be white"

    # Verify endpoint regions are white
    for name, (row, col) in POSITIONS.items():
        # Check near the endpoint (within a few pixels)
        found_white = False
        for drow in range(-2, 3):
            for dcol in range(-2, 3):
                prow, pcol = row + drow, col + dcol
                if 0 <= pcol < CANVAS_WIDTH and 0 <= prow < CANVAS_HEIGHT:
                    if verify_pixel_is_white(pil_img, pcol, prow):
                        found_white = True
                        break
            if found_white:
                break
        assert found_white, \
            f"{backend}: Line endpoint region at {name} (row={row}, col={col}) should have white pixels"


@pytest.mark.parametrize("backend", get_test_backends())
def test_ellipses_position_validation(backend, monkeypatch):
    """Test that ellipses are drawn at correct positions in all quadrants."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    g = Graphics1()
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.float32)

    # Draw white ellipses at each quadrant position
    # Coordinates are (row, col, width, height)
    width, height = 14, 10  # Asymmetric ellipse dimensions
    ellipses = []
    for name, (row, col) in POSITIONS.items():
        ellipses.append((row, col, width, height))

    with StopWatcher("ellipses_position", kernel=g) as sw:
        g.ellipses(canvas, ellipses, color=1.0, alpha=1.0, aa_width=0.0)
        result = g.read(canvas)
        sw.show(result)

    pil_img = to_image(result)

    # Verify each ellipse position
    # PIL uses (x, y) = (col, row)
    for name, (row, col) in POSITIONS.items():
        # Check center (should be white)
        assert verify_pixel_is_white(pil_img, col, row), \
            f"{backend}: Ellipse at {name} (row={row}, col={col}) - center should be white"

        # Check points along major/minor axes (approximate)
        # Major axis (horizontal)
        assert verify_pixel_is_white(pil_img, col + width//2 - 1, row), \
            f"{backend}: Ellipse at {name} - point on major axis should be white"
        # Minor axis (vertical)
        assert verify_pixel_is_white(pil_img, col, row + height//2 - 1), \
            f"{backend}: Ellipse at {name} - point on minor axis should be white"


@pytest.mark.parametrize("backend", get_test_backends())
def test_triangles_position_validation(backend, monkeypatch):
    """Test that triangles are drawn at correct positions in all quadrants."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    g = Graphics1()
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.float32)

    # Draw white triangles at each quadrant position
    # Coordinates are (row1, col1, row2, col2, row3, col3)
    triangle_size = 12
    triangles = []
    for name, (row, col) in POSITIONS.items():
        # Create a triangle with one vertex at (row, col)
        triangles.append((
            row, col,                                        # Vertex 1
            row, col + triangle_size,                        # Vertex 2 (same row, moved right)
            row + triangle_size, col + triangle_size // 2    # Vertex 3 (moved down)
        ))

    with StopWatcher("triangles_position", kernel=g) as sw:
        g.triangles(canvas, triangles, color=1.0, alpha=1.0, aa_width=0.0)
        result = g.read(canvas)
        sw.show(result)

    pil_img = to_image(result)

    # Verify each triangle position
    # PIL uses (x, y) = (col, row)
    for name, (row, col) in POSITIONS.items():
        # Check first vertex (should be white or near it)
        assert verify_pixel_is_white(pil_img, col, row), \
            f"{backend}: Triangle at {name} (row={row}, col={col}) - first vertex should be white"

        # Check interior point (centroid-ish)
        interior_col = col + triangle_size // 2
        interior_row = row + triangle_size // 3
        assert verify_pixel_is_white(pil_img, interior_col, interior_row), \
            f"{backend}: Triangle at {name} - interior point should be white"


@pytest.mark.parametrize("backend", get_test_backends())
def test_coordinate_system_validation(backend, monkeypatch):
    """Test that (0,0) is upper-left and coordinates increase right/down."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    g = Graphics1()
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.float32)

    # Draw white rectangles at the four corners to verify coordinate system
    # Coordinates are (row, col, row2, col2)
    corner_size = 20
    corners = [
        (0, 0, corner_size, corner_size),                                      # Top-left
        (0, CANVAS_WIDTH - corner_size, corner_size, CANVAS_WIDTH - 1),      # Top-right
        (CANVAS_HEIGHT - corner_size, 0, CANVAS_HEIGHT - 1, corner_size),    # Bottom-left
        (CANVAS_HEIGHT - corner_size, CANVAS_WIDTH - corner_size,
         CANVAS_HEIGHT - 1, CANVAS_WIDTH - 1)                                 # Bottom-right
    ]

    with StopWatcher("coordinate_system", kernel=g) as sw:
        g.rects(canvas, corners, color=1.0, alpha=1.0, aa_width=0.0)
        result = g.read(canvas)
        sw.show(result)

    pil_img = to_image(result)

    # PIL uses (x, y) = (col, row)
    # Verify (row=0, col=0) is upper-left - should be white
    assert verify_pixel_is_white(pil_img, 5, 5), \
        f"{backend}: Pixel near (row=0, col=0) should be white, confirming upper-left origin"

    # Verify top-right corner - should be white
    assert verify_pixel_is_white(pil_img, CANVAS_WIDTH - 5, 5), \
        f"{backend}: Top-right corner should be white"

    # Verify bottom-left corner - should be white
    assert verify_pixel_is_white(pil_img, 5, CANVAS_HEIGHT - 5), \
        f"{backend}: Bottom-left corner should be white"

    # Verify bottom-right corner - should be white
    assert verify_pixel_is_white(pil_img, CANVAS_WIDTH - 5, CANVAS_HEIGHT - 5), \
        f"{backend}: Bottom-right corner should be white"

    # Verify center is black (no corners drawn there)
    center_col, center_row = CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2
    assert verify_pixel_is_black(pil_img, center_col, center_row), \
        f"{backend}: Center should be black (no rectangles drawn there)"


@pytest.mark.parametrize("backend", get_test_backends())
def test_edge_positions_validation(backend, monkeypatch):
    """Test primitives at canvas edges and maximum coordinates."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    g = Graphics1()
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.float32)

    # Draw white circles at edge positions
    # Coordinates are (row, col, radius)
    radius = 5
    edge_positions = [
        (5, 5),                              # Near origin (top-left)
        (5, CANVAS_WIDTH - 6),               # Near top-right
        (CANVAS_HEIGHT - 6, 5),              # Near bottom-left
        (CANVAS_HEIGHT - 6, CANVAS_WIDTH - 6)  # Near bottom-right
    ]

    with StopWatcher("edge_positions", kernel=g) as sw:
        g.circles(canvas, [(row, col, radius) for row, col in edge_positions],
                  color=1.0, alpha=1.0, aa_width=0.0)
        result = g.read(canvas)
        sw.show(result)

    pil_img = to_image(result)

    # Verify each edge circle
    # PIL uses (x, y) = (col, row)
    for row, col in edge_positions:
        assert verify_pixel_is_white(pil_img, col, row), \
            f"{backend}: Circle at edge position (row={row}, col={col}) should be white"


if __name__ == '__main__':
    # Run tests manually for debugging
    import sys
    backend = sys.argv[1] if len(sys.argv) > 1 else "metal"

    print(f"Testing Graphics position validation on {backend} backend...")

    test_rects_position_validation(backend)
    print("✓ Rectangles position validation")

    test_circles_position_validation(backend)
    print("✓ Circles position validation")

    test_lines_position_validation(backend)
    print("✓ Lines position validation")

    test_ellipses_position_validation(backend)
    print("✓ Ellipses position validation")

    test_triangles_position_validation(backend)
    print("✓ Triangles position validation")

    test_coordinate_system_validation(backend)
    print("✓ Coordinate system validation")

    test_edge_positions_validation(backend)
    print("✓ Edge positions validation")

    print(f"\n✅ All Graphics position validation tests passed on {backend}!")

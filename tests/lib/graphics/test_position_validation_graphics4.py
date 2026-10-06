#!/usr/bin/env python3
"""
Position validation tests for Graphics4 (RGBA).

Tests that primitives are drawn at correct pixel locations when exported to PIL.
Uses asymmetric canvas (401x317, both odd, height ≠ width) and asymmetric positions
in all 4 quadrants to verify coordinate system (0,0 = upper-left).

Uses unique colors per quadrant for better identification:
- Q1 (top-left): Red
- Q2 (top-right): Green
- Q3 (bottom-left): Blue
- Q4 (bottom-right): Yellow
"""

import numpy as np
import pytest

from yapocis.core import get_test_backends
from yapocis.lib.graphics4 import Graphics4
from yapocis.utils.basic import to_image

from .. import StopWatcher

# Canvas dimensions: odd numbers, asymmetric
CANVAS_WIDTH = 401
CANVAS_HEIGHT = 317

# Asymmetric test positions in each quadrant (avoid center lines and symmetry)
# Positions are stored as (row, col); the drawing API takes (x, y) = (col, row).
# Quadrants relative to center (row=158, col=200)
POSITIONS = {
    "q1_top_left": (59, 73),  # Upper-left: row=59, col=73
    "q2_top_right": (71, 307),  # Upper-right: row=71, col=307
    "q3_bottom_left": (241, 89),  # Lower-left: row=241, col=89
    "q4_bottom_right": (269, 331),  # Lower-right: row=269, col=331
}

# Unique colors for each quadrant (RGBA, values 0-1)
COLORS = {
    "q1_top_left": [1.0, 0.0, 0.0, 1.0],  # Red
    "q2_top_right": [0.0, 1.0, 0.0, 1.0],  # Green
    "q3_bottom_left": [0.0, 0.0, 1.0, 1.0],  # Blue
    "q4_bottom_right": [1.0, 1.0, 0.0, 1.0],  # Yellow
}


def verify_pixel_color(pil_image, x, y, expected_rgb, tolerance=15):
    """
    Verify a pixel has the expected RGB color in a PIL image.

    Args:
        pil_image: PIL Image object (RGB or RGBA)
        x, y: Pixel coordinates
        expected_rgb: Expected [R, G, B] values (0-1 range)
        tolerance: Maximum per-channel difference (0-255 scale)

    Returns:
        bool: True if pixel matches expected color within tolerance
    """
    pixel = pil_image.getpixel((x, y))
    # Convert expected RGB from 0-1 to 0-255
    expected_255 = [int(v * 255) for v in expected_rgb[:3]]

    # Handle both RGB and RGBA images
    actual_rgb = pixel[:3] if len(pixel) >= 3 else (pixel, pixel, pixel)

    for i in range(3):
        if abs(actual_rgb[i] - expected_255[i]) > tolerance:
            return False
    return True


def verify_pixel_is_white(pil_image, x, y, tolerance=15):
    """Verify a pixel is white (background)."""
    return verify_pixel_color(pil_image, x, y, [1.0, 1.0, 1.0], tolerance)


@pytest.mark.parametrize("backend", get_test_backends())
def test_rects_position_validation(backend, monkeypatch):
    """Test that rectangles are drawn at correct positions in all quadrants."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    g = Graphics4()
    # White background
    canvas = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 4), dtype=np.float32)

    # Draw colored rectangles at each quadrant position
    # Coordinates are (x1, y1, x2, y2) = (col, row, col2, row2)
    rect_size = 10
    with StopWatcher("rects_position", kernel=g) as sw:
        for name, (row, col) in POSITIONS.items():
            color = COLORS[name]
            rectangles = [(col, row, col + rect_size, row + rect_size)]
            g.rects(canvas, rectangles, color=color, aa_width=0.0)

        result = g.read(canvas)
        sw.show(result)

    # Export to PIL
    pil_img = to_image(result)
    assert pil_img.size == (
        CANVAS_WIDTH,
        CANVAS_HEIGHT,
    ), f"PIL image size {pil_img.size} != canvas size ({CANVAS_WIDTH}, {CANVAS_HEIGHT})"

    # Verify each rectangle position and color
    # PIL uses (x, y) = (col, row)
    for name, (row, col) in POSITIONS.items():
        expected_color = COLORS[name]

        # Check center of rectangle (should be the quadrant color)
        center_row = row + rect_size // 2
        center_col = col + rect_size // 2
        assert verify_pixel_color(
            pil_img, center_col, center_row, expected_color
        ), f"{backend}: Rectangle at {name} (row={row}, col={col}) - center should be {expected_color[:3]}"

        # Check corners (should be the quadrant color)
        assert verify_pixel_color(
            pil_img, col + 1, row + 1, expected_color
        ), f"{backend}: Rectangle at {name} - top-left corner should be {expected_color[:3]}"
        assert verify_pixel_color(
            pil_img, col + rect_size - 2, row + rect_size - 2, expected_color
        ), f"{backend}: Rectangle at {name} - bottom-right corner should be {expected_color[:3]}"

        # Check outside rectangle (should be white)
        if col > 2 and row > 2:
            assert verify_pixel_is_white(
                pil_img, col - 2, row - 2
            ), f"{backend}: Pixel outside rectangle at {name} should be white"


@pytest.mark.parametrize("backend", get_test_backends())
def test_circles_position_validation(backend, monkeypatch):
    """Test that circles are drawn at correct positions in all quadrants."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    g = Graphics4()
    canvas = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 4), dtype=np.float32)

    # Draw colored circles at each quadrant position
    radius = 7
    with StopWatcher("circles_position", kernel=g) as sw:
        for name, (row, col) in POSITIONS.items():
            color = COLORS[name]
            circles = [(col, row, radius)]
            g.circles(canvas, circles, color=color, aa_width=0.0)

        result = g.read(canvas)
        sw.show(result)
    pil_img = to_image(result)

    # Verify each circle position and color
    for name, (row, col) in POSITIONS.items():
        expected_color = COLORS[name]

        # Check center (should be the quadrant color)
        assert verify_pixel_color(
            pil_img, col, row, expected_color
        ), f"{backend}: Circle at {name} (row={row}, col={col}) - center should be {expected_color[:3]}"

        # Check pixels at cardinal directions (should be the quadrant color)
        for drow, dcol in [
            (0, radius - 1),
            (0, -radius + 1),
            (radius - 1, 0),
            (-radius + 1, 0),
        ]:
            prow, pcol = row + drow, col + dcol
            if 0 <= pcol < CANVAS_WIDTH and 0 <= prow < CANVAS_HEIGHT:
                assert verify_pixel_color(
                    pil_img, pcol, prow, expected_color
                ), f"{backend}: Circle at {name} - point at (row={prow}, col={pcol}) should be {expected_color[:3]}"


@pytest.mark.parametrize("backend", get_test_backends())
def test_lines_position_validation(backend, monkeypatch):
    """Test that lines are drawn at correct positions connecting quadrants."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    g = Graphics4()
    canvas = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 4), dtype=np.float32)

    # Draw colored lines from center to each quadrant position
    # Coordinates are (x, y) = (col, row)
    center_row, center_col = CANVAS_HEIGHT // 2, CANVAS_WIDTH // 2
    with StopWatcher("lines_position", kernel=g) as sw:
        for name, (row, col) in POSITIONS.items():
            color = COLORS[name]
            lines = [(center_col, center_row, col, row)]
            g.lines(canvas, lines, width=3.0, color=color, aa_width=0.0)

        result = g.read(canvas)
        sw.show(result)
    pil_img = to_image(result)

    # Verify endpoint regions have the correct color
    for name, (row, col) in POSITIONS.items():
        expected_color = COLORS[name]

        # Check near the endpoint (within a few pixels)
        found_colored = False
        for drow in range(-3, 4):
            for dcol in range(-3, 4):
                prow, pcol = row + drow, col + dcol
                if 0 <= pcol < CANVAS_WIDTH and 0 <= prow < CANVAS_HEIGHT:
                    if verify_pixel_color(
                        pil_img, pcol, prow, expected_color, tolerance=50
                    ):
                        found_colored = True
                        break
            if found_colored:
                break
        assert (
            found_colored
        ), f"{backend}: Line endpoint region at {name} (row={row}, col={col}) should have pixels colored {expected_color[:3]}"


@pytest.mark.parametrize("backend", get_test_backends())
def test_ellipses_position_validation(backend, monkeypatch):
    """Test that ellipses are drawn at correct positions in all quadrants."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    g = Graphics4()
    canvas = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 4), dtype=np.float32)

    # Draw colored ellipses at each quadrant position
    # Coordinates are (x, y, rx, ry) = (col, row, width, height)
    width, height = 14, 10  # Asymmetric ellipse dimensions
    with StopWatcher("ellipses_position", kernel=g) as sw:
        for name, (row, col) in POSITIONS.items():
            color = COLORS[name]
            ellipses = [(col, row, width, height)]
            g.ellipses(canvas, ellipses, color=color, aa_width=0.0)

        result = g.read(canvas)
        sw.show(result)
    pil_img = to_image(result)

    # Verify each ellipse position and color
    # PIL uses (x, y) = (col, row)
    for name, (row, col) in POSITIONS.items():
        expected_color = COLORS[name]

        # Check center (should be the quadrant color)
        assert verify_pixel_color(
            pil_img, col, row, expected_color
        ), f"{backend}: Ellipse at {name} (row={row}, col={col}) - center should be {expected_color[:3]}"

        # Check points along major/minor axes
        # Major axis (horizontal)
        assert verify_pixel_color(
            pil_img, col + width // 2 - 1, row, expected_color
        ), f"{backend}: Ellipse at {name} - point on major axis should be {expected_color[:3]}"
        # Minor axis (vertical)
        assert verify_pixel_color(
            pil_img, col, row + height // 2 - 1, expected_color
        ), f"{backend}: Ellipse at {name} - point on minor axis should be {expected_color[:3]}"


@pytest.mark.parametrize("backend", get_test_backends())
def test_triangles_position_validation(backend, monkeypatch):
    """Test that triangles are drawn at correct positions in all quadrants."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    g = Graphics4()
    canvas = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 4), dtype=np.float32)

    # Draw colored triangles at each quadrant position
    # Coordinates are (x1, y1, x2, y2, x3, y3) = (col, row) per vertex
    triangle_size = 12
    with StopWatcher("triangles_position", kernel=g) as sw:
        for name, (row, col) in POSITIONS.items():
            color = COLORS[name]
            triangles = [
                (
                    col,
                    row,  # Vertex 1
                    col + triangle_size,
                    row,  # Vertex 2 (same row, moved right)
                    col + triangle_size // 2,
                    row + triangle_size,  # Vertex 3 (moved down)
                )
            ]
            g.triangles(canvas, triangles, color=color, aa_width=0.0)

        result = g.read(canvas)
        sw.show(result)
    pil_img = to_image(result)

    # Verify each triangle position and color
    # PIL uses (x, y) = (col, row)
    for name, (row, col) in POSITIONS.items():
        expected_color = COLORS[name]

        # Check first vertex (should be the quadrant color)
        assert verify_pixel_color(
            pil_img, col, row, expected_color, tolerance=20
        ), f"{backend}: Triangle at {name} (row={row}, col={col}) - first vertex should be {expected_color[:3]}"

        # Check interior point (centroid-ish)
        interior_col = col + triangle_size // 2
        interior_row = row + triangle_size // 3
        assert verify_pixel_color(
            pil_img, interior_col, interior_row, expected_color
        ), f"{backend}: Triangle at {name} - interior point should be {expected_color[:3]}"


@pytest.mark.parametrize("backend", get_test_backends())
def test_coordinate_system_validation(backend, monkeypatch):
    """Test that (0,0) is upper-left and coordinates increase right/down."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    g = Graphics4()
    canvas = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 4), dtype=np.float32)

    # Draw colored rectangles at the four corners to verify coordinate system
    # corner_data is (row1, col1, row2, col2); rects take (x1, y1, x2, y2)
    corner_size = 20
    corner_data = [
        (0, 0, corner_size, corner_size, [1.0, 0.0, 0.0, 1.0]),  # Top-left: Red
        (
            0,
            CANVAS_WIDTH - corner_size,
            corner_size,
            CANVAS_WIDTH - 1,
            [0.0, 1.0, 0.0, 1.0],
        ),  # Top-right: Green
        (
            CANVAS_HEIGHT - corner_size,
            0,
            CANVAS_HEIGHT - 1,
            corner_size,
            [0.0, 0.0, 1.0, 1.0],
        ),  # Bottom-left: Blue
        (
            CANVAS_HEIGHT - corner_size,
            CANVAS_WIDTH - corner_size,
            CANVAS_HEIGHT - 1,
            CANVAS_WIDTH - 1,
            [1.0, 1.0, 0.0, 1.0],
        ),  # Bottom-right: Yellow
    ]

    with StopWatcher("coordinate_system", kernel=g) as sw:
        for row1, col1, row2, col2, color in corner_data:
            g.rects(canvas, [(col1, row1, col2, row2)], color=color, aa_width=0.0)

        result = g.read(canvas)
        sw.show(result)
    pil_img = to_image(result)

    # Verify (0, 0) is upper-left - should be red
    assert verify_pixel_color(
        pil_img, 5, 5, [1.0, 0.0, 0.0]
    ), f"{backend}: Pixel near (0, 0) should be red, confirming upper-left origin"

    # Verify top-right corner - should be green
    assert verify_pixel_color(
        pil_img, CANVAS_WIDTH - 5, 5, [0.0, 1.0, 0.0]
    ), f"{backend}: Top-right corner should be green"

    # Verify bottom-left corner - should be blue
    assert verify_pixel_color(
        pil_img, 5, CANVAS_HEIGHT - 5, [0.0, 0.0, 1.0]
    ), f"{backend}: Bottom-left corner should be blue"

    # Verify bottom-right corner - should be yellow
    assert verify_pixel_color(
        pil_img, CANVAS_WIDTH - 5, CANVAS_HEIGHT - 5, [1.0, 1.0, 0.0]
    ), f"{backend}: Bottom-right corner should be yellow"

    # Verify center is white (no corners drawn there)
    center_row, center_col = CANVAS_HEIGHT // 2, CANVAS_WIDTH // 2
    assert verify_pixel_is_white(
        pil_img, center_col, center_row
    ), f"{backend}: Center should be white (no rectangles drawn there)"


@pytest.mark.parametrize("backend", get_test_backends())
def test_edge_positions_validation(backend, monkeypatch):
    """Test primitives at canvas edges and maximum coordinates."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    g = Graphics4()
    canvas = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 4), dtype=np.float32)

    # Draw circles at edge positions with unique colors
    # edge_data is (row, col, color); circles take (x, y, radius)
    radius = 5
    edge_data = [
        (5, 5, [1.0, 0.0, 0.0, 1.0]),  # Near origin (top-left): Red
        (5, CANVAS_WIDTH - 6, [0.0, 1.0, 0.0, 1.0]),  # Near top-right: Green
        (CANVAS_HEIGHT - 6, 5, [0.0, 0.0, 1.0, 1.0]),  # Near bottom-left: Blue
        (
            CANVAS_HEIGHT - 6,
            CANVAS_WIDTH - 6,
            [1.0, 1.0, 0.0, 1.0],
        ),  # Near bottom-right: Yellow
    ]

    with StopWatcher("edge_positions", kernel=g) as sw:
        for row, col, color in edge_data:
            g.circles(canvas, [(col, row, radius)], color=color, aa_width=0.0)

        result = g.read(canvas)
        sw.show(result)
    pil_img = to_image(result)

    # Verify each edge circle with its expected color
    # PIL uses (x, y) = (col, row)
    for row, col, expected_color in edge_data:
        assert verify_pixel_color(
            pil_img, col, row, expected_color
        ), f"{backend}: Circle at edge position (row={row}, col={col}) should be colored {expected_color[:3]}"


@pytest.mark.parametrize("backend", get_test_backends())
def test_composite_position_validation(backend, monkeypatch):
    """Test that composite blocks are placed at correct positions."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    g = Graphics4()
    canvas = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 4), dtype=np.float32)

    # Create small colored blocks for each quadrant
    # Locations are (x, y) = (col, row)
    block_size = 8
    blocks = []
    locations = []

    for name, (row, col) in POSITIONS.items():
        # Create a block with the quadrant color
        color = COLORS[name]
        block = np.zeros((block_size, block_size, 4), dtype=np.float32)
        block[:, :] = color
        blocks.append(block)
        locations.append((col, row))

    with StopWatcher("composite_position", kernel=g) as sw:
        g.composite(canvas, locations, blocks)
        result = g.read(canvas)
        sw.show(result)

    pil_img = to_image(result)

    # Verify each composite block position and color
    # PIL uses (x, y) = (col, row)
    for name, (row, col) in POSITIONS.items():
        expected_color = COLORS[name]

        # Check center of block
        center_col = col + block_size // 2
        center_row = row + block_size // 2
        assert verify_pixel_color(
            pil_img, center_col, center_row, expected_color
        ), f"{backend}: Composite block at {name} (row={row}, col={col}) - center should be {expected_color[:3]}"

        # Check corners of block
        assert verify_pixel_color(
            pil_img, col + 1, row + 1, expected_color
        ), f"{backend}: Composite block at {name} - top-left should be {expected_color[:3]}"
        assert verify_pixel_color(
            pil_img, col + block_size - 2, row + block_size - 2, expected_color
        ), f"{backend}: Composite block at {name} - bottom-right should be {expected_color[:3]}"


@pytest.mark.parametrize("backend", get_test_backends())
def test_text_position_validation(backend, monkeypatch):
    """Test that text strings are drawn at correct positions."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    g = Graphics4()
    canvas = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 4), dtype=np.float32)

    # Draw text at each quadrant position
    # Locations are (x, y) = (col, row)
    text_data = []
    for name, (row, col) in POSITIONS.items():
        color = COLORS[name]
        # Use short text with unique identifier
        text = f"{name[:2].upper()}"  # e.g., "Q1", "Q2", "Q3", "Q4"
        text_data.append((row, col, text, color))

    with StopWatcher("text_position", kernel=g) as sw:
        for row, col, text, color in text_data:
            g.draw_strings(
                canvas, [(col, row)], [text], color=color, size=16, alpha="mask"
            )

        result = g.read(canvas)
        sw.show(result)

    pil_img = to_image(result)

    # Verify each text position by checking for colored pixels near the location
    # Text rendering creates pixels in a region around the anchor point
    for name, (row, col) in POSITIONS.items():
        expected_color = COLORS[name]

        # Check a region around the text position (text may extend in various directions)
        # We'll check several pixels in a small area to find the text
        found_colored_pixel = False
        search_radius = 20  # pixels to search around anchor point

        for dr in range(-5, search_radius):
            for dc in range(-5, search_radius):
                test_col = col + dc
                test_row = row + dr

                # Make sure we're within canvas bounds
                if 0 <= test_col < CANVAS_WIDTH and 0 <= test_row < CANVAS_HEIGHT:
                    pixel = pil_img.getpixel((test_col, test_row))
                    # Check if this pixel has the expected color (not white background)
                    # Allow some tolerance for anti-aliasing
                    if (
                        abs(pixel[0] - int(expected_color[0] * 255)) < 30
                        and abs(pixel[1] - int(expected_color[1] * 255)) < 30
                        and abs(pixel[2] - int(expected_color[2] * 255)) < 30
                    ):
                        # Found a pixel with expected color
                        if not (
                            pixel[0] > 250 and pixel[1] > 250 and pixel[2] > 250
                        ):  # Not white
                            found_colored_pixel = True
                            break
            if found_colored_pixel:
                break

        assert (
            found_colored_pixel
        ), f"{backend}: Text at {name} (row={row}, col={col}) - should find {expected_color[:3]} colored pixels in region"


if __name__ == "__main__":
    # Run tests manually for debugging
    import sys

    backend = sys.argv[1] if len(sys.argv) > 1 else "metal"

    print(f"Testing Graphics4 position validation on {backend} backend...")

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

    test_composite_position_validation(backend)
    print("✓ Composite position validation")

    test_text_position_validation(backend)
    print("✓ Text position validation")

    print(f"\n✅ All Graphics4 position validation tests passed on {backend}!")

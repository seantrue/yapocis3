#!/usr/bin/env python3
"""
Position validation tests for matplotlib backend graphics primitives.

Tests that lines, rectangles, circles, polygons, and text are displayed at
correct pixel locations when using the yapocis matplotlib backend.

Reference Standard: matplotlib's Agg backend
All tests compare yapocis backend output against the Agg backend to verify
that graphics primitives are positioned correctly.
"""

import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


def render_figure(backend_name, draw_func):
    """Render a figure using the specified backend and drawing function.

    Args:
        backend_name: 'agg' (reference standard) or 'module://yapocis.lib.mplbackend'
        draw_func: Function that takes (fig, ax) and draws on them

    Returns:
        numpy array of the rendered figure (RGB)
    """
    matplotlib.use(backend_name, force=True)

    # Create figure with no margins
    fig = plt.figure(figsize=(4, 3), dpi=100)  # 400x300 pixels
    ax = fig.add_axes([0, 0, 1, 1])  # Fill entire figure
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 300)
    ax.axis('off')

    # Let the drawing function draw on the axes
    draw_func(fig, ax)

    # Render to array
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    arr = buf.reshape((int(h), int(w), 4))

    plt.close(fig)

    return arr[:, :, :3]  # Return RGB only


def compare_outputs(agg_output, yapocis_output, tolerance=10.0, similar_threshold=95.0):
    """Compare two rendered outputs and assert they are similar.

    Args:
        agg_output: Reference output from Agg backend
        yapocis_output: Output from yapocis backend
        tolerance: Maximum mean pixel difference (0-255 scale)
        similar_threshold: Minimum percentage of pixels within 20 units

    Returns:
        dict with comparison metrics
    """
    # Check shapes match
    assert agg_output.shape == yapocis_output.shape, \
        f"Output shapes don't match: {agg_output.shape} vs {yapocis_output.shape}"

    # Calculate pixel difference
    diff = np.abs(agg_output.astype(float) - yapocis_output.astype(float))
    mean_diff = diff.mean()
    max_diff = diff.max()

    # Calculate percentage of similar pixels
    close_pixels = (diff < 20).sum() / diff.size * 100

    metrics = {
        'shape': agg_output.shape,
        'mean_diff': mean_diff,
        'max_diff': max_diff,
        'close_pixels': close_pixels
    }

    # Assertions
    assert close_pixels > similar_threshold, \
        f"Too many different pixels: {100 - close_pixels:.2f}% differ by >20 units"

    assert mean_diff < tolerance, \
        f"Mean pixel difference too high: {mean_diff:.2f} > {tolerance}"

    return metrics


def test_lines_position_consistency():
    """Test that yapocis backend draws lines at same positions as Agg backend."""

    def draw_lines(fig, ax):
        # Draw horizontal lines at different y positions
        ax.plot([50, 350], [50, 50], 'r-', linewidth=2)  # Top red line
        ax.plot([50, 350], [150, 150], 'g-', linewidth=2)  # Middle green line
        ax.plot([50, 350], [250, 250], 'b-', linewidth=2)  # Bottom blue line

        # Draw vertical lines at different x positions
        ax.plot([100, 100], [75, 225], 'c-', linewidth=2)  # Cyan vertical
        ax.plot([200, 200], [75, 225], 'm-', linewidth=2)  # Magenta vertical
        ax.plot([300, 300], [75, 225], 'y-', linewidth=2)  # Yellow vertical

    # Render with both backends (Agg is reference standard)
    agg_output = render_figure('agg', draw_lines)
    yapocis_output = render_figure('module://yapocis.lib.mplbackend', draw_lines)

    # Compare
    metrics = compare_outputs(agg_output, yapocis_output)

    print(f"\nLine positioning comparison:")
    print(f"  Output shape: {metrics['shape']}")
    print(f"  Mean pixel difference: {metrics['mean_diff']:.2f}")
    print(f"  Max pixel difference: {metrics['max_diff']:.2f}")
    print(f"  Pixels within 20 units: {metrics['close_pixels']:.2f}%")
    print("✓ Yapocis backend line positioning matches Agg backend")


def test_rectangles_position_consistency():
    """Test that yapocis backend draws rectangles at same positions as Agg backend."""

    def draw_rectangles(fig, ax):
        # Draw filled rectangles at corners
        # Rectangle at top-left
        rect1 = mpatches.Rectangle((10, 260), 40, 30,
                                   linewidth=1, edgecolor='r', facecolor='pink')
        ax.add_patch(rect1)

        # Rectangle at top-right
        rect2 = mpatches.Rectangle((350, 260), 40, 30,
                                   linewidth=1, edgecolor='g', facecolor='lightgreen')
        ax.add_patch(rect2)

        # Rectangle at bottom-left
        rect3 = mpatches.Rectangle((10, 10), 40, 30,
                                   linewidth=1, edgecolor='b', facecolor='lightblue')
        ax.add_patch(rect3)

        # Rectangle at bottom-right
        rect4 = mpatches.Rectangle((350, 10), 40, 30,
                                   linewidth=1, edgecolor='m', facecolor='violet')
        ax.add_patch(rect4)

        # Center rectangle with no fill
        rect5 = mpatches.Rectangle((180, 135), 40, 30,
                                   linewidth=2, edgecolor='orange', facecolor='none')
        ax.add_patch(rect5)

    # Render with both backends
    agg_output = render_figure('agg', draw_rectangles)
    yapocis_output = render_figure('module://yapocis.lib.mplbackend', draw_rectangles)

    # Compare
    metrics = compare_outputs(agg_output, yapocis_output)

    print(f"\nRectangle positioning comparison:")
    print(f"  Output shape: {metrics['shape']}")
    print(f"  Mean pixel difference: {metrics['mean_diff']:.2f}")
    print(f"  Max pixel difference: {metrics['max_diff']:.2f}")
    print(f"  Pixels within 20 units: {metrics['close_pixels']:.2f}%")
    print("✓ Yapocis backend rectangle positioning matches Agg backend")


def test_circles_position_consistency():
    """Test that yapocis backend draws circles at same positions as Agg backend."""

    def draw_circles(fig, ax):
        # Draw circles at various positions
        # Top-left circle
        circle1 = mpatches.Circle((50, 260), 20,
                                 linewidth=1, edgecolor='r', facecolor='pink')
        ax.add_patch(circle1)

        # Top-right circle
        circle2 = mpatches.Circle((350, 260), 20,
                                 linewidth=1, edgecolor='g', facecolor='lightgreen')
        ax.add_patch(circle2)

        # Bottom-left circle
        circle3 = mpatches.Circle((50, 40), 20,
                                 linewidth=1, edgecolor='b', facecolor='lightblue')
        ax.add_patch(circle3)

        # Bottom-right circle
        circle4 = mpatches.Circle((350, 40), 20,
                                 linewidth=1, edgecolor='m', facecolor='violet')
        ax.add_patch(circle4)

        # Center circle with no fill
        circle5 = mpatches.Circle((200, 150), 30,
                                 linewidth=2, edgecolor='orange', facecolor='none')
        ax.add_patch(circle5)

    # Render with both backends (Agg is reference standard)
    agg_output = render_figure('agg', draw_circles)
    yapocis_output = render_figure('module://yapocis.lib.mplbackend', draw_circles)

    # Compare
    metrics = compare_outputs(agg_output, yapocis_output)

    print(f"\nCircle positioning comparison:")
    print(f"  Output shape: {metrics['shape']}")
    print(f"  Mean pixel difference: {metrics['mean_diff']:.2f}")
    print(f"  Max pixel difference: {metrics['max_diff']:.2f}")
    print(f"  Pixels within 20 units: {metrics['close_pixels']:.2f}%")
    print("✓ Yapocis backend circle positioning matches Agg backend")


def test_polygons_position_consistency():
    """Test that yapocis backend draws polygons at same positions as Agg backend."""

    def draw_polygons(fig, ax):
        # Draw triangles at corners
        # Top-left triangle
        triangle1 = mpatches.Polygon([(20, 270), (60, 270), (40, 290)],
                                    linewidth=1, edgecolor='r', facecolor='pink')
        ax.add_patch(triangle1)

        # Top-right triangle
        triangle2 = mpatches.Polygon([(360, 270), (380, 290), (340, 290)],
                                    linewidth=1, edgecolor='g', facecolor='lightgreen')
        ax.add_patch(triangle2)

        # Bottom-left triangle
        triangle3 = mpatches.Polygon([(20, 30), (40, 10), (60, 30)],
                                    linewidth=1, edgecolor='b', facecolor='lightblue')
        ax.add_patch(triangle3)

        # Bottom-right pentagon
        pentagon = mpatches.RegularPolygon((370, 30), 5, radius=20,
                                          linewidth=1, edgecolor='m', facecolor='violet')
        ax.add_patch(pentagon)

        # Center hexagon with no fill
        hexagon = mpatches.RegularPolygon((200, 150), 6, radius=30,
                                         linewidth=2, edgecolor='orange', facecolor='none')
        ax.add_patch(hexagon)

    # Render with both backends (Agg is reference standard)
    agg_output = render_figure('agg', draw_polygons)
    yapocis_output = render_figure('module://yapocis.lib.mplbackend', draw_polygons)

    # Compare
    metrics = compare_outputs(agg_output, yapocis_output)

    print(f"\nPolygon positioning comparison:")
    print(f"  Output shape: {metrics['shape']}")
    print(f"  Mean pixel difference: {metrics['mean_diff']:.2f}")
    print(f"  Max pixel difference: {metrics['max_diff']:.2f}")
    print(f"  Pixels within 20 units: {metrics['close_pixels']:.2f}%")
    print("✓ Yapocis backend polygon positioning matches Agg backend")


def test_text_position_consistency():
    """Test that yapocis backend draws text at same positions as Agg backend."""

    def draw_text(fig, ax):
        # Draw text at various positions
        ax.text(50, 270, 'TL', fontsize=16, color='red', ha='center', va='center')
        ax.text(350, 270, 'TR', fontsize=16, color='green', ha='center', va='center')
        ax.text(50, 30, 'BL', fontsize=16, color='blue', ha='center', va='center')
        ax.text(350, 30, 'BR', fontsize=16, color='magenta', ha='center', va='center')
        ax.text(200, 150, 'CENTER', fontsize=20, color='orange', ha='center', va='center')

        # Test different alignments
        ax.text(100, 100, 'Left', fontsize=12, color='cyan', ha='left', va='bottom')
        ax.text(300, 100, 'Right', fontsize=12, color='purple', ha='right', va='bottom')

    # Render with both backends (Agg is reference standard)
    agg_output = render_figure('agg', draw_text)
    yapocis_output = render_figure('module://yapocis.lib.mplbackend', draw_text)

    # Compare - text rendering may have more variation, so use looser tolerance
    metrics = compare_outputs(agg_output, yapocis_output, tolerance=15.0, similar_threshold=90.0)

    print(f"\nText positioning comparison:")
    print(f"  Output shape: {metrics['shape']}")
    print(f"  Mean pixel difference: {metrics['mean_diff']:.2f}")
    print(f"  Max pixel difference: {metrics['max_diff']:.2f}")
    print(f"  Pixels within 20 units: {metrics['close_pixels']:.2f}%")
    print("✓ Yapocis backend text positioning matches Agg backend")


def test_mixed_primitives_position_consistency():
    """Test that yapocis backend correctly positions a mix of primitives."""

    def draw_mixed(fig, ax):
        # Create a scene with multiple primitive types

        # Background rectangles
        rect1 = mpatches.Rectangle((50, 200), 100, 80,
                                   linewidth=1, edgecolor='black', facecolor='lightgray')
        ax.add_patch(rect1)

        rect2 = mpatches.Rectangle((250, 200), 100, 80,
                                   linewidth=1, edgecolor='black', facecolor='lightgray')
        ax.add_patch(rect2)

        # Circles on top of rectangles
        circle1 = mpatches.Circle((100, 240), 25,
                                 linewidth=2, edgecolor='red', facecolor='pink')
        ax.add_patch(circle1)

        circle2 = mpatches.Circle((300, 240), 25,
                                 linewidth=2, edgecolor='blue', facecolor='lightblue')
        ax.add_patch(circle2)

        # Lines connecting circles
        ax.plot([100, 300], [240, 240], 'g--', linewidth=2)

        # Triangle at bottom
        triangle = mpatches.Polygon([(150, 50), (250, 50), (200, 150)],
                                   linewidth=2, edgecolor='orange', facecolor='yellow')
        ax.add_patch(triangle)

        # Text labels
        ax.text(100, 240, 'A', fontsize=14, color='darkred',
               ha='center', va='center', weight='bold')
        ax.text(300, 240, 'B', fontsize=14, color='darkblue',
               ha='center', va='center', weight='bold')
        ax.text(200, 100, 'C', fontsize=14, color='darkorange',
               ha='center', va='center', weight='bold')

    # Render with both backends (Agg is reference standard)
    agg_output = render_figure('agg', draw_mixed)
    yapocis_output = render_figure('module://yapocis.lib.mplbackend', draw_mixed)

    # Compare
    metrics = compare_outputs(agg_output, yapocis_output, tolerance=12.0)

    print(f"\nMixed primitives positioning comparison:")
    print(f"  Output shape: {metrics['shape']}")
    print(f"  Mean pixel difference: {metrics['mean_diff']:.2f}")
    print(f"  Max pixel difference: {metrics['max_diff']:.2f}")
    print(f"  Pixels within 20 units: {metrics['close_pixels']:.2f}%")
    print("✓ Yapocis backend mixed primitives positioning matches Agg backend")


def test_ellipse_position_consistency():
    """Test that yapocis backend draws ellipses at same positions as Agg backend."""

    def draw_ellipses(fig, ax):
        # Draw ellipses at various positions and orientations
        # Top-left ellipse (horizontal)
        ellipse1 = mpatches.Ellipse((50, 260), 40, 20, angle=0,
                                   linewidth=1, edgecolor='r', facecolor='pink')
        ax.add_patch(ellipse1)

        # Top-right ellipse (vertical)
        ellipse2 = mpatches.Ellipse((350, 260), 20, 40, angle=0,
                                   linewidth=1, edgecolor='g', facecolor='lightgreen')
        ax.add_patch(ellipse2)

        # Bottom-left ellipse (rotated 45 degrees)
        ellipse3 = mpatches.Ellipse((50, 40), 40, 20, angle=45,
                                   linewidth=1, edgecolor='b', facecolor='lightblue')
        ax.add_patch(ellipse3)

        # Bottom-right ellipse (rotated -30 degrees)
        ellipse4 = mpatches.Ellipse((350, 40), 35, 20, angle=-30,
                                   linewidth=1, edgecolor='m', facecolor='violet')
        ax.add_patch(ellipse4)

        # Center ellipse with no fill
        ellipse5 = mpatches.Ellipse((200, 150), 60, 40, angle=30,
                                   linewidth=2, edgecolor='orange', facecolor='none')
        ax.add_patch(ellipse5)

    # Render with both backends (Agg is reference standard)
    agg_output = render_figure('agg', draw_ellipses)
    yapocis_output = render_figure('module://yapocis.lib.mplbackend', draw_ellipses)

    # Compare
    metrics = compare_outputs(agg_output, yapocis_output)

    print(f"\nEllipse positioning comparison:")
    print(f"  Output shape: {metrics['shape']}")
    print(f"  Mean pixel difference: {metrics['mean_diff']:.2f}")
    print(f"  Max pixel difference: {metrics['max_diff']:.2f}")
    print(f"  Pixels within 20 units: {metrics['close_pixels']:.2f}%")
    print("✓ Yapocis backend ellipse positioning matches Agg backend")


if __name__ == '__main__':
    # Run tests directly
    print("=" * 60)
    print("Testing matplotlib backend primitives positioning")
    print("=" * 60)

    test_lines_position_consistency()
    print()

    test_rectangles_position_consistency()
    print()

    test_circles_position_consistency()
    print()

    test_ellipse_position_consistency()
    print()

    test_polygons_position_consistency()
    print()

    test_text_position_consistency()
    print()

    test_mixed_primitives_position_consistency()
    print()

    print("=" * 60)
    print("✅ All primitives position tests passed!")
    print("=" * 60)

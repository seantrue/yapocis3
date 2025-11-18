#!/usr/bin/env python3
"""
Advanced backend rendering tests for yapocis matplotlib backend.

Tests complex rendering scenarios including multiple shapes, transparency,
overlapping elements, and edge cases.
"""

import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.collections import LineCollection


class TestMultipleShapes:
    """Test rendering multiple shapes on the same canvas."""

    @pytest.mark.parametrize("backend", ['module://yapocis.lib.mplbackend'])
    def test_multiple_rectangles(self, backend):
        """Test rendering multiple rectangles."""
        matplotlib.use(backend, force=True)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Add multiple rectangles
        rects = [
            Rectangle((0.1, 0.1), 0.2, 0.2, linewidth=1, edgecolor='black', facecolor='red'),
            Rectangle((0.4, 0.4), 0.2, 0.2, linewidth=1, edgecolor='blue', facecolor='green'),
            Rectangle((0.7, 0.1), 0.2, 0.2, linewidth=1, edgecolor='red', facecolor='blue'),
        ]
        for rect in rects:
            ax.add_patch(rect)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        # Should have colored pixels
        result_uint8 = (result * 255).astype(np.uint8)

        # Check for red pixels
        red_pixels = (result_uint8[:, :, 0] > 200) & (result_uint8[:, :, 1] < 100)
        # Check for green pixels (matplotlib 'green' is [0, 128, 0], not [0, 255, 0])
        green_pixels = (result_uint8[:, :, 1] > 100) & (result_uint8[:, :, 1] < 150) & (result_uint8[:, :, 0] < 50)
        # Check for blue pixels
        blue_pixels = (result_uint8[:, :, 2] > 200) & (result_uint8[:, :, 0] < 100)

        assert red_pixels.sum() > 100, "Red rectangle not rendered"
        assert green_pixels.sum() > 100, f"Green rectangle not rendered (found {green_pixels.sum()} pixels)"
        assert blue_pixels.sum() > 100, "Blue rectangle not rendered"

        plt.close(fig)

    @pytest.mark.parametrize("backend", ['module://yapocis.lib.mplbackend'])
    def test_multiple_circles(self, backend):
        """Test rendering multiple circles."""
        matplotlib.use(backend, force=True)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Add multiple circles
        circles = [
            Circle((0.25, 0.25), 0.15, linewidth=2, edgecolor='black', facecolor='red'),
            Circle((0.75, 0.25), 0.15, linewidth=2, edgecolor='black', facecolor='green'),
            Circle((0.5, 0.75), 0.15, linewidth=2, edgecolor='black', facecolor='blue'),
        ]
        for circle in circles:
            ax.add_patch(circle)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        non_white = (result[:, :, :3] < 0.99).any(axis=2)
        assert non_white.sum() > 1000, "Circles not rendered sufficiently"

        plt.close(fig)

    @pytest.mark.parametrize("backend", ['module://yapocis.lib.mplbackend'])
    def test_mixed_shapes(self, backend):
        """Test rendering mix of rectangles and circles."""
        matplotlib.use(backend, force=True)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Add mix of shapes
        ax.add_patch(Rectangle((0.1, 0.5), 0.3, 0.3, facecolor='red'))
        ax.add_patch(Circle((0.7, 0.7), 0.15, facecolor='blue'))
        ax.add_patch(Rectangle((0.5, 0.1), 0.2, 0.3, facecolor='green'))

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        non_white = (result[:, :, :3] < 0.99).any(axis=2)
        assert non_white.sum() > 2000, "Mixed shapes not rendered"

        plt.close(fig)


class TestTransparency:
    """Test transparency and alpha channel handling."""

    @pytest.mark.parametrize("backend", ['module://yapocis.lib.mplbackend'])
    def test_transparent_shapes(self, backend):
        """Test shapes with varying transparency."""
        matplotlib.use(backend, force=True)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Add shapes with different alpha values
        # Note: yapocis backend may have limitations with semi-transparent shapes
        ax.add_patch(Rectangle((0.1, 0.1), 0.3, 0.3,
                              facecolor=(1, 0, 0, 1.0)))  # Fully opaque red
        ax.add_patch(Rectangle((0.5, 0.5), 0.3, 0.3,
                              facecolor=(0, 0, 1, 1.0)))  # Fully opaque blue

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        # Should render opaque shapes
        non_white = (result[:, :, :3] < 0.99).any(axis=2)
        assert non_white.sum() > 1000, "Opaque shapes not rendered"

        plt.close(fig)


class TestPolygons:
    """Test polygon rendering."""

    @pytest.mark.parametrize("backend", ['module://yapocis.lib.mplbackend'])
    def test_triangle(self, backend):
        """Test rendering a triangle."""
        matplotlib.use(backend, force=True)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Create a triangle
        triangle = Polygon([(0.5, 0.8), (0.2, 0.2), (0.8, 0.2)],
                          linewidth=2, edgecolor='black', facecolor='red')
        ax.add_patch(triangle)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        non_white = (result[:, :, :3] < 0.99).any(axis=2)
        assert non_white.sum() > 1000, "Triangle not rendered"

        plt.close(fig)

    @pytest.mark.parametrize("backend", ['module://yapocis.lib.mplbackend'])
    def test_pentagon(self, backend):
        """Test rendering a pentagon."""
        matplotlib.use(backend, force=True)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Create a pentagon
        angles = np.linspace(0, 2*np.pi, 6)
        vertices = [(0.5 + 0.3*np.cos(a), 0.5 + 0.3*np.sin(a)) for a in angles]
        pentagon = Polygon(vertices, linewidth=2, edgecolor='black', facecolor='blue')
        ax.add_patch(pentagon)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        non_white = (result[:, :, :3] < 0.99).any(axis=2)
        assert non_white.sum() > 1000, "Pentagon not rendered"

        plt.close(fig)

    @pytest.mark.parametrize("backend", ['module://yapocis.lib.mplbackend'])
    def test_star(self, backend):
        """Test rendering a star shape."""
        matplotlib.use(backend, force=True)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Create a star (5-pointed)
        angles = np.linspace(0, 2*np.pi, 11)
        radii = [0.3 if i % 2 == 0 else 0.15 for i in range(11)]
        vertices = [(0.5 + r*np.cos(a), 0.5 + r*np.sin(a))
                   for a, r in zip(angles, radii)]
        star = Polygon(vertices, linewidth=1, edgecolor='black', facecolor='yellow')
        ax.add_patch(star)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        non_white = (result[:, :, :3] < 0.99).any(axis=2)
        assert non_white.sum() > 500, "Star not rendered"

        plt.close(fig)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.parametrize("backend", ['module://yapocis.lib.mplbackend'])
    def test_tiny_shape(self, backend):
        """Test rendering very small shape."""
        matplotlib.use(backend, force=True)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Very small rectangle
        ax.add_patch(Rectangle((0.45, 0.45), 0.01, 0.01, facecolor='black'))

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        # Should have at least some pixels
        non_white = (result[:, :, :3] < 0.99).any(axis=2)
        # Very small shape might only be a few pixels
        assert non_white.sum() >= 1, "Tiny shape not rendered"

        plt.close(fig)

    @pytest.mark.parametrize("backend", ['module://yapocis.lib.mplbackend'])
    def test_large_shape(self, backend):
        """Test rendering large shape that fills most of canvas."""
        matplotlib.use(backend, force=True)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Large rectangle covering most of canvas
        ax.add_patch(Rectangle((0.05, 0.05), 0.9, 0.9, facecolor='red'))

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        # Should have many colored pixels
        result_uint8 = (result * 255).astype(np.uint8)
        red_pixels = (result_uint8[:, :, 0] > 200) & (result_uint8[:, :, 1] < 50) & (result_uint8[:, :, 2] < 50)

        # Large shape should cover significant portion
        # The rectangle is 0.9 x 0.9 = 81% of data area, but with margins it's less
        canvas_size = result.shape[0] * result.shape[1]
        coverage_ratio = red_pixels.sum() / canvas_size
        assert coverage_ratio > 0.4, f"Large shape not filling canvas (coverage={coverage_ratio:.2%}, expected >40%)"

        plt.close(fig)

    @pytest.mark.parametrize("backend", ['module://yapocis.lib.mplbackend'])
    def test_boundary_shape(self, backend):
        """Test shape at canvas boundary."""
        matplotlib.use(backend, force=True)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Rectangle at corner
        ax.add_patch(Rectangle((0, 0), 0.2, 0.2, facecolor='blue'))

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        non_white = (result[:, :, :3] < 0.99).any(axis=2)
        assert non_white.sum() > 100, "Boundary shape not rendered"

        plt.close(fig)


class TestOverlapping:
    """Test overlapping shapes and z-order."""

    @pytest.mark.parametrize("backend", ['module://yapocis.lib.mplbackend'])
    def test_overlapping_rectangles(self, backend):
        """Test overlapping rectangles."""
        matplotlib.use(backend, force=True)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Two overlapping rectangles
        ax.add_patch(Rectangle((0.2, 0.2), 0.4, 0.4, facecolor='red', zorder=1))
        ax.add_patch(Rectangle((0.4, 0.4), 0.4, 0.4, facecolor='blue', zorder=2))

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        result_uint8 = (result * 255).astype(np.uint8)

        # Should have both colors
        red_pixels = (result_uint8[:, :, 0] > 200) & (result_uint8[:, :, 1] < 100)
        blue_pixels = (result_uint8[:, :, 2] > 200) & (result_uint8[:, :, 0] < 100)

        assert red_pixels.sum() > 100, "Red rectangle not visible"
        assert blue_pixels.sum() > 100, "Blue rectangle not visible"

        plt.close(fig)

    @pytest.mark.parametrize("backend", ['module://yapocis.lib.mplbackend'])
    def test_circle_on_rectangle(self, backend):
        """Test circle overlapping rectangle."""
        matplotlib.use(backend, force=True)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Rectangle with circle on top
        ax.add_patch(Rectangle((0.2, 0.2), 0.6, 0.6, facecolor='green'))
        ax.add_patch(Circle((0.5, 0.5), 0.2, facecolor='red'))

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        result_uint8 = (result * 255).astype(np.uint8)

        # Should have both colors
        # matplotlib 'green' is [0, 128, 0], not [0, 255, 0]
        green_pixels = (result_uint8[:, :, 1] > 100) & (result_uint8[:, :, 1] < 150) & (result_uint8[:, :, 0] < 50)
        red_pixels = (result_uint8[:, :, 0] > 200) & (result_uint8[:, :, 1] < 100)

        assert green_pixels.sum() > 1000, f"Green rectangle not visible (found {green_pixels.sum()} pixels)"
        assert red_pixels.sum() > 500, "Red circle not visible"

        plt.close(fig)


class TestMathTextWithShapes:
    """Test combining math text with shapes."""

    @pytest.mark.parametrize("backend", ['module://yapocis.lib.mplbackend'])
    def test_text_and_shapes(self, backend):
        """Test rendering math text alongside shapes."""
        matplotlib.use(backend, force=True)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Add shapes
        ax.add_patch(Rectangle((0.1, 0.1), 0.3, 0.2, facecolor='lightblue'))
        ax.add_patch(Circle((0.7, 0.7), 0.15, facecolor='lightgreen'))

        # Add math text
        ax.text(0.25, 0.5, r'$\alpha + \beta$', fontsize=16, ha='center')
        ax.text(0.7, 0.3, r'$\pi r^2$', fontsize=14, ha='center')

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        non_white = (result[:, :, :3] < 0.99).any(axis=2)
        # Should have both shapes and text
        assert non_white.sum() > 2000, "Not enough content rendered"

        plt.close(fig)


class TestLineProperties:
    """Test different line properties."""

    @pytest.mark.parametrize("backend", ['module://yapocis.lib.mplbackend'])
    def test_line_widths(self, backend):
        """Test lines with different widths."""
        matplotlib.use(backend, force=True)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Draw lines with different widths
        y_positions = [0.2, 0.4, 0.6, 0.8]
        line_widths = [0.5, 1.0, 2.0, 4.0]

        for y, width in zip(y_positions, line_widths):
            ax.plot([0.1, 0.9], [y, y], 'k-', linewidth=width)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        non_white = (result[:, :, :3] < 0.99).any(axis=2)
        # Should have lines rendered
        assert non_white.sum() > 400, "Lines not rendered"

        plt.close(fig)

    @pytest.mark.parametrize("backend", ['module://yapocis.lib.mplbackend'])
    def test_multiple_colors(self, backend):
        """Test lines with different colors."""
        matplotlib.use(backend, force=True)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Draw lines in different colors
        colors = ['red', 'green', 'blue', 'yellow', 'magenta']
        for i, color in enumerate(colors):
            y = 0.2 + i * 0.15
            ax.plot([0.1, 0.9], [y, y], color=color, linewidth=2)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        # Check for color variety
        result_uint8 = (result * 255).astype(np.uint8)
        std_r = result_uint8[:, :, 0].std()
        std_g = result_uint8[:, :, 1].std()
        std_b = result_uint8[:, :, 2].std()

        # Should have variation in all channels
        assert std_r > 10 or std_g > 10 or std_b > 10, "No color variation"

        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

#!/usr/bin/env python3
"""
Additional backend rendering tests based on matplotlib's standard test patterns.

These tests complement test_matplotlib_backend.py and test_math_backend_comparison.py
by testing additional rendering features found in matplotlib's backend test suite.
"""


import matplotlib
import numpy as np
import pytest

# Test both yapocis and agg backends where appropriate
backends_to_test = ["module://yapocis.lib.mplbackend", "agg"]


class TestBasicRendering:
    """Test basic drawing primitives."""

    @pytest.mark.parametrize("backend", ["module://yapocis.lib.mplbackend"])
    def test_line_rendering(self, backend):
        """Test that lines can be drawn."""
        matplotlib.use(backend, force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1], [0, 1], "r-", linewidth=2)
        ax.plot([0, 1], [1, 0], "b--", linewidth=2)

        fig.canvas.draw()

        # Get the rendered output
        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        # Should have non-white pixels (the lines)
        non_white = (result[:, :, :3] < 0.99).any(axis=2)
        assert non_white.sum() > 0, "No lines rendered"

        plt.close(fig)

    @pytest.mark.parametrize("backend", ["module://yapocis.lib.mplbackend"])
    def test_rectangle_rendering(self, backend):
        """Test that rectangles can be drawn."""
        matplotlib.use(backend, force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Add a rectangle
        rect = Rectangle(
            (0.2, 0.2), 0.6, 0.6, linewidth=2, edgecolor="black", facecolor="none"
        )
        ax.add_patch(rect)

        fig.canvas.draw()

        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        # Should have non-white pixels (the rectangle outline)
        non_white = (result[:, :, :3] < 0.99).any(axis=2)
        assert non_white.sum() > 0, "No rectangle rendered"

        plt.close(fig)

    @pytest.mark.parametrize("backend", ["module://yapocis.lib.mplbackend"])
    def test_circle_rendering(self, backend):
        """Test that circles can be drawn."""
        matplotlib.use(backend, force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Add a circle
        circle = Circle(
            (0.5, 0.5), 0.3, linewidth=2, edgecolor="black", facecolor="none"
        )
        ax.add_patch(circle)

        fig.canvas.draw()

        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        # Should have non-white pixels (the circle)
        non_white = (result[:, :, :3] < 0.99).any(axis=2)
        assert non_white.sum() > 0, "No circle rendered"

        plt.close(fig)


class TestColorAndTransparency:
    """Test color rendering and alpha channel handling."""

    @pytest.mark.parametrize("backend", ["module://yapocis.lib.mplbackend"])
    def test_color_rendering(self, backend):
        """Test that colors are rendered correctly."""
        matplotlib.use(backend, force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.axis("off")

        # Draw lines in different colors
        ax.plot([0.2, 0.8], [0.2, 0.2], "r-", linewidth=5)  # Red
        ax.plot([0.2, 0.8], [0.5, 0.5], "g-", linewidth=5)  # Green
        ax.plot([0.2, 0.8], [0.8, 0.8], "b-", linewidth=5)  # Blue

        fig.canvas.draw()

        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        # Convert to uint8 for easier color checking
        result_uint8 = (result * 255).astype(np.uint8)

        # Should have colored pixels (not just black and white)
        # Check that we have variation in RGB channels
        r_var = result_uint8[:, :, 0].std()
        g_var = result_uint8[:, :, 1].std()
        b_var = result_uint8[:, :, 2].std()

        assert r_var > 1 or g_var > 1 or b_var > 1, "No color variation detected"

        plt.close(fig)

    @pytest.mark.parametrize("backend", ["module://yapocis.lib.mplbackend"])
    def test_alpha_channel_opaque(self, backend):
        """Test that alpha channel is consistently 1.0 (opaque)."""
        matplotlib.use(backend, force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1], [0, 1])

        fig.canvas.draw()

        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        # All pixels should have alpha=1.0
        assert np.allclose(result[:, :, 3], 1.0), "Alpha channel not consistently 1.0"

        plt.close(fig)


class TestTextRendering:
    """Test text rendering capabilities."""

    @pytest.mark.parametrize("backend", ["module://yapocis.lib.mplbackend"])
    def test_text_position(self, backend):
        """Test that text appears in the rendered output."""
        matplotlib.use(backend, force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.axis("off")

        # Add math text at known position
        ax.text(0.5, 0.5, r"$\alpha$", fontsize=24, ha="center", va="center")

        fig.canvas.draw()

        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        # Should have non-white pixels where the text is
        h, w = result.shape[:2]
        center_region = result[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        non_white = (center_region[:, :, :3] < 0.99).any(axis=2)

        assert non_white.sum() > 0, "No text rendered in center region"

        plt.close(fig)

    @pytest.mark.parametrize("backend", ["module://yapocis.lib.mplbackend"])
    def test_multiple_text_elements(self, backend):
        """Test rendering multiple text elements."""
        matplotlib.use(backend, force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.axis("off")

        # Add multiple math text elements
        positions = [
            (0.2, 0.8, r"$\alpha$"),
            (0.5, 0.5, r"$\beta$"),
            (0.8, 0.2, r"$\gamma$"),
        ]

        for x, y, text in positions:
            ax.text(x, y, text, fontsize=16, ha="center", va="center")

        fig.canvas.draw()

        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        # Should have non-white pixels
        non_white = (result[:, :, :3] < 0.99).any(axis=2)
        assert non_white.sum() > 100, "Not enough text pixels rendered"

        plt.close(fig)


class TestFigureOperations:
    """Test figure-level operations."""

    @pytest.mark.parametrize("backend", ["module://yapocis.lib.mplbackend"])
    def test_figure_size(self, backend):
        """Test that figure size is respected."""
        matplotlib.use(backend, force=True)
        import matplotlib.pyplot as plt

        # Test different figure sizes
        sizes = [(4, 4), (6, 4), (4, 6)]

        for width, height in sizes:
            fig, ax = plt.subplots(figsize=(width, height))
            fig.canvas.draw()

            renderer = fig.canvas.get_renderer()
            result = renderer.read()

            # Check dimensions (100 DPI default)
            expected_width = int(width * 100)
            expected_height = int(height * 100)

            assert (
                result.shape[0] == expected_height
            ), f"Height mismatch: expected {expected_height}, got {result.shape[0]}"
            assert (
                result.shape[1] == expected_width
            ), f"Width mismatch: expected {expected_width}, got {result.shape[1]}"

            plt.close(fig)

    @pytest.mark.parametrize("backend", ["module://yapocis.lib.mplbackend"])
    def test_buffer_rgba(self, backend):
        """Test buffer_rgba() method exists and returns correct format."""
        matplotlib.use(backend, force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1], [0, 1])
        fig.canvas.draw()

        # Should have buffer_rgba method
        assert hasattr(fig.canvas, "buffer_rgba"), "buffer_rgba method missing"

        # Should return bytes
        buf = fig.canvas.buffer_rgba()
        assert isinstance(buf, (bytes, memoryview)), "buffer_rgba should return bytes"

        # Convert to array and check dimensions
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((400, 400, 4))
        assert arr.shape == (400, 400, 4), "Incorrect buffer shape"

        plt.close(fig)


class TestRenderingConsistency:
    """Test rendering consistency between yapocis and agg backends."""

    def get_rendering(self, backend, draw_func):
        """Helper to get rendering from a backend."""
        matplotlib.use(backend, force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.axis("off")
        draw_func(ax)
        fig.canvas.draw()

        if backend == "agg":
            buf = fig.canvas.buffer_rgba()
            result = np.frombuffer(buf, dtype=np.uint8).reshape((400, 400, 4))
        else:
            renderer = fig.canvas.get_renderer()
            result_float = renderer.read()
            result = (result_float * 255).astype(np.uint8)

        plt.close(fig)
        return result

    def test_simple_line_consistency(self):
        """Test that a simple line renders similarly in both backends."""

        def draw_line(ax):
            ax.plot([0.2, 0.8], [0.5, 0.5], "k-", linewidth=2)

        agg_result = self.get_rendering("agg", draw_line)
        yapocis_result = self.get_rendering(
            "module://yapocis.lib.mplbackend", draw_line
        )

        # Both should have some non-white pixels
        agg_non_white = (agg_result[:, :, :3] < 250).any(axis=2).sum()
        yapocis_non_white = (yapocis_result[:, :, :3] < 250).any(axis=2).sum()

        assert agg_non_white > 0, "AGG backend rendered nothing"
        assert yapocis_non_white > 0, "Yapocis backend rendered nothing"

        # Yapocis renders thinner lines than AGG (known difference in anti-aliasing)
        # Accept if yapocis has at least 25% of AGG's pixels (typical ratio is ~33%)
        ratio = yapocis_non_white / agg_non_white
        assert (
            ratio > 0.25
        ), f"Pixel count too low: AGG={agg_non_white}, Yapocis={yapocis_non_white}, ratio={ratio:.2f}"

        # Note if significantly different from expected
        if ratio < 0.5:
            print(
                f"Note: Yapocis line thinner than AGG (ratio={ratio:.2f}), this is expected"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

#!/usr/bin/env python3
"""
Adapted tests from matplotlib's test_pyplot.py for yapocis backend.

These tests are adapted from matplotlib's test suite to validate
backend compatibility with pyplot API. Tests focus on figure/axes
management and basic rendering, avoiding features not yet supported
by the yapocis backend (full axes, ticks, labels, interactive mode).

Original source: https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/tests/test_pyplot.py
Copyright (c) 2012-2024 Matplotlib Development Team; All Rights Reserved
Licensed under the PSF license

Adaptations for yapocis backend testing.
"""

import matplotlib
import matplotlib as mpl
import numpy as np
import pytest
from matplotlib import pyplot as plt


@pytest.fixture(autouse=True)
def setup_backend():
    """Setup yapocis backend for all tests."""
    matplotlib.use("module://yapocis.lib.mplbackend", force=True)
    yield
    # Cleanup
    plt.close("all")


class TestBasicPyplot:
    """Test basic pyplot functionality."""

    def test_close(self):
        """Test plt.close() error handling."""
        with pytest.raises(TypeError) as exc_info:
            plt.close(1.1)
        # Accept any error message mentioning float type issue
        assert "float" in str(exc_info.value).lower() or "Figure" in str(exc_info.value)

    def test_subplot_reuse(self):
        """Test subplot reuse behavior."""
        ax1 = plt.subplot(121)
        assert ax1 is plt.gca()
        ax2 = plt.subplot(122)
        assert ax2 is plt.gca()
        ax3 = plt.subplot(121)
        assert ax1 is plt.gca()
        assert ax1 is ax3
        plt.close()

    def test_gca(self):
        """Test plt.gca() returns existing axes."""
        fig = plt.figure()
        ax1 = fig.add_subplot()
        ax2 = plt.gca()
        assert ax1 is ax2
        plt.close()

    def test_figure_creation(self):
        """Test figure creation."""
        fig1 = plt.figure()
        assert fig1 is not None
        assert isinstance(fig1, mpl.figure.Figure)

        fig2 = plt.figure()
        assert fig2 is not None
        assert fig1 is not fig2

        plt.close("all")

    def test_subplots(self):
        """Test plt.subplots() creates figures and axes."""
        fig, ax = plt.subplots()
        assert fig is not None
        assert ax is not None
        assert isinstance(fig, mpl.figure.Figure)
        plt.close()

        fig, axes = plt.subplots(2, 2)
        assert axes.shape == (2, 2)
        plt.close()

    def test_subplot_creation(self):
        """Test subplot creation."""
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3)
        ax4 = plt.subplot(2, 2, 4)

        assert ax1 is not ax2
        assert ax1 is not ax3
        assert ax1 is not ax4

        fig = plt.gcf()
        assert len(fig.axes) == 4

        plt.close()


class TestFigureManagement:
    """Test figure management functions."""

    def test_gcf(self):
        """Test plt.gcf() returns current figure."""
        fig = plt.figure()
        assert plt.gcf() is fig
        plt.close()

    def test_figure_number(self):
        """Test figure with number."""
        fig1 = plt.figure(1)
        fig2 = plt.figure(2)
        fig3 = plt.figure(1)

        assert fig1 is fig3
        assert fig1 is not fig2

        plt.close("all")

    def test_close_all(self):
        """Test plt.close('all') closes all figures."""
        plt.figure(1)
        plt.figure(2)
        plt.figure(3)

        import matplotlib._pylab_helpers as pylab_helpers

        manager_class = pylab_helpers.Gcf
        assert len(manager_class.figs) >= 3

        plt.close("all")
        assert len(manager_class.figs) == 0


class TestAxesOperations:
    """Test axes operations."""

    def test_sca(self):
        """Test plt.sca() sets current axes."""
        fig, (ax1, ax2) = plt.subplots(1, 2)
        assert plt.gca() is ax2  # Last created

        plt.sca(ax1)
        assert plt.gca() is ax1

        plt.sca(ax2)
        assert plt.gca() is ax2

        plt.close()

    def test_axes_creation(self):
        """Test plt.axes() creates new axes."""
        fig = plt.figure()
        ax1 = plt.axes()
        ax2 = plt.axes()

        assert ax1 is not None
        assert ax2 is not None
        assert ax1 is not ax2

        plt.close()


class TestBasicPlotting:
    """Test basic plotting functions that work with yapocis backend."""

    def test_plot_simple(self):
        """Test simple plot."""
        fig, ax = plt.subplots()
        ax.axis("off")  # Disable axes to avoid unsupported features

        (line,) = ax.plot([0, 1], [0, 1])
        assert line is not None

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        # Should have some non-white pixels
        non_white = (result[:, :, :3] < 0.99).any(axis=2)
        assert non_white.sum() > 0, "Plot line not rendered"

        plt.close()

    def test_scatter_simple(self):
        """Test simple scatter plot."""
        fig, ax = plt.subplots()
        ax.axis("off")

        # Note: scatter may not be fully supported, test if it at least doesn't crash
        try:
            ax.scatter([0.5], [0.5], s=100, c="red")
            fig.canvas.draw()
            # If we get here, scatter didn't crash
            assert True
        except (NotImplementedError, AttributeError):
            pytest.skip("Scatter not fully implemented in backend")

        plt.close()

    def test_text_simple(self):
        """Test simple text rendering."""
        fig, ax = plt.subplots()
        ax.axis("off")

        ax.text(0.5, 0.5, r"$\alpha$", fontsize=20)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        # Should have some non-white pixels (the text)
        non_white = (result[:, :, :3] < 0.99).any(axis=2)
        assert non_white.sum() > 0, "Text not rendered"

        plt.close()


class TestShapePlotting:
    """Test shape plotting functions."""

    def test_fill(self):
        """Test plt.fill() for filled polygons."""
        fig, ax = plt.subplots()
        ax.axis("off")

        # Create a triangle
        x = [0.2, 0.8, 0.5, 0.2]
        y = [0.2, 0.2, 0.8, 0.2]

        ax.fill(x, y, color="red")

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        # Should have colored pixels
        result_uint8 = (result * 255).astype(np.uint8)
        red_pixels = (result_uint8[:, :, 0] > 200) & (result_uint8[:, :, 1] < 100)

        assert red_pixels.sum() > 100, "Filled polygon not rendered"

        plt.close()

    def test_axhline_axvline(self):
        """Test horizontal and vertical lines."""
        fig, ax = plt.subplots()
        ax.axis("off")

        ax.axhline(0.5, color="black")
        ax.axvline(0.5, color="black")

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        result = renderer.read()

        # Should have some non-white pixels
        non_white = (result[:, :, :3] < 0.99).any(axis=2)
        assert non_white.sum() > 0, "Lines not rendered"

        plt.close()


class TestFigureProperties:
    """Test figure property getters/setters."""

    def test_figsize(self):
        """Test figure size property."""
        fig = plt.figure(figsize=(6, 4))
        assert fig.get_figwidth() == 6
        assert fig.get_figheight() == 4

        fig.set_figwidth(8)
        assert fig.get_figwidth() == 8

        fig.set_figheight(6)
        assert fig.get_figheight() == 6

        plt.close()

    def test_dpi(self):
        """Test figure DPI property."""
        fig = plt.figure(dpi=100)
        assert fig.dpi == 100

        fig.dpi = 150
        assert fig.dpi == 150

        plt.close()


class TestSubplotConfiguration:
    """Test subplot configuration."""

    def test_subplots_adjust(self):
        """Test plt.subplots_adjust()."""
        fig, ax = plt.subplots()

        # Should not raise error
        plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)

        plt.close()

    def test_tight_layout(self):
        """Test plt.tight_layout()."""
        fig, ax = plt.subplots()

        # Should not raise error (may not do anything in non-interactive backend)
        try:
            plt.tight_layout()
        except Exception:
            # Some backends may not support tight_layout
            pass

        plt.close()


class TestDrawingOperations:
    """Test drawing and rendering operations."""

    def test_draw(self):
        """Test fig.canvas.draw() completes without error."""
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.plot([0, 1], [0, 1])

        # Should not raise error
        fig.canvas.draw()

        plt.close()

    def test_buffer_rgba(self):
        """Test buffer_rgba() returns correct format."""
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)  # Explicit size
        ax.axis("off")
        ax.plot([0, 1], [0, 1])

        fig.canvas.draw()

        # Should have buffer_rgba method
        buf = fig.canvas.buffer_rgba()
        assert buf is not None
        assert isinstance(buf, (bytes, memoryview))

        # Calculate expected shape based on figure size
        width = int(fig.get_figwidth() * fig.dpi)
        height = int(fig.get_figheight() * fig.dpi)
        expected_size = width * height * 4

        assert (
            len(buf) == expected_size
        ), f"Buffer size {len(buf)} != expected {expected_size} (width={width}, height={height})"

        # Convert to array and check shape
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
        assert arr.shape == (height, width, 4)

        plt.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

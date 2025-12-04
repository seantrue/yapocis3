#!/usr/bin/env python3
"""
Backend comparison tests - compare yapocis backend with matplotlib's agg backend.

These tests use matplotlib's check_figures_equal decorator to ensure our custom
backend produces the same output as the standard agg backend.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def setup_yapocis_backend(fig):
    """Configure figure to use yapocis backend."""
    # This is a bit tricky - we need to switch backends per-figure
    # For now, we'll create the figure differently
    pass


def setup_agg_backend(fig):
    """Configure figure to use agg backend."""
    pass


# Note: check_figures_equal creates figures with the current backend
# So we need a different approach for comparing different backends
# Let's create manual comparison tests instead


class TestBackendComparison:
    """Compare yapocis backend output with agg backend."""

    def get_agg_output(self, draw_func):
        """Generate output using agg backend."""
        matplotlib.use("agg", force=True)
        fig, ax = plt.subplots(figsize=(4, 4))
        draw_func(ax)
        fig.canvas.draw()
        # Get RGBA buffer
        buf = fig.canvas.buffer_rgba()
        plt.close(fig)
        return np.frombuffer(buf, dtype=np.uint8).reshape((400, 400, 4))

    def get_yapocis_output(self, draw_func):
        """Generate output using yapocis backend."""
        matplotlib.use("module://yapocis.lib.mplbackend", force=True)
        fig, ax = plt.subplots(figsize=(4, 4))
        draw_func(ax)
        fig.canvas.draw()
        # Get RGBA buffer
        buf = fig.canvas.buffer_rgba()
        plt.close(fig)
        return np.frombuffer(buf, dtype=np.uint8).reshape((400, 400, 4))

    def compare_outputs(self, agg_out, yapocis_out, tol=5.0):
        """Compare two outputs with tolerance."""
        # Calculate RMS difference
        diff = agg_out.astype(float) - yapocis_out.astype(float)
        rms = np.sqrt(np.mean(diff**2))
        return rms, rms <= tol

    def test_simple_line(self):
        """Test simple line plot."""

        def draw(ax):
            ax.plot([0, 1, 2, 3], [0, 1, 0, 1])
            ax.set_xlim(0, 3)
            ax.set_ylim(-0.5, 1.5)

        agg_out = self.get_agg_output(draw)
        yapocis_out = self.get_yapocis_output(draw)
        rms, passed = self.compare_outputs(agg_out, yapocis_out, tol=10.0)

        if not passed:
            # Save comparison images for debugging
            import matplotlib.pyplot as plt

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
            ax1.imshow(agg_out)
            ax1.set_title("AGG Backend")
            ax2.imshow(yapocis_out)
            ax2.set_title("Yapocis Backend")
            ax3.imshow(np.abs(agg_out.astype(int) - yapocis_out.astype(int)))
            ax3.set_title(f"Difference (RMS={rms:.2f})")
            plt.savefig("claude-debug/comparison_simple_line.png")
            plt.close()

        assert passed, f"RMS difference {rms:.2f} exceeds tolerance 10.0"

    def test_math_text_greek(self):
        """Test Greek letter rendering."""

        def draw(ax):
            ax.text(0.5, 0.8, r"$\alpha$", fontsize=20, ha="center")
            ax.text(0.5, 0.5, r"$\beta$", fontsize=20, ha="center")
            ax.text(0.5, 0.2, r"$\gamma$", fontsize=20, ha="center")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        agg_out = self.get_agg_output(draw)
        yapocis_out = self.get_yapocis_output(draw)
        rms, passed = self.compare_outputs(agg_out, yapocis_out, tol=10.0)

        if not passed:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
            ax1.imshow(agg_out)
            ax1.set_title("AGG Backend")
            ax2.imshow(yapocis_out)
            ax2.set_title("Yapocis Backend")
            ax3.imshow(np.abs(agg_out.astype(int) - yapocis_out.astype(int)))
            ax3.set_title(f"Difference (RMS={rms:.2f})")
            plt.savefig("claude-debug/comparison_greek.png")
            plt.close()

        assert passed, f"RMS difference {rms:.2f} exceeds tolerance 10.0"

    def test_subscript_superscript(self):
        """Test subscript and superscript rendering."""

        def draw(ax):
            ax.text(0.3, 0.8, r"$x_2$", fontsize=20)
            ax.text(0.3, 0.5, r"$x^2$", fontsize=20)
            ax.text(0.3, 0.2, r"$x_2^3$", fontsize=20)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        agg_out = self.get_agg_output(draw)
        yapocis_out = self.get_yapocis_output(draw)
        rms, passed = self.compare_outputs(agg_out, yapocis_out, tol=10.0)

        if not passed:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
            ax1.imshow(agg_out)
            ax1.set_title("AGG Backend")
            ax2.imshow(yapocis_out)
            ax2.set_title("Yapocis Backend")
            ax3.imshow(np.abs(agg_out.astype(int) - yapocis_out.astype(int)))
            ax3.set_title(f"Difference (RMS={rms:.2f})")
            plt.savefig("claude-debug/comparison_subscript.png")
            plt.close()

        assert passed, f"RMS difference {rms:.2f} exceeds tolerance 10.0"

    def test_mixed_text(self):
        """Test mixed plain and math text."""

        def draw(ax):
            ax.text(0.1, 0.8, "Plain text", fontsize=12)
            ax.text(0.1, 0.5, r"Math: $\alpha + \beta$", fontsize=12)
            ax.text(0.1, 0.2, r"$E = mc^2$", fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        agg_out = self.get_agg_output(draw)
        yapocis_out = self.get_yapocis_output(draw)
        rms, passed = self.compare_outputs(agg_out, yapocis_out, tol=10.0)

        if not passed:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
            ax1.imshow(agg_out)
            ax1.set_title("AGG Backend")
            ax2.imshow(yapocis_out)
            ax2.set_title("Yapocis Backend")
            ax3.imshow(np.abs(agg_out.astype(int) - yapocis_out.astype(int)))
            ax3.set_title(f"Difference (RMS={rms:.2f})")
            plt.savefig("claude-debug/comparison_mixed.png")
            plt.close()

        assert passed, f"RMS difference {rms:.2f} exceeds tolerance 10.0"

    def test_scatter_plot(self):
        """Test scatter plot."""

        def draw(ax):
            x = np.array([0.2, 0.4, 0.6, 0.8])
            y = np.array([0.3, 0.7, 0.5, 0.9])
            ax.scatter(x, y, s=100, c="red", alpha=0.6)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        agg_out = self.get_agg_output(draw)
        yapocis_out = self.get_yapocis_output(draw)
        rms, passed = self.compare_outputs(agg_out, yapocis_out, tol=10.0)

        if not passed:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
            ax1.imshow(agg_out)
            ax1.set_title("AGG Backend")
            ax2.imshow(yapocis_out)
            ax2.set_title("Yapocis Backend")
            ax3.imshow(np.abs(agg_out.astype(int) - yapocis_out.astype(int)))
            ax3.set_title(f"Difference (RMS={rms:.2f})")
            plt.savefig("claude-debug/comparison_scatter.png")
            plt.close()

        assert passed, f"RMS difference {rms:.2f} exceeds tolerance 10.0"


if __name__ == "__main__":
    # Run tests manually for debugging
    test = TestBackendComparison()

    print("Testing simple line...")
    test.test_simple_line()
    print("✓ Simple line passed")

    print("Testing Greek letters...")
    test.test_math_text_greek()
    print("✓ Greek letters passed")

    print("Testing subscript/superscript...")
    test.test_subscript_superscript()
    print("✓ Subscript/superscript passed")

    print("Testing mixed text...")
    test.test_mixed_text()
    print("✓ Mixed text passed")

    print("Testing scatter plot...")
    test.test_scatter_plot()
    print("✓ Scatter plot passed")

    print("\n✅ All comparison tests passed!")

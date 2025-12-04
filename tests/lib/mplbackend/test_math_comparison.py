#!/usr/bin/env python3
"""
Math text backend comparison - compare yapocis vs agg for MATH TEXT ONLY.

This focuses on the specific functionality we fixed: LaTeX-style math text rendering.
Regular text rendering and full plot support are separate concerns.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class TestMathTextComparison:
    """Compare yapocis backend math text output with agg backend."""

    def get_agg_output(self, draw_func):
        """Generate output using agg backend - math text only, no axes."""
        matplotlib.use("agg", force=True)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.axis("off")  # No axes to avoid regular text issues
        draw_func(ax)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        plt.close(fig)
        return np.frombuffer(buf, dtype=np.uint8).reshape((400, 400, 4))

    def get_yapocis_output(self, draw_func):
        """Generate output using yapocis backend - math text only, no axes."""
        matplotlib.use("module://yapocis.lib.mplbackend", force=True)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.axis("off")  # No axes to avoid regular text issues
        draw_func(ax)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        plt.close(fig)
        return np.frombuffer(buf, dtype=np.uint8).reshape((400, 400, 4))

    def compare_outputs(self, agg_out, yapocis_out, tol=20.0, save_prefix=None):
        """Compare two outputs with tolerance."""
        diff = agg_out.astype(float) - yapocis_out.astype(float)
        rms = np.sqrt(np.mean(diff**2))
        passed = rms <= tol

        if not passed or save_prefix:
            import matplotlib

            matplotlib.use("agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            ax1.imshow(agg_out)
            ax1.set_title("AGG Backend", fontsize=14)
            ax1.axis("off")
            ax2.imshow(yapocis_out)
            ax2.set_title("Yapocis Backend", fontsize=14)
            ax2.axis("off")
            diff_img = np.abs(agg_out.astype(int) - yapocis_out.astype(int))
            ax3.imshow(diff_img)
            ax3.set_title(f"Difference (RMS={rms:.2f}, tol={tol})", fontsize=14)
            ax3.axis("off")
            plt.tight_layout()
            filename = save_prefix or "comparison_failed"
            plt.savefig(f"claude-debug/{filename}.png", dpi=100, bbox_inches="tight")
            plt.close()

        return rms, passed

    def test_greek_letters(self):
        """Test Greek letter rendering matches agg backend."""

        def draw(ax):
            ax.text(0.5, 0.8, r"$\alpha$", fontsize=24, ha="center")
            ax.text(0.5, 0.5, r"$\beta$", fontsize=24, ha="center")
            ax.text(0.5, 0.2, r"$\gamma$", fontsize=24, ha="center")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        agg_out = self.get_agg_output(draw)
        yapocis_out = self.get_yapocis_output(draw)
        rms, passed = self.compare_outputs(
            agg_out, yapocis_out, tol=20.0, save_prefix="math_greek_comparison"
        )

        assert passed, f"Greek letter RMS difference {rms:.2f} exceeds tolerance 20.0"

    def test_subscripts(self):
        """Test subscript rendering matches agg backend."""

        def draw(ax):
            ax.text(0.5, 0.8, r"$x_2$", fontsize=24, ha="center")
            ax.text(0.5, 0.5, r"$H_2O$", fontsize=24, ha="center")
            ax.text(0.5, 0.2, r"$a_{sub}$", fontsize=24, ha="center")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        agg_out = self.get_agg_output(draw)
        yapocis_out = self.get_yapocis_output(draw)
        rms, passed = self.compare_outputs(
            agg_out, yapocis_out, tol=20.0, save_prefix="math_subscript_comparison"
        )

        assert passed, f"Subscript RMS difference {rms:.2f} exceeds tolerance 20.0"

    def test_superscripts(self):
        """Test superscript rendering matches agg backend."""

        def draw(ax):
            ax.text(0.5, 0.8, r"$x^2$", fontsize=24, ha="center")
            ax.text(0.5, 0.5, r"$E=mc^2$", fontsize=24, ha="center")
            ax.text(0.5, 0.2, r"$2^{10}$", fontsize=24, ha="center")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        agg_out = self.get_agg_output(draw)
        yapocis_out = self.get_yapocis_output(draw)
        rms, passed = self.compare_outputs(
            agg_out, yapocis_out, tol=20.0, save_prefix="math_superscript_comparison"
        )

        assert passed, f"Superscript RMS difference {rms:.2f} exceeds tolerance 20.0"

    def test_combined_scripts(self):
        """Test combined sub/superscripts."""

        def draw(ax):
            ax.text(0.5, 0.7, r"$x_2^3$", fontsize=28, ha="center")
            ax.text(0.5, 0.3, r"$a_i^j$", fontsize=28, ha="center")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        agg_out = self.get_agg_output(draw)
        yapocis_out = self.get_yapocis_output(draw)
        rms, passed = self.compare_outputs(
            agg_out, yapocis_out, tol=20.0, save_prefix="math_combined_comparison"
        )

        assert (
            passed
        ), f"Combined scripts RMS difference {rms:.2f} exceeds tolerance 20.0"

    def test_math_operators(self):
        """Test mathematical operators and symbols."""

        def draw(ax):
            ax.text(0.5, 0.8, r"$\alpha + \beta$", fontsize=20, ha="center")
            ax.text(0.5, 0.5, r"$\int_0^1 x dx$", fontsize=20, ha="center")
            ax.text(0.5, 0.2, r"$\sum_{i=1}^n i$", fontsize=20, ha="center")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        agg_out = self.get_agg_output(draw)
        yapocis_out = self.get_yapocis_output(draw)
        rms, passed = self.compare_outputs(
            agg_out, yapocis_out, tol=20.0, save_prefix="math_operators_comparison"
        )

        assert passed, f"Math operators RMS difference {rms:.2f} exceeds tolerance 20.0"


if __name__ == "__main__":
    # Run tests manually for debugging
    test = TestMathTextComparison()

    print("Testing Greek letters...")
    test.test_greek_letters()
    print("✓ Greek letters match agg backend")

    print("Testing subscripts...")
    test.test_subscripts()
    print("✓ Subscripts match agg backend")

    print("Testing superscripts...")
    test.test_superscripts()
    print("✓ Superscripts match agg backend")

    print("Testing combined sub/superscripts...")
    test.test_combined_scripts()
    print("✓ Combined scripts match agg backend")

    print("Testing math operators...")
    test.test_math_operators()
    print("✓ Math operators match agg backend")

    print("\n✅ All math text comparison tests passed!")
    print("\nComparison images saved to claude-debug/math_*_comparison.png")

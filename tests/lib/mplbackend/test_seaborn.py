#!/usr/bin/env python3
"""
Seaborn comparison tests - compare yapocis backend with matplotlib's agg backend
for seaborn plots.
"""

import matplotlib
import numpy as np
import pytest

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    pytest.skip(
        "seaborn not installed - skipping seaborn tests", allow_module_level=True
    )


class TestSeabornComparison:
    """Compare yapocis backend output with agg backend for seaborn plots."""

    def get_agg_output(self, draw_func, figsize=(6, 4)):
        """Generate output using agg backend."""
        import matplotlib.pyplot as plt_local

        matplotlib.use("agg", force=True)
        plt_local.clf()
        fig = draw_func()
        if fig is None:
            fig = plt_local.gcf()
        # Ensure consistent figure size
        fig.set_size_inches(figsize)
        fig.canvas.draw()
        # Get RGBA buffer
        buf = fig.canvas.buffer_rgba()
        width, height = fig.canvas.get_width_height()
        plt_local.close(fig)
        return np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))

    def get_yapocis_output(self, draw_func, figsize=(6, 4)):
        """Generate output using yapocis backend."""
        import matplotlib.pyplot as plt_local

        matplotlib.use("module://yapocis.lib.mplbackend", force=True)
        plt_local.clf()
        fig = draw_func()
        if fig is None:
            fig = plt_local.gcf()
        # Ensure consistent figure size
        fig.set_size_inches(figsize)
        fig.canvas.draw()
        # Get RGBA buffer
        buf = fig.canvas.buffer_rgba()
        width, height = fig.canvas.get_width_height()
        plt_local.close(fig)
        return np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))

    def compare_outputs(self, agg_out, yapocis_out, tol=10.0):
        """Compare two outputs with tolerance."""
        # Calculate RMS difference
        diff = agg_out.astype(float) - yapocis_out.astype(float)
        rms = np.sqrt(np.mean(diff**2))
        return rms, rms <= tol

    def test_lineplot(self):
        """Test seaborn lineplot."""
        # Load an example dataset with long-form data
        sns.set_theme(style="darkgrid")
        fmri = sns.load_dataset("fmri")

        # Create draw function that doesn't rely on external plt
        class DrawFunc:
            def __call__(self):
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
                sns.lineplot(
                    x="timepoint",
                    y="signal",
                    hue="region",
                    style="event",
                    data=fmri,
                    ax=ax,
                )
                return fig

        draw = DrawFunc()
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
            plt.savefig("seaborn_lineplot.png")
            plt.close()

        assert passed, f"RMS difference {rms:.2f} exceeds tolerance 10.0"

    def test_hexgrid(self):
        """Test seaborn hexgrid joint plot."""
        sns.set_theme(style="ticks")
        rs = np.random.RandomState(11)
        x = rs.gamma(2, size=1000)
        y = -0.5 * x + rs.normal(size=1000)

        class DrawFunc:
            def __call__(self):
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
                grid = sns.jointplot(x=x, y=y, kind="hex", color="#4CB391")
                return grid.figure

        draw = DrawFunc()
        agg_out = self.get_agg_output(draw, figsize=(6, 6))
        yapocis_out = self.get_yapocis_output(draw, figsize=(6, 6))
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
            plt.savefig("seaborn_hexgrid.png")
            plt.close()

        assert passed, f"RMS difference {rms:.2f} exceeds tolerance 10.0"

    def test_displot(self):
        """Test seaborn distribution plot (KDE)."""
        sns.set_theme(style="whitegrid")
        diamonds = sns.load_dataset("diamonds")

        class DrawFunc:
            def __call__(self):
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()

                grid = sns.displot(
                    data=diamonds,
                    x="carat",
                    hue="cut",
                    kind="kde",
                    height=6,
                    multiple="fill",
                    clip=(0, None),
                    palette="ch:rot=-.25,hue=1,light=.75",
                )
                return grid.figure

        draw = DrawFunc()
        agg_out = self.get_agg_output(draw, figsize=(6, 6))
        yapocis_out = self.get_yapocis_output(draw, figsize=(6, 6))
        rms, passed = self.compare_outputs(agg_out, yapocis_out, tol=10.0)

        if not passed:
            import matplotlib.pyplot as plt

            plt.clf()
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
            ax1.imshow(agg_out)
            ax1.set_title("AGG Backend")
            ax2.imshow(yapocis_out)
            ax2.set_title("Yapocis Backend")
            ax3.imshow(np.abs(agg_out.astype(int) - yapocis_out.astype(int)))
            ax3.set_title(f"Difference (RMS={rms:.2f})")
            plt.savefig("seaborn_displot.png")
            plt.close()

        assert passed, f"RMS difference {rms:.2f} exceeds tolerance 10.0"

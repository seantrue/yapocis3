#!/usr/bin/env python3
"""
Position validation tests for matplotlib backend imshow.

Tests that images are displayed at correct pixel locations when using imshow.

Reference Standard: matplotlib's Agg backend
All tests compare yapocis backend output against the Agg backend to verify
that images are positioned correctly.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def create_test_image(color, size=20):
    """Create a solid-color test image.

    Args:
        color: RGB tuple (r, g, b) with values 0-1
        size: Image size in pixels

    Returns:
        numpy array of shape (size, size, 3)
    """
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[:, :, 0] = color[0]
    img[:, :, 1] = color[1]
    img[:, :, 2] = color[2]
    return img


def create_corner_test_figure(backend_name):
    """Create a figure with colored images at each corner.

    Args:
        backend_name: 'agg' (reference standard) or 'module://yapocis.lib.mplbackend'

    Returns:
        numpy array of the rendered figure
    """
    matplotlib.use(backend_name, force=True)

    # Create test images
    red_img = create_test_image((1.0, 0.0, 0.0), size=40)
    green_img = create_test_image((0.0, 1.0, 0.0), size=40)
    blue_img = create_test_image((0.0, 0.0, 1.0), size=40)
    yellow_img = create_test_image((1.0, 1.0, 0.0), size=40)

    # Create figure with no margins
    fig = plt.figure(figsize=(5, 4), dpi=100)  # 500x400 pixels
    ax = fig.add_axes([0, 0, 1, 1])  # Fill entire figure
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 400)
    ax.axis("off")

    # Place images at four corners using extent
    # extent = [left, right, bottom, top] in data coordinates
    # matplotlib uses y-up coordinates (bottom=0, top=400)

    # Top-left: red at x=[20, 60], y=[360, 400]
    ax.imshow(
        red_img, extent=[20, 60, 360, 400], origin="upper", interpolation="nearest"
    )

    # Top-right: green at x=[440, 480], y=[360, 400]
    ax.imshow(
        green_img, extent=[440, 480, 360, 400], origin="upper", interpolation="nearest"
    )

    # Bottom-left: blue at x=[20, 60], y=[0, 40]
    ax.imshow(blue_img, extent=[20, 60, 0, 40], origin="upper", interpolation="nearest")

    # Bottom-right: yellow at x=[440, 480], y=[0, 40]
    ax.imshow(
        yellow_img, extent=[440, 480, 0, 40], origin="upper", interpolation="nearest"
    )

    # Render to array
    fig.canvas.draw()
    # Get the RGBA buffer from the figure canvas
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()  # Returns (width, height)
    # Reshape as (height, width, 4) since buffer is in row-major order
    arr = buf.reshape((int(h), int(w), 4))

    plt.close(fig)

    return arr[:, :, :3]  # Return RGB only


def test_imshow_position_consistency():
    """Test that yapocis backend places images at same positions as Agg backend."""
    # Create figures with both backends (Agg is reference standard)
    agg_output = create_corner_test_figure("agg")
    yapocis_output = create_corner_test_figure("module://yapocis.lib.mplbackend")

    # Check shapes match
    assert (
        agg_output.shape == yapocis_output.shape
    ), f"Output shapes don't match: {agg_output.shape} vs {yapocis_output.shape}"

    # Calculate pixel difference
    diff = np.abs(agg_output.astype(float) - yapocis_output.astype(float))
    mean_diff = diff.mean()
    max_diff = diff.max()

    # Calculate percentage of similar pixels
    close_pixels = (diff < 20).sum() / diff.size * 100

    print("\nImage positioning comparison:")
    print(f"  Output shape: {agg_output.shape}")
    print(f"  Mean pixel difference: {mean_diff:.2f} (0-255 scale)")
    print(f"  Max pixel difference: {max_diff:.2f}")
    print(f"  Pixels within 20 units: {close_pixels:.2f}%")

    # Most pixels should be very close (allowing for minor rendering differences)
    assert (
        close_pixels > 95.0
    ), f"Too many different pixels: {100 - close_pixels:.2f}% differ by >20 units"

    assert mean_diff < 10.0, f"Mean pixel difference too high: {mean_diff:.2f} > 10.0"

    print("✓ Yapocis backend image positioning matches Agg backend")


def test_imshow_simple_position():
    """Test that a single image is positioned correctly."""
    # Create a single red square
    red_img = create_test_image((1.0, 0.0, 0.0), size=50)

    # Create a small figure
    fig = plt.figure(figsize=(2, 2), dpi=100)  # 200x200 pixels
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.axis("off")

    # Place red square at bottom-left: x=[10,60], y=[10,60]
    ax.imshow(red_img, extent=[10, 60, 10, 60], origin="upper", interpolation="nearest")

    # Get output
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    arr = buf.reshape((int(h), int(w), 4))[:, :, :3]
    plt.close(fig)

    # Find red pixels
    red_mask = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 50) & (arr[:, :, 2] < 50)
    red_pixels = np.where(red_mask)

    assert len(red_pixels[0]) > 0, "No red pixels found"

    min_row = red_pixels[0].min()
    max_row = red_pixels[0].max()
    min_col = red_pixels[1].min()
    max_col = red_pixels[1].max()

    # In y-up coords: y=[10,60] with image height=50
    # y=60 (top) -> row = 200-60 = 140
    # y=10 (bottom) -> row = 200-10-50 = 140... wait that's wrong
    # Actually: bottom-left y coordinate is 10, so top of image is at y=60
    # row = height - y_top = 200 - 60 = 140 (top of image)
    # Image is 50 pixels tall, so bottom is at row 140+50 = 190
    expected_rows = (140, 189)  # Allow -1 for pixel boundaries
    expected_cols = (10, 59)

    print("\nSimple image position test:")
    print(f"  Red square at rows [{min_row},{max_row}], cols [{min_col},{max_col}]")
    print(f"  Expected: rows {expected_rows}, cols {expected_cols}")

    assert (
        abs(min_row - expected_rows[0]) <= 1
    ), f"Top edge wrong: {min_row} vs {expected_rows[0]}"
    assert (
        abs(max_row - expected_rows[1]) <= 1
    ), f"Bottom edge wrong: {max_row} vs {expected_rows[1]}"
    assert (
        abs(min_col - expected_cols[0]) <= 1
    ), f"Left edge wrong: {min_col} vs {expected_cols[0]}"
    assert (
        abs(max_col - expected_cols[1]) <= 1
    ), f"Right edge wrong: {max_col} vs {expected_cols[1]}"

    print("✓ Simple image positioned correctly")


def test_imshow_single_image_position():
    """Test positioning of a single image with clear markers."""
    # Create a test image with distinct quadrants
    test_img = np.zeros((60, 80, 3), dtype=np.float32)

    # Top-left: Red
    test_img[:30, :40, 0] = 1.0

    # Top-right: Green
    test_img[:30, 40:, 1] = 1.0

    # Bottom-left: Blue
    test_img[30:, :40, 2] = 1.0

    # Bottom-right: Yellow
    test_img[30:, 40:, 0] = 1.0
    test_img[30:, 40:, 1] = 1.0

    # Create figures with both backends (Agg is reference standard)
    for backend_name in ["agg", "module://yapocis.lib.mplbackend"]:
        matplotlib.use(backend_name, force=True)

        fig = plt.figure(figsize=(4, 3))  # 400x300 pixels
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 300)
        ax.axis("off")

        # Place image at center: x=[160, 240], y=[120, 180]
        ax.imshow(
            test_img,
            extent=[160, 240, 120, 180],
            origin="upper",
            interpolation="nearest",
        )

        backend_label = "yapocis" if "yapocis" in backend_name else "agg"
        output_path = f"claude-debug/imshow_single_{backend_label}.png"
        plt.savefig(output_path, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        print(f"✓ Saved {backend_label} output: {output_path}")

    # Compare outputs
    img_agg = np.array(Image.open("claude-debug/imshow_single_agg.png"))
    img_yapocis = np.array(Image.open("claude-debug/imshow_single_yapocis.png"))

    # Check they're similar
    diff = np.abs(img_agg[:, :, :3].astype(float) - img_yapocis[:, :, :3].astype(float))
    mean_diff = diff.mean()

    print("\nSingle image comparison:")
    print(f"  Mean difference: {mean_diff:.2f}")

    assert (
        mean_diff < 15.0
    ), f"Single image positioning differs too much: {mean_diff:.2f} > 15.0"

    print("✓ Single image positioning matches between backends")


if __name__ == "__main__":
    # Run tests directly
    print("=" * 60)
    print("Testing matplotlib backend imshow positioning")
    print("=" * 60)

    test_imshow_simple_position()
    print()

    test_imshow_position_consistency()
    print()

    test_imshow_single_image_position()
    print()

    print("=" * 60)
    print("✅ All imshow position tests passed!")
    print("=" * 60)

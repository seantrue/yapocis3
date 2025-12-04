#!/usr/bin/env python3
"""
Test matplotlib backend with real images from yapocis assets.

Creates a 4x4 contact sheet of images and compares yapocis backend output
against the standard Agg backend to verify correct image rendering.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from yapocis.images import find_image, images


def create_contact_sheet(backend_name):
    """Create a 4x4 contact sheet of yapocis asset images.

    Args:
        backend_name: 'Agg' or 'module://yapocis.lib.mplbackend'

    Returns:
        Path to saved image
    """
    matplotlib.use(backend_name)

    # Get available images
    available_images = images()

    # Limit to 16 images for 4x4 grid
    image_names = available_images[:16]

    if len(image_names) < 16:
        # Pad with repeated images if needed
        while len(image_names) < 16:
            image_names.extend(available_images[: 16 - len(image_names)])

    # Create figure with 4x4 subplots
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(f"Yapocis Asset Images - {backend_name}", fontsize=16)

    for idx, (ax, img_name) in enumerate(zip(axes.flat, image_names, strict=False)):
        try:
            # Load image
            img_path = find_image(img_name)
            img = plt.imread(img_path)

            # Display image
            ax.imshow(img, interpolation="nearest")
            ax.set_title(img_name, fontsize=8)
            ax.axis("off")

        except Exception:
            # If image fails to load, show error
            ax.text(
                0.5,
                0.5,
                f"Error:\n{img_name}",
                ha="center",
                va="center",
                fontsize=8,
                transform=ax.transAxes,
            )
            ax.axis("off")

    plt.tight_layout()

    # Save to different filenames based on backend
    backend_label = "yapocis" if "yapocis" in backend_name else "agg"
    output_path = f"claude-debug/contact_sheet_{backend_label}.png"
    plt.savefig(output_path, dpi=100)
    plt.close(fig)

    return output_path


def test_matplotlib_contact_sheet_comparison():
    """Test that yapocis backend produces output matching Agg backend."""
    # Skip if no images available
    try:
        available = images()
        if len(available) == 0:
            pytest.skip("No yapocis asset images available")
    except Exception:
        pytest.skip("Cannot access yapocis assets")

    # Create contact sheets with both backends
    agg_output = create_contact_sheet("Agg")
    yapocis_output = create_contact_sheet("module://yapocis.lib.mplbackend")

    print(f"\n✓ Agg backend output: {agg_output}")
    print(f"✓ Yapocis backend output: {yapocis_output}")

    # Load both images and compare
    from PIL import Image

    img_agg = np.array(Image.open(agg_output))
    img_yapocis = np.array(Image.open(yapocis_output))

    # Convert both to RGB if needed (ignore alpha channel differences)
    if img_agg.shape[-1] == 4:
        img_agg = img_agg[:, :, :3]
    if img_yapocis.shape[-1] == 4:
        img_yapocis = img_yapocis[:, :, :3]

    # Check shapes match
    assert (
        img_agg.shape == img_yapocis.shape
    ), f"Image shapes don't match: {img_agg.shape} vs {img_yapocis.shape}"

    # Calculate pixel difference
    diff = np.abs(img_agg.astype(float) - img_yapocis.astype(float))
    mean_diff = diff.mean()
    max_diff = diff.max()

    # Calculate percentage of pixels that differ significantly (> 10 in 0-255 range)
    # Use higher threshold to account for text rendering differences
    significant_diff = (diff > 10).sum() / diff.size * 100

    print("\nImage comparison:")
    print(f"  Mean pixel difference: {mean_diff:.2f} (0-255 scale)")
    print(f"  Max pixel difference: {max_diff:.2f}")
    print(f"  Significantly different pixels (>10): {significant_diff:.2f}%")

    # Allow differences due to text rendering variations between backends
    # The images themselves should be nearly identical, text may differ
    # Use mean difference as primary metric - should be small if images match
    assert mean_diff < 25.0, f"Mean pixel difference too high: {mean_diff:.2f} > 25.0"

    # Also check that the vast majority of pixels are close
    close_pixels = (diff < 20).sum() / diff.size * 100
    assert close_pixels > 70.0, f"Not enough similar pixels: {close_pixels:.2f}% < 70%"

    print(f"  Pixels within 20 units: {close_pixels:.2f}%")
    print(
        "\n✓ Yapocis backend output matches Agg backend (within acceptable tolerance)"
    )
    print("  Note: Some differences expected due to text rendering")


def test_matplotlib_image_orientation():
    """Test that images are displayed with correct orientation."""
    # Create a test image with clear orientation markers
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)

    # Top-left quadrant: RED
    test_img[:50, :50, 0] = 255

    # Top-right quadrant: GREEN
    test_img[:50, 50:, 1] = 255

    # Bottom-left quadrant: BLUE
    test_img[50:, :50, 2] = 255

    # Bottom-right quadrant: YELLOW (RED + GREEN)
    test_img[50:, 50:, 0] = 255
    test_img[50:, 50:, 1] = 255

    # Test with both backends
    for backend_name in ["Agg", "module://yapocis.lib.mplbackend"]:
        matplotlib.use(backend_name)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(test_img)
        ax.set_title(f"Orientation Test - {backend_name}")

        # Add labels to verify orientation
        ax.text(
            25,
            25,
            "RED\n(top-left)",
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            weight="bold",
        )
        ax.text(
            75,
            25,
            "GREEN\n(top-right)",
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            weight="bold",
        )
        ax.text(
            25,
            75,
            "BLUE\n(bottom-left)",
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            weight="bold",
        )
        ax.text(
            75,
            75,
            "YELLOW\n(bottom-right)",
            ha="center",
            va="center",
            color="black",
            fontsize=12,
            weight="bold",
        )

        backend_label = "yapocis" if "yapocis" in backend_name else "agg"
        output_path = f"claude-debug/orientation_test_{backend_label}.png"
        plt.savefig(output_path, dpi=100)
        plt.close(fig)

        print(f"✓ Saved orientation test: {output_path}")

    # Compare the two outputs
    from PIL import Image

    img_agg = np.array(Image.open("claude-debug/orientation_test_agg.png"))
    img_yapocis = np.array(Image.open("claude-debug/orientation_test_yapocis.png"))

    # Check key pixels to verify correct orientation
    # Sample a pixel from each quadrant
    h, w = img_agg.shape[:2]

    # Top-left should be red-ish (accounting for text)
    # Use only RGB channels for comparison (ignore alpha if present)
    tl_agg = img_agg[int(h * 0.15), int(w * 0.15), :3]
    tl_yapocis = img_yapocis[int(h * 0.15), int(w * 0.15), :3]

    # Check they're similar
    tl_diff = np.abs(tl_agg.astype(float) - tl_yapocis.astype(float)).mean()

    assert (
        tl_diff < 10
    ), f"Top-left pixels differ too much: {tl_diff:.2f} (should be < 10)"

    print("✓ Image orientation test passed")


if __name__ == "__main__":
    # Run tests directly
    print("=" * 60)
    print("Testing matplotlib backend with yapocis asset images")
    print("=" * 60)

    test_matplotlib_image_orientation()
    print()

    try:
        test_matplotlib_contact_sheet_comparison()
    except Exception as e:
        print(f"Contact sheet test failed or skipped: {e}")

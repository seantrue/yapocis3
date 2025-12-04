"""
Tests for yapocis matplotlib backend.

Tests math text rendering, subscripts, superscripts, and LaTeX-style formatting.
"""

import matplotlib
import numpy as np

matplotlib.use("module://yapocis.lib.mplbackend")
import matplotlib.pyplot as plt


def test_backend_loaded():
    """Sanity check: verify yapocis backend is loaded."""
    backend = matplotlib.get_backend()
    assert "yapocis" in backend.lower() or "template" in backend.lower()


def test_basic_text_rendering():
    """Test that basic plain text rendering works."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.text(0.5, 0.5, "Hello World", ha="center", va="center", fontsize=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Should not raise any errors
    fig.canvas.draw()
    plt.close(fig)


def test_math_text_greek_letters():
    """Test Greek letter rendering in math mode."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Test various Greek letters
    greek_letters = [
        (r"$\alpha$", "alpha"),
        (r"$\beta$", "beta"),
        (r"$\gamma$", "gamma"),
        (r"$\Delta$", "Delta"),
        (r"$\pi$", "pi"),
        (r"$\Sigma$", "Sigma"),
    ]

    for i, (math_text, name) in enumerate(greek_letters):
        y = 0.8 - i * 0.15
        ax.text(0.3, y, math_text, fontsize=16)
        ax.text(0.6, y, name, fontsize=12)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.canvas.draw()

    # Get the rendered array
    renderer = fig.canvas.get_renderer()
    buf = renderer.buffer_rgba()
    array = np.frombuffer(buf, dtype=np.uint8)
    array = array.reshape(renderer.height, renderer.width, 4)

    # Check that something was rendered (not all white)
    assert not np.all(array[:, :, :3] == 255), "Canvas should not be completely white"

    plt.close(fig)


def test_subscript():
    """Test subscript rendering."""
    fig, ax = plt.subplots(figsize=(6, 4))

    subscript_examples = [
        r"$x_2$",
        r"$H_2O$",
        r"$y_i$",
        r"$a_{sub}$",
    ]

    for i, text in enumerate(subscript_examples):
        y = 0.8 - i * 0.2
        ax.text(0.5, y, text, fontsize=20, ha="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.canvas.draw()

    # Get the rendered array
    renderer = fig.canvas.get_renderer()
    buf = renderer.buffer_rgba()
    array = np.frombuffer(buf, dtype=np.uint8)
    array = array.reshape(renderer.height, renderer.width, 4)

    # Check that something was rendered
    assert not np.all(array[:, :, :3] == 255), "Canvas should not be completely white"

    plt.close(fig)


def test_superscript():
    """Test superscript rendering."""
    fig, ax = plt.subplots(figsize=(6, 4))

    superscript_examples = [
        r"$x^2$",
        r"$E=mc^2$",
        r"$2^{10}$",
        r"$a^{super}$",
    ]

    for i, text in enumerate(superscript_examples):
        y = 0.8 - i * 0.2
        ax.text(0.5, y, text, fontsize=20, ha="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.canvas.draw()

    # Get the rendered array
    renderer = fig.canvas.get_renderer()
    buf = renderer.buffer_rgba()
    array = np.frombuffer(buf, dtype=np.uint8)
    array = array.reshape(renderer.height, renderer.width, 4)

    # Check that something was rendered
    assert not np.all(array[:, :, :3] == 255), "Canvas should not be completely white"

    plt.close(fig)


def test_combined_sub_superscript():
    """Test combined subscript and superscript."""
    fig, ax = plt.subplots(figsize=(6, 4))

    combined_examples = [
        r"$x_2^3$",
        r"$a_i^j$",
        r"$\sum_{i=1}^{n}$",
    ]

    for i, text in enumerate(combined_examples):
        y = 0.7 - i * 0.3
        ax.text(0.5, y, text, fontsize=24, ha="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.canvas.draw()

    # Get the rendered array
    renderer = fig.canvas.get_renderer()
    buf = renderer.buffer_rgba()
    array = np.frombuffer(buf, dtype=np.uint8)
    array = array.reshape(renderer.height, renderer.width, 4)

    # Check that something was rendered
    assert not np.all(array[:, :, :3] == 255), "Canvas should not be completely white"

    plt.close(fig)


def test_math_text_vs_plain_text():
    """Test that math text differs from plain text."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plain text (should render literal "$alpha$")
    ax.text(0.5, 0.7, "$alpha$", fontsize=20, ha="center")

    # Math text (should render Greek alpha symbol)
    ax.text(0.5, 0.3, r"$\alpha$", fontsize=20, ha="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.canvas.draw()

    # Get the rendered array
    renderer = fig.canvas.get_renderer()
    buf = renderer.buffer_rgba()
    array = np.frombuffer(buf, dtype=np.uint8)
    array = array.reshape(renderer.height, renderer.width, 4)

    # Extract upper and lower halves
    height = array.shape[0]
    upper_half = array[: height // 2, :, :]
    lower_half = array[height // 2 :, :, :]

    # They should be different (math text renders differently than plain)
    # This test will likely fail initially since math text isn't working
    difference = np.sum(np.abs(upper_half.astype(int) - lower_half.astype(int)))
    assert difference > 0, "Math text should render differently than plain text"

    plt.close(fig)


def test_fraction():
    """Test fraction rendering."""
    fig, ax = plt.subplots(figsize=(6, 4))

    fraction_examples = [
        r"$\frac{1}{2}$",
        r"$\frac{a}{b}$",
        r"$\frac{dy}{dx}$",
    ]

    for i, text in enumerate(fraction_examples):
        y = 0.7 - i * 0.3
        ax.text(0.5, y, text, fontsize=24, ha="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.canvas.draw()

    # Get the rendered array
    renderer = fig.canvas.get_renderer()
    buf = renderer.buffer_rgba()
    array = np.frombuffer(buf, dtype=np.uint8)
    array = array.reshape(renderer.height, renderer.width, 4)

    # Check that something was rendered
    assert not np.all(array[:, :, :3] == 255), "Canvas should not be completely white"

    plt.close(fig)


def test_ismath_parameter():
    """Test that ismath parameter is actually used."""
    fig, ax = plt.subplots(figsize=(4, 4))

    # This should trigger math text rendering
    text_obj = ax.text(0.5, 0.5, r"$\alpha$", fontsize=20, ha="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Force a draw to trigger the renderer
    fig.canvas.draw()

    # Get renderer and check if ismath would be set
    renderer = fig.canvas.get_renderer()

    # Check that we can at least call get_text_width_height_descent
    # with ismath parameter
    from matplotlib.font_manager import FontProperties

    prop = FontProperties(size=20)

    # This should handle ismath=True without crashing
    try:
        w, h, d = renderer.get_text_width_height_descent(r"$\alpha$", prop, ismath=True)
        assert w > 0 and h > 0, "Text should have non-zero dimensions"
    except NotImplementedError as e:
        # Acceptable if TeX mode explicitly not supported
        if "TeX" not in str(e):
            raise

    plt.close(fig)


def test_matplotlib_markers_example():
    """Test the matplotlib_markers.py example scenario."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # From examples/matplotlib_markers.py
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    ax.plot(x, y, "o-", label=r"$\alpha$ data")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\sin(x)$")
    ax.set_title(r"Plot with $\alpha$ in legend")
    ax.legend()
    ax.grid(True)

    # Should not raise any errors
    fig.canvas.draw()

    # Get the rendered array
    renderer = fig.canvas.get_renderer()
    buf = renderer.buffer_rgba()
    array = np.frombuffer(buf, dtype=np.uint8)
    array = array.reshape(renderer.height, renderer.width, 4)

    # Check that something was rendered
    assert not np.all(array[:, :, :3] == 255), "Canvas should not be completely white"

    plt.close(fig)


if __name__ == "__main__":
    # Run tests manually for debugging
    print("Testing matplotlib backend...")

    print("\n1. Testing backend loaded...")
    test_backend_loaded()
    print("✓ Backend loaded")

    print("\n2. Testing basic text rendering...")
    test_basic_text_rendering()
    print("✓ Basic text rendering works")

    print("\n3. Testing Greek letters (math text)...")
    try:
        test_math_text_greek_letters()
        print("✓ Greek letters work")
    except AssertionError as e:
        print(f"✗ Greek letters failed: {e}")

    print("\n4. Testing subscripts...")
    try:
        test_subscript()
        print("✓ Subscripts work")
    except AssertionError as e:
        print(f"✗ Subscripts failed: {e}")

    print("\n5. Testing superscripts...")
    try:
        test_superscript()
        print("✓ Superscripts work")
    except AssertionError as e:
        print(f"✗ Superscripts failed: {e}")

    print("\n6. Testing ismath parameter...")
    try:
        test_ismath_parameter()
        print("✓ ismath parameter handled")
    except Exception as e:
        print(f"✗ ismath parameter failed: {e}")

    print("\nAll tests completed!")

#!/usr/bin/env python3
"""
Comprehensive tests for convolve_kernels module and convolves interface.

Tests kernel generation, mathematical properties, and actual convolution operations.
"""
import numpy as np
import pytest

from yapocis.lib.convolve_kernels import (
    KERNEL_GENERATORS,
    generate_bartlett,
    generate_binomial,
    generate_blackman,
    generate_box,
    generate_convs_from_spec,
    generate_deriv1,
    generate_deriv2,
    generate_dog,
    generate_hamming,
    generate_hann,
    generate_kernel,
    generate_prewitt,
    generate_sharpen,
    generate_sobel,
    generate_triangle,
    generate_unsharp,
)


class TestKernelGenerators:
    """Test individual kernel generator functions."""

    def test_hamming_window(self):
        """Test Hamming window generation."""
        # Size 5 Hamming window
        kernel = generate_hamming(5)
        assert len(kernel) == 5
        # Hamming window is symmetric
        assert np.allclose(kernel[0], kernel[4])
        assert np.allclose(kernel[1], kernel[3])
        # Normalized to sum to 1
        assert np.sum(kernel) == pytest.approx(1.0)
        # Center value should be highest
        assert kernel[2] > kernel[1] > kernel[0]

    def test_hann_window(self):
        """Test Hann window generation."""
        kernel = generate_hann(5)
        assert len(kernel) == 5
        # Hann window is symmetric
        assert np.allclose(kernel[0], kernel[4])
        assert np.allclose(kernel[1], kernel[3])
        # Normalized to sum to 1
        assert np.sum(kernel) == pytest.approx(1.0)
        # Endpoints are near zero for Hann (but normalized so not exactly zero)
        assert kernel[0] < kernel[1]
        assert kernel[4] < kernel[3]

    def test_blackman_window(self):
        """Test Blackman window generation."""
        kernel = generate_blackman(7)
        assert len(kernel) == 7
        # Blackman window is symmetric
        assert np.allclose(kernel[0], kernel[6])
        assert np.allclose(kernel[1], kernel[5])
        assert np.allclose(kernel[2], kernel[4])
        # Normalized to sum to 1
        assert np.sum(kernel) == pytest.approx(1.0)

    def test_bartlett_window(self):
        """Test Bartlett (triangular) window generation."""
        kernel = generate_bartlett(5)
        assert len(kernel) == 5
        # Bartlett is triangular: increases then decreases
        assert kernel[0] < kernel[1] < kernel[2]
        assert kernel[2] > kernel[3] > kernel[4]
        # Symmetric
        assert np.allclose(kernel[0], kernel[4])
        # Normalized to sum to 1
        assert np.sum(kernel) == pytest.approx(1.0)

    def test_box_filter(self):
        """Test box (uniform) filter generation."""
        kernel = generate_box(7)
        assert len(kernel) == 7
        # All values equal
        assert np.allclose(kernel, 1.0 / 7)
        # Sums to 1
        assert np.sum(kernel) == pytest.approx(1.0)

    def test_triangle_filter(self):
        """Test triangular filter generation."""
        kernel = generate_triangle(5)
        assert len(kernel) == 5
        # Normalized to sum to 1
        assert np.sum(kernel) == pytest.approx(1.0)
        # Symmetric
        assert np.allclose(kernel[0], kernel[4])

    def test_binomial_filter(self):
        """Test binomial filter generation."""
        # Size 3: [1, 2, 1] / 4
        kernel3 = generate_binomial(3)
        assert len(kernel3) == 3
        assert np.allclose(kernel3, [0.25, 0.5, 0.25])
        assert np.sum(kernel3) == pytest.approx(1.0)

        # Size 5: [1, 4, 6, 4, 1] / 16
        kernel5 = generate_binomial(5)
        assert len(kernel5) == 5
        assert np.allclose(kernel5, [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16])
        assert np.sum(kernel5) == pytest.approx(1.0)

    def test_deriv1_kernels(self):
        """Test first derivative kernels."""
        # Size 3: central difference
        kernel3 = generate_deriv1(3)
        assert len(kernel3) == 3
        assert np.allclose(kernel3, [-0.5, 0.0, 0.5])
        # Sum should be zero (derivative)
        assert np.sum(kernel3) == pytest.approx(0.0)

        # Size 5: 4th order accurate
        kernel5 = generate_deriv1(5)
        assert len(kernel5) == 5
        # Anti-symmetric
        assert np.allclose(kernel5[0], -kernel5[4])
        assert np.allclose(kernel5[1], -kernel5[3])
        assert kernel5[2] == pytest.approx(0.0)
        assert np.sum(kernel5) == pytest.approx(0.0, abs=1e-6)

    def test_deriv2_kernels(self):
        """Test second derivative kernels."""
        # Size 3: discrete Laplacian
        kernel3 = generate_deriv2(3)
        assert len(kernel3) == 3
        assert np.allclose(kernel3, [1.0, -2.0, 1.0])
        # Sum should be zero (second derivative)
        assert np.sum(kernel3) == pytest.approx(0.0)

        # Size 5: higher order
        kernel5 = generate_deriv2(5)
        assert len(kernel5) == 5
        # Symmetric
        assert np.allclose(kernel5[0], kernel5[4])
        assert np.allclose(kernel5[1], kernel5[3])
        assert np.sum(kernel5) == pytest.approx(0.0, abs=1e-6)

    def test_sobel_kernel(self):
        """Test Sobel derivative kernel."""
        kernel = generate_sobel(3)
        assert len(kernel) == 3
        assert np.allclose(kernel, [-1.0, 0.0, 1.0])
        assert np.sum(kernel) == pytest.approx(0.0)

    def test_prewitt_kernel(self):
        """Test Prewitt derivative kernel."""
        kernel = generate_prewitt(3)
        assert len(kernel) == 3
        assert np.allclose(kernel, [-1.0, 0.0, 1.0])
        assert np.sum(kernel) == pytest.approx(0.0)

    def test_sharpen_kernels(self):
        """Test sharpening kernels."""
        kernel3 = generate_sharpen(3)
        assert len(kernel3) == 3
        # Center should be positive, neighbors negative
        assert kernel3[1] > 0
        assert kernel3[0] < 0 and kernel3[2] < 0

    def test_unsharp_kernels(self):
        """Test unsharp masking kernels."""
        kernel3 = generate_unsharp(3)
        assert len(kernel3) == 3
        # Center should be greater than 1
        assert kernel3[1] > 1.0
        # Neighbors should be negative
        assert kernel3[0] < 0 and kernel3[2] < 0

    def test_dog_kernels(self):
        """Test Difference of Gaussians kernels."""
        kernel5 = generate_dog(5)
        assert len(kernel5) == 5
        # Center should be positive, neighbors negative
        assert kernel5[2] > 0
        assert all(kernel5[i] < 0 for i in [0, 1, 3, 4])
        # Sum should be close to zero
        assert np.sum(kernel5) == pytest.approx(0.0, abs=1e-6)


class TestKernelRegistry:
    """Test the kernel generator registry."""

    def test_all_generators_registered(self):
        """Test that all expected generators are registered."""
        expected_types = [
            "hamming",
            "hann",
            "blackman",
            "bartlett",
            "box",
            "triangle",
            "binomial",
            "deriv1",
            "deriv2",
            "sobel",
            "prewitt",
            "sharpen",
            "unsharp",
            "dog",
        ]
        for kernel_type in expected_types:
            assert kernel_type in KERNEL_GENERATORS

    def test_generate_kernel_wrapper(self):
        """Test the generate_kernel wrapper function."""
        # Test valid kernel
        kernel = generate_kernel("hamming", 9)
        assert len(kernel) == 9
        assert kernel.dtype == np.float32

        # Test invalid kernel type
        with pytest.raises(ValueError, match="Unknown kernel type"):
            generate_kernel("invalid_type", 5)


class TestKernelProperties:
    """Test mathematical properties of generated kernels."""

    @pytest.mark.parametrize(
        "kernel_type,size",
        [
            ("box", 5),
            ("triangle", 7),
            ("binomial", 9),
            ("hamming", 11),
            ("hann", 13),
            ("blackman", 15),
            ("bartlett", 9),
        ],
    )
    def test_smoothing_kernels_sum_to_one(self, kernel_type, size):
        """Test that smoothing kernels sum to 1."""
        kernel = generate_kernel(kernel_type, size)
        assert np.sum(kernel) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize(
        "kernel_type,size",
        [
            ("deriv1", 3),
            ("deriv1", 5),
            ("deriv2", 3),
            ("deriv2", 5),
            ("sobel", 3),
            ("prewitt", 3),
            ("dog", 7),
        ],
    )
    def test_derivative_kernels_sum_to_zero(self, kernel_type, size):
        """Test that derivative kernels sum to 0."""
        kernel = generate_kernel(kernel_type, size)
        assert np.sum(kernel) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize(
        "kernel_type,size",
        [
            ("hamming", 9),
            ("hann", 9),
            ("blackman", 9),
            ("bartlett", 9),
            ("box", 9),
            ("triangle", 9),
            ("binomial", 9),
            ("deriv2", 9),
        ],
    )
    def test_kernel_symmetry(self, kernel_type, size):
        """Test that symmetric kernels are actually symmetric."""
        kernel = generate_kernel(kernel_type, size)
        # Check if symmetric (equal to reversed)
        assert np.allclose(kernel, kernel[::-1])

    @pytest.mark.parametrize(
        "kernel_type,size",
        [
            ("deriv1", 3),
            ("deriv1", 5),
            ("sobel", 3),
            ("prewitt", 3),
        ],
    )
    def test_first_derivative_antisymmetry(self, kernel_type, size):
        """Test that first derivative kernels are antisymmetric."""
        kernel = generate_kernel(kernel_type, size)
        # Check if antisymmetric: k[i] = -k[n-1-i]
        assert np.allclose(kernel, -kernel[::-1])


class TestConvsGeneration:
    """Test generating convs list from YAML specification."""

    def test_generate_convs_from_simple_spec(self):
        """Test generating convs from a simple specification."""
        spec = {
            "smoothing": [{"type": "box", "sizes": [3, 5, 7]}],
            "derivatives": [{"type": "deriv1", "sizes": [3, 5]}],
        }

        convs = generate_convs_from_spec(spec)

        # Should have 5 kernels total (3 box + 2 deriv1)
        assert len(convs) == 5

        # Check names
        names = [name for name, _ in convs]
        assert "box3" in names
        assert "box5" in names
        assert "box7" in names
        assert "deriv13" in names
        assert "deriv15" in names

        # Check kernel values
        for name, weights in convs:
            assert isinstance(weights, np.ndarray)
            assert weights.dtype == np.float32
            if "box" in name:
                size = int(name.replace("box", ""))
                assert len(weights) == size

    def test_generate_convs_with_custom_kernels(self):
        """Test generating convs with custom kernel specifications."""
        spec = {
            "custom": [
                {"name": "test3", "weights": [-1.0, 0.0, 1.0], "normalize": False},
                {
                    "name": "test5_norm",
                    "weights": [1.0, 2.0, 3.0, 2.0, 1.0],
                    "normalize": True,
                },
            ]
        }

        convs = generate_convs_from_spec(spec)

        # Should have 2 custom kernels
        assert len(convs) == 2

        # Check first custom kernel
        name1, weights1 = convs[0]
        assert name1 == "test3"
        assert np.allclose(weights1, [-1.0, 0.0, 1.0])

        # Check second custom kernel (normalized)
        name2, weights2 = convs[1]
        assert name2 == "test5_norm"
        # Should be normalized
        assert np.sum(weights2) == pytest.approx(1.0, abs=1e-6)

    def test_generate_convs_from_full_spec(self):
        """Test generating convs from a full specification with all categories."""
        spec = {
            "windows": [{"type": "hamming", "sizes": [5, 9]}],
            "smoothing": [{"type": "box", "sizes": [3, 5]}],
            "derivatives": [{"type": "deriv1", "sizes": [3]}],
            "highpass": [{"type": "sharpen", "sizes": [3]}],
            "bandpass": [{"type": "dog", "sizes": [5]}],
            "custom": [{"name": "edge3", "weights": [-1.0, 0.0, 1.0]}],
        }

        convs = generate_convs_from_spec(spec)

        # Should have 8 kernels total
        assert len(convs) == 8

        names = [name for name, _ in convs]
        assert "hamming5" in names
        assert "hamming9" in names
        assert "box3" in names
        assert "box5" in names
        assert "deriv13" in names
        assert "sharpen3" in names
        assert "dog5" in names
        assert "edge3" in names


class TestKernelApplications:
    """Test applying kernels to actual signal data."""

    def test_box_filter_on_signal(self):
        """Test box filter smoothing on a noisy signal."""
        # Create a simple signal with noise
        signal = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        kernel = generate_kernel("box", 3)

        # Apply convolution (valid mode to avoid edge effects)
        result = np.convolve(signal, kernel, mode="valid")

        # Result should be smoother
        assert len(result) == len(signal) - len(kernel) + 1
        # Middle values should be averages
        assert result[2] == pytest.approx((3.0 + 2.0 + 1.0) / 3.0, abs=1e-5)

    def test_derivative_on_ramp(self):
        """Test first derivative on a linear ramp."""
        # Linear ramp: derivative should be constant
        ramp = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        kernel = generate_kernel("deriv1", 3)

        # Apply correlation (not convolution) for derivative
        # Convolution reverses the kernel, correlation doesn't
        result = np.correlate(ramp, kernel, mode="valid")

        # Derivative of linear function is constant
        # For [-0.5, 0, 0.5] on [0,1,2,3,4,5], middle values should all be 1.0
        assert np.allclose(result, 1.0, atol=1e-6)

    def test_second_derivative_on_quadratic(self):
        """Test second derivative on a quadratic function."""
        # Quadratic: x^2, second derivative is constant (2)
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        quadratic = x * x

        kernel = generate_kernel("deriv2", 3)

        # Apply convolution
        result = np.convolve(quadratic, kernel, mode="valid")

        # Second derivative of x^2 is 2
        assert np.allclose(result, 2.0, atol=1e-5)

    def test_hamming_window_smoothing(self):
        """Test Hamming window for smoothing."""
        # Create impulse signal
        impulse = np.zeros(20, dtype=np.float32)
        impulse[10] = 1.0

        kernel = generate_kernel("hamming", 9)

        # Apply convolution
        result = np.convolve(impulse, kernel, mode="same")

        # Result should have Hamming window shape centered at position 10
        # Since kernel sums to 1, the impulse response reproduces the kernel shape
        assert result[10] == pytest.approx(np.max(kernel), abs=1e-5)  # Peak at center
        assert result[10] > result[9]  # Peak is maximum
        assert result[10] > result[11]
        assert result[5] < 0.01  # Low values far from center
        assert result[15] < 0.01

    def test_binomial_approximates_gaussian(self):
        """Test that binomial filter approximates Gaussian blur."""
        # Create impulse
        impulse = np.zeros(20, dtype=np.float32)
        impulse[10] = 1.0

        kernel = generate_kernel("binomial", 9)

        # Apply convolution
        result = np.convolve(impulse, kernel, mode="same")

        # Result should be smooth and symmetric
        assert result[10] == np.max(result)  # Peak at center
        # Symmetric around center
        for i in range(1, 5):
            assert np.abs(result[10 - i] - result[10 + i]) < 1e-5


class TestInterfaceIntegration:
    """Test integration with YAML specification and interface rendering."""

    def test_yaml_spec_completeness(self):
        """Test that YAML spec can generate expected number of kernels."""
        from yapocis.core.interface_context import load_interface_context

        context = load_interface_context("convolves")
        assert context is not None

        # Generate convs
        convs = generate_convs_from_spec(context)

        # Should have generated many kernels
        assert len(convs) > 50  # Based on our spec

        # All should be valid
        for name, weights in convs:
            assert isinstance(name, str)
            assert isinstance(weights, np.ndarray)
            assert weights.dtype == np.float32
            assert len(weights) > 0

    def test_interface_rendering(self):
        """Test that interface can be rendered with generated kernels."""
        from mako.template import Template

        from yapocis.core.interface import get_interface, load_interface
        from yapocis.core.interface_context import load_interface_context

        # Load and render interface
        interface_text = load_interface("convolves.interface")
        context = load_interface_context("convolves")
        convs = generate_convs_from_spec(context)
        context["convs"] = convs

        template = Template(interface_text)
        rendered = template.render(**context)

        # Parse rendered interface
        interface = get_interface(rendered)

        # Should have all kernels
        assert interface.name == "convolves"
        assert len(interface.kernels) == len(convs)

        # Check a few specific kernels
        kernel_names = [k.name for k in interface.kernels]
        assert "hamming5" in kernel_names
        assert "box3" in kernel_names
        assert "deriv13" in kernel_names
        assert "edge3" in kernel_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

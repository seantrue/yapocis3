#!/usr/bin/env python3
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.
"""
1D convolution kernel generation for signal processing.

Generates various windowing, filtering, and derivative kernels
for use with the convolves interface.
"""
import numpy as np


def generate_hamming(size):
    """Generate Hamming window of given size, normalized to sum to 1."""
    n = np.arange(size)
    weights = 0.54 - 0.46 * np.cos(2 * np.pi * n / (size - 1))
    return weights / np.sum(weights)


def generate_hann(size):
    """Generate Hann window of given size, normalized to sum to 1."""
    n = np.arange(size)
    weights = 0.5 * (1 - np.cos(2 * np.pi * n / (size - 1)))
    return weights / np.sum(weights)


def generate_blackman(size):
    """Generate Blackman window of given size, normalized to sum to 1."""
    n = np.arange(size)
    a0 = 0.42
    a1 = 0.5
    a2 = 0.08
    weights = (
        a0
        - a1 * np.cos(2 * np.pi * n / (size - 1))
        + a2 * np.cos(4 * np.pi * n / (size - 1))
    )
    return weights / np.sum(weights)


def generate_bartlett(size):
    """Generate Bartlett (triangular) window of given size, normalized to sum to 1."""
    n = np.arange(size)
    weights = 1 - np.abs((n - (size - 1) / 2) / ((size - 1) / 2))
    return weights / np.sum(weights)


def generate_box(size):
    """Generate box (uniform) filter of given size."""
    return np.ones(size) / size


def generate_triangle(size):
    """Generate triangular filter of given size."""
    weights = generate_bartlett(size)
    return weights / np.sum(weights)


def generate_binomial(size):
    """Generate binomial filter of given size."""
    if size == 3:
        weights = np.array([1.0, 2.0, 1.0])
    elif size == 5:
        weights = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
    elif size == 7:
        weights = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
    elif size == 9:
        weights = np.array([1.0, 8.0, 28.0, 56.0, 70.0, 56.0, 28.0, 8.0, 1.0])
    else:
        # For other sizes, use Pascal's triangle
        weights = np.ones(1)
        for _ in range(size - 1):
            weights = np.convolve(weights, [1, 1])
    return weights / np.sum(weights)


def generate_deriv1(size):
    """Generate first derivative (central difference) kernel."""
    if size == 3:
        return np.array([-0.5, 0.0, 0.5])
    elif size == 5:
        return np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0
    elif size == 7:
        return np.array([-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0]) / 60.0
    else:
        # Default to size 3
        return np.array([-0.5, 0.0, 0.5])


def generate_deriv2(size):
    """Generate second derivative (discrete Laplacian) kernel."""
    if size == 3:
        return np.array([1.0, -2.0, 1.0])
    elif size == 5:
        return np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / 12.0
    elif size == 7:
        return np.array([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0]) / 180.0
    else:
        # Default to size 3
        return np.array([1.0, -2.0, 1.0])


def generate_sobel(size):
    """Generate Sobel derivative kernel."""
    if size == 3:
        # Smooth with [1, 2, 1], differentiate with [-1, 0, 1]
        # This is the 1D component
        return np.array([-1.0, 0.0, 1.0])
    else:
        # For other sizes, fall back to central difference
        return generate_deriv1(size)


def generate_prewitt(size):
    """Generate Prewitt derivative kernel."""
    if size == 3:
        # Smooth with [1, 1, 1], differentiate with [-1, 0, 1]
        return np.array([-1.0, 0.0, 1.0])
    else:
        # For other sizes, fall back to central difference
        return generate_deriv1(size)


def generate_sharpen(size):
    """Generate sharpening kernel (identity + high-pass)."""
    if size == 3:
        # [0, -1, 0] + 3*[0, 1, 0] = [0, 2, 0] - [-1, 0, -1] = [1, 2, 1]
        return np.array([-1.0, 3.0, -1.0])
    elif size == 5:
        return np.array([-1.0, -1.0, 5.0, -1.0, -1.0])
    else:
        return np.array([-1.0, 3.0, -1.0])


def generate_unsharp(size):
    """Generate unsharp masking kernel (identity - low-pass)."""
    # Create a low-pass filter and subtract from identity
    smooth = generate_box(size)
    kernel = -smooth
    kernel[size // 2] += 2.0  # Add identity in center
    return kernel


def generate_dog(size):
    """Generate Difference of Gaussians approximation."""
    # Approximate with difference of box filters of different sizes
    if size == 5:
        return np.array([-1.0, -1.0, 4.0, -1.0, -1.0]) / 4.0
    elif size == 7:
        return np.array([-1.0, -1.0, -1.0, 6.0, -1.0, -1.0, -1.0]) / 6.0
    elif size == 9:
        return np.array([-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0]) / 8.0
    elif size == 11:
        return np.array(
            [-1.0, -1.0, -1.0, -1.0, -1.0, 10.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        ) / 10.0
    else:
        return np.array([-1.0, -1.0, 4.0, -1.0, -1.0]) / 4.0


# Kernel generator registry
KERNEL_GENERATORS = {
    "hamming": generate_hamming,
    "hann": generate_hann,
    "blackman": generate_blackman,
    "bartlett": generate_bartlett,
    "box": generate_box,
    "triangle": generate_triangle,
    "binomial": generate_binomial,
    "deriv1": generate_deriv1,
    "deriv2": generate_deriv2,
    "sobel": generate_sobel,
    "prewitt": generate_prewitt,
    "sharpen": generate_sharpen,
    "unsharp": generate_unsharp,
    "dog": generate_dog,
}


def generate_kernel(kernel_type, size):
    """
    Generate a 1D convolution kernel of the specified type and size.

    Args:
        kernel_type: Type of kernel (e.g., 'hamming', 'deriv1', 'box')
        size: Size of the kernel (odd integer)

    Returns:
        1D numpy array of float32 weights
    """
    generator = KERNEL_GENERATORS.get(kernel_type)
    if generator is None:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    weights = generator(size)
    return weights.astype(np.float32)


def generate_convs_from_spec(spec):
    """
    Generate convs list from YAML specification.

    Args:
        spec: Dictionary containing kernel specifications from YAML

    Returns:
        List of (name, weights) tuples for use in convolves interface
    """
    convs = []

    # Process each category of kernels
    for category in ["windows", "smoothing", "derivatives", "highpass", "bandpass"]:
        if category in spec:
            for kernel_spec in spec[category]:
                kernel_type = kernel_spec["type"]
                sizes = kernel_spec["sizes"]

                for size in sizes:
                    name = f"{kernel_type}{size}"
                    weights = generate_kernel(kernel_type, size)
                    convs.append((name, weights))

    # Process custom kernels
    if "custom" in spec:
        for custom_spec in spec["custom"]:
            name = custom_spec["name"]
            weights = np.array(custom_spec["weights"], dtype=np.float32)

            # Normalize if requested
            if custom_spec.get("normalize", False):
                weights = weights / np.sum(np.abs(weights))

            convs.append((name, weights))

    return convs


def generate_convolves_context(context):
    """
    Context generator for convolves interface.

    Generates signal processing convolution kernels from YAML specification
    and adds them to the context as 'convs' parameter.

    This function is invoked by the generic context generator system when
    rendering the convolves interface template.

    Args:
        context: Dictionary containing YAML context (windows, smoothing, etc.)

    Returns:
        Modified context dictionary with 'convs' parameter added

    Example:
        Input context (from convolves.yml):
            {
                'windows': [{'type': 'hamming', 'sizes': [5, 9]}],
                'smoothing': [{'type': 'box', 'sizes': [3, 5]}],
                ...
            }

        Output context:
            {
                'windows': [...],
                'smoothing': [...],
                'convs': [
                    ('hamming5', array([...])),
                    ('hamming9', array([...])),
                    ('box3', array([...])),
                    ('box5', array([...])),
                    ...
                ]
            }
    """
    convs = generate_convs_from_spec(context)
    context['convs'] = convs
    return context

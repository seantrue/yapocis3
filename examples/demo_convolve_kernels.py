#!/usr/bin/env python3
"""
Demonstration of 1D convolution kernels from the convolves interface.

This example applies various signal processing kernels to test signals
using the actual yapocis backend (OpenCL/Metal/CUDA/Numba) and
visualizes the results to show their different characteristics.
"""
import numpy as np
import matplotlib.pyplot as plt
from yapocis.lib.convolve_kernels import generate_kernel
import yapocis.core as core

# Load the convolves program (compiles kernels for current backend)
print("Loading convolves interface...")
prog = core.load_program("convolves")
print(f"Backend: {core.get_backend()}")
print(f"Device: {core.get_backend_device()[1]}")
print()


def create_test_signal(n_points=200):
    """
    Create a composite test signal with multiple features.

    Returns:
        Tuple of (x, signal) where signal contains:
        - Step edges
        - Linear ramps
        - Gaussian noise
        - Sine waves
    """
    x = np.linspace(0, 10, n_points)
    signal = np.zeros(n_points, dtype=np.float32)

    # Add step functions
    signal[40:80] = 1.0
    signal[120:160] = 0.7

    # Add ramps
    signal[80:120] = np.linspace(1.0, 0.0, 40)

    # Add sine wave
    signal += 0.2 * np.sin(2 * np.pi * x)

    # Add Gaussian noise
    np.random.seed(42)
    signal += 0.1 * np.random.randn(n_points)

    return x, signal.astype(np.float32)


def apply_kernel(signal, kernel_name, kernel_type=None):
    """
    Apply a convolution kernel to a signal using yapocis backend.

    Args:
        signal: Input signal (1D numpy array)
        kernel_name: Name of kernel to apply (e.g., 'hamming9')
        kernel_type: Type of kernel ('deriv1', 'deriv2', etc.) for special handling

    Returns:
        Convolved signal (same size as input)
    """
    # Get the kernel from the loaded program
    kernel = getattr(prog, kernel_name)

    # Apply convolution using backend
    result = kernel(signal)

    return result


def demo_smoothing_filters():
    """Demonstrate various smoothing filters."""
    print("=" * 70)
    print("SMOOTHING FILTERS DEMONSTRATION (using yapocis backend)")
    print("=" * 70)

    # Create noisy signal
    x, signal = create_test_signal()

    # Define kernels to test (using backend kernels)
    kernels = {
        'Original': None,
        'Box (5)': 'box5',
        'Box (11)': 'box11',
        'Hamming (9)': 'hamming9',
        'Hann (9)': 'hann9',
        'Binomial (9)': 'binomial9',
    }

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Smoothing Filters Comparison (Backend-Accelerated)', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    # Plot each result
    for idx, (name, kernel_name) in enumerate(kernels.items()):
        ax = axes[idx]

        if kernel_name is None:
            # Original signal
            result = signal
            ax.plot(x, result, 'b-', linewidth=1, alpha=0.7)
            ax.set_title(f'{name}\n(noisy input signal)', fontweight='bold')
        else:
            # Apply smoothing using backend
            result = apply_kernel(signal, kernel_name)

            # Plot original (faint) and filtered
            ax.plot(x, signal, 'gray', linewidth=0.5, alpha=0.3, label='Original')
            ax.plot(x, result, 'b-', linewidth=2, label='Filtered')

            # Get kernel size from name
            kernel_size = int(''.join(c for c in kernel_name if c.isdigit()))
            ax.set_title(f'{name}\nBackend: {core.get_backend()}', fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)

        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Position')
        ax.set_ylabel('Amplitude')

    plt.tight_layout()
    plt.savefig('claude-debug/smoothing_filters.png', dpi=150, bbox_inches='tight')
    print(f"Saved: claude-debug/smoothing_filters.png")
    print()

    return fig


def demo_derivative_filters():
    """Demonstrate derivative filters for edge detection."""
    print("=" * 70)
    print("DERIVATIVE FILTERS DEMONSTRATION (using yapocis backend)")
    print("=" * 70)

    # Create signal with clear edges
    x = np.linspace(0, 10, 200)
    signal = np.zeros(200, dtype=np.float32)

    # Create step edges
    signal[40:80] = 1.0
    signal[100:140] = 0.8
    signal[160:180] = 0.5

    # Add small amount of noise
    np.random.seed(42)
    signal += 0.02 * np.random.randn(200)

    # Define derivative kernels (using backend kernels)
    kernels = {
        'Original': None,
        'First Derivative (3-pt)': 'deriv13',
        'First Derivative (5-pt)': 'deriv15',
        'Sobel': 'sobel3',
        'Second Derivative (3-pt)': 'deriv23',
        'Second Derivative (5-pt)': 'deriv25',
    }

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Derivative Filters for Edge Detection (Backend-Accelerated)', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    # Plot each result
    for idx, (name, kernel_name) in enumerate(kernels.items()):
        ax = axes[idx]

        if kernel_name is None:
            # Original signal
            result = signal
            ax.plot(x, result, 'b-', linewidth=2)
            ax.set_title(f'{name}\n(step edges)', fontweight='bold')
            ax.set_ylim(-0.2, 1.2)
        else:
            # Apply derivative filter using backend
            result = apply_kernel(signal, kernel_name)

            # Plot result
            ax.plot(x, result, 'r-', linewidth=2, label='Derivative')
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_title(f'{name}\nBackend: {core.get_backend()}', fontweight='bold', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)

            # Add markers for detected edges (zero crossings or peaks)
            if 'First' in name or 'Sobel' in name:
                # Mark peaks (edges)
                peaks = np.where(np.abs(result) > 0.3 * np.max(np.abs(result)))[0]
                if len(peaks) > 0:
                    ax.plot(x[peaks], result[peaks], 'go', markersize=4, alpha=0.5)

        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Position')
        ax.set_ylabel('Amplitude')

    plt.tight_layout()
    plt.savefig('claude-debug/derivative_filters.png', dpi=150, bbox_inches='tight')
    print(f"Saved: claude-debug/derivative_filters.png")
    print()

    return fig


def demo_frequency_response():
    """Demonstrate frequency response of different filters."""
    print("=" * 70)
    print("FREQUENCY RESPONSE DEMONSTRATION (using yapocis backend)")
    print("=" * 70)

    # Create signals with different frequencies
    x = np.linspace(0, 10, 500)

    # Low frequency
    low_freq = 0.3 * np.sin(2 * np.pi * 1 * x)

    # Medium frequency
    med_freq = 0.3 * np.sin(2 * np.pi * 5 * x)

    # High frequency
    high_freq = 0.3 * np.sin(2 * np.pi * 15 * x)

    # Composite signal
    signal = (low_freq + med_freq + high_freq).astype(np.float32)

    # Test different filter types (using backend kernels)
    filters = {
        'Low-pass (Box 11)': 'box11',
        'Low-pass (Hamming 13)': 'hamming13',
        'High-pass (Sharpen 5)': 'sharpen5',
        'Band-pass (DoG 9)': 'dog9',
    }

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle('Frequency Response of Different Filters (Backend-Accelerated)', fontsize=16, fontweight='bold')

    # Plot original components
    ax = axes[0, 0]
    ax.plot(x[:100], signal[:100], 'b-', linewidth=1.5, label='Composite')
    ax.plot(x[:100], low_freq[:100], 'g--', linewidth=1, alpha=0.5, label='Low freq')
    ax.plot(x[:100], med_freq[:100], 'orange', linestyle='--', linewidth=1, alpha=0.5, label='Med freq')
    ax.plot(x[:100], high_freq[:100], 'r--', linewidth=1, alpha=0.5, label='High freq')
    ax.set_title('Original Signal\n(sum of 3 frequencies)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Position')
    ax.set_ylabel('Amplitude')

    # Plot filtered results
    for idx, (name, kernel_name) in enumerate(filters.items(), start=1):
        ax = axes[idx // 3, idx % 3]

        # Apply filter using backend
        result = apply_kernel(signal, kernel_name)

        # Plot
        ax.plot(x[:100], signal[:100], 'gray', linewidth=0.5, alpha=0.3, label='Original')
        ax.plot(x[:100], result[:100], 'b-', linewidth=2, label='Filtered')
        ax.set_title(f'{name}\nBackend: {core.get_backend()}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Position')
        ax.set_ylabel('Amplitude')

    # Hide unused subplot
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('claude-debug/frequency_response.png', dpi=150, bbox_inches='tight')
    print(f"Saved: claude-debug/frequency_response.png")
    print()

    return fig


def demo_window_comparison():
    """Compare different windowing functions."""
    print("=" * 70)
    print("WINDOWING FUNCTIONS COMPARISON")
    print("=" * 70)

    # Generate different windows
    size = 51
    windows = {
        'Bartlett (Triangular)': generate_kernel('bartlett', size),
        'Hamming': generate_kernel('hamming', size),
        'Hann': generate_kernel('hann', size),
        'Blackman': generate_kernel('blackman', size),
    }

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Windowing Functions Comparison', fontsize=16, fontweight='bold')

    # Plot window shapes
    x = np.arange(size)
    for name, window in windows.items():
        ax1.plot(x, window, linewidth=2, label=name, marker='o', markersize=3, alpha=0.7)

    ax1.set_title('Window Shapes (Time Domain)', fontweight='bold')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot frequency response (FFT of windows)
    for name, window in windows.items():
        # Pad window for better frequency resolution
        padded = np.pad(window, (0, 512 - len(window)), mode='constant')
        fft = np.fft.fft(padded)
        freq = np.fft.fftfreq(len(padded))

        # Plot magnitude in dB
        magnitude_db = 20 * np.log10(np.abs(fft[:256]) + 1e-10)
        magnitude_db -= np.max(magnitude_db)  # Normalize to 0 dB

        ax2.plot(freq[:256], magnitude_db, linewidth=2, label=name, alpha=0.7)

    ax2.set_title('Frequency Response (Magnitude)', fontweight='bold')
    ax2.set_xlabel('Normalized Frequency')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_ylim(-100, 5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('claude-debug/window_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: claude-debug/window_comparison.png")
    print()

    return fig


def demo_kernel_visualization():
    """Visualize the actual kernel weights."""
    print("=" * 70)
    print("KERNEL WEIGHTS VISUALIZATION")
    print("=" * 70)

    # Select representative kernels
    kernels = {
        'Box (7)': generate_kernel('box', 7),
        'Binomial (9)': generate_kernel('binomial', 9),
        'Hamming (11)': generate_kernel('hamming', 11),
        'First Derivative (5)': generate_kernel('deriv1', 5),
        'Second Derivative (5)': generate_kernel('deriv2', 5),
        'Sharpen (5)': generate_kernel('sharpen', 5),
    }

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Convolution Kernel Weights', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    # Plot each kernel
    for idx, (name, kernel) in enumerate(kernels.items()):
        ax = axes[idx]

        x = np.arange(len(kernel)) - len(kernel) // 2
        colors = ['red' if w < 0 else 'blue' for w in kernel]

        ax.bar(x, kernel, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_title(f'{name}\nSum: {np.sum(kernel):.3f}', fontweight='bold')
        ax.set_xlabel('Position')
        ax.set_ylabel('Weight')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (xi, yi) in enumerate(zip(x, kernel)):
            ax.text(xi, yi, f'{yi:.3f}', ha='center',
                   va='bottom' if yi >= 0 else 'top', fontsize=8)

    plt.tight_layout()
    plt.savefig('claude-debug/kernel_weights.png', dpi=150, bbox_inches='tight')
    print(f"Saved: claude-debug/kernel_weights.png")
    print()

    return fig


def print_kernel_properties():
    """Print mathematical properties of various kernels."""
    print("=" * 70)
    print("KERNEL MATHEMATICAL PROPERTIES")
    print("=" * 70)
    print()

    kernels = [
        ('box', 7),
        ('hamming', 9),
        ('binomial', 9),
        ('deriv1', 5),
        ('deriv2', 5),
        ('sharpen', 5),
        ('dog', 7),
    ]

    print(f"{'Kernel':<20} {'Size':<6} {'Sum':<12} {'Symmetric':<12} {'Purpose':<30}")
    print("-" * 80)

    for kernel_type, size in kernels:
        kernel = generate_kernel(kernel_type, size)
        kernel_sum = np.sum(kernel)
        is_symmetric = np.allclose(kernel, kernel[::-1])

        # Determine purpose
        if abs(kernel_sum - 1.0) < 1e-5:
            purpose = "Smoothing (preserves DC)"
        elif abs(kernel_sum) < 1e-5:
            purpose = "Derivative/High-pass (removes DC)"
        else:
            purpose = "General filtering"

        print(f"{kernel_type:<20} {size:<6} {kernel_sum:<12.6f} {str(is_symmetric):<12} {purpose:<30}")

    print()


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("CONVOLVES KERNEL DEMONSTRATION")
    print("Signal Processing with 1D Convolution Kernels")
    print("Using yapocis backend-accelerated convolution")
    print("=" * 70)
    print()

    # Print kernel properties
    print_kernel_properties()

    # Run demonstrations
    demo_kernel_visualization()
    demo_smoothing_filters()
    demo_derivative_filters()
    demo_frequency_response()
    demo_window_comparison()

    print("=" * 70)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)
    print()
    print(f"Backend used: {core.get_backend()}")
    print(f"Device: {core.get_backend_device()[1]}")
    print()
    print("Generated visualizations:")
    print("  - claude-debug/kernel_weights.png (kernel weight visualization)")
    print("  - claude-debug/smoothing_filters.png (backend-accelerated)")
    print("  - claude-debug/derivative_filters.png (backend-accelerated)")
    print("  - claude-debug/frequency_response.png (backend-accelerated)")
    print("  - claude-debug/window_comparison.png (kernel weight visualization)")
    print()
    print("Note: Smoothing, derivative, and frequency response demos use")
    print("      actual backend kernels (OpenCL/Metal/CUDA/Numba) for convolution.")
    print("      Kernel weight and window comparison demos visualize the weights.")
    print()
    print("To view the plots, use plt.show() or open the PNG files.")
    print()


if __name__ == '__main__':
    main()

    # Optionally show all plots interactively
    # plt.show()

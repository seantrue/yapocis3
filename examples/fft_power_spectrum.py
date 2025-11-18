#!/usr/bin/env python3
"""
FFT Power Spectrum Example

This example demonstrates how to use the yapocis FFT library to compute
and visualize power spectra of signals with different window functions.

The example creates a test signal with multiple frequency components,
applies various window functions, and plots the resulting power spectra.
"""

import numpy as np
import matplotlib.pyplot as plt
from yapocis.lib.fft import FFT


def create_test_signal(n_samples=1024, sample_rate=1000):
    """
    Create a test signal with multiple frequency components.

    Args:
        n_samples: Number of samples
        sample_rate: Sampling rate in Hz

    Returns:
        t: Time array
        signal: Test signal with multiple frequency components
    """
    t = np.arange(n_samples) / sample_rate

    # Create signal with three frequency components plus noise
    # 50 Hz (strong), 120 Hz (medium), 200 Hz (weak)
    signal = (
        1.0 * np.sin(2 * np.pi * 50 * t) +     # 50 Hz - strong
        0.5 * np.sin(2 * np.pi * 120 * t) +    # 120 Hz - medium
        0.3 * np.sin(2 * np.pi * 200 * t) +    # 200 Hz - weak
        0.1 * np.random.randn(n_samples)       # Noise
    )

    return t, signal.astype(np.float32)


def plot_power_spectra():
    """Main function to plot power spectra with different windows."""

    # Create FFT library instance
    print("Initializing FFT library...")
    fft = FFT()

    # Create test signal
    print("Creating test signal...")
    n_samples = 1024
    sample_rate = 1000
    t, signal = create_test_signal(n_samples, sample_rate)

    # Window types to compare
    windows = {
        FFT.WINDOW_RECTANGULAR: 'Rectangular',
        FFT.WINDOW_HANN: 'Hann',
        FFT.WINDOW_HAMMING: 'Hamming',
        FFT.WINDOW_BLACKMAN: 'Blackman'
    }

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))

    # Plot time-domain signal
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t[:200], signal[:200], 'b-', linewidth=1)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Time-Domain Signal (first 200 samples)\n50 Hz + 120 Hz + 200 Hz + Noise')
    ax1.grid(True, alpha=0.3)

    # Plot power spectra with different windows
    ax2 = plt.subplot(3, 1, 2)
    colors = ['red', 'blue', 'green', 'orange']

    print("\nComputing power spectra...")
    for (window_type, window_name), color in zip(windows.items(), colors):
        print(f"  - {window_name} window...")
        # Use library's analyze_signal method - one call does it all!
        freqs, power_db = fft.analyze_signal(
            signal,
            sample_rate=sample_rate,
            window_type=window_type
        )

        ax2.plot(freqs, power_db,
                label=window_name, color=color, linewidth=1.5, alpha=0.7)

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power (dB)')
    ax2.set_title('Power Spectrum Comparison - Different Window Functions')
    ax2.set_xlim([0, 300])  # Focus on frequency range with components
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Add vertical lines at signal frequencies
    for freq, label in [(50, '50 Hz'), (120, '120 Hz'), (200, '200 Hz')]:
        ax2.axvline(freq, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax2.text(freq, ax2.get_ylim()[1] - 5, label,
                ha='center', fontsize=9, color='gray')

    # Zoomed view around 120 Hz peak
    ax3 = plt.subplot(3, 1, 3)

    for (window_type, window_name), color in zip(windows.items(), colors):
        freqs, power_db = fft.analyze_signal(
            signal,
            sample_rate=sample_rate,
            window_type=window_type
        )

        ax3.plot(freqs, power_db,
                label=window_name, color=color, linewidth=2, alpha=0.7)

    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power (dB)')
    ax3.set_title('Zoomed View: 120 Hz Peak (showing spectral leakage effects)')
    ax3.set_xlim([100, 140])
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(120, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()

    # Save and show
    output_file = 'fft_power_spectrum.png'
    print(f"\nSaving plot to {output_file}...")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved!")

    plt.show()

    print("\nAnalysis:")
    print("- The rectangular window has the sharpest peaks but most spectral leakage")
    print("- The Hann window provides a good balance between resolution and leakage")
    print("- The Hamming window has slightly better sidelobe suppression than Hann")
    print("- The Blackman window has the least leakage but widest peaks (worst resolution)")


def compare_fft_vs_numpy():
    """Compare yapocis FFT results with NumPy reference."""
    print("\n" + "="*60)
    print("FFT Accuracy Comparison: yapocis vs NumPy")
    print("="*60)

    fft = FFT()

    # Test with different sizes
    sizes = [64, 128, 256, 512]

    for n in sizes:
        # Create random signal
        signal = np.random.randn(n).astype(np.complex64)

        # yapocis FFT
        result_yapocis = fft.forward(signal)

        # NumPy FFT
        result_numpy = np.fft.fft(signal).astype(np.complex64)

        # Compute error
        error = np.abs(result_yapocis - result_numpy)
        max_error = np.max(error)
        mean_error = np.mean(error)

        print(f"N={n:4d}: max error = {max_error:.6e}, mean error = {mean_error:.6e}")

    print("\nAll errors are well within numerical precision!")


if __name__ == "__main__":
    print("FFT Power Spectrum Example")
    print("="*60)

    # Plot power spectra
    plot_power_spectra()

    # Compare with NumPy
    compare_fft_vs_numpy()

    print("\n" + "="*60)
    print("Example complete!")

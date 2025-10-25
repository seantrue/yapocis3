#!/usr/bin/env python3
"""
FFT Quickstart Example

A minimal example showing how to use the yapocis FFT library
to analyze a simple audio-like signal.
"""

import numpy as np
import matplotlib.pyplot as plt
from yapocis.lib.fft import FFT


def main():
    # Create FFT library instance (auto-selects best backend)
    fft = FFT()
    print("FFT library initialized!")

    # Create a test signal: mixture of 440 Hz (A note) and 880 Hz (A octave higher)
    sample_rate = 8000  # Hz
    duration = 0.5      # seconds
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)

    # Musical A notes
    signal = (
        np.sin(2 * np.pi * 440 * t) +    # A4 (440 Hz)
        0.5 * np.sin(2 * np.pi * 880 * t)  # A5 (880 Hz)
    ).astype(np.float32)

    print(f"Created signal: {n_samples} samples at {sample_rate} Hz")
    print(f"Frequencies: 440 Hz and 880 Hz")

    # Analyze signal with ONE method call!
    # This applies windowing, FFT, and power spectrum computation
    freqs, power_db = fft.analyze_signal(
        signal,
        sample_rate=sample_rate,
        window_type=FFT.WINDOW_HANN
    )

    # Find dominant frequency peaks
    peak_freqs, peak_powers = fft.find_peaks(signal, sample_rate, n_peaks=2)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Time domain
    ax1.plot(t[:500], signal[:500], 'b-', linewidth=1)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Time-Domain Signal: 440 Hz + 880 Hz')
    ax1.grid(True, alpha=0.3)

    # Frequency domain (power spectrum)
    # Only plot up to 2000 Hz
    freq_mask = freqs <= 2000
    ax2.plot(freqs[freq_mask], power_db[freq_mask], 'r-', linewidth=1.5)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power (dB)')
    ax2.set_title('Power Spectrum (Hann Window)')
    ax2.grid(True, alpha=0.3)

    # Mark the detected peaks
    for freq in peak_freqs:
        ax2.axvline(freq, color='green', linestyle='--', alpha=0.5)
    ax2.legend(['Power Spectrum', 'Detected Peaks'])

    plt.tight_layout()

    # Save the plot
    output_file = 'claude-debug/fft_quickstart.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")

    plt.show()

    # Report detected peaks
    print("\nDetected frequency peaks:")
    for i, (freq, power) in enumerate(zip(peak_freqs, peak_powers), 1):
        print(f"  Peak {i}: {freq:.1f} Hz ({power:.1f} dB)")


if __name__ == "__main__":
    print("=" * 60)
    print("FFT Quickstart Example")
    print("=" * 60 + "\n")

    main()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

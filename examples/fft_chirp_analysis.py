#!/usr/bin/env python3
"""
FFT Chirp Signal Analysis

This example demonstrates FFT analysis of a chirp signal (frequency sweep).
It shows how frequency content changes over time using the yapocis FFT library's
spectrogram functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
from yapocis.lib.fft import FFT


def create_chirp_signal(duration=2.0, sample_rate=8000, f_start=100, f_end=1000):
    """
    Create a linear chirp signal (frequency sweep).

    Args:
        duration: Signal duration in seconds
        sample_rate: Sampling rate in Hz
        f_start: Starting frequency in Hz
        f_end: Ending frequency in Hz

    Returns:
        t: Time array
        signal: Chirp signal
    """
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)

    # Linear chirp: frequency increases linearly with time
    # phase(t) = 2*pi * (f_start*t + (f_end-f_start)/(2*duration) * t^2)
    k = (f_end - f_start) / duration  # Rate of frequency change
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
    signal = np.sin(phase).astype(np.float32)

    return t, signal


def plot_chirp_analysis():
    """Main function to analyze and plot chirp signal."""

    # Create FFT library instance
    print("Initializing FFT library...")
    fft = FFT()

    # Create chirp signal (100 Hz to 1000 Hz over 2 seconds)
    print("\nCreating chirp signal...")
    sample_rate = 8000
    duration = 2.0
    f_start = 100
    f_end = 1000
    t, signal = create_chirp_signal(duration, sample_rate, f_start, f_end)

    print(f"Signal: {len(signal)} samples, {duration}s")
    print(f"Frequency sweep: {f_start} Hz -> {f_end} Hz")

    # Generate spectrogram with ONE method call!
    print("\nGenerating spectrogram...")
    times, freqs, spectra = fft.spectrogram(
        signal,
        sample_rate=sample_rate,
        window_size=512,
        hop_size=128,  # 75% overlap
        window_type=FFT.WINDOW_HANN
    )
    print(f"Spectrogram generated: {spectra.shape[0]} time windows, {spectra.shape[1]} frequency bins")

    # Create figure
    fig = plt.figure(figsize=(14, 10))

    # 1. Time-domain signal
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, signal, 'b-', linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Chirp Signal: {f_start} Hz → {f_end} Hz')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, duration])

    # 2. Spectrogram (time-frequency representation)
    ax2 = plt.subplot(3, 1, 2)

    # Create mesh for spectrogram
    extent = [times[0], times[-1], freqs[0], freqs[-1]]
    im = ax2.imshow(spectra.T, aspect='auto', origin='lower',
                    extent=extent, cmap='viridis', interpolation='bilinear')

    # Add expected chirp line
    chirp_times = np.linspace(0, duration, 100)
    chirp_freqs = f_start + (f_end - f_start) * chirp_times / duration
    ax2.plot(chirp_times, chirp_freqs, 'r--', linewidth=2,
             label='Expected frequency', alpha=0.8)

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title('Spectrogram (Time-Frequency Analysis)')
    ax2.set_ylim([0, 1500])
    ax2.legend(loc='upper left')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Power (dB)')

    # 3. Power spectrum at specific time points
    ax3 = plt.subplot(3, 1, 3)

    # Plot spectra at 3 different time points
    time_points = [0.5, 1.0, 1.5]  # seconds
    colors = ['blue', 'green', 'red']

    for tp, color in zip(time_points, colors):
        # Find closest time index
        idx = np.argmin(np.abs(times - tp))
        actual_time = times[idx]

        # Expected frequency at this time
        expected_freq = f_start + (f_end - f_start) * actual_time / duration

        ax3.plot(freqs, spectra[idx, :], color=color, linewidth=1.5,
                label=f't={actual_time:.2f}s (≈{expected_freq:.0f} Hz)', alpha=0.7)

        # Mark expected frequency
        ax3.axvline(expected_freq, color=color, linestyle=':', alpha=0.5)

    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power (dB)')
    ax3.set_title('Power Spectra at Different Time Points')
    ax3.set_xlim([0, 1500])
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_file = 'fft_chirp_analysis.png'
    print(f"\nSaving plot to {output_file}...")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print("Plot saved!")

    plt.show()

    print("\nAnalysis complete!")
    print(f"The spectrogram clearly shows the frequency increasing from {f_start} Hz to {f_end} Hz")
    print("over the {:.1f} second duration.".format(duration))


if __name__ == "__main__":
    print("=" * 70)
    print("FFT Chirp Signal Analysis Example")
    print("=" * 70 + "\n")

    plot_chirp_analysis()

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

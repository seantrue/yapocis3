#!/usr/bin/env python3
"""
FFT Library Demo

This example demonstrates the high-level FFT library API,
which is simpler and more intuitive than using raw kernels.
"""

import numpy as np
import matplotlib.pyplot as plt
from yapocis.lib.fft import FFT


def demo_basic_analysis():
    """Demonstrate basic spectral analysis."""
    print("\n" + "="*60)
    print("Demo 1: Basic Spectral Analysis")
    print("="*60)

    fft = FFT()

    # Create test signal: 440 Hz (A4) + 880 Hz (A5)
    sample_rate = 8000
    duration = 0.5
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)

    signal = (
        np.sin(2 * np.pi * 440 * t) +
        0.5 * np.sin(2 * np.pi * 880 * t)
    ).astype(np.float32)

    # Analyze signal with ONE method call!
    freqs, power_db = fft.analyze_signal(
        signal,
        sample_rate=sample_rate,
        window_type=FFT.WINDOW_HANN
    )

    # Find dominant frequencies
    peak_freqs, peak_powers = fft.find_peaks(signal, sample_rate, n_peaks=2)

    print(f"Dominant frequencies found:")
    for i, (freq, power) in enumerate(zip(peak_freqs, peak_powers), 1):
        print(f"  Peak {i}: {freq:.1f} Hz ({power:.1f} dB)")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs, power_db, 'b-', linewidth=1)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (dB)')
    ax.set_title('Power Spectrum - Library API Demo')
    ax.set_xlim([0, 2000])
    ax.grid(True, alpha=0.3)

    # Mark peaks
    for freq in peak_freqs:
        ax.axvline(freq, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('claude-debug/fft_library_demo_basic.png', dpi=150)
    print("\nPlot saved to claude-debug/fft_library_demo_basic.png")
    plt.close()


def demo_spectrogram():
    """Demonstrate spectrogram generation."""
    print("\n" + "="*60)
    print("Demo 2: Spectrogram Generation")
    print("="*60)

    fft = FFT()

    # Create chirp signal (frequency sweep)
    sample_rate = 8000
    duration = 2.0
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)

    # Linear chirp from 200 to 1000 Hz
    f_start, f_end = 200, 1000
    k = (f_end - f_start) / duration
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
    signal = np.sin(phase).astype(np.float32)

    # Generate spectrogram with ONE method call!
    times, freqs, spec = fft.spectrogram(
        signal,
        sample_rate=sample_rate,
        window_size=512,
        hop_size=128,
        window_type=FFT.WINDOW_HANN
    )

    print(f"Spectrogram shape: {spec.shape}")
    print(f"Time range: {times[0]:.3f} to {times[-1]:.3f} seconds")
    print(f"Frequency range: {freqs[0]:.1f} to {freqs[-1]:.1f} Hz")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    extent = [times[0], times[-1], freqs[0], freqs[-1]]
    im = ax.imshow(spec.T, aspect='auto', origin='lower',
                   extent=extent, cmap='viridis', interpolation='bilinear')

    # Expected chirp line
    chirp_times = np.linspace(0, duration, 100)
    chirp_freqs = f_start + (f_end - f_start) * chirp_times / duration
    ax.plot(chirp_times, chirp_freqs, 'r--', linewidth=2, label='Expected')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Spectrogram - Library API Demo')
    ax.set_ylim([0, 1500])
    ax.legend()

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power (dB)')

    plt.tight_layout()
    plt.savefig('claude-debug/fft_library_demo_spectrogram.png', dpi=150)
    print("\nPlot saved to claude-debug/fft_library_demo_spectrogram.png")
    plt.close()


def demo_bandpass_filter():
    """Demonstrate bandpass filtering."""
    print("\n" + "="*60)
    print("Demo 3: Bandpass Filtering")
    print("="*60)

    fft = FFT()

    # Create signal with multiple frequencies
    sample_rate = 1000
    duration = 1.0
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)

    signal = (
        np.sin(2 * np.pi * 50 * t) +    # 50 Hz - will be filtered out
        np.sin(2 * np.pi * 100 * t) +   # 100 Hz - keep
        np.sin(2 * np.pi * 150 * t) +   # 150 Hz - keep
        np.sin(2 * np.pi * 300 * t)     # 300 Hz - will be filtered out
    ).astype(np.float32)

    # Apply bandpass filter with ONE method call!
    filtered = fft.bandpass_filter(
        signal,
        low_freq=80,
        high_freq=200,
        sample_rate=sample_rate
    )

    # Analyze both signals
    freqs_orig, power_orig = fft.analyze_signal(signal, sample_rate)
    freqs_filt, power_filt = fft.analyze_signal(filtered, sample_rate)

    print(f"Original signal frequency content:")
    peaks_orig, _ = fft.find_peaks(signal, sample_rate, n_peaks=4)
    for freq in sorted(peaks_orig):
        print(f"  {freq:.1f} Hz")

    print(f"\nFiltered signal frequency content (80-200 Hz passband):")
    peaks_filt, _ = fft.find_peaks(filtered, sample_rate, n_peaks=4)
    for freq in sorted(peaks_filt):
        print(f"  {freq:.1f} Hz")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Original spectrum
    ax1.plot(freqs_orig, power_orig, 'b-', linewidth=1)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power (dB)')
    ax1.set_title('Original Signal Spectrum')
    ax1.set_xlim([0, 400])
    ax1.grid(True, alpha=0.3)
    ax1.axvspan(80, 200, alpha=0.2, color='green', label='Passband')
    ax1.legend()

    # Filtered spectrum
    ax2.plot(freqs_filt, power_filt, 'r-', linewidth=1)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power (dB)')
    ax2.set_title('Filtered Signal Spectrum (80-200 Hz)')
    ax2.set_xlim([0, 400])
    ax2.grid(True, alpha=0.3)
    ax2.axvspan(80, 200, alpha=0.2, color='green', label='Passband')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('claude-debug/fft_library_demo_filter.png', dpi=150)
    print("\nPlot saved to claude-debug/fft_library_demo_filter.png")
    plt.close()


def demo_signal_metrics():
    """Demonstrate signal quality metrics."""
    print("\n" + "="*60)
    print("Demo 4: Signal Quality Metrics")
    print("="*60)

    fft = FFT()

    # Create clean signal
    sample_rate = 1000
    duration = 1.0
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)

    clean_signal = np.sin(2 * np.pi * 100 * t).astype(np.float32)
    noise = 0.1 * np.random.randn(n_samples).astype(np.float32)
    noisy_signal = clean_signal + noise

    # Compute metrics with simple method calls!
    energy_clean = fft.energy(clean_signal)
    energy_noise = fft.energy(noise)
    rms_clean = fft.rms(clean_signal)
    rms_noisy = fft.rms(noisy_signal)
    snr = fft.snr(clean_signal, noise)

    print(f"Clean signal:")
    print(f"  Energy: {energy_clean:.2f}")
    print(f"  RMS: {rms_clean:.4f}")

    print(f"\nNoise:")
    print(f"  Energy: {energy_noise:.2f}")

    print(f"\nNoisy signal:")
    print(f"  RMS: {rms_noisy:.4f}")

    print(f"\nSignal-to-Noise Ratio: {snr:.2f} dB")


def demo_coherence():
    """Demonstrate coherence analysis."""
    print("\n" + "="*60)
    print("Demo 5: Coherence Analysis")
    print("="*60)

    fft = FFT()

    # Create two signals with different correlation levels
    n = 1000
    signal1 = np.random.randn(n).astype(np.float32)

    # Signal 2: Mostly correlated with signal1, plus some noise
    signal2 = 0.9 * signal1 + 0.1 * np.random.randn(n).astype(np.float32)

    # Signal 3: Uncorrelated
    signal3 = np.random.randn(n).astype(np.float32)

    # Compute coherence with ONE method call each!
    freqs, coh_high = fft.coherence(signal1, signal2)  # Should be high
    _, coh_low = fft.coherence(signal1, signal3)       # Should be low

    mean_coh_high = np.mean(coh_high)
    mean_coh_low = np.mean(coh_low)

    print(f"Coherence between correlated signals: {mean_coh_high:.3f}")
    print(f"Coherence between uncorrelated signals: {mean_coh_low:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs, coh_high, 'g-', linewidth=1, label='Correlated signals', alpha=0.7)
    ax.plot(freqs, coh_low, 'r-', linewidth=1, label='Uncorrelated signals', alpha=0.7)
    ax.set_xlabel('Normalized Frequency')
    ax.set_ylabel('Coherence')
    ax.set_title('Coherence Analysis')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('claude-debug/fft_library_demo_coherence.png', dpi=150)
    print("\nPlot saved to claude-debug/fft_library_demo_coherence.png")
    plt.close()


if __name__ == "__main__":
    print("="*60)
    print("FFT Library API Demo")
    print("="*60)
    print("\nThis demo shows how the high-level FFT library makes")
    print("signal processing simple and intuitive!")

    demo_basic_analysis()
    demo_spectrogram()
    demo_bandpass_filter()
    demo_signal_metrics()
    demo_coherence()

    print("\n" + "="*60)
    print("All demos complete!")
    print("="*60)
    print("\nKey advantages of the library API:")
    print("  ✓ Simple, intuitive method names")
    print("  ✓ Sensible defaults (Hann window, dB scale, etc.)")
    print("  ✓ Automatic array conversions")
    print("  ✓ Comprehensive signal analysis tools")
    print("  ✓ One-line spectrograms and filters")
    print("="*60)

"""
FFT Library - High-level interface for FFT operations.

This module provides a convenient interface to yapocis FFT kernels for
signal processing and frequency analysis.

Example:
    >>> from yapocis.lib.fft import FFT
    >>> import numpy as np
    >>>
    >>> # Create FFT instance
    >>> fft = FFT()
    >>>
    >>> # Analyze a signal
    >>> signal = np.sin(2 * np.pi * 100 * t)
    >>> freqs, power = fft.analyze_signal(signal, sample_rate=1000)
    >>>
    >>> # Generate spectrogram
    >>> times, freqs, spec = fft.spectrogram(signal, sample_rate=1000)
"""

#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2024.

import os

import numpy as np

from .bound_kernels import BoundKernels


class FFT(BoundKernels):
    """
    High-level interface for FFT operations.

    This class provides convenient methods for common signal processing tasks
    including spectral analysis, windowing, and spectrogram generation.

    Attributes:
        INTERFACE: The kernel interface name ("fft")

    Example:
        >>> fft = FFT()
        >>> signal = np.random.randn(1024).astype(np.float32)
        >>> freqs, power_db = fft.analyze_signal(signal, sample_rate=1000)
    """

    INTERFACE = "fft"

    # CRITICAL FIX for OpenCL compiler optimization bug
    # Apple's OpenCL compiler has a bug with aggressive optimizations that causes
    # 2x amplification in batch FFT for sizes >= 64. Disabling optimizations fixes this.
    # This affects Apple M1/M2/M3 GPUs. The performance cost is acceptable (~10-20%).
    # Note: This option is OpenCL-specific and not needed for CUDA/Metal/Numbax
    def __init__(self, *args, **kwargs):
        backend = os.environ.get("YAPOCIS_BACKEND", "ocl")

        # Only apply optimization workaround for OpenCL backend
        if backend == "ocl":
            self.load_program_config = {"options": ["-cl-opt-disable"]}

        super().__init__(*args, **kwargs)

    # Window type constants for clarity
    WINDOW_RECTANGULAR = 0
    WINDOW_HANN = 1
    WINDOW_HAMMING = 2
    WINDOW_BLACKMAN = 3

    def forward(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute forward FFT of a complex signal.

        Args:
            signal: Complex input signal (np.complex64)

        Returns:
            Complex FFT result (np.complex64)

        Example:
            >>> signal = np.random.randn(256).astype(np.complex64)
            >>> fft_result = fft.forward(signal)
        """
        fft_size = len(signal)
        # For sizes > 256, we can't use local_size=fft_size (max work-group size is 256)
        # The kernel will use DFT fallback for sizes > 1024
        if fft_size <= 256:
            # Optimized FFT path with local memory
            return self.kmgr.fft1d(
                signal, 1, global_size=(fft_size,), local_size=(fft_size,)
            )
        else:
            # Let OpenCL choose work-group size for DFT fallback or large FFT
            return self.kmgr.fft1d(signal, 1, global_size=(fft_size,))

    def inverse(self, fft_result: np.ndarray) -> np.ndarray:
        """
        Compute inverse FFT.

        Args:
            fft_result: Complex FFT result (np.complex64)

        Returns:
            Time-domain signal (np.complex64)

        Example:
            >>> reconstructed = fft.inverse(fft_result)
        """
        fft_size = len(fft_result)
        # For sizes > 256, we can't use local_size=fft_size (max work-group size is 256)
        if fft_size <= 256:
            # Optimized FFT path with local memory
            return self.kmgr.fft1d(
                fft_result, -1, global_size=(fft_size,), local_size=(fft_size,)
            )
        else:
            # Let OpenCL choose work-group size for DFT fallback
            return self.kmgr.fft1d(fft_result, -1, global_size=(fft_size,))

    def batch_forward(self, signals: np.ndarray) -> np.ndarray:
        """
        Compute forward FFT for multiple signals in a single kernel launch.

        This method amortizes GPU launch overhead across many FFTs, providing
        10-100x speedup compared to processing signals individually.

        Args:
            signals: 2D array of complex signals (batch_size, fft_size)
                    Must be np.complex64 or convertible to it

        Returns:
            2D array of FFT results (batch_size, fft_size) as np.complex64

        Example:
            >>> # Process 100 signals of size 256 in one call
            >>> signals = np.random.randn(100, 256).astype(np.complex64)
            >>> fft_results = fft.batch_forward(signals)
            >>> assert fft_results.shape == (100, 256)

        Note:
            - Much faster than calling forward() in a loop
            - Best for processing spectrograms, batch signal processing
            - fft_size must be power of 2 and <= 256 (current limit)
        """
        signals = signals.astype(np.complex64)
        if signals.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {signals.shape}")

        batch_size, fft_size = signals.shape

        # Critical: Set threadgroup size to fft_size so each threadgroup processes one FFT
        return self.kmgr.fft1d_batch(
            signals,
            batch_size,
            fft_size,
            1,
            global_size=(batch_size * fft_size,),
            local_size=(fft_size,),
        )

    def batch_inverse(self, fft_results: np.ndarray) -> np.ndarray:
        """
        Compute inverse FFT for multiple FFT results in a single kernel launch.

        Args:
            fft_results: 2D array of FFT results (batch_size, fft_size)
                        Must be np.complex64 or convertible to it

        Returns:
            2D array of reconstructed signals (batch_size, fft_size) as np.complex64

        Example:
            >>> # Reconstruct 100 signals
            >>> reconstructed = fft.batch_inverse(fft_results)

        Note:
            - Much faster than calling inverse() in a loop
            - Useful for batch signal synthesis
        """
        fft_results = fft_results.astype(np.complex64)
        if fft_results.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {fft_results.shape}")

        batch_size, fft_size = fft_results.shape

        # Critical: Set threadgroup size to fft_size so each threadgroup processes one FFT
        return self.kmgr.fft1d_batch(
            fft_results,
            batch_size,
            fft_size,
            -1,
            global_size=(batch_size * fft_size,),
            local_size=(fft_size,),
        )

    def apply_window(
        self, signal: np.ndarray, window_type: int = WINDOW_HANN
    ) -> np.ndarray:
        """
        Apply window function to a signal.

        Args:
            signal: Input signal (np.float32)
            window_type: Window type (0=rectangular, 1=hann, 2=hamming, 3=blackman)

        Returns:
            Windowed signal (np.float32)

        Example:
            >>> windowed = fft.apply_window(signal, FFT.WINDOW_HANN)
        """
        return self.kmgr.window(signal, window_type)

    def compute_power_spectrum(self, fft_result: np.ndarray) -> np.ndarray:
        """
        Compute power spectrum from FFT result.

        Args:
            fft_result: Complex FFT result (np.complex64)

        Returns:
            Power spectrum (magnitude squared) (np.float32)

        Example:
            >>> power = fft.compute_power_spectrum(fft_result)
        """
        return self.kmgr.power_spectrum(fft_result)

    def analyze_signal(
        self,
        signal: np.ndarray,
        sample_rate: float = 1.0,
        window_type: int = WINDOW_HANN,
        return_db: bool = True,
        positive_only: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform complete spectral analysis of a real signal.

        This is a convenience method that applies windowing, computes FFT,
        and returns the power spectrum with frequency axis.

        Args:
            signal: Real input signal (will be converted to float32)
            sample_rate: Sampling rate in Hz (default: 1.0)
            window_type: Window type (default: WINDOW_HANN)
            return_db: Return power in dB scale (default: True)
            positive_only: Return only positive frequencies (default: True)

        Returns:
            freqs: Frequency array in Hz
            power: Power spectrum (dB if return_db=True, else linear)

        Example:
            >>> freqs, power_db = fft.analyze_signal(
            ...     signal, sample_rate=1000, window_type=FFT.WINDOW_HANN
            ... )
        """
        # Ensure float32
        signal = signal.astype(np.float32)
        n = len(signal)

        # Apply window
        windowed = self.apply_window(signal, window_type)

        # Convert to complex and compute FFT
        complex_signal = windowed.astype(np.complex64)
        fft_result = self.forward(complex_signal)

        # Compute power spectrum
        power = self.compute_power_spectrum(fft_result)

        # Convert to dB if requested
        if return_db:
            power = 10 * np.log10(power + 1e-10)

        # Create frequency axis
        freqs = np.fft.fftfreq(n, 1.0 / sample_rate)

        # Return only positive frequencies if requested
        if positive_only:
            positive_idx = freqs >= 0
            freqs = freqs[positive_idx]
            power = power[positive_idx]

        return freqs, power

    def find_peaks(
        self,
        signal: np.ndarray,
        sample_rate: float = 1.0,
        n_peaks: int = 5,
        window_type: int = WINDOW_HANN,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find the dominant frequency peaks in a signal.

        Args:
            signal: Real input signal
            sample_rate: Sampling rate in Hz
            n_peaks: Number of peaks to return
            window_type: Window type for analysis

        Returns:
            peak_freqs: Frequencies of the peaks (Hz)
            peak_powers: Power values at the peaks (dB)

        Example:
            >>> peak_freqs, peak_powers = fft.find_peaks(signal, sample_rate=1000, n_peaks=3)
            >>> print(f"Dominant frequencies: {peak_freqs} Hz")
        """
        freqs, power_db = self.analyze_signal(
            signal, sample_rate, window_type, return_db=True, positive_only=True
        )

        # Find top N peaks
        peak_indices = np.argsort(power_db)[-n_peaks:][::-1]

        peak_freqs = freqs[peak_indices]
        peak_powers = power_db[peak_indices]

        return peak_freqs, peak_powers

    def spectrogram(
        self,
        signal: np.ndarray,
        sample_rate: float = 1.0,
        window_size: int = 512,
        hop_size: int | None = None,
        window_type: int = WINDOW_HANN,
        return_db: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram (time-frequency representation) of a signal.

        Args:
            signal: Real input signal
            sample_rate: Sampling rate in Hz
            window_size: FFT window size (should be power of 2)
            hop_size: Step size between windows (default: window_size // 4)
            window_type: Window function type
            return_db: Return power in dB scale

        Returns:
            times: Time points for each window (seconds)
            freqs: Frequency array (Hz)
            spectrogram: 2D array of power values (time x frequency)

        Example:
            >>> times, freqs, spec = fft.spectrogram(
            ...     signal, sample_rate=8000, window_size=512
            ... )
            >>> # Plot with matplotlib
            >>> plt.imshow(spec.T, aspect='auto', origin='lower',
            ...            extent=[times[0], times[-1], freqs[0], freqs[-1]])
        """
        # Default hop size is 75% overlap
        if hop_size is None:
            hop_size = window_size // 4

        signal = signal.astype(np.float32)
        n_samples = len(signal)

        # Calculate number of windows
        n_windows = (n_samples - window_size) // hop_size + 1

        # Prepare output arrays
        freqs = np.fft.fftfreq(window_size, 1.0 / sample_rate)[: window_size // 2]
        n_freqs = len(freqs)
        spectrogram = np.zeros((n_windows, n_freqs), dtype=np.float32)
        times = np.zeros(n_windows, dtype=np.float32)

        # Use batch processing for efficiency (70x speedup on Metal, 55x on OpenCL)
        # Only use batch if we have enough windows to amortize overhead
        use_batch = n_windows >= 10 and window_size <= 256

        # Note: Async mode is not beneficial for spectrogram due to data dependencies
        # The batch_forward and compute_power_spectrum return values that are immediately needed
        # The batch processing path already provides significant speedup (70x on Metal)

        if use_batch:
            # Batch processing path - much faster for many windows
            # Pre-extract and window all signals
            windowed_signals = np.zeros((n_windows, window_size), dtype=np.float32)

            for i in range(n_windows):
                start = i * hop_size
                end = start + window_size
                window_signal = signal[start:end]
                times[i] = (start + end) / (2.0 * sample_rate)

                # Apply window function
                windowed_signals[i, :] = self.apply_window(window_signal, window_type)

            # Batch FFT - all windows at once
            complex_signals = windowed_signals.astype(np.complex64)
            fft_results = self.batch_forward(complex_signals)

            # Compute power spectrum for all windows
            for i in range(n_windows):
                power = self.compute_power_spectrum(fft_results[i])
                spectrogram[i, :] = power[:n_freqs]

        else:
            # Sequential processing path - better for few windows or large FFT sizes
            for i in range(n_windows):
                # Extract window
                start = i * hop_size
                end = start + window_size
                window_signal = signal[start:end]

                # Time point is center of window
                times[i] = (start + end) / (2.0 * sample_rate)

                # Apply window function
                windowed = self.apply_window(window_signal, window_type)

                # Compute FFT and power spectrum
                complex_signal = windowed.astype(np.complex64)
                fft_result = self.forward(complex_signal)
                power = self.compute_power_spectrum(fft_result)

                # Store positive frequencies only
                spectrogram[i, :] = power[:n_freqs]

        # Convert to dB if requested
        if return_db:
            spectrogram = 10 * np.log10(spectrogram + 1e-10)

        return times, freqs, spectrogram

    def estimate_frequency(
        self,
        signal: np.ndarray,
        sample_rate: float = 1.0,
        window_type: int = WINDOW_HANN,
    ) -> float:
        """
        Estimate the dominant frequency in a signal.

        Args:
            signal: Real input signal
            sample_rate: Sampling rate in Hz
            window_type: Window type for analysis

        Returns:
            Dominant frequency in Hz

        Example:
            >>> freq = fft.estimate_frequency(signal, sample_rate=1000)
            >>> print(f"Dominant frequency: {freq:.2f} Hz")
        """
        peak_freqs, _ = self.find_peaks(
            signal, sample_rate, n_peaks=1, window_type=window_type
        )
        return peak_freqs[0]

    def bandpass_filter(
        self,
        signal: np.ndarray,
        low_freq: float,
        high_freq: float,
        sample_rate: float = 1.0,
    ) -> np.ndarray:
        """
        Apply a simple bandpass filter in the frequency domain.

        This performs FFT, zeros out frequencies outside the band, and inverse FFT.

        Args:
            signal: Real input signal
            low_freq: Low cutoff frequency (Hz)
            high_freq: High cutoff frequency (Hz)
            sample_rate: Sampling rate in Hz

        Returns:
            Filtered signal (real)

        Example:
            >>> # Keep only 100-200 Hz components
            >>> filtered = fft.bandpass_filter(signal, 100, 200, sample_rate=1000)
        """
        signal = signal.astype(np.float32)
        n = len(signal)

        # Convert to complex and compute FFT
        complex_signal = signal.astype(np.complex64)
        fft_result = self.forward(complex_signal)

        # Create frequency mask
        freqs = np.fft.fftfreq(n, 1.0 / sample_rate)
        mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)

        # Apply mask
        fft_filtered = fft_result * mask.astype(np.complex64)

        # Inverse FFT
        result = self.inverse(fft_filtered)

        # Return real part
        return result.real.astype(np.float32)

    def coherence(
        self, signal1: np.ndarray, signal2: np.ndarray, window_type: int = WINDOW_HANN
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute magnitude-squared coherence between two signals.

        Coherence measures the correlation between two signals in the frequency domain.

        Args:
            signal1: First signal (real)
            signal2: Second signal (real)
            window_type: Window type for analysis

        Returns:
            freqs: Frequency array (normalized to [0, 0.5])
            coherence: Coherence values (0 to 1)

        Example:
            >>> freqs, coh = fft.coherence(signal1, signal2)
            >>> # High coherence (near 1) indicates strong correlation at that frequency
        """
        # Ensure same length
        n = min(len(signal1), len(signal2))
        signal1 = signal1[:n].astype(np.float32)
        signal2 = signal2[:n].astype(np.float32)

        # Apply window to both signals
        windowed1 = self.apply_window(signal1, window_type)
        windowed2 = self.apply_window(signal2, window_type)

        # Compute FFTs
        fft1 = self.forward(windowed1.astype(np.complex64))
        fft2 = self.forward(windowed2.astype(np.complex64))

        # Compute cross-spectrum and auto-spectra
        cross = fft1 * np.conj(fft2)
        auto1 = np.abs(fft1) ** 2
        auto2 = np.abs(fft2) ** 2

        # Coherence = |cross|^2 / (auto1 * auto2)
        coherence = np.abs(cross) ** 2 / (auto1 * auto2 + 1e-10)

        # Frequency axis (normalized)
        freqs = np.fft.fftfreq(n)

        # Return positive frequencies only
        positive_idx = freqs >= 0
        freqs = freqs[positive_idx]
        coherence = coherence[positive_idx]

        return freqs, coherence

    def energy(self, signal: np.ndarray) -> float:
        """
        Compute total energy of a signal.

        Args:
            signal: Real or complex signal

        Returns:
            Total energy (sum of squared magnitudes)

        Example:
            >>> energy = fft.energy(signal)
        """
        return float(np.sum(np.abs(signal) ** 2))

    def rms(self, signal: np.ndarray) -> float:
        """
        Compute root-mean-square (RMS) value of a signal.

        Args:
            signal: Real or complex signal

        Returns:
            RMS value

        Example:
            >>> rms_value = fft.rms(signal)
        """
        return float(np.sqrt(np.mean(np.abs(signal) ** 2)))

    def snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        """
        Compute signal-to-noise ratio in dB.

        Args:
            signal: Signal (real or complex)
            noise: Noise (real or complex)

        Returns:
            SNR in dB

        Example:
            >>> snr_db = fft.snr(clean_signal, noise)
        """
        signal_power = self.energy(signal)
        noise_power = self.energy(noise)
        return float(10 * np.log10(signal_power / (noise_power + 1e-10)))

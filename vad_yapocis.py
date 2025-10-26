"""
GPU-accelerated Voice Activity Detection using yapocis kernels.

This module implements a VAD segmenter that uses yapocis GPU kernels for:
- FFT-based spectrogram computation
- Bandpass variance extraction (formant detection)
- Log-variance normalization
- Feature smoothing

The algorithm is based on segment.py's AudioSegmenter but accelerated with
GPU computation and kernel fusion.
"""

#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

import numpy as np
import scipy.signal as sig
from typing import Optional, Tuple, List

from yapocis import core
from yapocis.lib.fft import FFT
from yapocis.lib.bound_kernels import BoundKernels

# Import minimal dependencies
try:
    # Try importing from segment.py if available
    from segment import (
        AbstractSegmenter, Segmentation, SecToIndex,
        textgrid, Sndfile, find_runs
    )
except ImportError:
    # Provide minimal implementations if segment.py not available
    print("Warning: segment.py not available, using minimal implementations")

    class SecToIndex:
        """Convert between seconds and array indices."""
        def __init__(self, frequency=None, duration=None):
            if frequency:
                self.frequency = frequency
                self.duration = 1.0 / frequency
            else:
                self.frequency = 1.0 / duration
                self.duration = duration

        def index(self, seconds):
            return int(seconds * self.frequency)

        def seconds(self, index):
            return float(index) * self.duration

    class Segmentation:
        """Minimal segmentation result container."""
        def __init__(self, audio_data, samplerate, signal_slices, noise_slices, text_grid=None):
            self.audio_data = audio_data
            self.samplerate = samplerate
            self.signal_slices = signal_slices
            self.noise_slices = noise_slices
            self.tg = text_grid
            if audio_data is not None:
                self.total_seconds = audio_data.shape[0] / float(samplerate)
                self.nsamples = audio_data.shape[0]
            else:
                self.total_seconds = 0
                self.nsamples = 0

        @property
        def signal_seconds(self):
            seconds = 0
            for (l, r) in self.signal_slices:
                seconds += (r - l)
            return seconds

        @property
        def noise_seconds(self):
            seconds = 0
            for (l, r) in self.noise_slices:
                seconds += (r - l)
            return seconds

        def drop_audio(self):
            self.audio_data = None

    class AbstractSegmenter:
        """Minimal base class for segmenters."""
        def __init__(self):
            self.timers = {}
            self.segmentation = None

        def lap_timer(self, label):
            """Context manager for timing operations."""
            import time
            from contextlib import contextmanager

            @contextmanager
            def timer():
                start = time.time()
                yield
                elapsed = time.time() - start
                if label in self.timers:
                    self.timers[label] += elapsed
                else:
                    self.timers[label] = elapsed

            return timer()

        def __call__(self, *args, **kwargs):
            return self.call(*args, **kwargs)

        def call(self, *args, **kwargs):
            raise NotImplementedError

    # Minimal textgrid support
    class textgrid:
        class TextGrid:
            def __init__(self):
                self.tiers = []

            def add_tier(self, tier):
                self.tiers.append(tier)

        class IntervalTier:
            def __init__(self, start, end, name):
                self.start = start
                self.end = end
                self.name = name
                self.intervals = []

            def add_interval(self, interval):
                self.intervals.append(interval)

        class PointTier:
            def __init__(self, start, end, name):
                self.start = start
                self.end = end
                self.name = name
                self.points = []

        class Interval:
            def __init__(self, start, end, label):
                self.start = start
                self.end = end
                self.label = label

    # Minimal audio file support
    try:
        from secondary.audio.sndfile import Sndfile
    except ImportError:
        import wave
        import struct

        class Sndfile:
            """Minimal WAV file reader."""
            def __init__(self, file_or_path, mode='r', **kwargs):
                if mode == 'r':
                    if isinstance(file_or_path, str):
                        self.wav = wave.open(file_or_path, 'rb')
                    else:
                        self.wav = wave.open(file_or_path, 'rb')
                    self.samplerate = self.wav.getframerate()
                    self.channels = self.wav.getnchannels()
                    self.sampwidth = self.wav.getsampwidth()
                    self._len = self.wav.getnframes()
                else:
                    raise NotImplementedError("Write mode not implemented in minimal Sndfile")

            def __len__(self):
                return self._len

            def read(self, nframes, dtype=np.float32):
                """Read audio frames."""
                frames = self.wav.readframes(nframes)
                if self.sampwidth == 2:
                    # 16-bit PCM
                    data = np.frombuffer(frames, dtype=np.int16)
                    data = data.astype(dtype) / 32768.0
                elif self.sampwidth == 4:
                    # 32-bit float
                    data = np.frombuffer(frames, dtype=np.float32)
                    if dtype != np.float32:
                        data = data.astype(dtype)
                else:
                    raise ValueError(f"Unsupported sample width: {self.sampwidth}")

                if self.channels > 1:
                    data = data.reshape(-1, self.channels)[:, 0]  # Take first channel

                return data

            def close(self):
                if hasattr(self, 'wav'):
                    self.wav.close()

    def find_runs(binary_array):
        """Find contiguous runs of True values in a boolean array."""
        # Pad with False at both ends
        padded = np.concatenate([[False], binary_array, [False]])
        # Find edges
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        return list(zip(starts, ends))


class AudioKernels(BoundKernels):
    """High-level interface to audio processing kernels."""
    INTERFACE = "audio"


class YapocisAudioSegmenter(AbstractSegmenter):
    """
    GPU-accelerated audio segmentation using yapocis kernels.

    This implementation replaces scipy-based spectral analysis with:
    - FFT library for spectrogram computation (with batch processing for 55-70x speedup)
    - Fused GPU kernels for variance extraction and normalization
    - GPU-accelerated box filter smoothing (when win parameter is provided)

    The segmentation algorithm remains compatible with AudioSegmenter but
    provides significant speedup for large audio files.

    Performance Notes:
        - Batch FFT provides 55-70x speedup for spectrogram computation
        - For frequency_buckets <= 256: All backends work well
        - For frequency_buckets > 256: Recommend Metal or Numbax
        - OpenCL has optimal performance for FFT sizes <= 256

    Backend Recommendations:
        - Metal: Best for batch operations (70x speedup), optimal for macOS
        - OpenCL: Good for sizes <= 256, limited for larger FFTs
        - Numbax: Excellent for CPU-only or small FFTs
        - CUDA: Good for NVIDIA GPUs (similar to Metal performance)

    Args:
        min_formant_duration: Minimum duration (sec) for a formant to be considered speech
        max_gap: Maximum gap (sec) between formants to bridge
        front_porch: Time (sec) to add before detected speech
        back_porch: Time (sec) to add after detected speech
        formant_band: Frequency range (Hz) for formant detection (default: 200-4800 Hz)
        log_var_formant_threshold: Percentile threshold for log-variance (0-1)
        frequency_buckets: Number of FFT bins for spectrogram (default: 1024)
            Note: For OpenCL, values <= 256 are optimal. For larger values, use Metal or Numbax.
        win: Optional smoothing window (NumPy array). If provided, a GPU box filter
            of the same size is applied. If None, no smoothing is applied.
        use_kernel_fusion: If True, use fused VAD kernel (default: True)
    """

    SPEC_FRAME_DURATION = 0.032

    def __init__(
        self,
        min_formant_duration: float = 0.1,
        max_gap: float = 0.25,
        front_porch: float = 0.25,
        back_porch: float = 0.25,
        formant_band: Tuple[int, int] = (200, 4800),
        log_var_formant_threshold: float = 0.4,
        frequency_buckets: int = 1024,
        win: Optional[np.ndarray] = None,
        use_kernel_fusion: bool = True
    ):
        self.min_formant_duration = min_formant_duration
        self.max_gap = max_gap
        self.front_porch = front_porch
        self.back_porch = back_porch
        self.formant_band = formant_band
        self.logvar_formant_threshold = log_var_formant_threshold
        self.frequency_buckets = frequency_buckets
        self.win = win
        self.use_kernel_fusion = use_kernel_fusion
        self.threshold = None
        self.segmentation = None

        # Initialize yapocis kernels
        self.fft = FFT()
        self.audio_kernels = AudioKernels()

        super(YapocisAudioSegmenter, self).__init__()

        # Check for non-optimal configuration
        import os
        backend = os.environ.get("YAPOCIS_BACKEND", "numbax")
        if backend == "ocl" and frequency_buckets > 256:
            import warnings
            warnings.warn(
                f"OpenCL FFT performance is suboptimal for frequency_buckets={frequency_buckets} (> 256). "
                "For best performance, use Metal (YAPOCIS_BACKEND=metal) or reduce frequency_buckets to 256.",
                UserWarning
            )

    def drop_audio(self):
        """Drop references to large data buffers."""
        self.speech = self.filtered = self.rspeech = self.logvar = self.spectrum = None
        if self.segmentation:
            self.segmentation.drop_audio()

    def load_audio(self, audio: str) -> np.ndarray:
        """Load audio file and set up frequency parameters."""
        with self.lap_timer("load audio"):
            sf = Sndfile(open(audio, "rb"))
            self.samplerate = sf.samplerate
            nsamples = len(sf)
            self.total_duration = nsamples / float(self.samplerate)

            # Set frequency range for formants
            self.lower = int(self.formant_band[0] / (self.samplerate / float(self.frequency_buckets)))
            self.higher = int(self.formant_band[1] / (self.samplerate / float(self.frequency_buckets)))

            audio_data = sf.read(nsamples, dtype=np.float32)
            return audio_data

    def compute_spectrogram_yapocis(self, audio_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram using yapocis FFT library.

        Returns:
            spectrum: 2D power spectrum (freq x time)
            freqs: Frequency array
            t: Time array
        """
        with self.lap_timer("compute spectrogram (GPU)"):
            # Add small jitter to avoid zero variance (same as original)
            JITTERFRAME = 12345
            jitter = np.random.random(min(len(audio_data), JITTERFRAME)) * 0.00001
            jitter = np.array(jitter, audio_data.dtype)

            # Add noise in place
            len_data = len(audio_data)
            mangle_frame = len(jitter)
            for i in range(0, len_data, mangle_frame):
                njitter = min(mangle_frame, len_data - i)
                audio_data[i:i + njitter] += jitter[:njitter]

            # Compute spectrogram using yapocis FFT
            window_size = self.frequency_buckets
            hop_size = self.frequency_buckets // 2

            times, freqs, spec = self.fft.spectrogram(
                audio_data,
                sample_rate=self.samplerate,
                window_size=window_size,
                hop_size=hop_size,
                window_type=FFT.WINDOW_HANN,
                return_db=False  # We want linear power for variance computation
            )

            # Remove the noise
            for i in range(0, len_data, mangle_frame):
                njitter = min(mangle_frame, len_data - i)
                audio_data[i:i + njitter] -= jitter[:njitter]

            # Convert to format expected by downstream processing
            # yapocis returns (time x freq), we need (freq x time)
            spectrum = spec.T

            # Convert to dB scale for compatibility
            spectrum = 10 * np.log10(spectrum + 1e-10)

            return spectrum, freqs, times

    def compute_vad_features_fused(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Compute VAD features using fused GPU kernel.

        This combines bandpass extraction, variance computation, log transform,
        and normalization in a single GPU kernel for efficiency.
        """
        with self.lap_timer("compute VAD features (fused GPU kernel)"):
            # Prepare spectrum for GPU (needs to be float32, C-contiguous)
            spectrum = np.ascontiguousarray(spectrum, dtype=np.float32)
            height, width = spectrum.shape

            # Allocate output
            features = np.empty(width, dtype=np.float32)

            # Run fused kernel
            self.audio_kernels.kmgr.fused_vad_features(
                spectrum,
                self.lower, self.higher,
                width, height,
                features
            )

            # Normalize to [0, 1]
            min_val = features.min()
            max_val = features.max()
            self.logvar_max = max_val

            if max_val - min_val > 1e-10:
                features = (features - min_val) / (max_val - min_val)
            else:
                features = np.zeros_like(features)

            return features

    def compute_vad_features_separate(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Compute VAD features using separate kernel calls.

        This is less efficient but more flexible for debugging.
        """
        with self.lap_timer("compute VAD features (separate kernels)"):
            # Extract bandpass region
            bandpass = spectrum[self.lower:self.higher, :]

            # Compute variance on CPU (could be GPU kernel)
            var = bandpass.var(axis=0).astype(np.float32)

            # Apply log and normalize on GPU
            logvar = np.log(var + 1e-10).astype(np.float32)
            min_val = logvar.min()
            max_val = logvar.max()
            self.logvar_max = max_val

            if max_val - min_val > 1e-10:
                logvar = (logvar - min_val) / (max_val - min_val)
            else:
                logvar = np.zeros_like(logvar)

            return logvar

    def smooth_features(self, features: np.ndarray) -> np.ndarray:
        """Apply smoothing to VAD features using GPU smooth_1d kernel."""
        if self.win is not None:
            # Use GPU box filter with window size from provided window
            window_size = len(self.win)
            with self.lap_timer("smooth features (GPU)"):
                # Ensure features are float32 for GPU
                features_f32 = features.astype(np.float32)
                smoothed = self.audio_kernels.kmgr.smooth_1d(features_f32, window_size)
            return smoothed
        else:
            # No smoothing
            return features

    def find_speech(self, audio_data: np.ndarray) -> List[Tuple[float, float]]:
        """
        Detect speech regions using thresholded log-variance.

        Returns:
            List of (start_sec, end_sec) tuples for detected speech
        """
        with self.lap_timer("threshold formants"):
            # Threshold based on percentile
            v = list(self.logvar)
            threshold_index = int(len(v) * self.logvar_formant_threshold)
            self.threshold = sorted(v)[threshold_index]

            # Apply smoothing
            self.filtered = self.smooth_features(self.logvar)

            # Binarize based on threshold
            speech = np.zeros_like(self.logvar)
            above_threshold = self.filtered > self.threshold
            speech[np.where(above_threshold)] = 1.0
            self.speech = speech

            # Find runs of speech
            signal_slices = find_runs(above_threshold)

            # Convert to seconds
            signal_slices = [
                (self.spectrumc.seconds(l), self.spectrumc.seconds(r))
                for (l, r) in signal_slices
            ]

        return signal_slices

    def create_textgrid(self):
        """Create Praat TextGrid annotations."""
        with self.lap_timer("create textgrid"):
            self.tg = tg = textgrid.TextGrid()
            self.formant_tier = textgrid.IntervalTier(0, self.total_duration, "svFormant")
            tg.add_tier(self.formant_tier)
            self.signal_tier = textgrid.IntervalTier(0, self.total_duration, "svSignal")
            tg.add_tier(self.signal_tier)
            self.noise_tier = textgrid.IntervalTier(0, self.total_duration, "svNoise")
            tg.add_tier(self.noise_tier)
            tg.add_tier(textgrid.PointTier(0, self.total_duration, "svText"))

    def segment_merge(self, signal_slices: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Post-process speech segments: remove short segments, bridge gaps, add porches.
        """
        with self.lap_timer("segment and merge"):
            # Add annotations for raw detections
            for (start, stop) in signal_slices:
                self.formant_tier.add_interval(textgrid.Interval(start, stop, "F"))

            # Remove very short formant peaks
            signal_slices = [[l, r] for (l, r) in signal_slices if r - l > self.min_formant_duration]
            if len(signal_slices) == 0:
                return []

            # Bridge short gaps
            bridged = [signal_slices[0]]
            for l, r in signal_slices[1:]:
                if l - bridged[-1][1] <= self.max_gap:
                    bridged[-1][1] = r
                else:
                    bridged.append([l, r])

            # Add front and back porch
            bridged = [[max(0.0, l - self.front_porch), r + self.back_porch] for (l, r) in bridged]
            signal_slices = bridged

            # Merge overlapping segments
            bridged = [signal_slices[0]]
            for l, r in signal_slices[1:]:
                if l <= bridged[-1][1]:
                    bridged[-1][1] = max(bridged[-1][1], r)
                else:
                    bridged.append([l, r])

            signal_slices = [(round(l, 4), round(r, 4)) for (l, r) in bridged]

            # Create binary speech mask
            rspeech = np.zeros_like(self.logvar)
            for (start, stop) in signal_slices:
                self.signal_tier.add_interval(textgrid.Interval(start, stop, "S"))
                l, r = self.spectrumc.index(start), self.spectrumc.index(stop)
                rspeech[l:r] = 1.0
            self.rspeech = rspeech

            if self.speech is None:
                self.speech = rspeech

        return signal_slices

    def extract_noise(self, signal_slices: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Extract non-speech (noise) regions."""
        with self.lap_timer("extract non speech"):
            noise_slices = []
            lastr = 0.0
            for start, stop in signal_slices:
                self.noise_tier.add_interval(textgrid.Interval(lastr, start, "N"))
                noise_slices.append((lastr, start))
                lastr = stop

            if lastr < self.total_duration:
                self.noise_tier.add_interval(textgrid.Interval(lastr, self.total_duration, "N"))
                noise_slices.append((lastr, self.total_duration))

        return noise_slices

    def call(self, audio: str) -> Segmentation:
        """
        Run VAD on audio file.

        Args:
            audio: Path to audio file

        Returns:
            Segmentation object with detected speech/noise regions
        """
        self.audio = audio

        # Load audio
        audio_data = self.load_audio(audio)

        # Compute spectrogram using yapocis FFT
        spectrum, freqs, t = self.compute_spectrogram_yapocis(audio_data)
        self.spectrum = spectrum
        self.freqs = freqs
        self.t = t

        # Determine frame duration
        try:
            spec_frame_duration = t[1] - t[0]
        except IndexError:
            spec_frame_duration = self.SPEC_FRAME_DURATION
        self.spec_frame_duration = spec_frame_duration

        # Compute VAD features (log-variance of bandpass power)
        if self.use_kernel_fusion:
            self.logvar = self.compute_vad_features_fused(spectrum)
        else:
            self.logvar = self.compute_vad_features_separate(spectrum)

        # Check if audio has meaningful content
        # Note: logvar_max is the raw (un-normalized) maximum
        # The threshold should be relative to the FFT size since larger FFTs
        # spread power across more bins, reducing variance per bin
        # A more robust check: if the range is too small, there's no variation
        logvar_range = self.logvar.max() - self.logvar.min() if len(self.logvar) > 0 else 0
        if logvar_range < 1e-6:
            # No variation in log-variance - likely silence or uniform noise
            self.create_textgrid()
            self.segmentation = Segmentation(
                audio_data, self.samplerate, [], [], self.tg
            )
            return self.segmentation

        # Set up time conversion
        self.spectrumc = SecToIndex(duration=self.spec_frame_duration)

        # Find speech regions
        signal_slices = self.find_speech(audio_data)

        # Create TextGrid
        self.create_textgrid()

        # Post-process segments
        signal_slices = self.segment_merge(signal_slices)

        # Extract noise regions
        noise_slices = self.extract_noise(signal_slices)

        # Create segmentation result
        self.segmentation = Segmentation(
            audio_data, self.samplerate,
            signal_slices, noise_slices,
            self.tg
        )

        return self.segmentation


# Convenience function
def segment_audio_gpu(
    audio_path: str,
    **kwargs
) -> Segmentation:
    """
    Segment audio file using GPU-accelerated VAD.

    Args:
        audio_path: Path to audio file
        **kwargs: Additional arguments passed to YapocisAudioSegmenter

    Returns:
        Segmentation object

    Example:
        >>> seg = segment_audio_gpu("speech.wav", formant_band=(300, 3400))
        >>> print(f"Found {len(seg.signal_slices)} speech segments")
        >>> print(f"Total speech: {seg.signal_seconds:.2f}s")
    """
    segmenter = YapocisAudioSegmenter(**kwargs)
    return segmenter(audio_path)

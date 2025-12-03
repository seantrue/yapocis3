# FFT

GPU-accelerated Fast Fourier Transform for signal processing.

## Overview

The `FFT` class provides GPU-accelerated FFT operations including forward/inverse transforms, spectral analysis, windowing, and spectrogram generation.

## Class Reference

```python
from yapocis.lib.fft import FFT

fft = FFT()
```

## Window Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `FFT.WINDOW_RECTANGULAR` | 0 | No windowing |
| `FFT.WINDOW_HANN` | 1 | Hann window (default) |
| `FFT.WINDOW_HAMMING` | 2 | Hamming window |
| `FFT.WINDOW_BLACKMAN` | 3 | Blackman window |

## Core Methods

### forward

Compute forward FFT.

```python
fft_result = fft.forward(signal)
```

**Parameters:**
- `signal` - Complex input signal (np.complex64)

**Returns:** Complex FFT result (np.complex64)

### inverse

Compute inverse FFT.

```python
signal = fft.inverse(fft_result)
```

### batch_forward / batch_inverse

Process multiple signals in parallel.

```python
# Process 100 signals of size 256
signals = np.random.randn(100, 256).astype(np.complex64)
fft_results = fft.batch_forward(signals)
```

10-100x faster than processing individually.

## Analysis Methods

### analyze_signal

Complete spectral analysis of a real signal.

```python
freqs, power = fft.analyze_signal(
    signal,
    sample_rate=1.0,
    window_type=FFT.WINDOW_HANN,
    return_db=True,
    positive_only=True
)
```

**Returns:** (frequencies, power_spectrum) arrays

### find_peaks

Find dominant frequency peaks.

```python
peak_freqs, peak_powers = fft.find_peaks(
    signal,
    sample_rate=1000,
    n_peaks=5
)
```

### estimate_frequency

Estimate dominant frequency.

```python
freq = fft.estimate_frequency(signal, sample_rate=1000)
```

### spectrogram

Compute time-frequency representation.

```python
times, freqs, spec = fft.spectrogram(
    signal,
    sample_rate=8000,
    window_size=512,
    hop_size=128,
    window_type=FFT.WINDOW_HANN,
    return_db=True
)
```

## Signal Processing Methods

### apply_window

Apply window function.

```python
windowed = fft.apply_window(signal, FFT.WINDOW_HANN)
```

### compute_power_spectrum

Get power spectrum from FFT result.

```python
power = fft.compute_power_spectrum(fft_result)
```

### bandpass_filter

Simple frequency-domain bandpass filter.

```python
filtered = fft.bandpass_filter(signal, low_freq=100, high_freq=500, sample_rate=1000)
```

### coherence

Compute coherence between two signals.

```python
freqs, coh = fft.coherence(signal1, signal2)
```

## Utility Methods

### energy

Total signal energy.

```python
e = fft.energy(signal)  # Sum of squared magnitudes
```

### rms

Root-mean-square value.

```python
rms_val = fft.rms(signal)
```

### snr

Signal-to-noise ratio in dB.

```python {.no-exec}
snr_db = fft.snr(clean_signal, noise)
```

## Usage Examples

### Basic Spectral Analysis

```python
import numpy as np
from yapocis.lib.fft import FFT

# Create test signal with 100 Hz and 250 Hz components
sample_rate = 1000
t = np.linspace(0, 1, sample_rate).astype(np.float32)
signal = np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 250 * t)

# Analyze
fft = FFT()
freqs, power_db = fft.analyze_signal(signal, sample_rate=sample_rate)

# Find peaks
peak_freqs, _ = fft.find_peaks(signal, sample_rate=sample_rate, n_peaks=2)
print(f"Detected frequencies: {peak_freqs}")  # Should be ~100 Hz and ~250 Hz
```

### Audio Spectrogram

```python
from yapocis.lib.fft import FFT
import matplotlib.pyplot as plt

fft = FFT()

# Load or generate audio signal
sample_rate = 8000
signal = np.random.randn(sample_rate * 5).astype(np.float32)  # 5 seconds

# Compute spectrogram
times, freqs, spec = fft.spectrogram(
    signal,
    sample_rate=sample_rate,
    window_size=256,
    hop_size=64
)

# Plot
plt.imshow(spec.T, aspect='auto', origin='lower',
           extent=[times[0], times[-1], freqs[0], freqs[-1]])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Power (dB)')
plt.show()
```

### Noise Filtering

```python
import numpy as np
from yapocis.lib.fft import FFT

fft = FFT()

# Create time array
sample_rate = 1000
t = np.linspace(0, 1, sample_rate).astype(np.float32)

# Signal with noise
clean = np.sin(2 * np.pi * 100 * t)
noisy = clean + 0.5 * np.random.randn(len(t)).astype(np.float32)

# Keep only 80-120 Hz
filtered = fft.bandpass_filter(noisy, 80, 120, sample_rate=sample_rate)
print(f"Filtered signal shape: {filtered.shape}")
```

## Performance Notes

- FFT sizes up to 256 use optimized local memory path
- Larger sizes fall back to DFT
- Batch processing amortizes GPU launch overhead
- OpenCL backend uses `-cl-opt-disable` to work around Apple GPU bugs

## Underlying Kernel: fft

The `FFT` class uses the `fft` interface.

### Interface Definition

```
interface fft {
    # 1D FFT (direction: 1=forward, -1=inverse)
    kernel fft1d(in complex64 input, length_of int input, out_like input,
                 int direction);
    extension res as fft1d(resident complex64 input, length_of int input,
                           resident complex64 output, int direction);

    # Batch 1D FFT
    kernel fft1d_batch(in complex64 input, int batch_size, int fft_size,
                       int direction, out_like input);

    # Apply window function (0=rect, 1=hann, 2=hamming, 3=blackman)
    kernel window(in float input, int window_type, out_like input);
    extension res as window(resident float input, int window_type,
                            resident float output);

    # Compute power spectrum from FFT output
    kernel power_spectrum(in complex64 fft_output, out_like float fft_output);
    extension res as power_spectrum(resident complex64 fft_output,
                                    resident float power);
};
```

### Kernel Parameters

**fft1d:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `complex64*` | Input complex signal |
| `length` | `int` | Signal length (derived) |
| `output` | `complex64*` | Output FFT result |
| `direction` | `int` | 1 = forward, -1 = inverse |

**window:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `float*` | Input signal |
| `window_type` | `int` | 0=rect, 1=hann, 2=hamming, 3=blackman |
| `output` | `float*` | Windowed signal |

### Direct Kernel Access

```python
from yapocis.core import load_program
import numpy as np

# Load the fft interface
prog = load_program("fft")

# Forward FFT (positional args)
signal = np.random.randn(256).astype(np.complex64)
fft_result = prog.fft1d(signal, 1)  # direction=1

# Inverse FFT
reconstructed = prog.fft1d(fft_result, -1)  # direction=-1

# Batch FFT
batch = np.random.randn(100, 256).astype(np.complex64)
batch_fft = prog.fft1d_batch(batch, 100, 256, 1)

# Apply window
real_signal = np.random.randn(256).astype(np.float32)
windowed = prog.window(real_signal, 1)  # Hann window

# Power spectrum
power = prog.power_spectrum(fft_result)
```

## See Also

- {doc}`convolve-kernels` - 1D convolution kernels

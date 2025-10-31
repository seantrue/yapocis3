# Additional Async Pipeline Candidates

## Analysis of yapocis/lib for Async Opportunities

Examined all libraries excluding CNN-based models (vgg16, resnet50, retinanet).

## High Priority Candidates

### 1. Median Filter (`median.py`) ⭐⭐⭐

**Function**: `median3x3fast()`

**Current code**:
```python
def median3x3fast(self, image: Image1, iterations: int = 1) -> Image1:
    m3 = self.kmgr.median3x3_res
    input = image
    output = np.empty_like(input)
    m3.write(input)
    m3.write(output)
    while True:
        m3(input, output, resident=(input, output))
        iterations -= 1
        if iterations <= 0:
            m3.read(output)
            return output.copy()
        input, output = output, input
```

**Async opportunity**:
- Loop with `iterations` kernel calls
- Uses resident buffers (perfect for async)
- Ping-pong pattern already optimized
- **Expected speedup**: 1.2-1.5x for 5+ iterations

**Implementation**:
```python
use_async = os.environ.get('YAPOCIS_ASYNC', '').lower() in ('1', 'true', 'yes')

if use_async:
    with self.kmgr.async_mode():
        for i in range(iterations):
            m3(input, output, resident=(input, output))
            input, output = output, input
            iterations -= 1
```

**Complexity**: Low (very similar to anisodiff pattern)

---

### 2. FFT Spectrogram (`fft.py`) ⭐⭐⭐

**Function**: `spectrogram()`

**Current code** (line 363-410):
```python
# Batch processing path - much faster for many windows
windowed_signals = np.zeros((n_windows, window_size), dtype=np.float32)

for i in range(n_windows):
    start = i * hop_size
    window_signal = signal[start:start + window_size]
    windowed_signals[i] = window_signal

# Apply windowing function to all signals at once
self.apply_window(windowed_signals, window_type)

# Batch FFT
complex_signals = np.zeros((n_windows, window_size), dtype=np.complex64)
complex_signals[:, :] = windowed_signals[:]
fft_results = self.batch_forward(complex_signals)

# Extract power spectrum for each window
for i in range(n_windows):
    power = self.compute_power_spectrum(fft_results[i:i+1])
    spectrogram[i, :] = power[0, :n_freqs]
```

**Async opportunity**:
- Multiple kernel calls in sequence:
  1. apply_window (1 kernel)
  2. batch_forward (1 kernel for all windows)
  3. compute_power_spectrum (loop of n_windows kernels)
- Sequential processing of independent windows
- **Expected speedup**: 1.3-1.6x for multi-window analysis

**Implementation**:
```python
use_async = os.environ.get('YAPOCIS_ASYNC', '').lower() in ('1', 'true', 'yes')

if use_async:
    with self.kmgr.async_mode():
        self.apply_window(windowed_signals, window_type)
        complex_signals[:, :] = windowed_signals[:]
        fft_results = self.batch_forward(complex_signals)
        for i in range(n_windows):
            power = self.compute_power_spectrum(fft_results[i:i+1])
            spectrogram[i, :] = power[0, :n_freqs]
```

**Complexity**: Medium (multiple kernel types, needs careful testing)

---

### 3. Graphics Pipeline (`graphics4.py`) ⭐⭐

**Note**: Need to examine the file, but graphics operations often have multiple sequential kernels

**Likely opportunities**:
- Drawing multiple shapes (circles, lines, rectangles)
- Text rendering with multiple glyphs
- Compositing operations

**Expected speedup**: 1.2-1.4x for multi-shape scenes

**Complexity**: Medium (need to examine actual code patterns)

---

## Medium Priority Candidates

### 4. HSI Color Space Conversion (`hsi.py`) ⭐

**Functions**: `rgb2hsi3()`, `hsi2rgb3()`

**Current code**:
```python
def rgb2hsi3(self, rgb: Image3) -> Image3:
    hsi = self.kmgr.rgb2hsi3(rgb, global_size=rgb.size // rgb.shape[-1])
    hsi = hsi.copy()
    self.purge()
    return hsi
```

**Async opportunity**:
- Single kernel calls
- **Low benefit** unless used in sequence with other operations
- More valuable when combined in a pipeline (e.g., with anisodiff_rgbI)

**Expected speedup**: 1.0-1.05x (minimal - single kernel)

**Priority**: Low standalone, but anisodiff already uses this in pipeline

---

### 5. Colormath Transforms (`colormath.py`) ⭐

**Functions**: Multiple color space transforms

**Async opportunity**:
- Single kernel calls per transform
- **Low benefit** unless multiple transforms in sequence
- Useful for batch color correction workflows

**Expected speedup**: 1.0-1.05x per transform

**Priority**: Low (single kernels)

---

## Low Priority / No Benefit

### 6. Gradient (`gradient.py`) ❌

**Reason**: Single kernel call only
```python
def gradient(self, a, stride=3):
    return self.kmgr.gradient(a, stride)
```

**Async benefit**: None (overhead > benefit)

---

### 7. Operators (`operators.py`) ❌

**Reason**: Simple arithmetic operations (add, subtract, multiply)

**Async benefit**: None (single kernel calls)

---

### 8. Zero-Crossing (`zc.py`) ❌

**Reason**: Single kernel operations

**Async benefit**: None unless used in pipeline (already optimized in gaussian.zcsdog)

---

### 9. Mandelbrot (`mandelbrot.py`) ❌

**Reason**: Single compute-intensive kernel

**Async benefit**: None (GPU already fully utilized)

---

## Already Optimized ✅

### 10. Anisodiff (`anisodiff.py`) ✅

**Status**: Already implemented (this session)

**Speedup achieved**: 1.09-1.31x

---

### 11. Gaussian (`gaussian.py`) ✅

**Status**: Already implemented (this session)

**Functions optimized**: `gauss_image()`, `zcsdog()`

**Speedup achieved**: 1.02-1.39x

---

## Recommended Implementation Order

### Phase 1: High Value, Low Complexity
1. ✅ **Anisodiff** - DONE (1.31x speedup)
2. ✅ **Gaussian** - DONE (1.39x speedup on zcsdog)
3. **Median3x3** - Similar to anisodiff, easy implementation
4. **FFT Spectrogram** - Good speedup potential

### Phase 2: Graphics and Visualization
5. **Graphics4** - Need to examine code first
6. **Graphics** - Need to examine code first

### Phase 3: Deferred (CNN-based)
- VGG16
- ResNet50
- RetinaNet

### Not Worth Implementing
- Single kernel operations (gradient, colormath, hsi, zc, operators)
- Compute-bound operations (mandelbrot)

---

## Implementation Template

For future async additions:

```python
import os

class YourClass(BoundKernels):
    def your_method(self, image, iterations=10):
        # Check if async mode is enabled
        use_async = os.environ.get('YAPOCIS_ASYNC', '').lower() in ('1', 'true', 'yes')

        # Setup buffers
        kernel = self.kmgr.your_kernel
        input = image
        output = np.empty_like(input)
        kernel.write(input)
        kernel.write(output)

        if use_async:
            # Async path: batch all iterations
            with self.kmgr.async_mode():
                for i in range(iterations):
                    kernel(input, output, resident=(input, output))
                    input, output = output, input
        else:
            # Sync path: traditional
            for i in range(iterations):
                kernel(input, output, resident=(input, output))
                input, output = output, input

        kernel.read(output)
        return output
```

---

## Summary

**High priority** (recommended next):
- Median3x3 (easy, good speedup)
- FFT Spectrogram (moderate complexity, good speedup)

**Medium priority**:
- Graphics pipelines (need examination)

**Low priority**:
- Single kernel operations (minimal benefit)

**Deferred**:
- CNN models (per user request)

**Expected overall impact**: 1.2-1.5x speedup on iterative filters and multi-kernel pipelines when `YAPOCIS_ASYNC=1` is enabled.

# Examples

Ready-to-run example scripts demonstrating yapocis features.

## Quick Start Examples

### simple_yapocis.py

Minimal example showing the yapocis programming model:

```python
import numpy as np
from yapocis.ocl.kernel_manager import load_program

# Create arrays
a = np.random.rand(50000)
b = np.random.rand(50000).astype(np.float32)

# Load and run kernel
demo = load_program("demo")
a_plus_b = demo.sum(a, b)

# Verify result
print(np.linalg.norm(a_plus_b - (a + b)))  # Should be ~0
```

Key concepts:

- `load_program()` compiles and loads a kernel interface
- Arrays are automatically transferred to/from GPU
- Results match NumPy operations

### simple_numpy.py

Shows direct OpenCL kernel calls without the library abstraction.

### simple_openclose.py

Demonstrates explicit buffer management for fine-grained control.

## Image Processing Examples

### gaussians.py

Comprehensive image processing pipeline demonstrating:

- Median filtering for noise removal
- HSI color space conversion
- Gaussian blur at multiple scales
- Difference of Gaussians (DoG) edge detection
- Gradient magnitude and direction

```python {.no-exec}
from yapocis.lib.gaussian import Gaussian
from yapocis.lib.gradient import Gradient
from yapocis.lib.hsi import HSI
from yapocis.lib.median import Median

# Initialize processors
median = Median()
gradient = Gradient()
hsi = HSI()
gaussian = Gaussian()

# Process image
filtered = median.median3x3(noisy_image, iterations=100)
h, s, i = hsi.rgb2hsi(r, g, b)
blur = gaussian.gauss_image(i, scale=3)
grad, angle = gradient.gradient(i, stride=5)
```

## Fractal Examples

### mandelbrot.py

GPU-accelerated Mandelbrot set computation with Tkinter visualization.

Demonstrates:

- Using `import_program()` to import kernel as a Python function
- Comparison between NumPy, serial Python, and GPU implementations
- Interactive fractal exploration

```python {.no-exec}
from yapocis.ocl.kernel_manager import import_program

# Import kernel as Python callable
import_program("mandelbrot", mandelbrot="calc_fractal_opencl")

# Use like a regular function
output = calc_fractal_opencl(complex_coords, maxiter=256)
```

**Performance comparison (1024x1024):**

- Serial Python: ~10 seconds
- Vectorized NumPy: ~0.5 seconds
- GPU (OpenCL): ~0.02 seconds

## FFT Examples

### fft_quickstart.py

Minimal FFT example analyzing a two-tone signal:

```python {.no-exec}
from yapocis.lib.fft import FFT

fft = FFT()

# Create test signal (440 Hz + 880 Hz)
signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)

# One-liner analysis
freqs, power_db = fft.analyze_signal(signal, sample_rate=8000)

# Find peaks
peak_freqs, peak_powers = fft.find_peaks(signal, sample_rate, n_peaks=2)
```

### fft_power_spectrum.py

Detailed power spectrum analysis with windowing options.

![FFT: Power Spectrum](/_static/images/fft_library_demo_spectrogram.png)

### fft_chirp_analysis.py

Analyze frequency-varying (chirp) signals using short-time FFT.

![FFT: Chirp](/_static/images/fft_library_demo_filter.png)

### fft_library_demo.py

Comprehensive FFT library demonstration covering all features.

![FFT: Demo](/_static/images/fft_library_demo_basic.png)

## Performance Benchmarks

### median_performance.py

Multi-backend median filter benchmark:

```bash
python examples/median_performance.py
python examples/median_performance.py --backends ocl,metal
python examples/median_performance.py --devices "Apple M4"
```

**Sample output:**

![Median Performance](/_static/images/median_performance.png)

Features:

- Automatic backend/device discovery
- Comparison against scipy baseline
- Async resident buffer optimization
- Seaborn visualization

### zcs_performance.py

Zero-crossing detection performance across backends and image sizes.

![ZCS: Performance](/_static/images/zcs_performance.png)

## Matplotlib Backend Examples

Yapocis includes a GPU-accelerated Matplotlib backend.

### matplotlib_markers.py

Demonstrates scatter plot marker rendering:

```python {.no-exec}
import matplotlib

matplotlib.use("module://yapocis.lib.mplbackend")
import matplotlib.pyplot as plt

# Now plots are GPU-accelerated
plt.scatter(x, y, s=200, c=z, marker="o")
plt.savefig("matplotlib_markers.png")
```

![Matplotlib Markers](/_static/images/matplotlib_markers.png)

### matplotlib_filled_paths.py

Complex path rendering with fills and strokes.

![Matplotlib Filled Paths](/_static/images/matplot_filled_paths.png)

### matplotlib_histogram.py

Histogram rendering demonstration.

![Matplotlib Histogram](/_static/images/matplotlib_histogram.png)

## Neural Network Examples

### vgg16.py

VGG16 image classification on GPU:

```python {.no-exec}
from yapocis.lib.vgg16 import VGG16

model = VGG16()
model.load_weights("models/vgg16/weights.npz")

# Classify image
predictions = model.classify(image)
top5 = model.top_k(predictions, k=5)
```

### resnet50.py

ResNet50 inference with batch normalization fusion.

## Convolution Kernel Examples

### demo_convolve_kernels.py

Demonstrates 1D convolution kernel generation and application:

- Hamming, Hann, Blackman windows
- Box filters
- Custom kernel weights
- Performance comparison

## Running Examples

Most examples can be run directly:

```bash
# From repository root
python examples/mandelbrot.py
python examples/gaussians.py
python examples/fft_quickstart.py

# Performance benchmarks
python examples/median_performance.py --backends metal,ocl
python examples/zcs_performance.py
```

## Example Output Files

Several examples generate output files in the `examples/` directory:

| File                       | Description                   |
|----------------------------|-------------------------------|
| `median_performance.png`   | Median filter benchmark chart |
| `zcs_performance.png`      | Zero-crossing benchmark chart |
| `matplotlib_markers.png`   | Scatter plot markers          |
| `matplot_filled_paths.png` | Path rendering                |
| `matplotlib_histogram.png` | Histogram rendering           |

## See Also

- {doc}`../quickstart/index` - Getting started guide
- {doc}`../library/index` - Library reference
- {doc}`../cli/index` - Command-line tools

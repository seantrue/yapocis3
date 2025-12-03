# Working with Images

Yapocis is designed for image processing. This guide covers loading, processing, and saving images.

## Loading Images

### Using the Test Image

For documentation examples and testing:

```python
from yapocis.utils.basic import get_test_image

# Returns a float32 RGB array normalized to [0, 1]
img = get_test_image()
print(f"Shape: {img.shape}")  # (height, width, 3)
print(f"Dtype: {img.dtype}")  # float32
print(f"Range: [{img.min():.2f}, {img.max():.2f}]")  # [0.00, 1.00]
```

### Using Asset Managers

The asset managers provide flexible ways to find and load images:

```python
from yapocis.images import find_image, images
from yapocis.utils.basic import imread

# List all available images
print(images())  # ['test.jpg', 'lena.png', ...]

# Find image path (partial matching)
path = find_image("test")

# Load and normalize to float32 [0, 1]
img = imread(path)

# Optionally resize to max dimension
img = imread(find_image("test"), longest=1024)
```

### Using PIL/Pillow

For more control over loading:

```python
from PIL import Image
from yapocis.images import find_image
import numpy as np

# Load via asset manager and convert to numpy
pil_img = Image.open(find_image("test"))
img = np.array(pil_img).astype(np.float32) / 255.0
```

## Image Formats

Yapocis works with NumPy arrays. Standard conventions:

| Channels | Shape | Example |
|----------|-------|---------|
| Grayscale | (H, W) | `(512, 512)` |
| RGB | (H, W, 3) | `(512, 512, 3)` |
| RGBA | (H, W, 4) | `(512, 512, 4)` |

**Data types:**
- `float32`: Normalized to [0, 1] (preferred for processing)
- `uint8`: Range [0, 255] (for display/saving)

## Processing Images

### Gaussian Blur

```python
from yapocis.lib.gaussian import Gaussian
from yapocis.utils.basic import get_test_image

img = get_test_image()
gaussian = Gaussian()

# Blur with scale=3 (kernel half-width)
blurred = gaussian.gauss_image(img[:,:,0], scale=3)  # Operates on grayscale
print(f"Blurred shape: {blurred.shape}")
```

### Median Filter

```python
from yapocis.lib.median import Median
from yapocis.utils.basic import get_test_image

img = get_test_image()
median = Median()

# Apply 3x3 median filter to grayscale channel
filtered = median.median3x3fast(img[:,:,0])
print(f"Filtered shape: {filtered.shape}")
```

### Color Conversion

```python
from yapocis.lib.hsi import HSI
from yapocis.utils.basic import get_test_image

img = get_test_image()
hsi = HSI()

# RGB to HSI
hsi_img = hsi.rgb2hsi3(img)

# Modify saturation
hsi_img[:, :, 1] *= 1.5  # Boost saturation

# Back to RGB
result = hsi.hsi2rgb3(hsi_img)
print(f"Result shape: {result.shape}")
```

### Edge Detection

```python
from yapocis.lib.gradient import Gradient
from yapocis.utils.basic import get_test_image

img = get_test_image()
gradient = Gradient()

# Compute gradient magnitude and direction
mag, direction = gradient.gradient(img[:,:,0])
print(f"Gradient magnitude shape: {mag.shape}")
```

## Saving Images

### Using show_array (development)

```python
from yapocis.utils.basic import show_array, get_test_image

result = get_test_image()  # Any array works

# Saves to viewing queue with title
show_array(result, "processed_image")
```

### Using PIL

```python {.no-exec}
from PIL import Image
import numpy as np

# Convert float32 [0,1] to uint8 [0,255]
output = (result * 255).clip(0, 255).astype(np.uint8)

# Handle different channel counts
if len(output.shape) == 2:
    mode = "L"
elif output.shape[2] == 3:
    mode = "RGB"
else:
    mode = "RGBA"

Image.fromarray(output, mode=mode).save("output.png")
```

### Using yapocis utilities

```python {.no-exec}
from yapocis.utils.basic import imwrite

imwrite("output.png", result)
```

## Working with Channels

### Split channels

```python {.no-exec}
# For RGB image
r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

# Process individually
r_processed = some_operation(r)

# Recombine
result = np.stack([r_processed, g, b], axis=2)
```

### Convert grayscale to RGB

```python {.no-exec}
# Grayscale (H, W) to RGB (H, W, 3)
gray = img.mean(axis=2)  # Or use proper luminance weights
rgb = np.stack([gray, gray, gray], axis=2)
```

### Add alpha channel

```python {.no-exec}
# RGB to RGBA
alpha = np.ones_like(img[:, :, 0])
rgba = np.concatenate([img, alpha[:, :, np.newaxis]], axis=2)
```

## Memory Considerations

Large images consume significant GPU memory:

| Size | Channels | Memory (float32) |
|------|----------|------------------|
| 1024x1024 | 3 | 12 MB |
| 4096x4096 | 3 | 192 MB |
| 8192x8192 | 4 | 1 GB |

**Tips:**
- Process in tiles for very large images
- Use `float32` not `float64`
- Release buffers when done

# HSI

Hue-Saturation-Intensity color space conversion.

## Overview

The HSI (Hue, Saturation, Intensity) color space separates chromatic information (hue, saturation) from luminance (intensity), making it useful for color-based image processing.

## Class Reference

```python
from yapocis.lib.hsi import HSI

hsi = HSI()
```

## Methods

### rgb2hsi3

Convert RGB image to HSI.

```python
from yapocis.lib.hsi import HSI
from yapocis.utils.basic import get_test_image

rgb_image = get_test_image()
hsi = HSI()
hsi_image = hsi.rgb2hsi3(rgb_image)
print(f"HSI image shape: {hsi_image.shape}")
```

**Parameters:**
- `rgb_image` - (H, W, 3) float32 array with values in [0, 1]

**Returns:** (H, W, 3) float32 array with HSI values

### hsi2rgb3

Convert HSI image back to RGB.

```python
# Convert back to RGB (using hsi_image from previous example)
rgb_back = hsi.hsi2rgb3(hsi_image)
print(f"RGB image shape: {rgb_back.shape}")
```

**Parameters:**
- `hsi_image` - (H, W, 3) float32 array

**Returns:** (H, W, 3) float32 array with RGB values

## Example

```python
import numpy as np
from yapocis.lib.hsi import HSI
from yapocis.utils.basic import get_test_image

# Load RGB image
rgb = get_test_image()

# Convert to HSI
hsi_conv = HSI()
hsi_image = hsi_conv.rgb2hsi3(rgb)

# Access components
hue = hsi_image[:, :, 0]        # [0, 1] - color angle
saturation = hsi_image[:, :, 1]  # [0, 1] - color purity
intensity = hsi_image[:, :, 2]   # [0, 1] - brightness

# Modify saturation (increase by 50%)
hsi_image[:, :, 1] = np.clip(hsi_image[:, :, 1] * 1.5, 0, 1)

# Convert back to RGB
saturated_rgb = hsi_conv.hsi2rgb3(hsi_image)
```

## HSI Components

| Component | Range | Description |
|-----------|-------|-------------|
| Hue (H) | [0, 1] | Color angle (0=red, 0.33=green, 0.67=blue) |
| Saturation (S) | [0, 1] | Color purity (0=gray, 1=pure color) |
| Intensity (I) | [0, 1] | Brightness (0=black, 1=white) |

## Use Cases

### Intensity-Based Filtering

Process only brightness, preserve color:

```python
from yapocis.lib.anisodiff import Anisodiff

ad = Anisodiff()

# Filter only intensity channel
hsi_image = hsi.rgb2hsi3(rgb)
hsi_image[:, :, 2] = ad.anisodiff_n(hsi_image[:, :, 2], n=10)
result = hsi.hsi2rgb3(hsi_image)
```

### Color Selection

Select pixels by hue:

```python
hsi_image = hsi.rgb2hsi3(rgb)
hue = hsi_image[:, :, 0]

# Select red colors (hue near 0 or 1)
red_mask = (hue < 0.1) | (hue > 0.9)
```

### Saturation Adjustment

Increase or decrease color intensity:

```python
hsi_image = hsi.rgb2hsi3(rgb)

# Desaturate (toward grayscale)
hsi_image[:, :, 1] *= 0.5

# Or boost saturation
hsi_image[:, :, 1] = np.clip(hsi_image[:, :, 1] * 1.5, 0, 1)

result = hsi.hsi2rgb3(hsi_image)
```

## Related Classes

The `Anisodiff` class provides convenience methods that use HSI internally:

```python
from yapocis.lib.anisodiff import Anisodiff

ad = Anisodiff()

# Filter intensity only
result = ad.anisodiff_rgbI_n(rgb, n=10)

# Filter intensity and saturation
result = ad.anisodiff_rgbIS_n(rgb, n=10)
```

## Underlying Kernel: hsi

The `HSI` class uses the `hsi` interface.

### Interface Definition

```
interface hsi {
    kernel rgb2hsi(in float r, in float g, in float b,
                   out_like float r, out_like float r, out_like float r);
    kernel hsi2rgb(in float h, in float s, in float i,
                   out_like h, out_like h, out_like h);

    kernel rgb2hsi3(depth_of rgb, in float rgb, out_like rgb);
    kernel hsi2rgb3(depth_of hsi, in float hsi, out_like hsi);
};
```

### Kernel Parameters

**Separate channel versions (rgb2hsi, hsi2rgb):**

| Parameter | Type | Description |
|-----------|------|-------------|
| `r/h` | `float*` | Red / Hue channel |
| `g/s` | `float*` | Green / Saturation channel |
| `b/i` | `float*` | Blue / Intensity channel |

**Interleaved versions (rgb2hsi3, hsi2rgb3):**

| Parameter | Type | Description |
|-----------|------|-------------|
| `depth` | `int` | Channel depth (derived from `depth_of`) |
| `rgb/hsi` | `float*` | Interleaved (H, W, 3) image |

### Kernel Variants

| Kernel | Description |
|--------|-------------|
| `rgb2hsi` | Separate R,G,B planes to separate H,S,I planes |
| `hsi2rgb` | Separate H,S,I planes to separate R,G,B planes |
| `rgb2hsi3` | Interleaved (H,W,3) RGB to HSI |
| `hsi2rgb3` | Interleaved (H,W,3) HSI to RGB |

### Direct Kernel Access

```python
from yapocis.core import load_program
from yapocis.utils.basic import get_test_image

rgb_image = get_test_image()

# Load the hsi interface
prog = load_program("hsi")

# Interleaved conversion (positional args)
hsi_result = prog.rgb2hsi3(rgb_image)
rgb_result = prog.hsi2rgb3(hsi_result)
print(f"HSI shape: {hsi_result.shape}, RGB shape: {rgb_result.shape}")
```

## See Also

- {doc}`colormath` - Additional color space transforms
- {doc}`anisodiff` - Intensity-based filtering
- {doc}`channels` - Channel manipulation

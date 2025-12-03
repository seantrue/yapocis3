# Colormath

Professional color space transformations.

## Overview

The `Colormath` class provides GPU-accelerated color space conversions between RGB and CIE color spaces (XYZ, xyY), supporting multiple RGB working spaces and white points.

## Class Reference

```python
from yapocis.lib.colormath import Colormath

cm = Colormath()
```

## Methods

### transform_named

Apply a named color transformation.

```python
from yapocis.lib.colormath import Colormath
from yapocis.utils.basic import get_test_image

cm = Colormath()
rgb_image = get_test_image()
xyz = cm.transform_named("srgb_d65.rgb2xyz", rgb_image)
print(f"XYZ shape: {xyz.shape}")
```

**Parameters:**
- `name` - Transform name (see table below)
- `array` - (H, W, 3) float32 array

**Returns:** Transformed (H, W, 3) array

### xyz2xyY

Convert XYZ to xyY (chromaticity coordinates + luminance).

```python
# Using xyz from previous example
xyY = cm.xyz2xyY(xyz)
print(f"xyY shape: {xyY.shape}")
```

### xyY2xyz

Convert xyY back to XYZ.

```python
# Using xyY from previous example
xyz_back = cm.xyY2xyz(xyY)
print(f"XYZ back shape: {xyz_back.shape}")
```

## Available Transforms

### sRGB (D65 illuminant)

| Name | Description |
|------|-------------|
| `srgb_d65.rgb2xyz` | sRGB to XYZ |
| `srgb_d65.xyz2rgb` | XYZ to sRGB |

### Adobe RGB

| Name | Description |
|------|-------------|
| `adobe1998_d65.rgb2xyz` | Adobe RGB 1998 to XYZ |
| `adobe1998_d65.xyz2rgb` | XYZ to Adobe RGB 1998 |
| `adobe_d50.rgb2xyz` | Adobe RGB (D50) to XYZ |
| `adobe_d50.xyz2rgb` | XYZ to Adobe RGB (D50) |

### Apple RGB

| Name | Description |
|------|-------------|
| `apple_d65.rgb2xyz` | Apple RGB to XYZ |
| `apple_d65.xyz2rgb` | XYZ to Apple RGB |
| `applergb_d50.rgb2xyz` | Apple RGB (D50) to XYZ |
| `applergb_d50.xyz2rgb` | XYZ to Apple RGB (D50) |

### ProPhoto RGB

| Name | Description |
|------|-------------|
| `prophoto_d50.rgb2xyz` | ProPhoto RGB to XYZ |
| `prophoto_d50.xyz2rgb` | XYZ to ProPhoto RGB |

### Wide Gamut

| Name | Description |
|------|-------------|
| `wide_d50.rgb2xyz` | Wide Gamut RGB to XYZ |
| `wide_d50.xyz2rgb` | XYZ to Wide Gamut RGB |

### Broadcast Standards

| Name | Description |
|------|-------------|
| `ntsc_c.rgb2xyz` | NTSC (Illuminant C) to XYZ |
| `palsecam_d65.rgb2xyz` | PAL/SECAM to XYZ |
| `smpte-c_d65.rgb2xyz` | SMPTE-C to XYZ |

### Other Working Spaces

| Name | Description |
|------|-------------|
| `best_d50.rgb2xyz` | Best RGB to XYZ |
| `beta_d50.rgb2xyz` | Beta RGB to XYZ |
| `bruce_d65.rgb2xyz` | Bruce RGB to XYZ |
| `cie_e.rgb2xyz` | CIE RGB (E illuminant) to XYZ |
| `colormatch_d50.rgb2xyz` | ColorMatch RGB to XYZ |
| `donrgb3_d50.rgb2xyz` | Don RGB 3 to XYZ |
| `eci_d50.rgb2xyz` | ECI RGB to XYZ |
| `ektaspaceps5_d50.rgb2xyz` | EktaSpace PS5 to XYZ |

### Identity

| Name | Description |
|------|-------------|
| `unity` | Pass-through (no transformation) |

## Example

```python
import numpy as np
from yapocis.lib.colormath import Colormath
from yapocis.utils.basic import get_test_image

# Load sRGB image
rgb = get_test_image()

cm = Colormath()

# Convert sRGB to XYZ
xyz = cm.transform_named("srgb_d65.rgb2xyz", rgb)

# Convert to chromaticity (xyY)
xyY = cm.xyz2xyY(xyz)

# x, y are chromaticity coordinates
# Y is luminance
x = xyY[:, :, 0]
y = xyY[:, :, 1]
Y = xyY[:, :, 2]

# Convert to different RGB space
adobe_rgb = cm.transform_named("adobe1998_d65.xyz2rgb", xyz)
```

## Color Space Pipeline

```
sRGB -> XYZ -> Lab (for perceptual work)
            -> xyY (for chromaticity analysis)
            -> Other RGB (for color management)
```

## White Points

| Illuminant | Color Temperature | Description |
|------------|-------------------|-------------|
| D50 | 5003K | Graphic arts standard |
| D65 | 6504K | Daylight, display standard |
| C | 6774K | Average daylight (legacy) |
| E | - | Equal energy (theoretical) |

## Underlying Kernel: colormath

The `Colormath` class uses the `colormath` interface.

### Interface Definition

```
interface colormath {
    kernel transform(in float input, int transform_id, out_like float input);
    alias transform_res as transform(resident float input, int transform_id,
                                      resident float output);

    kernel xyz2xyY(in float input, out_like float input);
    alias xyz2xyY_res as xyz2xyY(resident float input, resident float output);

    kernel xyY2xyz(in float input, out_like float input);
    alias xyY2xyz_res as xyY2xyz(resident float input, resident float output);
};
```

### Kernel Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `float*` | Input image (H, W, 3) |
| `transform_id` | `int` | Index into transform lookup table |
| `output` | `float*` | Output image (H, W, 3) |

### Transform IDs

Transform IDs are internal indices. Use `transform_named()` for named access:

```python {.no-exec}
# Library method (recommended)
cm.transform_named("srgb_d65.rgb2xyz", image)

# Direct kernel with ID (advanced)
prog.transform(image, 0)  # ID depends on compiled order
```

### Direct Kernel Access

```python
from yapocis.core import load_program
from yapocis.utils.basic import get_test_image
from yapocis.lib.colormath import Colormath

# Load test image and convert to XYZ first
rgb_image = get_test_image()
cm = Colormath()
xyz_image = cm.transform_named("srgb_d65.rgb2xyz", rgb_image)

# Load the colormath interface
prog = load_program("colormath")

# XYZ to xyY conversion (positional args)
xyY = prog.xyz2xyY(xyz_image)
print(f"xyY shape: {xyY.shape}")
```

## See Also

- {doc}`hsi` - HSI color space
- {doc}`channels` - Channel manipulation

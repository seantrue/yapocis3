# Channels

Channel and plane manipulation utilities.

## Overview

The `channels` module provides functions for splitting, joining, and processing image channels. It handles the
differences between PIL ordering and NumPy array layouts.

## Functions

### split_channels

Split an RGB image into separate R, G, B arrays.

```python
from yapocis.lib.channels import split_channels
from yapocis.utils.basic import get_test_image

rgb_image = get_test_image()
r, g, b = split_channels(rgb_image)
print(f"Channel shapes: r={r.shape}, g={g.shape}, b={b.shape}")
```

**Parameters:**

- `rgb` - (H, W, 3) float32 array

**Returns:** Tuple of three (H, W) arrays (copies)

### join_channels

Join separate channels into an RGB image.

```python
from yapocis.lib.channels import join_channels

# Using r, g, b from previous example
rgb_joined = join_channels(r, g, b)
print(f"Joined RGB shape: {rgb_joined.shape}")
```

**Parameters:**

- `r`, `g`, `b` - (H, W) float32 arrays

**Returns:** (H, W, 3) float32 array

### split_planes

Split an image into individual planes (works for any channel count).

```python
from yapocis.lib.channels import split_planes
from yapocis.utils.basic import get_test_image

image = get_test_image()
planes = split_planes(image)  # Returns tuple of 2D arrays
print(f"Number of planes: {len(planes)}")
```

**Parameters:**

- `image` - 2D or 3D float32 array

**Returns:** Tuple of (H, W) arrays, or single array if 2D input

### join_planes

Join planes back into a multi-channel image.

```python
from yapocis.lib.channels import join_planes

# Using planes from previous example
image_rejoined = join_planes(*planes)
print(f"Rejoined image shape: {image_rejoined.shape}")
```

**Returns:** (H, W, N) float32 array where N = number of planes

### Imageops.histeq

Histogram equalization for contrast enhancement.

```python

from yapocis.lib.imageops import Imageops
from yapocis.utils.basic import get_test_image
import numpy as np

gray_image = np.mean(get_test_image(), axis=2).astype(np.float32)
equalized = Imageops().histeq(gray_image, nbr_bins=65536)
print(f"Equalized shape: {equalized.shape}")
```

**Parameters:**

- `im` - 2D float32 array with values in [0, 1]
- `nbr_bins` - Number of histogram bins (default: 2^16)

**Returns:** Equalized 2D array

## Example: Per-Channel Processing

```python
import numpy as np
from yapocis.lib.channels import split_channels, join_channels
from yapocis.lib.anisodiff import Anisodiff
from yapocis.utils.basic import get_test_image

# Load RGB image
rgb = get_test_image()

# Split into channels
r, g, b = split_channels(rgb)

# Process each channel
ad = Anisodiff()
r_filtered = ad.anisodiff_n(r, n=10)
g_filtered = ad.anisodiff_n(g, n=10)
b_filtered = ad.anisodiff_n(b, n=10)

# Rejoin
result = join_channels(r_filtered, g_filtered, b_filtered)
```

## Example: Histogram Equalization

```python
from yapocis.lib.channels import split_planes, join_planes
from yapocis.lib.imageops import Imageops
from yapocis.lib.hsi import HSI
from yapocis.utils.basic import get_test_image

# Load RGB image
rgb = get_test_image()

# Convert to HSI to equalize luminance only (preserves color)
hsi_conv = HSI()
h, s, i = split_planes(hsi_conv.rgb2hsi3(rgb))

# Equalize the intensity plane
ops = Imageops()
i_eq = ops.histeq(i)

# Rejoin and convert back
hsi_eq = join_planes(h, s, i_eq)
result = hsi_conv.hsi2rgb3(hsi_eq)
```

## Example: RGBA Processing

```python
import numpy as np
from yapocis.lib.channels import split_planes, join_planes
from yapocis.utils.basic import get_test_image

# Load RGB and add alpha channel
rgb = get_test_image()
alpha = np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
rgba = np.dstack([rgb, alpha])

# Split into R, G, B, A
r, g, b, a = split_planes(rgba)

# Process RGB, preserve alpha
r_processed, g_processed, b_processed = r * 1.2, g * 0.9, b * 1.1

# Rejoin with original alpha
result = join_planes(r_processed, g_processed, b_processed, a)
```

## Notes

- All functions return copies of data (not views)
- Arrays are made contiguous for GPU compatibility
- Works with float32 arrays

## See Also

- {doc}`hsi` - HSI color space (uses channels internally)
- {doc}`colormath` - Color transformations

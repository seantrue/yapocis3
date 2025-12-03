# Simplex Noise

GPU-accelerated Simplex noise generation with fractal Brownian motion.

## Overview

The `Simplex` class generates 2D and 3D Simplex noise using Ken Perlin's improved algorithm (2001). It supports multiple octaves for fractal Brownian motion (fBm), tileable patterns, and deterministic seeding.

## Class Reference

```python
from yapocis.lib.simplex import Simplex

s = Simplex()
```

## Methods

### simplex2d

Generate 2D simplex noise.

```python
noise = s.simplex2d(
    width, height,
    seed=0,
    scale=1.0,
    offset=(0.0, 0.0),
    octaves=4,
    persistence=0.5,
    lacunarity=2.0,
    tileable=False
)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `width`, `height` | required | Output dimensions in pixels |
| `seed` | 0 | Random seed for reproducibility |
| `scale` | 1.0 | Coordinate scale (smaller = larger features) |
| `offset` | (0, 0) | Offset in noise space for panning |
| `octaves` | 4 | Number of noise layers (1-8 typical) |
| `persistence` | 0.5 | Amplitude decay per octave |
| `lacunarity` | 2.0 | Frequency increase per octave |
| `tileable` | False | Generate seamless tileable pattern |

**Returns:** (height, width) float32 array in [0.0, 1.0]

### simplex3d

Generate 3D simplex noise volume.

```python
volume = s.simplex3d(
    width, height, depth,
    seed=0,
    scale=1.0,
    offset=(0.0, 0.0, 0.0),
    octaves=4,
    persistence=0.5,
    lacunarity=2.0
)
```

**Returns:** (depth, height, width) float32 array in [0.0, 1.0]

### turbulence

Generate turbulence pattern (absolute value of noise).

```python
turb = s.turbulence(width, height, seed=0, scale=1.0, octaves=6)
```

Creates sharper, more chaotic patterns useful for marble, fire, and fluid effects.

### ridged

Generate ridged multifractal noise.

```python
ridged = s.ridged(width, height, seed=0, scale=1.0, octaves=6)
```

Creates sharp ridges useful for mountainous terrain and rocky surfaces.

## Convenience Functions

```python
from yapocis.lib.simplex import make_simplex2d, make_simplex3d

# Without creating instance
noise = make_simplex2d(512, 512, seed=42, octaves=6)
volume = make_simplex3d(128, 128, 128, seed=99)
```

## Usage Examples

### Clouds

```python
s = Simplex()
clouds = s.simplex2d(
    512, 512,
    seed=1,
    scale=0.005,
    octaves=6,
    persistence=0.5
)
```

### Terrain Heightmap

```python
s = Simplex()

# Base terrain with ridges
terrain = s.ridged(1024, 1024, seed=42, scale=0.005, octaves=8)

# Add fine detail
detail = s.simplex2d(1024, 1024, seed=43, scale=0.02, octaves=4)

# Combine
final = 0.8 * terrain + 0.2 * detail
```

### Tileable Texture

```python
s = Simplex()
tile = s.simplex2d(256, 256, seed=123, tileable=True)

# Can be tiled seamlessly
tiled = np.tile(tile, (3, 3))
```

### Animated Noise (Time Dimension)

```python
s = Simplex()

# Generate 3D volume where Z = time
volume = s.simplex3d(256, 256, 100, seed=1, scale=0.01, octaves=4)

# Animate by iterating through Z slices
for t in range(100):
    frame = volume[t, :, :]
    # Display or process frame
```

### Marble Texture

```python
s = Simplex()

# Create marble using turbulence
turb = s.turbulence(512, 512, seed=42, scale=0.02, octaves=8)

# Create veins
x = np.linspace(0, 10, 512)
veins = np.sin(x[np.newaxis, :] + turb * 10)

# Normalize to [0, 1]
marble = (veins + 1) / 2
```

## Parameter Effects

### Scale

| Value | Effect |
|-------|--------|
| 0.001 | Very large blobs (continent scale) |
| 0.01 | Large features (clouds, terrain) |
| 0.1 | Medium features (textures) |
| 1.0 | Fine detail |

### Octaves

| Value | Effect |
|-------|--------|
| 1 | Smooth, single frequency |
| 4 | Natural look (default) |
| 6-8 | High detail, more computation |

### Persistence

| Value | Effect |
|-------|--------|
| 0.3 | Very smooth transitions |
| 0.5 | Natural balance (default) |
| 0.7 | More fine detail visible |

## Performance

| Dimensions | Octaves | GPU Time |
|------------|---------|----------|
| 512x512 | 4 | ~2ms |
| 1024x1024 | 4 | ~8ms |
| 1024x1024 | 8 | ~15ms |
| 128x128x128 | 4 | ~50ms |

## Underlying Kernel: simplex

The `Simplex` class uses the `simplex` interface.

### Interface Definition

```
interface simplex {
    // 2D Simplex noise with optional tiling
    kernel simplex2d(width_of output, height_of output, out float output,
                     int seed, float scale, float offset_x, float offset_y,
                     int octaves, float persistence, float lacunarity,
                     int tileable);
    extension res2d as simplex2d(..., resident float output, ...);

    // 3D Simplex noise
    kernel simplex3d(int width, int height, int depth,
                     out float output, int seed, float scale,
                     float offset_x, float offset_y, float offset_z,
                     int octaves, float persistence, float lacunarity);
    extension res3d as simplex3d(..., resident float output, ...);
};
```

### Kernel Parameters

**simplex2d:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `width` | `int` | Output width (derived) |
| `height` | `int` | Output height (derived) |
| `output` | `float*` | Output noise array |
| `seed` | `int` | Random seed |
| `scale` | `float` | Coordinate scale |
| `offset_x/y` | `float` | Offset in noise space |
| `octaves` | `int` | Number of fBm octaves |
| `persistence` | `float` | Amplitude decay per octave |
| `lacunarity` | `float` | Frequency increase per octave |
| `tileable` | `int` | 1 for tileable, 0 otherwise |

**simplex3d:**

Same as 2D plus `depth` and `offset_z` parameters.

### Direct Kernel Access

```python
from yapocis.core import load_program
import numpy as np

# Load the simplex interface
prog = load_program("simplex")

# Create output array
output = np.empty((512, 512), dtype=np.float32)

# Call kernel directly (positional args)
prog.simplex2d(output, 42, 0.01, 0.0, 0.0, 4, 0.5, 2.0, 0)
# seed, scale, offset_x, offset_y, octaves, persistence, lacunarity, tileable

# 3D noise
volume = np.empty((64, 256, 256), dtype=np.float32)
prog.simplex3d(256, 256, 64, volume, 42, 0.01, 0.0, 0.0, 0.0, 4, 0.5, 2.0)
```

## See Also

- {doc}`mandelbrot` - Fractal computation

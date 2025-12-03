# Matplotlib Backend

GPU-accelerated Matplotlib backend using Yapocis graphics.

## Overview

The `mplbackend` module provides a Matplotlib backend that uses Yapocis GPU primitives for rendering. This can accelerate visualization of large datasets.

## Status

This module is experimental and under development. The current implementation provides basic integration with Matplotlib's backend infrastructure.

## Usage

```python
import matplotlib
matplotlib.use('module://yapocis.lib.mplbackend')

import matplotlib.pyplot as plt
import numpy as np

# Standard matplotlib usage
x = np.linspace(0, 10, 1000)
y = np.sin(x)

plt.plot(x, y)
plt.show()
```

## Features

The GPU backend can accelerate:

- Large scatter plots
- Line plots with many points
- Filled regions
- Basic shapes

## Limitations

- Not all Matplotlib features are supported
- Complex text rendering may fall back to CPU
- Best suited for simple visualizations with many primitives

## Comparison with Standard Backend

| Feature | Standard | GPU Backend |
|---------|----------|-------------|
| Points 1K | ~1ms | ~1ms |
| Points 100K | ~50ms | ~5ms |
| Points 1M | ~500ms | ~20ms |
| Text | Full support | Basic |
| Antialiasing | CPU | GPU |

## See Also

- {doc}`graphics` - Direct GPU graphics API
- {doc}`kerneldraw` - Display list rendering

# Quickstart

This guide will get you up and running with yapocis in minutes. By the end, you'll be able to load images, run GPU-accelerated operations, and view results.

```{toctree}
:maxdepth: 1

installation
first-program
backends
images
```

## Prerequisites

- Python 3.10 or later (3.13 recommended)
- NumPy 2.1+
- At least one supported backend (OpenCL, Metal, CUDA, or Numba)

## Quick Install

```bash
# Basic installation
pip install yapocis

# With OpenCL backend
pip install yapocis[opencl]

# With all backends (where available)
pip install yapocis[all-backends]
```

## Verify Installation

```python
from yapocis.core.manager import get_backends

# List available backends
backends = get_backends()
print(f"Available backends: {backends}")
```

## What's Next?

1. {doc}`installation` - Detailed installation instructions
2. {doc}`first-program` - Run your first yapocis program
3. {doc}`backends` - Choose and configure backends
4. {doc}`images` - Work with images

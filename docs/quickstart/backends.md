# Choosing a Backend

Yapocis supports four compute backends. This guide helps you choose the right one for your use case.

## Available Backends

| Backend | Best For | Platforms | Performance |
|---------|----------|-----------|-------------|
| OpenCL | Cross-platform GPU | Linux, macOS, Windows | Excellent |
| Metal | Apple GPUs | macOS only | Excellent |
| CUDA | NVIDIA GPUs | Linux, Windows | Excellent |
| Numba | CPU fallback | All | Good |

## Backend Selection

### Environment Variable (Recommended)

```bash
export YAPOCIS_BACKEND=ocl    # OpenCL
export YAPOCIS_BACKEND=metal  # Metal
export YAPOCIS_BACKEND=cuda   # CUDA
export YAPOCIS_BACKEND=numba  # Numba
```

### Programmatic Selection

```python
from yapocis.core import manager

# Specify backend when loading
prog = manager.load_program("mandelbrot", backend="metal")
```

### Automatic Selection

If no backend is specified, yapocis tries backends in order:
1. OpenCL (if available)
2. Metal (on macOS)
3. CUDA (if available)
4. Numba (always available)

## Device Selection

For backends with multiple devices (e.g., multiple GPUs):

```bash
# By index
export YAPOCIS_DEVICE=0

# By name (partial match)
export YAPOCIS_DEVICE="AMD"
```

## Listing Available Backends

```python
from yapocis.core.manager import get_backends

backends = get_backends()
print(f"Available: {backends}")
# Example output: ['ocl', 'metal', 'numba']
```

## Backend Details

### OpenCL

**Pros:**
- Works on AMD, Intel, and NVIDIA GPUs
- Works on Intel/AMD CPUs
- Most portable option

**Cons:**
- macOS support deprecated by Apple
- Driver quality varies

**Best for:** Cross-platform applications, AMD hardware

### Metal

**Pros:**
- Native Apple GPU support
- Excellent performance on Apple Silicon
- Well-maintained by Apple

**Cons:**
- macOS only
- Requires PyMetallic compatibility layer

**Best for:** macOS-only applications, Apple Silicon optimization

### CUDA

**Pros:**
- Best NVIDIA GPU performance
- Mature ecosystem
- Extensive documentation

**Cons:**
- NVIDIA only
- Requires CUDA toolkit installation

**Best for:** NVIDIA GPU users, ML/AI workflows

### Numba

**Pros:**
- No GPU required
- Works everywhere Python works
- Easy debugging

**Cons:**
- Slower than GPU backends
- Single-threaded by default

**Best for:** Development, testing, CI servers, CPU-only machines

## Performance Comparison

Rough performance ratios (mandelbrot 1024x1024, lower is better):

| Backend | Relative Time |
|---------|---------------|
| Metal (M1) | 1.0x |
| OpenCL (discrete GPU) | 1.2x |
| CUDA (RTX 3080) | 0.8x |
| Numba (CPU) | 15-20x |

*Actual results vary significantly by hardware and workload.*

## Troubleshooting

### Backend not detected

```python
# Check why a backend failed
import os
os.environ["YAPOCIS_BACKEND"] = "ocl"

try:
    from yapocis.ocl import kernel_manager
    print("OpenCL available")
except ImportError as e:
    print(f"OpenCL not available: {e}")
```

### Wrong device selected

```bash
# List devices
yapocis-tools devices --list
```

### Performance issues

1. Check you're using GPU, not CPU OpenCL
2. Verify device selection
3. Check for thermal throttling

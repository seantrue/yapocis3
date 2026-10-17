# Installation

## Requirements

- **Python**: 3.10 or later (3.13 recommended for development)
- **NumPy**: 2.1+ (automatically installed)
- **Backend**: At least one of OpenCL, Metal, CUDA, or Numba

## Basic Installation

Install yapocis from PyPI:

```bash
pip install yapocis
```

This installs the core framework without any GPU backend. You'll need to add a backend separately.

## Backend-Specific Installation

### OpenCL (Recommended for cross-platform)

```bash
pip install yapocis[opencl]
```

**Requirements:**
- OpenCL runtime (usually included with GPU drivers)
- PyOpenCL

**Platforms:** Linux, macOS, Windows

### Metal (macOS only)

```bash
pip install yapocis[metal]
```

**Requirements:**
- macOS 10.14+
- PyMetallic

### CUDA (NVIDIA GPUs)

```bash
pip install yapocis[cuda]
```

**Requirements:**
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- PyCUDA

### Numba (CPU fallback)

```bash
pip install yapocis[numba]
```

**Requirements:**
- None beyond Python (Numba is pure Python with LLVM)

This is useful for:
- Development without GPU hardware
- Testing on CI servers
- Debugging kernel logic

## Combined Installation

```bash
# All available backends
pip install yapocis[all-backends]

# Mac-specific (OpenCL + Metal + Numba)
pip install yapocis[mac-backends]

# Linux-specific (OpenCL + CUDA + Numba)
pip install yapocis[linux-backends]
```

## Development Installation

For contributing to yapocis:

```bash
# Clone the repository
git clone https://github.com/seantrue/yapocis3.git
cd yapocis3

# Using make (recommended)
make install-dev

# Or manually with uv
uv venv .venv --python 3.13
source .venv/bin/activate
uv sync --group dev
```

## Verify Installation

```python
# Check available backends
from yapocis.core import get_backends
print(f"Backends: {get_backends()}")

# Check specific backend
from yapocis.core import manager
try:
    prog = manager.load_program("mandelbrot", backend="ocl")
    print("OpenCL backend working")
except Exception as e:
    print(f"OpenCL not available: {e}")
```

## Troubleshooting

### OpenCL not found

Install OpenCL ICD loader:

```bash
# Ubuntu/Debian
sudo apt install ocl-icd-opencl-dev

# macOS (included with Xcode)
xcode-select --install

# Windows
# Install GPU drivers from AMD/Intel/NVIDIA
```

### Metal not available

Metal requires macOS 10.14+ and Apple Silicon or AMD GPU.

### CUDA errors

Ensure CUDA Toolkit version matches your GPU driver:

```bash
nvidia-smi  # Check driver version
nvcc --version  # Check CUDA toolkit version
```

### Numba compilation errors

Update LLVM:

```bash
pip install --upgrade numba llvmlite
```

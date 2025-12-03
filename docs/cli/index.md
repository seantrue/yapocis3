# Command-Line Tools

Yapocis provides several CLI tools for development, debugging, and asset management.

## Quick Reference

| Command | Description |
|---------|-------------|
| `yapocis-tools` | Unified development tools (backends, assets, interfaces) |
| `yapocis-validate` | Test validation management |
| `yapocis-imagewatcher` | Live image queue viewer |
| `yapocis-queue` | Analyze test output files |
| `yapocis-performance` | Performance tracking and analysis |
| `yapocis-report` | Generate reports |
| `yapocis-doctest` | Run documentation example tests |

## yapocis-tools

Unified CLI for inspecting backends, devices, assets, and interfaces.

### Subcommands

#### backends

List available compute backends and devices:

```bash
$ yapocis-tools backends
================================================================================
Available Backends and Devices
================================================================================

✓ OCL
  [0] Apple M4 Max

✓ METAL ←
  [0] Apple M4 Max ←

✗ CUDA
  Error: No module named 'pycuda'

✓ NUMBAX
  [0] CPU: arm

Legend:
  ✓  = Backend available
  ✗  = Backend not available
  ← = Currently selected (backend or device)
```

Options:
- `--json` - Output in JSON format
- `--verbose` - Show environment variable help

#### devices

List OpenCL/Metal/CUDA devices:

```bash
$ yapocis-tools devices --list
Apple M4 Max

$ yapocis-tools devices
Apple M4 Max
  Platform: Apple
  Type: GPU
  Max work group size: 1024
  ...
```

#### assets

Find and list yapocis assets (interfaces, kernels, images, icons, fonts):

```bash
# List all assets
$ yapocis-tools assets --list --all
## INTERFACES
zcs.interface
graphics4.interface
colormath.interface
...

## KERNELS
anisodiff
colormath
convolvesep
...

## IMAGES
test.jpg
lena.png
...

# Find specific asset
$ yapocis-tools assets test.jpg --images
image     test.jpg   /path/to/yapocis/images/test.jpg

# Find interface
$ yapocis-tools assets mandelbrot --interfaces
interface mandelbrot /path/to/yapocis/interfaces/mandelbrot.interface
```

Options:
- `--list` - List assets instead of finding
- `--all` - Include all asset types
- `--interfaces` - Search interfaces only
- `--kernels` - Search kernels only
- `--images` - Search images only
- `--icons` - Search icons only
- `--fonts` - Search fonts only

#### interfaces

Generate interface documentation and inspect kernel signatures:

```bash
# List all interfaces
$ yapocis-tools interfaces --list
# Available Interfaces
  - mandelbrot.interface
  - gradient.interface
  - gaussian.interface
  ...

# Explain interface
$ yapocis-tools interfaces mandelbrot --explain
## mandelbrot
Interface: mandelbrot
Kernel: mandelbrot untiled
  Param 0: complex64 * q
  Coerce 'q' to complex64
  Before call, copy buffer from array to GPU
  Pass as parameter of type *complex64

  Param derived: short * out_like_q
  Allocate an array like 'q' (opencl arg position=1, dtype=int16)
  After call, copy buffer from GPU into array
  Return as int16 (tuple position 0)

  Param 1: int maxiter
```

Options:
- `--list` - List available interfaces
- `--explain` - Show detailed parameter documentation
- `--stub` - Generate Python type stubs

#### models

Download and convert pre-trained neural network weights:

```bash
# Download VGG16 weights
$ yapocis-tools models vgg16
Downloading VGG16 from PyTorch...
Converting weights to yapocis format...
Saved to models/vgg16/weights.npz

# Download ResNet50 (with batch norm fusion)
$ yapocis-tools models resnet50 --output my_models/
```

## yapocis-validate

Manage validation references for regression testing.

### Subcommands

#### info

Show validation system status:

```bash
$ yapocis-validate info
Validation System Information
=============================
Validation directory: /path/to/validation
Raw queue directory: /path/to/queue/raw

References: 42
Pending results: 5
```

#### list

List validation references:

```bash
$ yapocis-validate list
test_mandelbrot [ocl]
test_mandelbrot [metal]
test_gradient [ocl]
...
```

#### promote

Promote test results to validation references:

```bash
# Promote specific test
$ yapocis-validate promote "test_mandelbrot"
✓ Promoted test_mandelbrot (agnostic)

# Promote for specific backend
$ yapocis-validate promote "test_mandelbrot" --backends ocl
✓ Promoted test_mandelbrot [ocl]

# Promote all pending results
$ yapocis-validate promote all
```

Options:
- `--backends` - Target specific backends
- `--backend-specific` - Create backend-specific references

#### validate

Run systematic validation against references:

```bash
$ yapocis-validate validate
Validating test_mandelbrot [ocl]... ✓ PASS
Validating test_gradient [ocl]... ✓ PASS
Validating test_median [metal]... ✗ FAIL (tolerance exceeded)
```

## yapocis-imagewatcher

Live viewer for test output images in the queue directory.

```bash
# Watch default queue
$ yapocis-imagewatcher

# Watch specific directory
$ yapocis-imagewatcher --queue queue/rendered

# Filter by name
$ yapocis-imagewatcher graphics

# Show only validation failures
$ yapocis-imagewatcher --validation-failed
```

Features:
- Auto-refreshes on file changes
- Side-by-side comparison with references
- Diff visualization for failures
- Filter by test name or backend

## yapocis-queue

Analyze test output files (NPZ arrays) for data quality:

```bash
# Analyze default queue
$ yapocis-queue queue/raw
test_mandelbrot [ocl].npz
  Shape: (768, 1024)
  Dtype: int16
  Range: [0, 255]
  NaN: 0, Inf: 0

# Filter by pattern
$ yapocis-queue --filter "cuda"

# Show only files with NaN values
$ yapocis-queue --nan-only

# Summary statistics only
$ yapocis-queue --summary
Files: 42
With NaN: 0
With Inf: 0
```

Options:
- `--filter PATTERN` - Filter by filename pattern
- `--nan-only` - Only show files with NaN values
- `--summary` - Summary statistics only
- `--dtype TYPE` - Filter by dtype
- `--verbose` - Show all array details

## yapocis-performance

Track and analyze kernel performance over time.

```bash
# Run performance tests
$ yapocis-performance run

# Show performance report
$ yapocis-performance report

# Compare backends
$ yapocis-performance compare --backends ocl,metal
```

## yapocis-doctest

Run and validate documentation code examples:

```bash
# Run all documentation examples
$ yapocis-doctest

# Run examples from specific file
$ yapocis-doctest --file library/gaussian.md

# List available examples
$ yapocis-doctest --list

# Run with verbose output
$ yapocis-doctest --verbose
```

Options:
- `--file FILE` - Run examples from specific documentation file
- `--list` - List available example files
- `--verbose` - Show detailed output
- `--dry-run` - Parse but don't execute examples

## yapocis-report

Generate various reports:

```bash
$ yapocis-report
```

## Environment Variables

CLI tools respect these environment variables:

| Variable | Description |
|----------|-------------|
| `YAPOCIS_BACKEND` | Default backend (ocl, metal, cuda, numba) |
| `YAPOCIS_DEVICE` | Default device name or index |
| `YAPOCIS_PROJECT` | Additional asset search path |
| `YAPOCIS_CACHE` | Kernel cache directory |

Example:

```bash
# Use Metal backend
$ YAPOCIS_BACKEND=metal yapocis-tools backends

# Use specific device
$ YAPOCIS_DEVICE="Apple M4 Max" python my_script.py
```

## See Also

- {doc}`../quickstart/index` - Getting started guide
- {doc}`../examples/index` - Example scripts
- {doc}`../architecture/index` - Architecture overview

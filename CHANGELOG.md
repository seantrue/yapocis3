# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-17

### Changed

- **Breaking:** the graphics API uses one coordinate convention, x = column and y = row, for shapes and `Graphics4` compositing on every backend, matching text and the docs. Code that swapped coordinates per backend should drop the swap.
- **Breaking:** `histeq` moved from `yapocis.lib.channels` to the `Imageops` class (`Imageops().histeq(...)`)
- `YAPOCIS_BACKEND` accepts the same aliases as `get_manager()`: `numba`, `opencl`, `metallic` and `mtl`
- `core.get_manager(backend)` and `load_program(..., backend=...)` honour the backend argument instead of always using `YAPOCIS_BACKEND`
- Kernel, interface and asset search order is `$YAPOCIS_PROJECT`, then the current directory, then the package, as documented, so a project can override a built-in kernel
- pymetallic is installed by default on macOS, so the Metal backend works after `pip install yapocis`
- `split_channels` returns a tuple, as annotated
- Compiled Mako modules are keyed by template content, so an edited template is never served from a stale cache
- Built-in Numba kernels load by package name; existing Numba caches still load

### Added

- Extension projects: multi-directory kernel and template search through `YAPOCIS_PROJECT`, and a scaffold command for new projects
- `Imageops` methods `histmatch`, `clamp_pixels`, `stretch` and `binarize`
- `.pyi` stub generation for interfaces (`--stubs`, `--write-lib`)
- FP16 image I/O: `imread(..., dtype=np.float16)` converts in slabs, at about a quarter of the peak memory; `byte_scale` and `array_to_image` accept float16
- `yapocis-cuda-cards`: per-card CUDA benchmark across GPUs and CUDA toolkits
- `canonical_backend()` and `BACKEND_ALIASES` in `yapocis.core`
- `src/yapocis/images/CREDITS.md` records where each demo image came from

### Fixed

- Colour-space transform matrices were applied transposed on every backend. `xyz2xyY` returned Z instead of Y as its third channel, and `xyY2xyz` swapped its Y and Z outputs.
- CUDA: shared helper functions were defined twice, so the HSI kernel failed to compile and left the CUDA context unusable
- CUDA: bounds guards for out-of-range threads, plus blending and multi-GPU fixes
- CUDA: the gradient angle was computed with swapped axes, mirroring the other backends' angle across the diagonal
- `Anisodiff.anisodiff_n(img, n)` ran n-1 iterations; RGB anisotropic diffusion returned garbage on CUDA and was wrong on OpenCL and Metal
- Zero-length lines were skipped on Metal, OpenCL and CUDA; they are now drawn as dots, as on Numba
- The Numba gradient fallback (used without numba) used one-sided differences, giving half-strength gradients and wrong angles
- `imwrite` was broken for every dtype; it now saves [0, 1] data, mapping NaN to 0 and clipping out-of-range values
- `ctype_to_nptype` maps `f16` to float16
- `yapocis-cuda-cards --help` gave the wrong separator for cards in `--matrix` (`+`, not `,`)
- `yapocis-doctest --help` examples pointed at a `newdocs/` directory that doesn't exist

### Documentation

- README examples rewritten so they run on every backend. The broken code fence is fixed, CUDA is correctly described as PyCUDA, and links are absolute so they work on PyPI.
- Fixed `get_backends` imports on five pages, the `channels` examples, and references to APIs that don't exist (`yapocis.assets.add_asset_path`, `parse_interface_file`, `yapocis.numba`)
- Documented `yapocis-cuda-cards` and the real `yapocis-doctest` options
- Read the Docs builds on Python 3.13 and installs the package, so the API pages have content. The docs build with no Sphinx warnings.
- Corrected the Perona–Malik attribution in the anisodiff kernels (Pietro Perona and Jitendra Malik, 1990)

### Licensing

- The 13 classifier demo photos are replaced with CC0 images
- `lib/typeset.py` carries freetype-py's BSD notice
- Copyright is "Sean D. True and Swapwizard LLC", 2011–2026

### Build

- The `dev` extra includes the docs tooling; twine 7 is required for publishing (it accepts Metadata-Version 2.5)
- `make test-smoke` runs the smoke-marked tests; stale Makefile targets fixed

### Known issues

- OpenCL FFT gives wrong results on the NVIDIA Tesla P40 (Pascal). All 8 OpenCL tests in `tests/lib/fft` fail there: 127 of 128 output values are wrong. The same tests pass with CUDA on the P40 and with OpenCL on an RTX 3080. The bug predates 1.0.0: it reproduces on the P40 at `8b049af`. Until it is fixed, use the CUDA backend for FFT on Pascal-generation cards. Other Pascal cards are untested.

## [0.3.8] - 2025-12-03

### Documentation

- Complete documentation rewrite with executable examples
- Add `yapocis-doctest` for validating documentation examples
- Update AI.md with documentation rewrite contribution

### Added

- Tests for `median3x3`

## [0.3.7] - 2025-12-01

### Added

- Add multi-backend CNN kernel support with dispatch traits
- Add multiscale zero-crossing benchmark and improve median performance
- Add unified yapocis-performance tool for async benchmarking
- Add model download command to yapocis-tools
- Add graphics4 performance benchmark to yapocis-performance
- Add async_write support with three-level resolution across all backends
- Add anti-aliasing support for line primitives
- Complete anti-aliasing support for all graphics primitives
- Add CNN dimension specifiers and fix coordinate handling
- Add Simplex noise generation (2D/3D) across all backends

### Documentation

- Add Cython optimization analysis and results

### Fixed

- Correct dimension ordering in OpenCL stride-2 convolution kernels
- Suppress CUDA warning #177 for unreferenced functions
- Suppress CUDA warning #68 for integer sign conversions
- Suppress CUDA warning #550 for unused variables
- Correct bounds checking in CUDA convolvesep kernels
- Correct bounds checking in Metal and OpenCL convolvesep kernels
- Correct coordinate calculation in CUDA ZCS kernels
- Matplotlib backend coordinate system and DPI handling
- Add line width support to matplotlib backend
- Add antialiasing helper functions for CUDA graphics kernels
- Resolve CUDA context stack cleanup error and pytest deprecation warning
- Matplotlib backend - dashed lines, image orientation, and rendering
- Correct alpha blending in graphics4 kernels
- *(numbax)* Correct composite function bounds checking for text rendering

### Miscellaneous Tasks

- Remove standalone model download scripts
- Bump version to 0.3.7

### Refactored

- Extract identical kernel methods to KernelBase (Phase 1)
- Unify dimension handlers with _package_dimension hook (Phase 2)
- Phase 3 - template method pattern for kernel execution

### Testing

- Add comprehensive async modes validation and performance tests
- Parametrize compute tests by backend
- Parametrize processing tests by backend using monkeypatch
- Convert backend parameterized tests to use monkeypatch fixture

## [0.3.6] - 2025-10-31

### Miscellaneous Tasks

- Bump version to 0.3.6

## [0.3.5] - 2025-10-31

### Miscellaneous Tasks

- Bump version to 0.3.5

## [0.3.4] - 2025-10-31

### Miscellaneous Tasks

- Bump version to 0.3.4

## [0.3.3] - 2025-10-31

### Added

- Modernize package infrastructure with hatchling and enhanced tooling
- Add automated changelog generation with git-cliff
- Fix CUDA backend kernel compilation and backend isolation
- Add argparse to yapocis-imagewatcher with --queue option
- Add comprehensive library performance tests with device tracking
- Enhance performance reports with device info and millisecond timing
- Add ABC classes for BufferManager and DeviceManager
- Add performance marker to skip slow tests by default
- Add Metal kernel packages and enhanced performance reporting
- Add kernel package support to high-level library classes
- Add CUDA kernel packages for zero-overhead GPU calls
- Implement yapocis-package code generator for kernel packages
- Add kernel package support to all library classes
- Add unified yapocis-tools CLI
- Add convolves interface with 58 signal processing kernels
- Add complete FFT implementation with high-level library API
- Add complete audio processing pipeline with GPU-accelerated FFT and VAD
- Add VGG16 neural network support with Metal and OpenCL backends
- Add ResNet50 support and fix critical CNN kernel dispatch bugs
- Add asynchronous kernel execution support across all backends
- Add async API compatibility to Numba backend
- Add async execution support to image processing pipelines
- Optimize VGG16Resident with pre-written weights for 3x speedup
- Add ResNet50Resident with pre-written weights optimization

### Documentation

- Update README, CONTRIBUTORS, and AI contribution tracking
- Rewrite README with current goals and HISTORY reference
- Update AI.md with recent bug fixes
- Document compiled kernel packages in CLAUDE.md
- Document path discovery for src layout compatibility
- Add RetinaNet example with torchvision comparison and performance analysis

### Fixed

- Let cuda be an install target
- CUDA backend row-major addressing and truetrace kernel port
- Remove duplicate URL keys in pyproject.toml
- Update repository URLs from yapocis to yapocis3
- Prevent YAPOCIS_DEVICE environment pollution between tests
- Re-enable CUDA device manager caching to prevent context invalidation
- Handle alpha channel in HSI color space conversion kernels
- Use 2D coordinate bounds for separable convolution margins
- Remove CUDA context cleanup to prevent resource access errors
- Implement CUDA resource management to prevent buffer leaks
- Implement fft1d_batch for Numbax backend to fix test failures
- Resolve FFT kernel compilation and runtime errors across all backends
- Resolve FFT kernel compilation and runtime errors across all backends

### Miscellaneous Tasks

- Update CHANGELOG with latest commits
- Restore code_cache/README.md
- Bump version to 0.3.3

### Performance

- Add kernel instance caching to OpenCL kernel manager

### Refactored

- Rename YAPOCIS_ENGINE to YAPOCIS_DEVICE and enhance backend listing
- Migrate scripts to yapocis.scripts with CLI entry points
- Relocate infer_wh_from_args and enhance core API
- Modernize type annotations to Python 3.9+ native types
- Simplify CLI script names for better usability
- Extract common KernelManager functionality into base class
- Consolidate OpenCL managers using DeviceManagerBase ABC
- Consolidate Metal and CUDA managers using DeviceManagerBase ABC
- Centralize data directory path discovery
- Implement YAML-based context generators and backend demo
- Consolidate kernel directories under unified src/yapocis/kernels/ structure
- Remove deprecated backend-specific kernel directories
- [**breaking**] Remove kernelpack infrastructure for simplified codebase

### Testing

- Add comprehensive tests for operators interface
- Enumerate and test all available devices for each backend

## [0.3.1] - 2025-10-05

### Added

- Add enhanced test infrastructure from main branch
- Complete SDFPath and Mandelbrot kernel support
- Enhance StopWatcher with context manager checkpoints and auto-logging
- Add comprehensive performance tracking system
- Enhance performance tracking for reproducibility

### Documentation

- Remove legacy requirements.txt files and update installation instructions

### Fixed

- Complete numba HSI backend and validation framework test
- Normalize npz filename to lowercase in imagewatcher
- Resolve validation promotion filename mismatch
- Prevent environment variable leakage between tests

### Refactored

- Standardize StopWatcher pattern and enable validation framework
- Migrate performance tracking from JSON to DuckDB
- Standardize StopWatcher pattern in test_graphics4.py
- Extend StopWatcher pattern to kerneldraw tests
- Complete StopWatcher pattern audit and refactoring
- Centralize random test data generation with reproducible seeding
- Complete StopWatcher pattern refactoring across remaining test files

### Build

- Remove obsolete setup.py in favor of pyproject.toml

<!-- generated by git-cliff -->

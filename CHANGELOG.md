# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

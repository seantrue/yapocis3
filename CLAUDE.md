# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Yapocis is a multi-backend GPU/CPU computing framework for image processing, graphics operations and parallel processing. It provides a unified Python interface to OpenCL, CUDA, Metal, and Numba backends through an IDL-like interface definition language and Mako template system.

## Key Commands

### Environment Setup
```bash
make install-dev                    # Full development setup (Python 3.13)
make install-backends               # Install all backend dependencies
uv sync --extra opencl --group dev  # OpenCL only
uv sync --extra metal --group dev   # Metal only
```

### Code Quality
```bash
make format      # isort + black
make lint        # ruff check
make lint-fix    # ruff with auto-fix
make typecheck   # mypy
make check       # All checks (format-check + lint + typecheck)
make fix         # Auto-fix all issues
```

### Testing
```bash
make test                # All tests
make test-ocl            # OpenCL backend
make test-metal          # Metal backend
make test-numba          # Numba backend
make test-smoke          # Quick smoke tests (~22 tests)

# Single test
pytest tests/lib/graphics/test_graphics.py::test_smoke_line -v

# With validation
pytest --validate tests/
```

### Building
```bash
make build       # Build wheel + sdist
make release     # Interactive release workflow
make publish     # PyPI publication
```

## Architecture Overview

### Core Abstractions (`src/yapocis/core/`)
- **KernelManagerBase**: Template rendering, interface parsing, program caching
- **KernelBase**: Buffer management, parameter mapping, execution dispatch
- **DeviceManagerBase**: Device context, buffer lifecycle, resource tracking

### Backend Implementations
Each backend (`ocl/`, `metallic/`, `numbax/`, `cu/`) inherits from base classes:
- `kernel_manager.py` → Template resolution and compilation
- `kernel.py` → Backend-specific execution
- `device_manager.py` → Device context management

### Template Pipeline
```
Interface name → Load YAML context → Render interface spec (Mako) →
Parse interface → Get kernel manager → Render kernel source (Mako) →
Compile/cache program → Register callables
```

### High-Level Libraries (`src/yapocis/lib/`)
- `BoundKernels` base class binds interfaces to backends
- Respects `YAPOCIS_BACKEND` environment variable
- Libraries: graphics, convolve, colormath, gaussian, fft, truetrace, etc.

## Critical Constraints

- **Column vs Row Major**: OpenCL and Metal are column-major; CUDA is row-major
- **Named Arguments**: Bound kernels do NOT take named arguments (except meta-kernel control). Library methods may use keyword arguments.
- **Auto-derivation**: Yapocis kernels derive parameters automatically. Don't explicitly specify derivable parameters (M=3, N=2, etc.)
- **Bug Fixes**: Do not fix tests by using alternate backends. The bug is still a bug.

## Environment Variables

```bash
YAPOCIS_BACKEND=ocl|metal|cuda|numba   # Backend selection
YAPOCIS_DEVICE=<name_or_index>         # Device selection
YAPOCIS_CACHE=<directory>              # Kernel cache directory
YAPOCIS_PROJECT=<directory>            # Additional asset root
YAPOCIS_KERNELPACK=1|true|yes          # Enable compiled kernel packages
YAPOCIS_VALIDATION                     # Enable reproducible test seeding
```

## Test Organization

Tests are in `tests/` with structure: `tests/{category}/{subcategory}/`

**Categories**: backend/, lib/, kernelpack/, async/, performance/, regression/, integration/

**Markers**:
```python
@pytest.mark.smoke       # Quick validation (~22 tests)
@pytest.mark.ocl         # OpenCL-specific
@pytest.mark.metal       # Metal-specific
@pytest.mark.validation  # Regression validation
@pytest.mark.performance # Benchmarks (opt-in)
```

## Validation Framework

```bash
# Run with validation
pytest --validate tests/

# Manage references
yapocis-validate info
yapocis-validate list
yapocis-validate promote "test_name"
```

## CLI Tools

```bash
yapocis-tools backends              # List backends and devices
yapocis-tools interfaces --list     # List interfaces
yapocis-validate info               # Validation system status
yapocis-report                      # Performance reports
yapocis-cuda-cards --list           # Per-card CUDA benchmark: show cards, toolkits, plan
yapocis-cuda-cards                  # 12 on all cards, 13 on the 3060; writes performance/cuda_cards_<stamp>.*
yapocis-imagewatcher                # Image viewer for debugging
```

## Development Notes

- Put test scripts and data in `claude-debug/` - don't delete
- Add `.md` files in `claude-debug/` to git, not code/data files
- Update `AI.md` when creating commits
- Create commits only when explicitly requested
- Never delete the performance database

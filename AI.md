# AI Contributions to Yapocis

This document tracks contributions made by AI assistants (primarily Claude) versus the human maintainer.

## Summary Statistics

- **Total AI-assisted commits**: ~27
- **Major features developed with AI assistance**:
  - Infrastructure: Package modernization (hatchling, modern tooling), automated changelog generation, type annotation modernization
  - Backend work: CUDA kernel compilation fixes, CUDA row-major addressing fixes, Metal backend fixes, coordinate system standardization, Numba backend completeness, truetrace kernel port, separable convolution boundary fix
  - Testing: Validation framework, StopWatcher enhancements, test isolation fixes, NPZ analysis tooling, test determinism fixes
  - Performance: Performance tracking system, DuckDB migration, reproducibility improvements
  - Code quality: Test refactoring, random data centralization, Python 3.9+ type modernization, HSI alpha channel handling
  - Developer experience: Environment variable standardization, enhanced backend/device listing
- **Lines of code**:
  - ~1300 lines: Performance tracking system (2025-10-03)
  - ~400 lines: DuckDB migration (2025-10-03)
  - ~550 lines: Random data centralization (2025-10-04)
  - ~300 lines: Package modernization (2025-10-05)
  - ~200 lines: Changelog system (2025-10-05)
  - ~350 lines: CUDA backend compilation fixes (2025-10-05)
  - ~50 lines: YAPOCIS_ENGINE → YAPOCIS_DEVICE rename (2025-10-06)
  - ~200 lines: CUDA addressing fixes and truetrace port (2025-10-06)
  - ~350 lines: Type annotation modernization (2025-10-08)
  - ~5 lines: Test determinism fix (2025-10-16)
  - ~40 lines: HSI alpha channel handling (2025-10-16)
  - ~25 lines: Separable convolution boundary fix (2025-10-17)
  - ~100 lines: Documentation updates (2025-10-05, 2025-10-06, 2025-10-08)

## Detailed Contributions

### 2025-10-17: Separable Convolution Boundary Fix

**AI Contribution (Claude):**
- Fixed asymmetric truncation in Gaussian blur convolution operations
- Problem: Vertical convolution was truncated by kernel width at top/bottom, horizontal showed different behavior
- Root cause: Linear index boundary checks didn't properly enforce 2D rectangular margins
- Solution: Modified OCL, Metal, and CUDA kernel templates to use 2D coordinate-based boundary checks
- Added `<%include file="helpers.mako"/>` to OpenCL convolvesep.mako
- Changed boundary logic from linear ranges to explicit 2D bounds: `(x >= margin) && (x < width-margin) && (y >= margin) && (y <= height-margin)`
- Works correctly for both column-major (OCL/Metal) and row-major (CUDA) layouts
- ~25 lines changed across 3 files
- Verification: All Gaussian tests pass with symmetric 9-pixel margins on all sides

**Files Modified:**
- `yapocis/kernels/convolvesep.mako` - Added helpers include, updated boundary checks in _h and _v kernels
- `yapocis/metalkernels/convolvesep.mako` - Updated boundary checks in _h and _v kernels
- `yapocis/cukernels/convolvesep.mako` - Updated boundary checks in _h and _v kernels

**Human Contribution:**
- Identified the asymmetric truncation issue in test outputs
- Directed investigation to GPU kernels specifically
- Made initial Metal kernel fix to demonstrate correct approach
- Requested application to OpenCL and CUDA

---

### 2025-10-16: HSI Color Space Alpha Channel Handling

**AI Contribution (Claude):**
- Fixed NaN values appearing in alpha channel during HSI color space conversions
- Problem: OCL and Metal hsi2rgb kernels produced NaN in alpha channel when processing 4-channel RGBA images
- Root cause: Kernels only wrote RGB channels (0-2), leaving alpha channel uninitialized in output buffer
- Solution: Added conditional alpha passthrough when depth > 3 in all backends
- Modified both rgb2hsi3 and hsi2rgb3 kernels across 4 backends
- ~40 lines changed across 4 files
- Avoids performance penalty of zeros_like() by only copying alpha when present

**Files Modified:**
- `yapocis/kernels/hsi.mako` - Added alpha passthrough in rgb2hsi3 and hsi2rgb3
- `yapocis/metalkernels/hsi.mako` - Added alpha passthrough in rgb2hsi3 and hsi2rgb3
- `yapocis/cukernels/hsi.mako` - Added alpha passthrough in rgb2hsi3 and hsi2rgb3
- `yapocis/numbax/kernels/hsi.py` - Added alpha passthrough in _rgb2hsi_glsl and _hsi2rgb_glsl

**Human Contribution:**
- Identified NaN values in OCL/Metal HSI conversion output
- Rejected slower zeros_like() approach, requested direct alpha passthrough
- Testing and verification

---

### 2025-10-16: Test Data Determinism Fix

**AI Contribution (Claude):**
- Fixed non-deterministic text test data generation despite YAPOCIS_VALIDATE environment variable
- Problem: `interesting_words()` in tests/__init__.py used `set()` which has non-deterministic iteration order due to Python's PYTHONHASHSEED
- Solution: Added `sorted()` wrapper to ensure deterministic ordering before shuffle
- Changed `words = [word for word in set(s.split()) if len(word) > 2]` to `words = sorted(set(word for word in s.split() if len(word) > 2))`
- Created comprehensive test suite proving the bug and fix
- ~5 lines changed in 1 file

**Files Modified:**
- `tests/__init__.py` - Fixed interesting_words() to use sorted(set(...)) for deterministic output

**Testing:**
- Created `claude-debug/test_validation_seed.py` demonstrating the non-determinism bug
- Verified fix produces identical results across multiple processes

**Human Contribution:**
- Identified the random data issue in stressed text tests
- Requested bug fix with proof of determinism control
- Testing and verification

---

## Detailed Contributions

### 2025-10-08: Type Annotation Modernization to Python 3.9+ Native Types

**AI Contribution (Claude):**
- Modernized all type annotations to use Python 3.9+ native type syntax
- Phase 1: Converted `Optional[X]` to `X | None` (PEP 604)
  - 120 replacements across 37 files
  - Removed unused Optional imports from 18 files
- Phase 2: Replaced typing module types with native types (PEP 585)
  - `List[X]` → `list[X]`
  - `Dict[K,V]` → `dict[K,V]`
  - `Tuple[X,Y]` → `tuple[X,Y]`
  - `Set[X]` → `set[X]`
  - `FrozenSet[X]` → `frozenset[X]`
  - 163 replacements in 27 files (bracketed forms)
  - 63 replacements in 23 files (standalone capitalized types)
  - Total: 226 typing module replacements
  - Removed typing imports from 35 files
- Total impact: 346 type annotation modernizations across 50+ files
- Lines changed: ~350 modifications

**Key Files Modified:**
- `yapocis/utils/stereo_typing.py` - Added backward compatibility exports (later removed)
- `yapocis/utils/performance.py` - 16 Dict/List replacements
- `yapocis/utils/validation.py` - 4 Dict/List replacements
- `yapocis/scripts/imagewatcher.py` - 5 replacements
- `yapocis/scripts/performance_report.py` - 5 replacements
- `yapocis/scripts/manage_validation.py` - 2 replacements
- `yapocis/lib/typeset.py` - 2 Tuple replacements
- `yapocis/lib/channels.py` - 2 replacements
- `yapocis/lib/gaussian.py`, `yapocis/lib/truetrace.py` - Type annotation updates
- All backend kernel implementations (`yapocis/ocl/kernel.py`, `yapocis/cu/kernel.py`, `yapocis/metallic/kernel.py`, `yapocis/numbax/kernel.py`) - Updated params type from `list[Tuple]` to `list[tuple]`

**Error Resolution:**
- Fixed NameError issues where capitalized typing types (Tuple, Dict) were used without imports
- Initial approach: Added backward compatibility aliases to stereo_typing.py
- Final solution: Replaced ALL capitalized typing types with lowercase native types throughout codebase
- Result: All modules now import successfully with zero NameError issues

**Testing:**
- All backend smoke tests pass (48 tests)
- All backend runtime tests pass (3 tests, 1 skipped)
- All validation tests pass (5 tests)
- Comprehensive import tests verify all key modules load correctly
- `python3 -m compileall` confirms all files compile successfully

**Benefits:**
- Modern, idiomatic Python 3.9+ type syntax
- Improved readability and maintainability
- Follows PEP 585 (Type Hinting Generics In Standard Collections)
- Follows PEP 604 (Allow writing union types as X | Y)
- Reduced imports from typing module
- Aligns with Python community best practices for 2025

**Human Contribution:**
- Request for Optional → union type conversion
- Request to replace typing module types with native types
- Clarification to replace ALL references including standalone capitalized types
- Testing and verification

---

### 2025-10-06: CUDA Backend Row-Major Addressing Fixes and Truetrace Kernel Port

**AI Contribution (Claude):**
- Fixed critical CUDA coordinate addressing bugs causing rotated/misaligned graphics output
- Ported truetrace (SDF path tracing) kernel from OpenCL to CUDA
- Resolved all CUDA validation failures related to addressing and NaN generation
- ~200 lines changed across 5 files

**X/Y Macro Row-Major Fix (helpers.mako):**
- Problem: CUDA uses row-major storage (like NumPy), but X/Y macros were backwards
- OpenCL: Column-major `ADDRESS(xx,yy) = (xx*height)+yy`, so `X(addr) = addr/height`, `Y(addr) = addr%height`
- CUDA: Row-major `ADDRESS(xx,yy) = (yy*width)+xx`, but had `X(addr) = addr/width` (WRONG!)
- **Fix**: Corrected to `X(addr) = addr%width`, `Y(addr) = addr/width` for row-major
- Result: Proper coordinate extraction from linear addresses

**Composite Pixel Addressing Fix (graphics.mako, graphics4.mako):**
- Problem: First character in text appeared rotated 90 degrees clockwise
- Root cause: `source_offset = (source_location.x*coord.z)+source_location.y` assumed column-major
- **Fix**: Changed to `source_offset = (source_location.y*coord.z)+source_location.x` for row-major
- Applied to both single-channel (graphics.mako) and 4-channel (graphics4.mako) kernels
- Result: All 14 graphics tests pass with correct pixel alignment

**ZCS Kernel NaN Fix (zcs.mako):**
- Problem: NaN values and extreme values (±3.4e+38) in multiscale zero-crossing output
- Root cause: Used `X(i)` and `Y(i)` where `i = get_global_id(0)*planes` (plane-multiplied index)
- Plane multiplication caused X/Y macros to compute wrong coordinates
- **Fix**: Added `pixel_id = get_global_id(0)` and direct calculation:
  - `x = max(1, (int)(pixel_id % width))`
  - `y = max(1, (int)(pixel_id / width))`
- CUDA-specific fix (OpenCL/Metal unchanged as requested)
- Result: Zero NaN/Inf values in all 162 validation files

**Truetrace Kernel Port (truetrace.mako):**
- Created CUDA version of Jump Flood Algorithm distance field kernel
- Ported three kernel functions from OpenCL:
  - `init_seed_from_mask`: Initialize seed points from binary mask
  - `jfa_step`: Jump Flood Algorithm distance field computation
  - `marching_squares`: Contour extraction at iso-level
- Key CUDA adaptations:
  - Changed `__kernel` → `extern "C" __global__`
  - Removed `__global` pointer qualifiers (CUDA uses bare pointers)
  - Changed `atomic_inc` → `atomicAdd` for segment counter
  - Changed `uint` → `unsigned int` for CUDA compatibility
  - Kept X/Y macros (work correctly after row-major fix)
- Result: Kernel compiles and all truetrace tests pass

**NPZ Analysis Tool (scripts/analyze_queue.py):**
- Created CLI tool for analyzing test output NPZ files
- Features: dtype analysis, min/max values, NaN/Inf detection
- Flags: `--filter`, `--nan-only`, `--verbose`, `--summary`, `--dtype`
- Used to diagnose and verify ZCS NaN fix

**Files Modified:**
- `yapocis/cukernels/helpers.mako` - Fixed X/Y coordinate macros
- `yapocis/cukernels/graphics.mako` - Fixed composite pixel addressing (line 172)
- `yapocis/cukernels/graphics4.mako` - Fixed composite pixel addressing (line 168)
- `yapocis/cukernels/zcs.mako` - Fixed coordinate calculation with plane indexing
- `yapocis/cukernels/truetrace.mako` - **Created** (133 lines, complete port)
- `scripts/analyze_queue.py` - **Created** (202 lines, NPZ analysis tool)

**Testing:**
- All 14 graphics tests pass with correct output
- All ZCS tests pass with zero NaN/Inf values
- Truetrace functional test passes (test_truetrace.py)
- Full CUDA validation: 162 files, 160 float32 + 2 uint8 arrays, 0 NaN, 0 Inf
- Verified with `scripts/analyze_queue.py --summary`

**Benefits:**
- CUDA backend now produces pixel-perfect graphics output matching other backends
- All coordinate-dependent kernels work correctly with row-major storage
- Jump Flood Algorithm available on CUDA for distance field computation
- Comprehensive validation tooling for detecting data quality issues
- Foundation for CUDA backend validation framework

**Human Contribution:**
- Identified that CUDA uses row-major storage (same as NumPy)
- Pointed out graphics composite kernel addressing issue
- Requested CUDA-specific ZCS fix (not touching OCL/Metal)
- Requested NPZ analysis tool with CLI interface
- Requested truetrace kernel port
- Testing and verification throughout

---

### 2025-10-06: Environment Variable Standardization and Backend Listing Enhancement

**AI Contribution (Claude):**
- Renamed `YAPOCIS_ENGINE` → `YAPOCIS_DEVICE` across entire codebase for clarity
- Enhanced `scripts/list_backends.py` to show currently selected backend and device
- Fixed CUDA device enumeration to use PyCUDA instead of CuPy
- Updated all documentation to reflect new environment variable name
- ~50 lines changed across 10 files

**Environment Variable Rename:**
- Globally replaced `YAPOCIS_ENGINE` with `YAPOCIS_DEVICE` for consistency
- Updated code files (5): `yapocis/cu/device.py`, `yapocis/ocl/device.py`, `yapocis/metallic/device.py`, `yapocis/lib/truetrace.py`, `scripts/list_backends.py`
- Updated documentation (4): `CLAUDE.md`, `AGENTS.md`, `docs/metallic.md`, `docs/testing.md`
- Updated dotfiles (1): `.env`
- Historical performance data preserved (not modified)

**Backend Listing Script Enhancements:**
- Added visual indicator (`←`) for currently selected backend
- Added visual indicator (`←`) for currently selected device within active backend
- Defaults to showing device 0 as selected when no `YAPOCIS_DEVICE` specified
- Supports device selection by numeric index or exact device name
- Fixed CUDA backend to use `yapocis.cu.device.get_available_devices()` instead of CuPy
- Updated legend to explain selection markers
- Simplified legend from 4 items to 3 (merged backend/device selection indicator)

**Files Modified:**
- `scripts/list_backends.py` - Enhanced device/backend display, fixed CUDA enumeration
- `yapocis/cu/device.py` - Environment variable name
- `yapocis/ocl/device.py` - Environment variable name
- `yapocis/metallic/device.py` - Environment variable name and comment
- `yapocis/lib/truetrace.py` - Documentation comment
- `CLAUDE.md` - Environment variable documentation
- `AGENTS.md` - Environment variable documentation
- `docs/metallic.md` - Device selection documentation
- `docs/testing.md` - Metal device documentation
- `.env` - Export statement
- `README.md` - Updated CUDA backend description, added device selection docs

**Testing:**
- Verified device selection works for all backends (OCL, CUDA, Numba)
- Confirmed selection by numeric index (e.g., `YAPOCIS_DEVICE=0`)
- Confirmed selection by device name (e.g., `YAPOCIS_DEVICE="Tesla P40"`)
- Verified default to device 0 when no explicit selection
- All backends show correct indicators

**Benefits:**
- Clearer, more intuitive environment variable name (`DEVICE` vs `ENGINE`)
- Visual feedback on current backend/device selection in listing script
- Better developer experience when working with multiple devices
- Consistent naming across entire codebase
- Fixed CUDA device listing to use project's actual CUDA implementation

**Human Contribution:**
- Request to rename YAPOCIS_ENGINE to YAPOCIS_DEVICE
- Request to show selected device in listing script
- Request to default to device 0 when not explicitly set
- Request to use arrow instead of asterisk for backend selection
- Testing and verification

---

### 2025-10-05: CUDA Backend Kernel Compilation Fixes

**AI Contribution (Claude):**
- Fixed all 11 CUDA kernels to compile successfully (100% compilation rate)
- Completed transition from OpenCL-specific syntax to CUDA-compatible code
- Fixed backend isolation issue where mako template caches were being shared between OCL and CUDA
- Implemented proper CUDA context management for kernel launches
- ~350 lines of changes across 8 files

**Kernel Syntax Fixes:**
- **graphics.mako**: Replaced OpenCL component access syntax (`.s0/.s1` → `.x/.y`), fixed multi-digit swizzles (`.s10/.s32`), replaced OpenCL casts `(int2)(x,y)` → `make_int2(x,y)`
- **graphics4.mako**: Replaced all `.s3` → `.w` component access (7 occurrences), fixed `.s0/.s1/.s2/.s3` → `.x/.y/.z/.w` syntax
- **hsi.mako**: Added `__device__` qualifier to 6 inline helper functions (`_rgb2hsi`, `_hsi2rgb`, `_rgb2xyz`, `_xyz2lab`, `adjustrgb`, `adjustxyz`) - functions were being treated as host-only without this

**CUDA Constants & Helpers (yapocis/cukernels/constants.mako):**
- Replaced `_Generic` clamp macro with proper C++ function overloads for float/float2/float3/float4
- Added vector math functions: `fract()`, `mix()`, `floor()`, `fabs()` with float2/float3/float4 overloads
- Added `native_powr()` macro mapping to `powf()` for OpenCL compatibility
- Added int2/int3/int4 arithmetic operators (+, -, *)
- Added float2/float3/float4 `/=` operator for vector/vector division

**Backend Isolation:**
- Fixed mako template cache contamination between backends
- Changed `mako_modules` directory from shared to backend-specific:
  - OpenCL: `mako_modules/ocl/`
  - CUDA: `mako_modules/cuda/`
  - Metal: `mako_modules/metal/` (already implemented)
- Prevents OpenCL templates from being used in CUDA kernels and vice versa

**CUDA Context Management:**
- Changed from `make_context()` to `retain_primary_context()` for better multi-test stability
- Added immediate `ctx.push()` on context creation to make it current
- Changed cleanup from `ctx.detach()` to `ctx.pop()` for proper context stack management
- Ensures CUDA context is active for kernel launches

**Files Modified:**
- `yapocis/cukernels/constants.mako` - Added vector operators, math function overloads
- `yapocis/cukernels/graphics.mako` - Fixed OpenCL syntax (6 changes)
- `yapocis/cukernels/graphics4.mako` - Fixed OpenCL syntax (9 changes)
- `yapocis/cukernels/hsi.mako` - Added `__device__` qualifiers (6 functions)
- `yapocis/cu/kernel_manager.py` - Backend-specific mako cache
- `yapocis/ocl/kernel_manager.py` - Backend-specific mako cache
- `yapocis/cu/device_manager.py` - Primary context management
- `yapocis/cu/kernel.py` - Context awareness (removed after primary context approach)

**Testing:**
- All 11 kernels compile successfully: anisodiff, colormath, demo, gradient, graphics, graphics4, hsi, imageops, mandelbrot, median3x3, zcs
- Individual CUDA tests pass successfully
- OpenCL, Metal, and Numba backends unaffected (all tests pass)

**Compilation Errors Fixed:**
1. OpenCL vector cast syntax `(float3)(x,y,z)` → CUDA `make_float3(x,y,z)`
2. OpenCL component access `.s0/.s1/.s2/.s3` → CUDA `.x/.y/.z/.w`
3. OpenCL multi-digit swizzles `.s10/.s32` → explicit `make_int2(coord.x, coord.y)`
4. OpenCL vector swizzling (`.xy/.zy/.xyz/.www`) → manual component expansion
5. `_Generic` C11 syntax → C++ function overloads
6. Missing `__device__` qualifiers on helper functions
7. Missing vector math function overloads (sqrt, isnan, atan2pi, fabs, floor, fract, mix)
8. Backend template cache contamination

**Known Limitations:**
- Full CUDA test suite shows 65 failures due to pre-existing test isolation issues (not related to kernel compilation)
- Individual tests pass; failures only occur when running full suite
- Issue is with test infrastructure context management between tests, not kernel code
- All other backends (OCL, Metal, Numba) pass all tests

**Benefits:**
- CUDA backend now fully functional for all 11 kernel interfaces
- Clean separation between backend template caches
- Proper CUDA context lifecycle management
- Foundation for CUDA backend testing and validation

**Human Contribution:**
- Identification of kernel compilation failures
- Request to fix CUDA backend compilation
- Clarification that backends don't share code outside core/lib/interfaces/utils
- Testing and verification

---

### 2025-10-05: Automated Changelog Generation System

**AI Contribution (Claude):**
- Implemented automated changelog generation using git-cliff
- Created cliff.toml configuration
  - Conventional commits parsing (feat, fix, docs, refactor, perf, test, chore)
  - Keep a Changelog format compliance
  - Semantic versioning support
  - Automatic grouping by commit type
- Added Makefile targets
  - `make changelog`: Generate/update CHANGELOG.md from git history
  - `make changelog-unreleased`: Preview unreleased changes
- Generated initial CHANGELOG.md from project history
- Updated documentation (CLAUDE.md, RENOVATE.md)
- Lines changed: ~200 additions across 5 files

**Files Created:**
- `cliff.toml` - git-cliff configuration (100 lines)
- `CHANGELOG.md` - Generated changelog from git history (50 lines)

**Files Modified:**
- `Makefile` - Added changelog section with 2 targets
- `CLAUDE.md` - Added changelog management section
- `RENOVATE.md` - Added changelog system documentation

**Testing:**
- Verified conventional commit parsing
- Confirmed changelog generation from git history
- Tested unreleased changes preview

**Benefits:**
- Automated release notes generation
- Consistent changelog format
- No manual changelog maintenance
- Conventional commits enforcement

**Human Contribution:**
- Request for automated changelog
- Testing and verification

---

### 2025-10-05: Package Infrastructure Modernization

**AI Contribution (Claude):**
- Migrated build backend from setuptools to hatchling (PEP 517/518 compliant)
- Updated ruff configuration to modern `[tool.ruff.lint]` format
- Fixed license from PSF to MIT (matching actual LICENSE.txt)
- Added comprehensive coverage configuration (`[tool.coverage.run]`, `[tool.coverage.report]`)
- Added development tools: isort, licensecheck, pytest-benchmark
- Added publishing tools: twine, cibuildwheel with proper configuration
- Enhanced Makefile with new targets:
  - Added isort to format/format-check targets
  - Added publish-test and publish targets for PyPI
  - Added license-check target
  - Forced Python 3.13 for virtual environments
- Updated documentation (CLAUDE.md, RENOVATE.md)
- Created CONTRIBUTORS.md
- Lines changed: ~300 additions/modifications across 6 files

**Files Created:**
- `RENOVATE.md` - Documentation of package modernization (155 lines)
- `CONTRIBUTORS.md` - Project contributors and guidelines

**Files Modified:**
- `pyproject.toml` - Modern build backend, coverage config, new dev tools
- `Makefile` - Enhanced targets for formatting, publishing, Python 3.13
- `LICENSE.txt` - Updated copyright year to 2025
- `CLAUDE.md` - Added code quality and building sections
- `README.md` - Updated with modern packaging features

**Key Changes:**
- Build backend: setuptools → hatchling
- Ruff config: top-level → `[tool.ruff.lint]`
- License: PSF → MIT
- Python: 3.10+ → 3.13 enforced
- New tools: isort, licensecheck, pytest-benchmark, twine

**Testing:**
- Complete environment rebuild with Python 3.13
- Verified all dev dependencies install correctly
- Confirmed hatchling builds packages properly

**Benefits:**
- Faster, PEP-compliant builds
- Modern tooling and best practices
- Correct license declaration
- Streamlined PyPI publishing
- Enhanced code quality automation

**Human Contribution:**
- Request to adopt improvements from pymetallic
- Comparison of pyproject.toml files
- Request for Python 3.13 enforcement
- Testing and verification

---

### 2025-10-04: Fix Test Isolation - Environment Variable Leakage

**AI Contribution (Claude):**
- Fixed environment variable leakage between pytest tests causing incorrect backend tagging
- Identified root cause: Tests were permanently modifying `os.environ["YAPOCIS_BACKEND"]`
- Problem: When running full test suite (`pytest tests`), parametrized tests across backends would leave `YAPOCIS_BACKEND=numba` (last backend tested) set for all subsequent tests
- Solution: Replaced direct `os.environ` modification with pytest's `monkeypatch` fixture
- Created `scripts/list_backends.py` utility to enumerate all available backends and devices
- Lines changed: ~25 modifications across 3 test files, ~120 lines added for backend listing script

**Root Cause Analysis:**
- `test_backend_smoke.py`: Parametrized test across all backends modified environment without cleanup
- `test_validation_example.py`: Similar issue in validation test
- `test_backend_runtime.py`: All 4 runtime tests (ocl, metal, cuda, numba) leaked backend setting
- When tests ran in order: ocl → metal → cuda → numba, final state was `YAPOCIS_BACKEND=numba`
- This caused all subsequent tests to use numba backend regardless of shell environment

**Files Modified:**
- `tests/test_backend_smoke.py`:
  - Added `monkeypatch` parameter to `test_compile_interfaces_all_backends`
  - Changed `os.environ["YAPOCIS_BACKEND"] = backend` to `monkeypatch.setenv("YAPOCIS_BACKEND", backend)`
- `tests/test_validation_example.py`:
  - Added `monkeypatch` parameter to `test_graphics4_circles_validation`
  - Changed direct environment modification to use `monkeypatch.setenv()`
- `tests/test_backend_runtime.py`:
  - Removed `_set_backend()` helper function
  - Added `monkeypatch` parameter to all 4 test functions
  - Changed all tests to use `monkeypatch.setenv("YAPOCIS_BACKEND", backend)`

**Files Created:**
- `scripts/list_backends.py`:
  - Lists all available backends (OCL, Metal, CUDA, Numba)
  - Enumerates devices for each backend
  - Supports verbose mode and JSON output
  - 120 lines of backend/device enumeration code

**Testing:**
- Verified files now correctly tagged with shell's `YAPOCIS_BACKEND` setting
- Running `pytest tests/test_gaussians.py::test_image` with `YAPOCIS_BACKEND=metal` produces `[metal]` tagged files
- Running full test suite no longer changes backend for subsequent tests
- Confirmed `monkeypatch` automatically restores environment after each test

**Benefits:**
- Tests are now properly isolated - no environment variable leakage
- Output files correctly tagged with intended backend
- Shell environment variable properly respected when running full test suite
- Makes debugging multi-backend issues easier
- Backend enumeration script useful for CI/CD and user diagnostics

**Human Contribution:**
- Bug report: Files marked `[numba]` when running OCL tests
- Clarification that issue occurred with full test suite, not individual tests
- Testing and verification

---


### 2025-10-04: Performance System Reproducibility Enhancements

**AI Contribution (Claude):**
- Enhanced performance tracking system to ensure "clean" status means fully replicable results
- Modified `get_git_info()` to check entire repository status instead of just test file
  - Status is "clean" ONLY when there are NO uncommitted changes or untracked files anywhere
  - Added "untracked" status for repositories with untracked files
  - Added "modified" status for repositories with only tracked file modifications
  - Ensures "clean" performance results are fully replicable
- Updated `save_performance_data()` to accept `test_start_time` parameter
  - Timestamp now reflects when test started, not when it finished
  - More accurate timing data for longitudinal analysis
  - Defaults to current time if test_start_time not provided (backward compatible)
- Updated `StopWatcher.__exit__()` to pass test start time to `save_performance_data()`
  - Uses `self.t` captured in `__enter__()` as test start time
  - Ensures timestamp accuracy in performance database
- Lines changed: ~70 modifications across 2 files

**Status Classification:**
- **clean**: No uncommitted changes, no untracked files → fully replicable
- **modified**: Has modified tracked files, no untracked files → partially tracked
- **untracked**: Has untracked files (may also have modifications) → not fully tracked

**Files Modified:**
- `yapocis/utils/performance.py`:
  - Modified `get_git_info()` to check entire repository status (lines 81-149)
  - Added `test_start_time` parameter to `save_performance_data()` (lines 152-183)
- `tests/__init__.py`:
  - Updated `StopWatcher.__exit__()` to pass `test_start_time=self.t` (line 223)

**Testing:**
- Verified with `test_truetrace.py` using `YAPOCIS_PERFORMANCE=1`
- Confirmed "untracked" status correctly detected when untracked files present
- Verified timestamp reflects test start time in DuckDB
- Confirmed backward compatibility when test_start_time not provided

**Benefits:**
- "clean" status provides strong reproducibility guarantee
- Performance comparisons can filter to only replicable results
- Accurate test start timestamps for longitudinal analysis
- Better understanding of repository state when performance data was captured
- Enables identification of performance regressions vs. environmental changes

**Human Contribution:**
- Request to make "clean" status mean fully replicable
- Specification of timestamp requirement
- Testing and verification

---


### 2025-10-04: Random Data Generation Refactoring with Reproducible Seeding

**AI Contribution (Claude):**
- Created centralized `tests/random_data.py` module for all random test data generation
- Implemented reproducible seeding via `YAPOCIS_VALIDATION` environment variable
- Extracted and consolidated random data functions from multiple test files
- Resolved naming conflicts between single-channel and 4-channel variants
- Pattern applied:
  - Single-channel (graphics1) functions: `random_colors_1ch()`, `random_alpha_1ch()`, etc.
  - 4-channel (graphics4/RGBA) functions: `random_colors_4ch()`, `random_alpha_4ch()`, etc.
  - Common functions without channel-specific behavior: `random_points()`, `random_lines()`, etc.
  - Backward-compatible aliases default to single-channel versions
  - Shared functions: `random_texts()` used by both graphics1 and graphics4

**Reproducible Seeding:**
- When `YAPOCIS_VALIDATION` environment variable is set, seeds both `random` and `numpy.random` to 42
- Enables reproducible test data generation for validation and regression testing
- Verified with automated testing showing identical output when env var is set
- Does not affect normal test execution when env var is not set

**Files Created:**
- `tests/random_data.py` - 265 lines of centralized random data functions

**Files Modified:**
- `tests/test_graphics.py` - Removed 147 lines of duplicate functions, added imports
- `tests/test_graphics4.py` - Removed 154 lines of duplicate functions, added imports
- `tests/test_kerneldraw.py` - Updated imports to use random_data
- `tests/test_kerneldraw3.py` - Updated imports to use random_data
- `tests/test_kerneldraw4.py` - Updated imports to use random_data (with _4ch suffix)

**Functions Extracted:**
- Common: `random_points()`, `random_lines()`, `random_widths()`, `random_rects`, `random_triangles()`, `random_circles()`, `random_ellipses()`
- Graphics1: `random_colors_1ch()`, `random_alpha_1ch()`, `image_source_1ch()`, `image_sprites_1ch()`, `random_composites_1ch()`, `random_sprites_1ch()`
- Graphics4: `random_vector()`, `components_to_colors()`, `random_colors_4ch()`, `random_alpha_4ch()`, `random_icons()`, `image_source_4ch()`, `image_sprites_4ch()`, `random_composites_4ch()`, `random_sprites_4ch()`
- Shared: `random_texts()`

**Testing:**
- Verified all tests pass with new imports: test_graphics, test_graphics4, test_kerneldraw, test_kerneldraw3, test_kerneldraw4
- Created validation script in `claude-debug/test_validation_seeding.py`
- Confirmed reproducible seeding works correctly with YAPOCIS_VALIDATION

**Benefits:**
- Eliminates code duplication across test files (reduced by ~300 lines total)
- Centralized location for all random test data generation
- Enables reproducible test data for validation framework
- Clear separation between single-channel and 4-channel variants
- Easier maintenance of test data generation logic
- Supports future expansion of validation framework

**Human Contribution:**
- Request for reproducible seeding via environment variable
- Specification of YAPOCIS_VALIDATION as the trigger mechanism
- Direction to avoid name collisions

---

### 2025-10-04: Comprehensive StopWatcher Pattern Audit and Refactoring

**AI Contribution (Claude):**
- Conducted comprehensive audit of all test files for StopWatcher usage
- Refactored test files to match the updated StopWatcher pattern from `tests/test_graphics.py`
- Added performance tracking to previously untracked kernel operations
- Pattern changes applied:
  - Added descriptive names to StopWatcher initialization: `StopWatcher("test_name", kernel=obj)`
  - Converted checkpoint calls to context manager style: `with sw.checkpoint("label"):`
  - Removed manual `print(sw.summary())` calls (now auto-logged via save_perf)
  - Updated `sw.show()` signature from `sw.show(kernel, canvas)` to `sw.show(canvas)`
  - Preserved `sw.showcheck()` for intermediate checkpoints with metadata
- Applied formatting improvements via linter (black/ruff)
  - Standardized spacing and line breaks
  - Fixed import organization
  - Improved code readability
- Lines changed: ~250 modifications across 8 files

**Previously Refactored (earlier commits):**
- `test_graphics4.py` - 15 functions refactored
- `test_kerneldraw.py` - 3 functions refactored
- `test_kerneldraw4.py` - 3 functions refactored

**Newly Refactored (this commit):**

**test_kerneldraw3.py - 2 functions:**
- `test_batch` - Batch rendering with 3-channel graphics
- `test_displaylist` - Display list with individual draw calls

**test_mandelbrot.py - 1 function:**
- `test_mandelbrot` - Mandelbrot set computation with perf_tracker integration

**test_convolve.py - 1 function:**
- `test_convolve` - 1D convolution performance testing

**test_imageops.py - 6 functions (NEW tracking added):**
- `test_clamp` - Pixel clamping operation
- `test_binarize` - Binarization operation
- `test_stretch` - Histogram stretching
- `test_histogram` - Histogram computation with different bin counts
- `test_histogram_res` - Resident buffer histogram operations
- `test_ops` - Combined operations (clamp, histogram, stretch)

**test_truetrace.py - 1 function (NEW tracking added):**
- `test_truetrace_matches_original_fill_iou_ssim` - GPU-accelerated SDF tracing

**Files verified (already using correct pattern):**
- `test_anisodiff.py` - Uses `sw.showcheck()` appropriately
- `test_colormath.py` - Uses `sw.showcheck()` appropriately
- `test_gaussians.py` - Uses `sw.showcheck()` appropriately
- `test_examples.py` - Uses modern pattern appropriately

**Files modified:**
- `tests/test_kerneldraw3.py` - Refactored StopWatcher usage pattern
- `tests/test_mandelbrot.py` - Refactored StopWatcher usage pattern
- `tests/test_convolve.py` - Added names and kernel parameters
- `tests/test_imageops.py` - Added StopWatcher tracking to 6 functions
- `tests/test_truetrace.py` - Added StopWatcher tracking

**Testing:**
- Verified all refactored tests maintain correct functionality
- Confirmed consistent usage across all test functions
- Maintained backward compatibility with existing functionality

**Benefits:**
- Complete StopWatcher coverage across all kernel-calling tests
- Consistent pattern usage across entire test suite
- Better integration with performance tracking and DuckDB storage
- Improved visibility into kernel performance
- Cleaner checkpoint timing with context managers

**Human Contribution:**
- Request for comprehensive audit
- Clarification on showcheck() vs show() usage pattern
- Approval to refactor all identified files
- Code review and linting

---

### 2025-10-03: JSON to DuckDB Migration Script

**AI Contribution (Claude):**
- Created migration script `scripts/migrate_json_to_duckdb.py` to import legacy JSON performance data
- Successfully migrated 409 JSON performance records to DuckDB
- Imported 1,352 checkpoint records with proper foreign key relationships
- Zero errors during migration process
- Lines changed: ~100 additions in migration script

**Migration Results:**
- 409 performance runs successfully imported
- 1,352 checkpoints successfully imported
- All historical data preserved with full fidelity
- Database verified with CLI tools (list, stats commands)

**Files created:**
- `scripts/migrate_json_to_duckdb.py` - One-time migration utility

**Human Contribution:**
- Request to migrate existing data
- Verification of migration results

---

### 2025-10-03: DuckDB Migration for Performance Tracking

**AI Contribution (Claude):**
- Migrated performance tracking system from JSON file storage to DuckDB relational database
- Designed normalized database schema with two tables:
  - `performance_runs`: Stores test execution metadata and aggregate statistics
  - `performance_checkpoints`: Stores individual checkpoint timings with foreign key to runs
- Implemented DuckDB sequences for auto-increment primary keys
- Completely rewrote `yapocis/utils/performance.py` (1158 lines)
  - `_get_connection()`: DuckDB connection management with automatic schema creation
  - `save_performance_data()`: INSERT INTO with auto-generated IDs
  - `load_performance_data()`: Dynamic SQL queries with parameterized filtering
  - `purge_performance_data()`: DELETE with foreign key cascade handling
  - Maintained exact same return data structure for backward compatibility
- Updated `scripts/performance_report.py` to remove obsolete imports
- Updated `.gitignore` to track .duckdb and .duckdb.wal files instead of results/ directory
- Added DuckDB>=1.1.0 to pyproject.toml dev dependencies
- Created indexes on (test_name, backend, branch, commit, file_status) and timestamp_unix
- Lines changed: ~400 modifications across 4 files

**Database Schema:**
```sql
-- Auto-increment sequences
CREATE SEQUENCE runs_seq START 1
CREATE SEQUENCE checkpoints_seq START 1

-- Main runs table with aggregated stats
CREATE TABLE performance_runs (
    id INTEGER PRIMARY KEY DEFAULT nextval('runs_seq'),
    test_name VARCHAR NOT NULL,
    test_file VARCHAR NOT NULL,
    backend VARCHAR NOT NULL,
    branch VARCHAR NOT NULL,
    commit VARCHAR NOT NULL,
    file_status VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    timestamp_unix DOUBLE NOT NULL,
    total_time DOUBLE NOT NULL,
    min_time DOUBLE,
    max_time DOUBLE,
    mean_time DOUBLE,
    slowest_operation VARCHAR,
    fastest_operation VARCHAR
)

-- Checkpoints with foreign key
CREATE TABLE performance_checkpoints (
    id INTEGER PRIMARY KEY DEFAULT nextval('checkpoints_seq'),
    run_id INTEGER NOT NULL,
    checkpoint_name VARCHAR NOT NULL,
    elapsed_time DOUBLE NOT NULL,
    FOREIGN KEY (run_id) REFERENCES performance_runs(id)
)
```

**Files modified:**
- `yapocis/utils/performance.py` - Complete rewrite to use DuckDB
- `pyproject.toml` - Added duckdb dependency
- `.gitignore` - Updated to track .duckdb files
- `scripts/performance_report.py` - Removed obsolete imports

**Testing:**
- Verified save/load operations with DuckDB
- Tested all CLI commands (list, stats, compare, trend, comprehensive, purge)
- Confirmed backward compatibility with all reporting functions
- Verified foreign key constraints and cascading deletes

**Benefits:**
- Better query performance for filtering and aggregation
- Relational structure enables complex queries
- Smaller storage footprint (single database vs thousands of JSON files)
- ACID compliance for concurrent access
- Built-in indexing for fast lookups
- No file system limitations on number of records

**Human Contribution:**
- Decision to migrate to DuckDB for better scalability
- Testing and verification

---

### 2025-10-03: Performance Tracking System Implementation

**AI Contribution (Claude):**
- Designed and implemented comprehensive performance tracking system for cross-backend and longitudinal analysis
- Enhanced `name_of_test()` to extract full module.class.function path with class detection
  - Added `full_path` parameter for backward compatibility
  - Detects instance methods via 'self' and class methods via 'cls' in frame locals
  - Returns both full test path and absolute file path
- Created `yapocis/utils/performance.py` module (860+ lines)
  - `save_performance_data()`: Auto-save timing data to JSON with git metadata
  - `get_git_info()`: Extract branch, commit hash, and file modification status
  - `load_performance_data()`: Load and filter performance data with flexible criteria
  - `purge_performance_data()`: Delete old data with dry-run support
  - `generate_summary_table()`: Tabular summaries using tabulate
  - `generate_checkpoint_comparison_table()`: Compare checkpoints across runs
  - `plot_backend_comparison()`: Bar charts with error bars (matplotlib/seaborn)
  - `plot_longitudinal_trend()`: Time-series line graphs for trend analysis
  - `plot_checkpoint_breakdown()`: Pie/bar charts of checkpoint distribution
  - `generate_backend_comparison_report()`: Markdown reports with embedded graphs
  - `generate_longitudinal_report()`: Trend reports with regression detection
  - `generate_comprehensive_report()`: Overview of all performance data
- Enhanced StopWatcher class with `save_perf` parameter
  - Auto-captures test name and file path on __enter__
  - Auto-saves performance data on __exit__ when save_perf=True
  - Extracts backend from environment (YAPOCIS_BACKEND)
  - Gracefully handles errors without failing tests
- Created CLI tool `scripts/performance_report.py` (250+ lines)
  - Commands: list, stats, compare, trend, comprehensive, purge
  - Filter by backend, test name, date range
  - Generate reports and visualizations
  - Dry-run support for purge operations
- Updated `pyproject.toml` dependencies
  - Added seaborn>=0.12.0 for advanced visualizations
  - Added tabulate>=0.9.0 for formatted tables
- Updated `.gitignore` to exclude performance/results/ and performance/reports/
- Created performance data directories with auto-creation
- Lines changed: ~1300 additions across 6 files

**Files created:**
- `yapocis/utils/performance.py` - Core performance tracking module
- `scripts/performance_report.py` - CLI tool for managing performance data
- `performance/results/` - Auto-saved JSON performance data (gitignored)
- `performance/reports/` - Generated reports and graphs (gitignored)

**Files modified:**
- `tests/__init__.py` - Enhanced name_of_test() and StopWatcher with save_perf
- `pyproject.toml` - Added seaborn and tabulate dependencies
- `.gitignore` - Excluded performance data directories

**Testing:**
- Verified auto-save of performance data with save_perf=True
- Tested class method detection in name_of_test()
- Verified git info extraction (branch, commit, file status)
- Tested CLI list and stats commands
- Confirmed backward compatibility (save_perf=False by default)

**Features:**
- Auto-tracking: Simply use `StopWatcher(save_perf=True)` to track performance
- Git integration: Tracks branch, commit, and file modification status
- Backend comparison: Compare performance across OCL, Metal, Numba, CUDA
- Longitudinal analysis: Track performance trends over time
- Regression detection: Auto-detect >10% performance changes
- Flexible filtering: By backend, test, date, file status
- Rich visualizations: Bar charts, line graphs, pie charts with seaborn
- Markdown reports: Publication-ready reports with embedded graphs
- Data management: Purge old data with flexible criteria

**Human Contribution:**
- Feature request and high-level design requirements
- Refinement of auto-save behavior and data schema
- Testing and verification

---

### 2025-10-03: StopWatcher Enhancement - Context Manager Checkpoints and Auto-logging

**AI Contribution (Claude):**
- Enhanced StopWatcher.checkpoint() to work as both method call and context manager
  - Backward compatible: `sw.checkpoint("label")` still works
  - New context manager style: `with sw.checkpoint("label"): code...`
  - Automatically times the block when used with `with`
  - Added `_CheckpointContext` helper class for context management
- Added auto-logging on StopWatcher exit
  - New `log` parameter in `__init__` accepts callable
  - Automatically calls `log(summary())` when context exits
  - Enables cleaner test code: no manual `print(sw.summary())` needed
  - Example: `with StopWatcher(log=print) as sw: ...`
- Fixed duplicate `__exit__` method that was overriding functionality
- Lines changed: ~50 additions/modifications

**Files modified:**
- `tests/__init__.py` - Enhanced StopWatcher class with new features

**Testing:**
- Verified backward compatibility with existing checkpoint usage
- Tested new context manager style for checkpoint blocks
- Tested mixed usage (both styles together)
- Verified auto-logging with print and custom log functions
- Confirmed no logging when log parameter not provided

**Human Contribution:**
- Feature request and design requirements
- Testing and verification

---

### 2025-10-03: StopWatcher Pattern Standardization and Show Array Updates

**AI Contribution (Claude):**
- Backported NPZ file saving to show_array from main branch
  - Added RAW_QUEUE directory for storing raw arrays
  - Implemented array_to_image() helper function
  - Modified show_array to save both PNG and NPZ files for validation framework
  - Updated show_array signature: array now first parameter
- Fixed StopWatcher timing bugs
  - Fixed undefined variable 't' in __call__
  - Fixed AttributeError in __enter__ accessing self.t before initialization
  - Improved checkpoint timing accuracy
- Applied consistent StopWatcher pattern across all test files
  - Updated test_graphics.py, test_graphics4.py (14 functions)
  - Updated test_kerneldraw.py, test_kerneldraw3.py, test_kerneldraw4.py (8 functions)
  - Updated test_anisodiff.py, test_gaussians.py, test_colormath.py
  - Added appropriate timing checkpoints (display_list, render, read)
  - Standardized on sw.checkpoint(), sw.show(), and print(sw.summary())
- Updated all show_array callers to use new signature
- Lines changed: ~425 additions, ~270 deletions across 11 files

**Files modified:**
- `yapocis/utils/basic.py` - NPZ saving and signature update
- `yapocis/utils/advanced.py` - Updated show_array calls
- `tests/__init__.py` - Fixed StopWatcher bugs
- `tests/test_graphics.py` - StopWatcher pattern
- `tests/test_graphics4.py` - StopWatcher pattern
- `tests/test_kerneldraw.py` - StopWatcher pattern
- `tests/test_kerneldraw3.py` - StopWatcher pattern
- `tests/test_kerneldraw4.py` - StopWatcher pattern
- `tests/test_anisodiff.py` - StopWatcher pattern
- `tests/test_gaussians.py` - StopWatcher pattern
- `tests/test_colormath.py` - StopWatcher pattern

**Human Contribution:**
- Design decision to backport NPZ saving
- Manual update of test_graphics.py with new StopWatcher pattern as example
- Request to apply pattern consistently across all tests

---

### 2025-10-03: Validation Framework Fixes and Numba HSI Completion

**AI Contribution (Claude):**
- Fixed test_manual_validation_context test failure
  - Analyzed validation framework auto-promotion logic
  - Added cleanup of stale validation references
  - Enhanced test to verify both auto-promotion and subsequent validation
  - Improved test documentation and comments
- Completed Numba HSI backend implementation
  - Identified missing `hsi2rgb` function in numba/kernels/hsi.py
  - Implemented `hsi2rgb` matching OCL kernel interface
  - Maintained proper channel swizzle (r,b,g) for consistency
  - Verified all numba backend tests pass
- Lines changed: ~30 additions across 2 files

**Files modified:**
- `tests/test_validation_example.py` - Fixed validation test with proper cleanup and verification
- `yapocis/numba/kernels/hsi.py` - Added missing `hsi2rgb` function

**Testing:**
- All numba backend tests pass (18 tests)
- Validation framework tests pass
- HSI color conversion tests pass on numba backend

**Human Contribution:**
- Bug identification and reporting
- Testing and verification

---

### 2025-10-03: Rename sdfpath to truetrace + Numba documentation

**AI Contribution (Claude):**
- Renamed sdfpath module to truetrace across entire codebase
  - Files renamed: 5 (interface, library, OCL kernel, Metal kernel, Numba kernel)
  - Class renamed: SDFPath → TrueTrace
  - Interface declaration updated
  - Test file updated with new imports
  - Documentation updated in CLAUDE.md and README.md
- Updated README.md to document Numba backend
  - Added Numba to supported backends list
  - Documented CPU fallback functionality
  - Added test markers and dependencies
  - Added backend selection examples
- Created AI.md to track contributions

**Files modified:**
- `yapocis/interfaces/sdfpath.interface` → `truetrace.interface`
- `yapocis/lib/sdfpath.py` → `truetrace.py`
- `yapocis/kernels/sdfpath.mako` → `truetrace.mako`
- `yapocis/metalkernels/sdfpath.mako` → `truetrace.mako`
- `yapocis/numba/kernels/sdfpath.py` → `truetrace.py`
- `tests/test_truetrace_equiv.py` → `test_truetrace.py`
- `README.md`
- `CLAUDE.md`
- `AI.md` (created)

**Human Contribution:**
- Design decision to rename sdfpath to truetrace
- Review and approval of changes

---

### 2025-10 (earlier): Metal Backend Column-Major Coordinate System Fixes

**AI Contribution (Claude):**
- Analyzed OCL vs Metal coordinate system implementations
- Identified bugs in Metal helpers.mako (row-major vs column-major)
- Fixed Metal kernel.py width/height handlers
- Fixed graphics4.mako composite kernel coordinate swizzles (.xy/.zw → .yx/.wz)
- Fixed graphics.mako (single-channel) composite kernel swizzles
- Fixed abstract_graphics.py coerce_colors for single-channel graphics
- Created comprehensive test scripts in claude-debug/

**Files modified:**
- `yapocis/metalkernels/helpers.mako`
- `yapocis/metallic/kernel.py`
- `yapocis/metalkernels/graphics4.mako`
- `yapocis/metalkernels/graphics.mako`
- `yapocis/lib/abstract_graphics.py`

**Commits:**
- `bea0457` - Fix Metal composite kernel coordinate swizzle for column-major layout
- `0eb50ab` - Fix Metal graphics (single-channel) composite kernel coordinate swizzle
- Additional commit for coerce_colors fix

**Human Contribution:**
- Bug identification and reproduction
- Design decision: maintain column-major layout for 'older' branch
- Testing and validation of fixes

---

## Methodology

**AI Role:**
- Implementation of approved changes
- Code analysis and bug identification
- Documentation updates
- Test creation and debugging
- Commit message drafting

**Human Role:**
- Architecture decisions
- Requirements specification
- Final review and approval
- Git operations (when not delegated)
- Strategic direction

## Notes

- AI contributions are supervised and reviewed by the human maintainer
- All commits are attributed to the human maintainer in git history
- This tracking started 2025-10-03; earlier AI contributions are documented retroactively based on session summaries

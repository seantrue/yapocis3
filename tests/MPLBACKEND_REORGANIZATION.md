# Matplotlib Backend Tests Reorganization

## Overview

Consolidated all matplotlib backend (mplbackend) tests into a dedicated `tests/lib/mplbackend/` directory. Previously these tests were scattered across `tests/backend/`, `tests/integration/`, and `tests/lib/graphics/`.

## Changes Made

### New Directory Structure

Created `tests/lib/mplbackend/` with all matplotlib backend tests:

```
tests/lib/mplbackend/
├── __init__.py
├── test_advanced.py         # Advanced rendering scenarios
├── test_backend.py          # Core mplbackend functionality
├── test_comparison.py       # Backend comparison tests
├── test_images.py           # Image integration tests
├── test_math_comparison.py  # Math rendering comparison
├── test_pyplot.py           # Pyplot adaptation tests
└── test_rendering.py        # Rendering tests
```

### Files Moved

**From `tests/backend/` (4 files):**
- `test_advanced.py` → `tests/lib/mplbackend/test_advanced.py`
- `test_comparison.py` → `tests/lib/mplbackend/test_comparison.py`
- `test_math_comparison.py` → `tests/lib/mplbackend/test_math_comparison.py`
- `test_rendering.py` → `tests/lib/mplbackend/test_rendering.py`

**From `tests/integration/` (2 files):**
- `test_matplotlib_backend.py` → `tests/lib/mplbackend/test_backend.py` (renamed)
- `test_pyplot_adapted.py` → `tests/lib/mplbackend/test_pyplot.py` (renamed)

**From `tests/lib/graphics/` (1 file):**
- `test_matplotlib_images.py` → `tests/lib/mplbackend/test_images.py` (renamed)

### Backend Directory Cleaned Up

`tests/backend/` now contains only core backend infrastructure tests:
- `test_smoke.py` - Interface compilation tests across all backends
- `test_runtime.py` - Basic backend execution tests

These are the only tests that apply to **all** backends (OCL, Metal, CUDA, Numba). All matplotlib-specific backend tests are now in `tests/lib/mplbackend/`.

## Test Counts

- **Mplbackend tests**: 69 tests
- **Backend tests** (smoke + runtime): 64 tests
- **Total tests**: 655 (unchanged)

## Benefits

### 1. Clear Organization

All matplotlib backend tests are now in one logical location under `tests/lib/` alongside other library component tests (graphics, processing, compute, etc.).

### 2. Easy to Find and Run

```bash
# Run all matplotlib backend tests
pytest tests/lib/mplbackend/

# Run specific mplbackend test category
pytest tests/lib/mplbackend/test_backend.py
pytest tests/lib/mplbackend/test_pyplot.py
pytest tests/lib/mplbackend/test_rendering.py
```

### 3. Logical Separation

- `tests/backend/` - Core backend infrastructure (applies to all backends)
- `tests/lib/mplbackend/` - Matplotlib backend library tests (specific to mplbackend)
- `tests/lib/graphics/` - Graphics primitive tests (Graphics1, Graphics4, etc.)

### 4. Better Test Discovery

Tests are grouped by what they test rather than by type:
- Before: matplotlib tests split across backend/, integration/, and graphics/
- After: All matplotlib backend tests in lib/mplbackend/

## Running Tests

### All Mplbackend Tests

```bash
# Run all 69 mplbackend tests
pytest tests/lib/mplbackend/
```

### Specific Test Categories

```bash
# Core backend functionality (10 tests)
pytest tests/lib/mplbackend/test_backend.py

# Pyplot adaptation (22 tests)
pytest tests/lib/mplbackend/test_pyplot.py

# Advanced rendering scenarios
pytest tests/lib/mplbackend/test_advanced.py

# Image integration
pytest tests/lib/mplbackend/test_images.py

# Rendering tests
pytest tests/lib/mplbackend/test_rendering.py

# Comparison tests
pytest tests/lib/mplbackend/test_comparison.py
pytest tests/lib/mplbackend/test_math_comparison.py
```

### Pattern Matching

```bash
# Run all tests with "matplotlib" or "mpl" in the path
pytest -k "mplbackend" tests/

# Run specific test within mplbackend
pytest tests/lib/mplbackend/test_backend.py::test_backend_loaded
```

## Test Content

### test_backend.py (Core Tests)
- Backend loading verification
- Basic text rendering
- Math text (Greek letters, subscripts, superscripts)
- LaTeX-style formatting
- Fractions and complex math expressions

### test_pyplot.py (Pyplot Tests)
- Pyplot API adaptation
- Figure and axes creation
- Plot types (line, scatter, bar, etc.)
- Legends, labels, titles
- Multiple subplots

### test_advanced.py (Advanced Rendering)
- Multiple shapes on same canvas
- Transparency and alpha blending
- Overlapping elements
- Complex compositions
- Edge cases

### test_rendering.py (Rendering Tests)
- Shape rendering (rectangles, circles, lines)
- Path rendering
- Color and style handling
- Coordinate transformations

### test_comparison.py (Backend Comparison)
- Compare mplbackend output with standard matplotlib backends
- Consistency checks
- Output format validation

### test_math_comparison.py (Math Rendering)
- Math text rendering comparison
- Font handling
- Symbol rendering accuracy

### test_images.py (Image Tests)
- Image loading and display
- Image transformations
- Color mapping
- Array to image conversion

## Updated Documentation

The test suite README and other documentation files will reflect this new organization:
- `tests/README.md` - Updated directory structure
- `tests/MIGRATION_SUMMARY.md` - Notes new mplbackend directory

## Verification

All tests verified working after reorganization:

```bash
# Verify mplbackend tests collect
pytest --collect-only tests/lib/mplbackend/
# 69 tests collected ✓

# Verify backend tests collect
pytest --collect-only tests/backend/
# 64 tests collected ✓

# Verify total test count
pytest --collect-only tests/
# 655 tests collected ✓
```

## Date

November 6, 2025

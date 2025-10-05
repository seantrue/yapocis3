# AI Contributions to Yapocis

This document tracks contributions made by AI assistants (primarily Claude) versus the human maintainer.

## Summary Statistics

- **Total AI-assisted commits**: ~20
- **Major features developed with AI assistance**:
  - Infrastructure: Package modernization (hatchling, modern tooling), automated changelog generation
  - Backend work: Metal backend fixes, coordinate system standardization, Numba backend completeness
  - Testing: Validation framework, StopWatcher enhancements, test isolation fixes
  - Performance: Performance tracking system, DuckDB migration, reproducibility improvements
  - Code quality: Test refactoring, random data centralization
- **Lines of code**:
  - ~1300 lines: Performance tracking system (2025-10-03)
  - ~400 lines: DuckDB migration (2025-10-03)
  - ~550 lines: Random data centralization (2025-10-04)
  - ~300 lines: Package modernization (2025-10-05)
  - ~200 lines: Changelog system (2025-10-05)
  - ~100 lines: Documentation updates (2025-10-05)

## Detailed Contributions

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

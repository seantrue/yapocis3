# AI Contributions to Yapocis

This document tracks contributions made by AI assistants (primarily Claude) versus the human maintainer.

## Summary Statistics

- **Total AI-assisted commits**: ~13
- **Major features developed with AI assistance**: Metal backend fixes, coordinate system standardization, renamed truetrace module, validation framework fixes, numba backend completeness, StopWatcher enhancements, performance tracking system, DuckDB migration, test refactoring
- **Lines of code**: ~1300 lines added for performance tracking (2025-10-03), ~400 lines modified for DuckDB migration (2025-10-03), ~100 lines refactored in test_graphics4.py (2025-10-04)

## Detailed Contributions

### 2025-10-04: StopWatcher Pattern Refactoring in test_graphics4.py

**AI Contribution (Claude):**
- Refactored all 15 test functions in `tests/test_graphics4.py` to match the updated StopWatcher pattern from `tests/test_graphics.py`
- Pattern changes applied:
  - Added descriptive names to StopWatcher initialization: `StopWatcher("test_name", kernel=grph)`
  - Converted checkpoint calls to context manager style: `with sw.checkpoint("label"):`
  - Removed manual `print(sw.summary())` calls (now auto-logged via save_perf)
  - Updated `sw.show()` signature from `sw.show(grph, canvas)` to `sw.show(canvas)`
- Applied formatting improvements via linter (black/ruff)
  - Standardized spacing and line breaks
  - Fixed import organization
  - Improved code readability
- Lines changed: ~100 modifications across 1 file

**Refactored test functions:**
- `test_lines`, `test_rects`, `test_circles`, `test_triangles`, `test_ellipses`
- `test_composite`, `test_composite_sprites`, `test_composite_mosaic`
- `test_simple_icons`, `test_random_icons`
- `test_simple_texts`, `test_text_rotation`
- `test_texts_stressed`, `test_texts_stressed_rotation`, `test_typesetting`
- `test_one`

**Files modified:**
- `tests/test_graphics4.py` - Refactored StopWatcher usage pattern

**Testing:**
- Verified test_one passes with new pattern
- Confirmed consistent usage across all test functions
- Maintained backward compatibility with existing functionality

**Benefits:**
- Consistent StopWatcher usage across test suite
- Cleaner checkpoint timing with context managers
- Better integration with performance tracking system
- Improved code readability and maintainability

**Human Contribution:**
- Request to refactor based on test_graphics.py pattern
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

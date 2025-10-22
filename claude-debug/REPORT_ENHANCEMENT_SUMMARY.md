# Performance Report Enhancement - Backend Comparison

## Summary

Enhanced the performance reporting system to provide comprehensive cross-backend (branch) comparison reports. Added a new `compare-backends` command to the report script that generates detailed comparisons across all available backends (OpenCL, Metal, CUDA, Numbax).

**Date**: 2025-10-17
**Files Modified**: 2
**New Features**: 1 command, 1 function

## Changes Made

### 1. Added New Command: `compare-backends`

**File**: `yapocis/scripts/report.py`

Added a new command to the CLI tool that generates cross-backend performance comparisons:

```bash
# Generate cross-backend comparison for all tests
python yapocis/scripts/report.py compare-backends

# Filter by test pattern
python yapocis/scripts/report.py compare-backends --test=graphics

# Filter by branch
python yapocis/scripts/report.py compare-backends --branch=main

# Limit results per backend
python yapocis/scripts/report.py compare-backends --limit=50
```

**Features**:
- Compares performance across all available backends
- Groups results by test name
- Calculates speedups relative to fastest backend (baseline)
- Filters by test pattern and/or git branch
- Generates both tables and visualizations

### 2. Added New Function: `generate_backend_cross_comparison_report()`

**File**: `yapocis/utils/performance.py`

Created comprehensive backend comparison report generator with the following features:

**Input Parameters**:
- `test_pattern` (optional): Filter tests by name pattern (partial match)
- `branch` (optional): Filter by git branch
- `limit` (default: 100): Maximum results per backend
- `output_path` (optional): Custom output location

**Report Contents**:

1. **Summary Statistics**
   - Number of tests analyzed
   - Available backends
   - Total runs
   - Applied filters

2. **Performance Comparison Visualization**
   - Grouped bar chart with error bars
   - Shows mean time ± standard deviation
   - Color-coded by backend
   - Limited to first 15 tests for readability

3. **Results Table**
   - Rows: Test names
   - Columns: Backends (ocl, metal, cuda, numbax)
   - Format: `mean±std (speedup) [runs]`
   - Speedup calculated relative to fastest backend
   - Fastest backend labeled as "baseline"

4. **Backend Summary**
   - Average performance across all tests per backend
   - Overall statistics

5. **Interpretation Guide**
   - Explanation of speedup calculation
   - Notes on baseline selection
   - Best practices for comparison

## Report Format Example

```markdown
# Backend Cross-Comparison Report

Generated: 2025-10-17 21:30:00

## Summary

- Tests analyzed: 10
- Backends: metal, ocl
- Total runs: 200
- Filtered by pattern: `graphics`

## Performance Comparison

![Backend Comparison](backend_cross_comparison_20251017_213000.png)

## Results Table

| Test                   | metal                         | ocl                           |
|------------------------|-------------------------------|-------------------------------|
| test_graphics4.circles | 4.23±0.12ms (baseline) [50]   | 4.50±0.15ms (0.94x) [50]     |
| test_graphics4.rects   | 3.87±0.10ms (baseline) [50]   | 4.10±0.12ms (0.94x) [50]     |

## Backend Summary

- **metal**: Average 4.05ms across 2 tests
- **ocl**: Average 4.30ms across 2 tests

## Interpretation

- **Speedup**: Ratio of baseline (fastest) backend time to current backend time
- **Baseline**: The fastest backend for each test (shown as speedup 1.0x)
- **N/A**: Backend has no performance data for this test
- Lower times and higher speedups indicate better performance
```

## Usage Examples

### Example 1: Compare All Backends for Graphics Tests

```bash
python yapocis/scripts/report.py compare-backends --test=graphics
```

**Output**:
- Markdown report in `performance/reports/backend_cross_comparison_YYYYMMDD_HHMMSS.md`
- PNG chart in `performance/reports/backend_cross_comparison_YYYYMMDD_HHMMSS.png`

**Use Case**: See how Metal vs OpenCL performance compares for all graphics-related tests.

### Example 2: Compare All Tests on Main Branch

```bash
python yapocis/scripts/report.py compare-backends --branch=main
```

**Output**: Report showing performance across all backends for tests run on the main branch only.

**Use Case**: Exclude experimental branch results, focus on stable performance.

### Example 3: Quick Comparison (Limited Results)

```bash
python yapocis/scripts/report.py compare-backends --test=convol --limit=20
```

**Output**: Report with up to 20 runs per backend for convolution-related tests.

**Use Case**: Fast overview without processing hundreds of runs.

## Implementation Details

### Data Flow

1. **Query Database**
   - Queries `performance_runs` table in DuckDB
   - Filters by test pattern and/or branch
   - Orders by test name, backend, timestamp (DESC)

2. **Aggregate Results**
   - Groups data by test name and backend
   - Calculates mean, std, min, max for each group
   - Tracks number of runs

3. **Calculate Speedups**
   - For each test, finds fastest backend (baseline)
   - Calculates speedup as: `baseline_time / backend_time`
   - Baseline gets speedup of 1.0x (or labeled "baseline")

4. **Generate Visualizations**
   - Creates grouped bar chart using matplotlib/seaborn
   - Shows up to 15 tests (for readability)
   - Error bars show standard deviation
   - Different color per backend

5. **Format Table**
   - Uses tabulate for GitHub-flavored markdown
   - Cells: `mean±std (speedup) [runs]`
   - Example: `4.23±0.12ms (1.05x) [50]`

6. **Write Report**
   - Markdown file with embedded chart reference
   - Self-contained (chart saved alongside)
   - Timestamped filename prevents overwrites

### Database Schema

Queries this table structure:

```sql
CREATE TABLE performance_runs (
    id INTEGER PRIMARY KEY,
    test_name VARCHAR NOT NULL,
    test_file VARCHAR NOT NULL,
    backend VARCHAR NOT NULL,           -- 'ocl', 'metal', 'cuda', 'numbax'
    device_name VARCHAR,
    driver_version VARCHAR,
    branch VARCHAR NOT NULL,            -- Git branch
    commit VARCHAR NOT NULL,            -- Git commit hash
    file_status VARCHAR NOT NULL,       -- 'clean', 'modified', 'untracked'
    timestamp TIMESTAMP NOT NULL,
    timestamp_unix DOUBLE NOT NULL,
    total_time DOUBLE NOT NULL,
    ...
)
```

## Advantages Over Existing Reports

| Feature | Old `compare` Command | New `compare-backends` Command |
|---------|----------------------|--------------------------------|
| Scope | Single test only | All tests or filtered pattern |
| Grouping | By run | By test name |
| Baseline | Fixed | Dynamic (fastest per test) |
| Speedup | No | Yes (vs baseline) |
| Cross-test | No | Yes |
| Branch filter | No | Yes |

## Integration with Existing Tools

The new command complements existing report commands:

- `list`: List raw performance data
- `stats`: Show statistics grouped by test/backend/device
- `compare <test>`: Detailed comparison for ONE specific test
- `trend <test>`: Longitudinal trends over time for ONE test
- `comprehensive`: Overview of ALL data with aggregation
- **`compare-backends` (NEW)**: Cross-backend comparison grouped by test

## Example Workflow

1. **Run tests on multiple backends**:
   ```bash
   YAPOCIS_BACKEND=ocl pytest tests/test_graphics4.py
   YAPOCIS_BACKEND=metal pytest tests/test_graphics4.py
   ```

2. **Generate cross-backend comparison**:
   ```bash
   python yapocis/scripts/report.py compare-backends --test=graphics
   ```

3. **View results**:
   ```bash
   open performance/reports/backend_cross_comparison_*.md
   ```

4. **Analyze**:
   - Which backend is fastest for each test?
   - How consistent are results (std deviation)?
   - Are there specific tests where one backend excels?

## Future Enhancements

Potential additions:

1. **HTML Output**: Interactive reports with sortable tables
2. **Statistical Tests**: t-tests to determine if differences are significant
3. **Historical Comparison**: Show trends over time per backend
4. **Device Breakdown**: Group by backend AND device
5. **Checkpoint Analysis**: Compare specific checkpoints across backends
6. **Export to CSV**: For external analysis in Excel/R/Python

## Dependencies

Required packages (already listed in pyproject.toml):
- `duckdb`: Database queries
- `matplotlib`: Chart generation
- `seaborn`: Enhanced styling
- `tabulate`: Markdown table formatting

## Backward Compatibility

✅ **Fully backward compatible**:
- Existing commands unchanged
- Existing reports work as before
- New command is additive only
- No database schema changes (uses existing tables)

## Testing

To verify the implementation:

```bash
# Test the new command exists
python yapocis/scripts/report.py compare-backends --help

# Test with mock data (if any exists)
python yapocis/scripts/report.py compare-backends

# Test filtering
python yapocis/scripts/report.py compare-backends --test=test
python yapocis/scripts/report.py compare-backends --branch=main
```

## Files Modified

1. **yapocis/scripts/report.py** (+18 lines)
   - Added `cmd_compare_backends()` function
   - Added argument parser for `compare-backends` command
   - Added command to dispatcher dict
   - Updated help examples

2. **yapocis/utils/performance.py** (+293 lines)
   - Added `generate_backend_cross_comparison_report()` function
   - Comprehensive report generation with statistics and visualization

## Conclusion

The new `compare-backends` command provides a powerful tool for analyzing performance differences across backends. It automatically:

- Identifies the fastest backend for each test (baseline)
- Calculates relative speedups
- Shows statistical consistency (via std deviation)
- Generates visual comparisons
- Supports filtering by test pattern and branch

This enhancement makes it easy to answer questions like:
- "Is Metal faster than OpenCL for graphics tests?"
- "Which backend should I use for convolution operations?"
- "How do backend performances compare on the main branch?"

All while maintaining full backward compatibility with existing reporting tools.

---

**Status**: ✅ **Complete and Ready to Use**
**Documentation**: Updated in report.py help text
**Testing**: Manual verification successful

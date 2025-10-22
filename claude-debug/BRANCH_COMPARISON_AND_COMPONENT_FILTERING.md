# Performance Report Enhancements: Branch Comparison & Component Filtering

## Summary

Added two major enhancements to the performance reporting system:

1. **Branch Comparison Reports** - Compare performance across different git branches
2. **Component Filtering** - Select all tests matching a component (e.g., `graphics4` matches all graphics4 tests)

**Date**: 2025-10-17
**Files Modified**: 2
**New Features**: 1 command, 1 function, enhanced filtering

## Changes Made

### 1. New Command: `compare-branches`

Compare performance across different git branches to detect regressions or improvements.

**Usage**:
```bash
# Compare all branches (all backends, all tests)
python yapocis/scripts/report.py compare-branches

# Compare specific branches
python yapocis/scripts/report.py compare-branches --branches main feature-x

# Compare branches for specific backend
python yapocis/scripts/report.py compare-branches --backend=metal --branches main dev

# Compare branches for component
python yapocis/scripts/report.py compare-branches --component=graphics4

# Compare branches for test pattern
python yapocis/scripts/report.py compare-branches --test=convol --backend=ocl
```

**Options**:
- `--test`: Filter by test name (partial match)
- `--component`: Filter by component name (matches test name or file path)
- `--backend`: Filter by specific backend (default: all backends)
- `--branches`: Specific branches to compare (default: all branches)
- `--limit`: Maximum results per branch (default: 100)

### 2. Component Filtering (All Commands)

Added `--component` parameter to both `compare-backends` and `compare-branches` commands.

**How It Works**:
- Component name matches against both `test_name` and `test_file` columns
- Example: `--component=graphics4` matches:
  - `test_graphics4.test_circles`
  - `test_graphics4.test_rects`
  - Any test in file containing "graphics4"

**Difference from `--test`**:
- `--test`: Matches only test name (e.g., `test_graphics4.test_batch`)
- `--component`: Matches test name OR file path (e.g., all tests in `test_graphics4.py`)

**Usage**:
```bash
# Compare backends for all graphics4 tests
python yapocis/scripts/report.py compare-backends --component=graphics4

# Compare branches for all convolution tests
python yapocis/scripts/report.py compare-branches --component=convol --backend=metal
```

## Implementation Details

### New Function: `generate_branch_comparison_report()`

**Location**: `yapocis/utils/performance.py`

**Function Signature**:
```python
def generate_branch_comparison_report(
    test_pattern: str | None = None,
    backend: str | None = None,
    component: str | None = None,
    branches: list[str] | None = None,
    limit: int = 100,
    output_path: Path | None = None
) -> Path
```

**Report Structure**:

1. **Summary**
   - Tests analyzed
   - Branches compared
   - Backends included
   - Total runs
   - Applied filters

2. **Visualization**
   - Single backend: Grouped bar chart by branch
   - Multiple backends: Grouped by backend:branch combinations
   - Shows mean ± std deviation
   - Up to 12 tests (for readability)

3. **Results Table**
   - Single backend: Columns = branches
   - Multiple backends: Columns = backend:branch pairs
   - Format: `mean±std (speedup) [runs]`
   - Baseline: Fastest branch (or backend:branch) per test

4. **Branch Summary**
   - Average performance across all tests per branch

5. **Interpretation Guide & Use Cases**

### Enhanced `generate_backend_cross_comparison_report()`

**Updated Signature**:
```python
def generate_backend_cross_comparison_report(
    test_pattern: str | None = None,
    branch: str | None = None,
    component: str | None = None,  # NEW
    limit: int = 100,
    output_path: Path | None = None
) -> Path
```

**Changes**:
- Added `component` parameter
- Query now includes: `AND (test_name LIKE ? OR test_file LIKE ?)`
- Summary shows component filter when applied

## Report Format Examples

### Example 1: Branch Comparison (Single Backend)

```markdown
# Branch Comparison Report

Generated: 2025-10-17 21:45:00

## Summary

- Tests analyzed: 5
- Branches: feature-metal, main
- Backends: metal
- Total runs: 100
- Filtered by component: `graphics4`

## Results Table

| Test                   | feature-metal             | main                      |
|------------------------|---------------------------|---------------------------|
| test_graphics4.circles | 4.15±0.10ms (baseline) [25] | 4.50±0.12ms (0.92x) [25]  |
| test_graphics4.rects   | 3.80±0.08ms (baseline) [25] | 4.10±0.11ms (0.93x) [25]  |

## Branch Summary

- **feature-metal**: Average 3.98ms across 5 test-backend combinations
- **main**: Average 4.30ms across 5 test-backend combinations

## Use Cases

- **Feature branch comparison**: See if your feature branch introduced performance regressions
- **Before/after comparison**: Compare performance before and after a specific change
- **Release validation**: Ensure release branches maintain or improve performance
```

### Example 2: Branch Comparison (Multiple Backends)

```markdown
# Branch Comparison Report

## Results Table

| Test         | metal:main | metal:dev | ocl:main | ocl:dev |
|--------------|------------|-----------|----------|---------|
| test_convol  | 5.2±0.1ms (baseline) | 5.3±0.1ms (0.98x) | 6.1±0.2ms (0.85x) | 6.2±0.2ms (0.84x) |
```

### Example 3: Component Filtering

```bash
# Select all graphics4 tests across all backends
python yapocis/scripts/report.py compare-backends --component=graphics4
```

**Matches**:
- `test_graphics4.test_circles`
- `test_graphics4.test_rects`
- `test_graphics4.test_triangles`
- `test_graphics4.test_ellipses`
- `test_graphics4.test_lines`
- `test_graphics4.test_composite`

**Does NOT match**:
- `test_graphics.test_circles` (different component)
- `test_median3x3.test_median` (unrelated)

## Use Cases

### Use Case 1: Feature Branch Performance Check

**Scenario**: You've created a feature branch with Metal kernel packages. Want to verify no regressions.

```bash
# Compare your feature branch against main for Metal backend
python yapocis/scripts/report.py compare-branches \
  --backend=metal \
  --branches main feature-kernel-packages \
  --component=graphics4
```

**Result**: Report shows if feature branch is faster/slower than main for graphics4 tests.

### Use Case 2: Backend Selection per Branch

**Scenario**: Testing if Metal is faster than OpenCL on main branch.

```bash
# Compare backends on main branch only
python yapocis/scripts/report.py compare-backends \
  --branch=main \
  --component=convol
```

**Result**: Shows which backend (Metal vs OpenCL) is faster for convolution tests on main.

### Use Case 3: Cross-Branch, Cross-Backend Analysis

**Scenario**: Comprehensive analysis of performance across all dimensions.

```bash
# Compare all branches and backends
python yapocis/scripts/report.py compare-branches --component=graphics4
```

**Result**: Matrix showing backend:branch combinations, identifying best performer.

### Use Case 4: Component-Based Testing

**Scenario**: Run all tests for a specific component.

```bash
# Get all graphics4 tests regardless of exact test name
python yapocis/scripts/report.py compare-backends --component=graphics4

# vs specific test pattern (less comprehensive)
python yapocis/scripts/report.py compare-backends --test=test_graphics4.test_circles
```

**Difference**:
- `--component=graphics4`: All graphics4 tests (circles, rects, triangles, etc.)
- `--test=test_circles`: Only tests with "circles" in name

## Database Query Changes

### Component Filtering Logic

```sql
-- Old query (test pattern only)
SELECT * FROM performance_runs
WHERE test_name LIKE '%graphics%'

-- New query (component matching)
SELECT * FROM performance_runs
WHERE (test_name LIKE '%graphics4%' OR test_file LIKE '%graphics4%')
```

**Benefits**:
- Matches tests by file path, not just test name
- More flexible grouping (all tests in a file)
- Better for module/component-level analysis

## Comparison Table Formats

### Single Backend Format

**Table Structure**:
```
| Test       | Branch1  | Branch2  | Branch3  |
|------------|----------|----------|----------|
| test_foo   | 4.2ms... | 4.5ms... | 4.1ms... |
```

**Baseline**: Fastest branch per test

### Multiple Backend Format

**Table Structure**:
```
| Test       | be1:br1  | be1:br2  | be2:br1  | be2:br2  |
|------------|----------|----------|----------|----------|
| test_foo   | 4.2ms... | 4.5ms... | 5.1ms... | 5.3ms... |
```

**Baseline**: Fastest backend:branch combination per test

## Visualization Differences

### Single Backend Visualization

- **X-axis**: Test names
- **Bars**: Grouped by branch
- **Colors**: One color per branch
- **Legend**: Branch names
- **Title**: "Branch Performance Comparison ({backend})"

### Multiple Backend Visualization

- **X-axis**: Test names
- **Bars**: Grouped by backend:branch combinations
- **Colors**: One color per combination
- **Legend**: "backend:branch" labels
- **Title**: "Branch Performance Comparison (All Backends)"

## Files Modified

### 1. `yapocis/scripts/report.py`

**Changes** (+47 lines):
- Added `cmd_compare_branches()` function
- Updated `cmd_compare_backends()` to pass component parameter
- Added `compare-branches` command parser with options
- Added `--component` option to `compare-backends` parser
- Updated command dispatcher
- Updated help examples

### 2. `yapocis/utils/performance.py`

**Changes** (+427 lines):
- Enhanced `generate_backend_cross_comparison_report()`:
  - Added `component` parameter
  - Updated query to support component filtering
  - Updated error messages and report summary

- Added `generate_branch_comparison_report()`:
  - Full branch comparison implementation
  - Supports single/multiple backend modes
  - Component and test pattern filtering
  - Visualization generation
  - Comprehensive markdown report

## Command Reference

### Updated Commands

```bash
# compare-backends (enhanced with --component)
python yapocis/scripts/report.py compare-backends [OPTIONS]

Options:
  --test TEST              Filter by test name (partial match)
  --component COMPONENT    Filter by component (matches name/path)
  --branch BRANCH          Filter by git branch
  --limit LIMIT            Maximum results per backend (default: 100)

# compare-branches (NEW)
python yapocis/scripts/report.py compare-branches [OPTIONS]

Options:
  --test TEST              Filter by test name (partial match)
  --component COMPONENT    Filter by component (matches name/path)
  --backend BACKEND        Filter by specific backend
  --branches BR [BR ...]   Specific branches to compare
  --limit LIMIT            Maximum results per branch (default: 100)
```

### Existing Commands (Unchanged)

```bash
list                   List performance data
stats                  Show performance statistics
compare TEST           Generate backend comparison for single test
trend TEST             Generate longitudinal trend report
comprehensive          Generate comprehensive report
purge                  Delete old performance data
```

## Integration with Existing Reports

| Command | Scope | Grouping | Filtering |
|---------|-------|----------|-----------|
| `compare <test>` | Single test | By run | Test name (exact) |
| `compare-backends` | All tests | By test & backend | Test/component/branch |
| `compare-branches` (NEW) | All tests | By test & branch | Test/component/backend |
| `comprehensive` | All tests | Aggregated | Test pattern |

## Backward Compatibility

✅ **Fully backward compatible**:
- All existing commands work unchanged
- Existing reports unaffected
- New parameters are optional
- No database schema changes
- Existing function signatures preserved (new parameters added at end)

## Testing Verification

```bash
# Verify commands exist
python yapocis/scripts/report.py --help | grep compare

# Verify component filtering
python yapocis/scripts/report.py compare-backends --help | grep component

# Verify branch comparison
python yapocis/scripts/report.py compare-branches --help
```

**Expected Output**:
- `compare-backends` and `compare-branches` both listed
- `--component` option shown in help
- All options properly documented

## Future Enhancements

Potential additions:

1. **Wildcard Components**: Support `*` for patterns (e.g., `graphics*` matches graphics, graphics4)
2. **Multiple Components**: Allow `--component graphics4 --component median3x3`
3. **Regex Support**: Full regex matching for complex patterns
4. **Component Auto-Discovery**: List all available components
5. **Diff Mode**: Show only tests with significant differences between branches
6. **Threshold Filters**: Only show tests with >10% difference

## Best Practices

### When to Use Component Filtering

**Use `--component`** when:
- Testing all variations of a module (e.g., all graphics4 tests)
- Analyzing file-level performance
- Grouping related tests

**Use `--test`** when:
- Searching for specific test name
- Need exact match or specific pattern
- Testing single function

### When to Use Branch Comparison

**Compare branches** to:
- Verify feature branches don't regress performance
- Track performance across releases
- Compare optimization attempts
- Validate refactoring didn't slow down code

**Compare backends** to:
- Choose best backend for deployment
- Understand backend-specific performance
- Validate backend implementations

## Conclusion

These enhancements provide powerful new ways to analyze performance data:

1. **Branch Comparison** - Essential for continuous performance monitoring across development lifecycle
2. **Component Filtering** - Simplifies analysis of related tests without complex patterns

Together, they enable comprehensive performance analysis:
- Compare your feature branch vs main
- See if Metal is faster than OpenCL for graphics tests
- Track performance regressions across releases
- Validate optimization efforts

All while maintaining full backward compatibility with existing reporting infrastructure.

---

**Status**: ✅ **Complete and Ready to Use**
**Files Modified**: 2
**Lines Added**: ~474
**New Commands**: 1
**Enhanced Commands**: 1
**Backward Compatible**: Yes

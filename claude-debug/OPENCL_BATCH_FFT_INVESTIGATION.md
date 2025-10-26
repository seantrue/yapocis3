# OpenCL Batch FFT Investigation Summary

## Problem Statement

OpenCL batch FFT produces incorrect results (2x amplification for size 64+, NaN for other cases in comprehensive tests).

---

## Key Findings

### ✅ What Works

1. **Non-batch OpenCL FFT**: Works perfectly for all sizes (4-1024) after adding explicit `local_size` parameter
   - Error: 0.000000e+00 for sizes 64, 128
   - Required fix: Added `local_size=(fft_size,)` to `forward()` and `inverse()` methods

2. **Batch FFT for small sizes (4-32)**: Works perfectly
   - Size 4: error = 0.000000e+00
   - Size 8: error = 0.000000e+00
   - Size 16: error = 0.000000e+00
   - Size 32: error = 0.000000e+00

3. **Work-group configuration**: Correct
   - For batch_size=1, fft_size=64 with global_size=(64,), local_size=(64,)
   - Creates exactly 1 work-group (verified via debug kernel)
   - group_id = 0, lid ranges 0-63 as expected

4. **Data passing**: Correct
   - Simple copy kernel transfers data correctly
   - No duplication or corruption in input buffer

### ❌ What Fails

**Batch FFT for sizes ≥ 64**: Produces 2x amplified results
- Size 64: error = 1.000000 (exactly 2x expected value)
- Size 128: error = 3.000000
- Size 256: error = 6.991570

**Pattern**: For impulse at position 0, expected result is [1,1,1,1...], actual result is [2, 1.995, 1.981, ...]

---

## Investigation Steps

### 1. Fixed non-batch FFT local_size issue ✓
**Problem**: OpenCL non-batch FFT was broken (huge values, NaN)
**Cause**: Missing `local_size` parameter - OpenCL used default work-group size instead of fft_size
**Fix**: Added explicit `local_size=(fft_size,)` to `forward()` and `inverse()` in `src/yapocis/lib/fft.py`

**Result**: Non-batch FFT now works perfectly for all sizes

### 2. Identified size boundary ✓
**Test**: Swept FFT sizes from 4 to 256
**Finding**: Batch FFT works perfectly for sizes 4-32, fails starting at size 64

### 3. Compared non-batch vs batch kernels ✓
**Finding**: Algorithms are identical
- Same FFT butterfly operations
- Same bit-reversal logic
- Same twiddle factor calculations

### 4. Increased shared memory allocation
**Test**: Changed batch kernel from 256 to 1024 elements (matching non-batch)
**Result**: No effect - still fails at size 64+

### 5. Forced non-batch indexing
**Test**: Made batch kernel use exact same indexing as non-batch for batch_size=1
**Code change**:
```c
shared_data[lid] = (batch_size == 1) ? input[rev_idx] : input[batch_offset + rev_idx];
output[(batch_size == 1) ? lid : (batch_offset + lid)] = result;
```
**Result**: No effect - still 2x amplification

### 6. Verified work-group configuration ✓
**Test**: Created debug kernel to print group_id and lid
**Finding**: Exactly 1 work-group, lid 0-63, as expected

### 7. Verified data transfer ✓
**Test**: Simple copy kernel
**Finding**: Data transfers correctly with no duplication

---

## Hypotheses Tested & Ruled Out

❌ **Missing local_size** - Fixed for non-batch, but batch still fails
❌ **Shared memory size** - Increasing from 256 to 1024 had no effect
❌ **Indexing logic** - Forcing non-batch indexing had no effect
❌ **Multiple work-groups** - Verified only 1 work-group created
❌ **Data duplication** - Copy kernel shows data is correct
❌ **Work-group size limit** - Device supports up to 256, we're using 64

---

## Remaining Hypotheses

### 1. Parameter passing issue
**Theory**: The `batch_size` or `fft_size` parameters might be read incorrectly by the kernel
**Evidence**: Batch works for size ≤32, fails for ≥64 (specific boundary)
**Test needed**: Write parameters to output to verify values

### 2. Compiler optimization bug
**Theory**: OpenCL compiler might optimize batch kernel differently than non-batch
**Evidence**: Identical code produces different results
**Test needed**: Try disabling optimizations, add compiler barriers

### 3. Shared memory access pattern
**Theory**: Access pattern for sizes >32 might trigger different cache/memory behavior
**Evidence**: Exact size boundary at 64 (2× warp size on some GPUs)
**Test needed**: Analyze memory access patterns, try different access orders

### 4. Barrier synchronization timing
**Theory**: For larger work-groups, barrier timing might be different
**Evidence**: Size 32 works (typical warp/wavefront size), 64+ fails
**Test needed**: Add extra barriers, try different synchronization patterns

---

## Current Status

- **Non-batch OpenCL FFT**: ✅ **FIXED** - Works perfectly for all sizes
- **Batch OpenCL FFT**: ⚠️ **PARTIALLY WORKING** - Works for sizes 4-32, fails for 64+
- **Metal Batch FFT**: ✅ **FULLY WORKING** - 78-230x speedup, perfect accuracy

---

## Files Modified

1. **`src/yapocis/lib/fft.py`** ✅
   - Added `local_size` to `forward()` and `inverse()` methods
   - Fixes non-batch FFT

2. **`src/yapocis/kernels/fft.mako`** ⚠️
   - Changed shared_data from [256] to [1024]
   - Added debug/workaround code (currently active)
   - **TODO**: Revert debug code once issue is found

---

## Recommended Next Steps

### Immediate (for production use)
1. **Document limitation**: OpenCL batch FFT only works for sizes ≤ 32
2. **Add size check**: Automatically fall back to loop of non-batch FFTs for sizes > 32
3. **Update tests**: Skip batch tests for OpenCL sizes > 32

### For fixing the bug
1. **Write parameter values to output** to verify batch_size and fft_size are correct
2. **Try disabling compiler optimizations** with `-cl-opt-disable`
3. **Test with explicit memory fences** around critical sections
4. **Compare assembly/IR** between working (size 32) and broken (size 64) versions
5. **Test on different OpenCL device** to see if it's Apple M1-specific

---

## Code to Revert

The following debug/workaround code in `src/yapocis/kernels/fft.mako` should be reverted once the issue is solved:

**Line 207**: Change back from 1024 to 256:
```c
__local float2 shared_data[256];  // Original value
```

**Line 233**: Remove size limit change:
```c
if (n <= 256) {  // Original value
```

**Line 238-239**: Remove conditional indexing:
```c
shared_data[lid] = input[batch_offset + rev_idx];  // Original code
```

**Line 272**: Remove conditional output:
```c
output[batch_offset + lid] = result;  // Original code
```

---

## Performance Impact

**Current Workaround**: Use loop of non-batch FFTs for sizes > 32

**Performance cost**:
- Size 64: ~64 individual kernel launches vs 1 batch launch
- Estimated: 10-20x slower than working batch implementation
- Still faster than CPU for large batches

**When batch FFT is fixed**:
- Expect 50-100x speedup for typical batch sizes (like Metal)

---

## Conclusion

OpenCL batch FFT has a mysterious bug that appears at size 64+, causing 2x amplification of results. Despite extensive investigation:
- Work-group configuration is correct
- Data passing is correct
- Kernel code is identical to working non-batch version
- Yet results differ

The bug is likely related to a subtle interaction between work-group size, shared memory access patterns, and possibly OpenCL compiler optimizations or Apple M1-specific behavior.

**Recommendation**: Use Metal backend for batch FFT in production (works perfectly), continue investigating OpenCL issue as time permits.

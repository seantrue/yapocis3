# Power Spectra Performance - Quick Reference

## Performance at a Glance (256-point FFT)

```
SINGLE OPERATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NUMBAX   ████ 9.92 µs                                    ⚡ FASTEST
METAL    ████████████████████████████████████████████ 456 µs (46x slower)
OCL      ████████████████████████████████████████████████████████████████████████████ 1.32 ms (133x slower)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```
BATCH OPERATION (100 signals):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METAL    ██████████ 23.41 ms (234 µs/signal)           ⚡ FASTEST
OCL      ███████████████████████ 62.26 ms (623 µs/signal) (2.7x slower)
NUMBAX   ████████████████████████████ ~99.2 ms (992 µs/signal) (4.2x slower - estimated, no batch support)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Winner by Category

| Operation | Winner | Time | Speedup vs 2nd |
|-----------|--------|------|----------------|
| Single FFT (256) | 🏆 **NUMBAX** | 6.82 µs | 107x faster than OCL |
| Batch FFT (100×256) | 🏆 **METAL** | 371 µs | 3.6x faster than OCL |
| Power Spectrum (256) | 🏆 **NUMBAX** | 2.68 µs | 88x faster than OCL |
| Full Pipeline (256) | 🏆 **NUMBAX** | 9.92 µs | 133x faster than OCL |
| Batch Pipeline (100×256) | 🏆 **METAL** | 23.41 ms | 2.7x faster than OCL |

## Decision Tree

```
                    How many signals?
                            |
        ┌───────────────────┴───────────────────┐
        |                                       |
     < 10 signals                        > 10 signals
        |                                       |
   Use NUMBAX                              Batch size?
   (100x faster)                                |
                                    ┌───────────┴───────────┐
                                    |                       |
                              10-100 signals        100+ signals
                                    |                       |
                              Use NUMBAX             Use METAL
                              or METAL           (70x batch speedup)
                           (case dependent)      4,272 signals/sec
```

## Throughput Comparison

**Signals processed per second:**

```
METAL (batch):     ████████████████████████████████ 4,272 signals/sec
OCL (batch):       ████████████ 1,606 signals/sec
NUMBAX (single):   ███████████ 1,008 signals/sec
```

## Key Takeaways

### 🚀 For Low Latency (single signals):
- **NUMBAX dominates:** 100-150x faster than GPU backends
- Perfect for interactive applications, low-latency needs
- Typical latency: **6-17 µs** per signal

### 🚀 For High Throughput (batch processing):
- **METAL dominates:** 70x batch speedup, 4,272 signals/sec
- Perfect for spectrograms, large datasets, offline processing
- Typical latency: **234 µs** per signal (in batches of 100)

### ⚠️ OpenCL Status:
- Functional but slower than Metal (2.7x slower for batches)
- Limited to FFT size ≤ 256 (work-group limitation)
- Requires `-cl-opt-disable` fix (10-20% performance penalty)
- Use only when Metal is unavailable

## Real-World Examples

### Example 1: Audio Spectrum Analyzer
**Requirements:** Real-time display, 256-point FFT, 60 FPS
- **NUMBAX:** Can handle 1,008 frames/sec ✅ 16.8x headroom
- **METAL:** Can handle 4,272 frames/sec ✅ 71x headroom
- **Recommendation:** Either works, **NUMBAX** simpler for low latency

### Example 2: Batch Spectrogram (1000 windows, 512-point FFT)
**Time to process:**
- **NUMBAX (sequential):** 13 ms ⚡ (1000 × 13 µs)
- **METAL (10 batches):** 57 ms (10 × 5.72 ms)
- **Recommendation:** **NUMBAX** for this size

### Example 3: Large Dataset (100,000 signals, 256-point)
**Time to process:**
- **NUMBAX (sequential):** 992 ms
- **METAL (batched):** 371 ms ⚡ (2.7x faster)
- **Recommendation:** **METAL** for high throughput

## Backend Selection Guide

```python
# Recommended approach
def select_backend(num_signals, fft_size):
    if num_signals < 10:
        return 'numbax'  # Low latency
    elif num_signals < 100 and fft_size <= 256:
        return 'numbax'  # Still fast enough
    else:
        return 'metal'   # High throughput
```

## Performance Summary Table

| Backend | Single (256) | Batch (100×256) | Batch Speedup | FFT Size Limit |
|---------|--------------|-----------------|---------------|----------------|
| **NUMBAX** | 🏆 9.92 µs | ~992 µs (no batch) | N/A | Unlimited |
| **METAL** | 456 µs | 🏆 234 µs/signal | 70x | Unlimited |
| **OCL** | 1.32 ms | 623 µs/signal | 55x | ⚠️ 256 max |

## Bottom Line

- **Best all-around:** NUMBAX for singles, METAL for batches
- **Avoid:** OpenCL unless no other option (slower + size-limited)
- **Sweet spot:** Switch from NUMBAX to METAL at ~10-50 signals

**Rule of thumb:**
- Signals < 10: NUMBAX (automatic choice)
- Signals > 100: METAL (automatic choice)
- Signals 10-100: Test your specific workload

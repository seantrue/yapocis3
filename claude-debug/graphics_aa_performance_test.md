# Graphics Anti-Aliasing Performance Test

A new performance test has been added to compare anti-aliasing performance across all backends and async modes.

## Usage

```bash
# Run on all available backends
yapocis-performance graphics-aa

# Run on specific backends
yapocis-performance graphics-aa --backends metal,ocl

# Run with custom iterations (default is 5)
yapocis-performance graphics-aa --iterations 10

# Run with fewer iterations for quick testing
yapocis-performance graphics-aa --iterations 2
```

## What it tests

The test compares performance across:

- **AA widths**: 0.0 (no AA), 0.25, 0.5, 1.0
- **Primitive types**: circles, rects, triangles, ellipses
- **Async modes**: async-both (YAPOCIS_ASYNC=1) vs no-async (YAPOCIS_ASYNC=0)
- **GPU backends only**: OCL, Metal, CUDA (numbax excluded as CPU fallback)

## Test Configuration

- Canvas size: 1024x1024
- Primitives per test: 5000
- Iterations: Configurable via `--iterations` (default: 5)

## Output

The test generates two plots in the performance directory:

1. **graphics_aa_performance_by_primitive.png** - 4 subplots showing execution time vs AA width for each primitive type, comparing async vs sync modes
2. **graphics_aa_performance_speedup.png** - Grouped bar chart showing async speedup across all primitives and AA widths

## Example Output

```
METAL: Apple M1 Pro

  Circles:
    aa=0.00: async=0.2285s, sync=0.2288s, speedup=1.00x
    aa=0.25: async=0.3437s, sync=0.3439s, speedup=1.00x
    aa=0.50: async=0.3443s, sync=0.3433s, speedup=1.00x
    aa=1.00: async=0.3452s, sync=0.3438s, speedup=1.00x

  Rects:
    aa=0.00: async=0.2598s, sync=0.2596s, speedup=1.00x
    aa=0.25: async=0.2793s, sync=0.2753s, speedup=0.99x
    ...
```

## Performance Observations

From initial testing on Metal (Apple M1 Pro):

- **Anti-aliasing overhead**: Approximately 50% increase in execution time (0.23s → 0.35s for circles)
- **AA width scaling**: Performance is relatively constant across AA widths (0.25, 0.5, 1.0)
- **Async benefit**: Minimal on Metal for graphics operations (1.00-1.02x speedup)
- **Primitive comparison**:
  - Fastest: Rects (~0.26s no AA, ~0.27s with AA)
  - Slowest: Triangles (~0.35s no AA, ~0.37s with AA)

The relatively flat scaling across AA widths suggests the smoothstep calculation overhead is minimal compared to the distance field computation.

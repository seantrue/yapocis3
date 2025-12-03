# Interface Basics

Interface files define how Python code calls kernel functions. They specify parameter types, directions, and relationships.

## Basic Syntax

```
interface <name> {
    kernel <kernel_name>(<parameters>);
};
```

## Parameter Directions

| Direction | Meaning |
|-----------|---------|
| `in` | Read-only input buffer |
| `out` | Write-only output buffer (allocated by caller) |
| `inout` | Read-write buffer (modified in place) |
| `out_like <param>` | Output with same shape as another parameter |
| `resident` | Buffer stays on GPU between calls |

## Examples

### Simple Input/Output

```
interface mandelbrot {
    kernel mandelbrot(in complex64 q, out_like short q, int maxiter);
};
```

- `in complex64 q` - Input complex array
- `out_like short q` - Output short array, same shape as `q`
- `int maxiter` - Scalar integer parameter

### In-Place Modification

```
interface blur {
    kernel blur(inout float img, float sigma);
};
```

- `inout float img` - Image is modified in place
- `float sigma` - Scalar blur amount

### Dimension Extraction

```
interface gradient {
    kernel gradient(width_of img, height_of img, in float img, out_like float img);
};
```

- `width_of img` - Automatically extracts width from img
- `height_of img` - Automatically extracts height from img

## Dimension Hints

| Hint | Meaning |
|------|---------|
| `width_of <param>` | Width (columns) of the parameter |
| `height_of <param>` | Height (rows) of the parameter |
| `depth_of <param>` | Depth (3rd dimension) of the parameter |
| `size_of <param>` | Total element count |
| `length_of <param>` | Length of 1D array |

## Data Types

| Interface Type | NumPy Type | OpenCL Type |
|---------------|------------|-------------|
| `float` | `float32` | `float` |
| `float2` | `float32` (2 channels) | `float2` |
| `float4` | `float32` (4 channels) | `float4` |
| `int` | `int32` | `int` |
| `int32` | `int32` | `int` |
| `short` | `int16` | `short` |
| `complex64` | `complex64` | `float2` |

## Extensions

Extensions create alternate entry points for a kernel:

```
interface graphics4 {
    kernel lines(inout float a, ...);
    extension res as lines(resident float a, ...);
};
```

This creates:
- `prog.lines(canvas, ...)` - Normal version
- `prog.lines_res(canvas, ...)` - Version using resident buffer

## Multiple Kernels

An interface can define multiple kernels:

```
interface imageops {
    kernel histogram(in float img, out int hist);
    kernel normalize(inout float img);
    kernel threshold(in float img, out_like float img, float level);
};
```

Access as `prog.histogram()`, `prog.normalize()`, `prog.threshold()`.

## Comments

Use `#` for comments:

```
interface example {
    # This kernel does something useful
    kernel useful(in float data, out_like float data);
};
```

## Next Steps

- {doc}`template-basics` - Write the kernel implementation
- {doc}`/kernel-development/interface-files` - Full interface reference

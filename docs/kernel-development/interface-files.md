# Interface Files

Complete reference for yapocis interface files.

## Syntax

```
interface <interface_name> {
    kernel <kernel_name>(<parameter_list>);
    extension <alias> as <kernel_name>(<modified_params>);
};
```

## Parameter Specification

Each parameter has the form: `[role] <type> <name>`

### Roles

| Role | Description | GPU Buffer | Data Transfer |
|------|-------------|------------|---------------|
| `in` | Input only | Read-only | Host → Device |
| `out` | Output only | Write-only | Device → Host |
| `inout` | Read and write | Read-write | Both directions |
| `out_like <param>` | Output shaped like param | Write-only | Device → Host |
| `resident` | Persistent on device | Read-write | Stays on device |

### Types

| Interface Type | NumPy | OpenCL | CUDA | Metal |
|---------------|-------|--------|------|-------|
| `float` | `float32` | `float` | `float` | `float` |
| `float2` | `(n, 2) float32` | `float2` | `float2` | `float2` |
| `float3` | `(n, 3) float32` | `float3` | `float3` | `float3` |
| `float4` | `(n, 4) float32` | `float4` | `float4` | `float4` |
| `int` | `int32` | `int` | `int` | `int` |
| `int32` | `int32` | `int` | `int` | `int` |
| `short` | `int16` | `short` | `short` | `short` |
| `complex64` | `complex64` | `float2` | `float2` | `float2` |

### Dimension Hints

Automatically extract array dimensions:

| Hint | Extracts | From Array Shape |
|------|----------|------------------|
| `width_of <p>` | `shape[1]` | Columns |
| `height_of <p>` | `shape[0]` | Rows |
| `depth_of <p>` | `shape[2]` | Depth |
| `size_of <p>` | `prod(shape)` | Total elements |
| `length_of <p>` | `shape[0]` | 1D length |
| `dim_N_of <p>` | `shape[N]` | Specific dimension |

## Examples

### Simple Processing

```
interface threshold {
    kernel threshold(
        in float img,
        out_like float img,
        float level
    );
};
```

### 2D with Dimensions

```
interface blur {
    kernel gaussian(
        width_of img,
        height_of img,
        in float img,
        out_like float img,
        float sigma
    );
};
```

### Multiple Outputs

```
interface gradient {
    kernel gradient(
        width_of img,
        height_of img,
        in float img,
        int reach,
        out_like float img,    # gradient magnitude
        out_like float img     # gradient direction
    );
};
```

### Extensions

```
interface graphics4 {
    kernel lines(
        width_of a,
        height_of a,
        inout float a,
        width_of lines,
        in float lines,
        in float distance,
        in float aa_width,
        in float rcolor
    );

    # Resident buffer version
    extension res as lines(
        width_of a,
        height_of a,
        resident float a,
        width_of lines,
        in float lines,
        in float distance,
        in float aa_width,
        in float rcolor
    );
};
```

Usage:
```python {.no-exec}
prog.lines(canvas, ...)       # Normal
prog.lines_res(canvas, ...)   # Resident buffer
```

### Templated Kernels

For parameterized kernels using Mako context:

```
interface convolves {
    kernel convolve(
        width_of img,
        height_of img,
        in float img,
        out_like float img
    );
};
```

Load with context:
```python {.no-exec}
prog = manager.load_program("convolves", kernel_size=5)
```

## Scalar Parameters

Parameters without a role are scalars:

```
interface mandelbrot {
    kernel mandelbrot(
        in complex64 q,
        out_like short q,
        int maxiter         # Scalar int
    );
};
```

## Comments

Use `#` for comments:

```
interface example {
    # This is a comment
    kernel process(in float data, out_like float data);
};
```

## Parser Details

The interface parser (`yapocis.core.interface_parser`) handles:

- Legacy spellings: `widthof` → `width_of`, `outlike` → `out_like`
- Whitespace normalization
- Multi-line definitions
- Mako preprocessing for templated interfaces

## Best Practices

1. **Name consistently** - Interface and primary kernel names should match
2. **Order parameters logically** - Dimensions, inputs, outputs, scalars
3. **Use extensions sparingly** - Only for performance-critical paths
4. **Document complex interfaces** - Use comments for non-obvious parameters

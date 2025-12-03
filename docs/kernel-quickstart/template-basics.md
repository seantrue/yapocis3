# Template Basics

Kernel templates use Mako syntax to generate backend-specific code. This guide covers the essentials.

## Basic Structure

A minimal OpenCL kernel template:

```c
/* Description of what this kernel does */

__kernel void kernel_name(
    __global const float *input,
    __global float *output)
{
    int gid = get_global_id(0);
    output[gid] = input[gid] * 2.0f;
}
```

## Common OpenCL Patterns

### 1D Processing

```c
__kernel void process1d(__global const float *in, __global float *out)
{
    int i = get_global_id(0);
    out[i] = process(in[i]);
}
```

### 2D Processing

```c
__kernel void process2d(
    int width, int height,
    __global const float *in,
    __global float *out)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        int idx = y * width + x;
        out[idx] = process(in[idx]);
    }
}
```

### Neighborhood Operations

```c
__kernel void blur3x3(
    int width, int height,
    __global const float *in,
    __global float *out)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x > 0 && x < width-1 && y > 0 && y < height-1) {
        float sum = 0.0f;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                sum += in[(y+dy) * width + (x+dx)];
            }
        }
        out[y * width + x] = sum / 9.0f;
    }
}
```

## Mako Features

### Including Other Templates

```mako
<%include file="helpers.mako" />
<%include file="constants.mako" />
```

### Generating Multiple Versions

```mako
%for planes in ["", 2, 3, 4]:
__kernel void process${planes}(__global FLOAT${planes}* data)
{
    // Works with 1, 2, 3, or 4 channel data
}
%endfor
```

This generates `process`, `process2`, `process3`, `process4`.

### Conditional Code

```mako
%if use_fast_math:
    result = native_sqrt(value);
%else:
    result = sqrt(value);
%endif
```

### Template Variables

Access variables passed from Python:

```mako
__kernel void convolve(__global float *img)
{
    // Kernel size from template context
    const int KERNEL_SIZE = ${kernel_size};
}
```

## Helper Macros

Yapocis provides standard helpers via `helpers.mako`:

```c
<%include file="helpers.mako" />

// Column-major addressing
int idx = ADDRESS(x, y);  // = x * height + y

// Extract coordinates from linear index
int x = X(i);
int y = Y(i);
```

## Multi-Channel Support

Handle varying channel counts:

```mako
%for planes in [1, 2, 3, 4]:
<%
    suffix = "" if planes == 1 else str(planes)
    float_type = f"float{suffix}" if planes > 1 else "float"
%>
__kernel void process${suffix}(__global ${float_type}* data)
{
    // Process ${planes} channel(s)
}
%endfor
```

## Backend Differences

### OpenCL

```c
__kernel void foo(__global float *data)
{
    int gid = get_global_id(0);
}
```

### CUDA

```c
__global__ void foo(float *data)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
}
```

### Metal

```c
kernel void foo(device float *data [[buffer(0)]],
                uint gid [[thread_position_in_grid]])
{
    // Metal uses different syntax
}
```

### Numba

Numba kernels are pure Python:

```python {.no-exec}
@numba.jit(nopython=True, parallel=True)
def foo(data, output):
    for i in numba.prange(len(data)):
        output[i] = data[i] * 2.0
```

## Next Steps

- {doc}`testing-basics` - Test your kernel
- {doc}`/kernel-development/mako-templates` - Advanced Mako usage
- {doc}`/kernel-development/opencl-kernels` - OpenCL specifics

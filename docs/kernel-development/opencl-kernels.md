# OpenCL Kernels

OpenCL is the primary backend for yapocis. This guide covers OpenCL-specific patterns.

## Basic Structure

```c
__kernel void kernel_name(
    __global const float *input,
    __global float *output)
{
    int gid = get_global_id(0);
    output[gid] = input[gid] * 2.0f;
}
```

## Work Item Functions

| Function | Returns |
|----------|---------|
| `get_global_id(dim)` | Global work item ID |
| `get_local_id(dim)` | Local work item ID |
| `get_group_id(dim)` | Work group ID |
| `get_global_size(dim)` | Total work items |
| `get_local_size(dim)` | Work group size |
| `get_num_groups(dim)` | Number of work groups |

## Memory Spaces

| Qualifier | Description |
|-----------|-------------|
| `__global` | GPU main memory |
| `__local` | Work group shared memory |
| `__private` | Per work-item (default) |
| `__constant` | Read-only, cached |

## 2D Kernels

```c
__kernel void process2d(
    int width,
    int height,
    __global const float *input,
    __global float *output)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    // Column-major addressing
    int idx = x * height + y;
    output[idx] = process(input[idx]);
}
```

## Local Memory

```c
__kernel void blur_local(
    int width,
    int height,
    __global const float *input,
    __global float *output)
{
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Shared memory tile
    __local float tile[18][18];  // 16x16 + 1 pixel border

    // Load tile (with bounds checking)
    // ...

    barrier(CLK_LOCAL_MEM_FENCE);

    // Process using local memory
    // ...
}
```

## Vector Types

```c
// Built-in vector types
float2 v2 = (float2)(1.0f, 2.0f);
float4 v4 = (float4)(1.0f, 2.0f, 3.0f, 4.0f);

// Component access
float x = v4.x;  // or v4.s0
float y = v4.y;  // or v4.s1
float2 xy = v4.xy;

// Vector operations
float4 result = v4 * 2.0f;
float dot = dot(v4.xyz, v4.xyz);
```

## Math Functions

| Function | Description |
|----------|-------------|
| `sqrt(x)` | Square root |
| `fabs(x)` | Absolute value |
| `fmin(x, y)` | Minimum |
| `fmax(x, y)` | Maximum |
| `clamp(x, lo, hi)` | Clamp to range |
| `sin(x)`, `cos(x)` | Trigonometric |
| `exp(x)`, `log(x)` | Exponential |
| `atan2(y, x)` | Arc tangent |
| `native_sqrt(x)` | Fast approximate sqrt |

## Image Sampling

```c
__kernel void sample_image(
    __read_only image2d_t input,
    sampler_t sampler,
    __global float4 *output)
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    float4 pixel = read_imagef(input, sampler, coord);
    output[coord.y * get_global_size(0) + coord.x] = pixel;
}
```

## Atomic Operations

```c
__kernel void histogram(
    __global const uchar *image,
    __global int *histogram)
{
    int gid = get_global_id(0);
    uchar value = image[gid];
    atomic_inc(&histogram[value]);
}
```

## Column-Major Addressing

Yapocis uses column-major order (Fortran-style):

```c
// Address calculation
int idx = x * height + y;

// Extract coordinates
int x = idx / height;
int y = idx % height;
```

Via helpers:

```c
<%include file="helpers.mako" />

int idx = ADDRESS(x, y);
int x = X(idx);
int y = Y(idx);
```

## Bounds Checking

Always check bounds for 2D kernels:

```c
if (x < 0 || x >= width || y < 0 || y >= height) return;
```

For neighborhood operations:

```c
if (x < radius || x >= width - radius ||
    y < radius || y >= height - radius) {
    output[ADDRESS(x, y)] = input[ADDRESS(x, y)];
    return;
}
```

## Debugging

OpenCL 1.2+ supports printf:

```c
if (get_global_id(0) == 0) {
    printf("Width: %d, Height: %d\\n", width, height);
}
```

## Performance Tips

1. **Coalesce memory access** - Adjacent work items access adjacent memory
2. **Use local memory** - For data reused by work group
3. **Avoid divergent branches** - All work items should take same path
4. **Prefer vector operations** - float4 operations can be faster
5. **Use native math** - `native_sqrt()` faster than `sqrt()` but less precise

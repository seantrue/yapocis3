# Metal Kernels

Metal Shading Language (MSL) kernels for Apple GPUs.

## Key Differences from OpenCL

| Feature | OpenCL | Metal |
|---------|--------|-------|
| Kernel declaration | `__kernel void` | `kernel void` |
| Global memory | `__global float*` | `device float*` |
| Constant memory | `__constant` | `constant` |
| Local memory | `__local` | `threadgroup` |
| Thread ID | `get_global_id(0)` | Attribute `[[ thread_position_in_grid ]]` |
| Buffer binding | Implicit | Explicit `[[ buffer(N) ]]` |

## Basic Structure

```cpp
<%include file="constants.mako" />

kernel void kernel_name(
    device const float *input [[ buffer(0) ]],
    device float *output [[ buffer(1) ]],
    uint gid [[ thread_position_in_grid ]])
{
    output[gid] = input[gid] * 2.0f;
}
```

## Buffer Attributes

Metal requires explicit buffer binding indices:

```cpp
kernel void process(
    constant int &width [[ buffer(0) ]],      // Scalar by reference
    constant int &height [[ buffer(1) ]],
    const device float* input [[ buffer(2) ]],  // Read-only array
    device float* output [[ buffer(3) ]],       // Read-write array
    uint i [[ thread_position_in_grid ]])
{
    // ...
}
```

Buffer indices must match the order in the interface file.

## Thread Position Attributes

| Attribute | Equivalent |
|-----------|------------|
| `thread_position_in_grid` | `get_global_id()` |
| `thread_position_in_threadgroup` | `get_local_id()` |
| `threadgroup_position_in_grid` | `get_group_id()` |
| `threads_per_threadgroup` | `get_local_size()` |

## Column-Major Addressing

Metal uses the same column-major order as OpenCL:

```cpp
// Metal addressing (column-major)
#define ADDRESS(xx,yy) (((xx)*height)+(yy))

#define X(addr) ((addr)/height)
#define Y(addr) ((addr)%height)
```

Include `helpers.mako` for these macros.

## 2D Kernels

```cpp
<%include file="constants.mako" />
<%include file="helpers.mako" />

kernel void process2d(
    constant int &width [[ buffer(0) ]],
    constant int &height [[ buffer(1) ]],
    const device FLOAT* input [[ buffer(2) ]],
    device FLOAT* output [[ buffer(3) ]],
    uint i [[ thread_position_in_grid ]])
{
    int x = X(i);
    int y = Y(i);

    if (x >= width || y >= height) return;

    output[i] = process(input[i]);
}
```

## Vector Types

Metal uses C++ constructor syntax:

```cpp
// Metal vector construction
float4 v = float4(1.0f, 2.0f, 3.0f, 4.0f);
float2 xy = float2(1.0f, 2.0f);

// Component access
float x = v.x;  // or v[0]
float y = v.y;  // or v[1]
float2 xy = v.xy;

// Swizzling
float4 reversed = v.wzyx;
```

## Constants

The `constants.mako` provides compatibility macros:

```cpp
#define ZERO 0.0f
#define ZERO2 float2(ZERO, ZERO)
#define ZERO3 float3(ZERO, ZERO, ZERO)
#define ZERO4 float4(ZERO, ZERO, ZERO, ZERO)

#define FLOAT float
#define FLOAT2 float2
#define FLOAT3 float3
#define FLOAT4 float4
```

## Math Functions

Metal has most OpenCL math functions built-in. The constants file provides compatibility for missing ones:

```cpp
// Provided by constants.mako
#define atan2pi(y,x) (atan2((y),(x)) / M_PI_F)
```

## Conditional Selection

Metal uses `select()` instead of the ternary operator for vectors:

```cpp
// OpenCL style (ternary)
grad[i] = isnan(gr) ? ZERO4 : gr;

// Metal style (select)
grad[i] = select(gr, ZERO4, isnan(gr));
```

Note: `select(a, b, cond)` returns `b` when `cond` is true, opposite of most select functions!

## Threadgroup Memory

```cpp
kernel void blur_local(
    constant int &width [[ buffer(0) ]],
    constant int &height [[ buffer(1) ]],
    const device float *input [[ buffer(2) ]],
    device float *output [[ buffer(3) ]],
    uint2 gid [[ thread_position_in_grid ]],
    uint2 lid [[ thread_position_in_threadgroup ]])
{
    // Shared memory tile
    threadgroup float tile[18][18];

    // Load tile
    // ...

    threadgroup_barrier(mem_flags::mem_device);

    // Process using local memory
    // ...
}
```

## Atomic Operations

```cpp
// Compatibility shim from constants.mako
static inline void atomic_inc(device atomic_int* p) {
    (void)atomic_fetch_add_explicit(p, 1, memory_order_relaxed);
}

// Usage
kernel void histogram(
    device const uchar *image [[ buffer(0) ]],
    device atomic_int *histogram [[ buffer(1) ]],
    uint gid [[ thread_position_in_grid ]])
{
    uchar value = image[gid];
    atomic_inc(&histogram[value]);
}
```

## Multi-Channel Kernels

```cpp
<%include file="constants.mako" />
<%include file="helpers.mako" />

%for planes in ["",2,3,4]:
kernel void gradient${planes}(
    constant int &width [[ buffer(0) ]],
    constant int &height [[ buffer(1) ]],
    const device FLOAT${planes}* input [[ buffer(2) ]],
    constant int &reach [[ buffer(3) ]],
    device FLOAT${planes}* grad [[ buffer(4) ]],
    device FLOAT${planes}* theta [[ buffer(5) ]],
    uint i [[ thread_position_in_grid ]])
{
    int x = X(i);
    int y = Y(i);
    FLOAT${planes} gr = ZERO${planes};
    // ... computation ...
    grad[i] = select(gr, ZERO${planes}, isnan(gr));
}
%endfor
```

## Metal Standard Library

Metal includes `<metal_stdlib>`:

```cpp
#include <metal_stdlib>
using namespace metal;

// Now have access to:
// - Math functions: sin, cos, sqrt, pow, etc.
// - Geometric functions: dot, cross, normalize, length
// - Common functions: abs, min, max, clamp, mix
```

## Performance Tips

1. **Use half precision** - Metal supports `half` type natively
2. **Minimize buffer bindings** - Pack related data together
3. **Use threadgroup memory** - For data shared within work group
4. **Avoid divergent branches** - SIMD execution model

## Porting from OpenCL

Key changes when porting:

1. Add `#include <metal_stdlib>` and `using namespace metal;`
2. Change `__kernel` to `kernel`
3. Change `__global` to `device`
4. Add buffer binding attributes `[[ buffer(N) ]]`
5. Replace `get_global_id(0)` with thread position attribute
6. Use `select()` instead of ternary for vector conditionals
7. Include `constants.mako` for compatibility macros

# CUDA Kernels

CUDA kernels in yapocis are translated from OpenCL with compatibility shims.

## Key Differences from OpenCL

| Feature | OpenCL | CUDA |
|---------|--------|------|
| Kernel declaration | `__kernel void` | `extern "C" __global__ void` |
| Thread ID | `get_global_id(0)` | Mapped via shim |
| Local memory | `__local` | `__shared__` |
| Barrier | `barrier(CLK_LOCAL_MEM_FENCE)` | `__syncthreads()` |
| **Storage order** | **Column-major** | **Row-major** |

## Basic Structure

```cuda
<%include file="constants.mako"/>
<%include file="helpers.mako"/>

extern "C" __kernel void kernel_name(
    const float *input,
    float *output)
{
    int gid = get_global_id(0);
    output[gid] = input[gid] * 2.0f;
}
```

The `constants.mako` file provides OpenCL compatibility:

```cuda
#define __kernel __global__
#define __local  __shared__
#define barrier(x) __syncthreads()
#define get_global_id(N) ...  // Mapped to blockIdx/threadIdx
```

## Row-Major Addressing

**Critical**: CUDA uses row-major order (matching NumPy), opposite of OpenCL/Metal:

```cuda
// CUDA addressing (row-major)
#define ADDRESS(xx,yy) (((yy)*height)+(xx))

#define X(addr) ((addr)%height)
#define Y(addr) ((addr)/height)
```

This means the same kernel code produces the same results because the shims handle the difference, but be aware when debugging.

## 2D Kernels

```cuda
<%include file="constants.mako"/>
<%include file="helpers.mako"/>

extern "C" __kernel
void process2d(int width, int height,
               const FLOAT* input,
               FLOAT* output)
{
    size_t i = get_global_id(0);
    int x = X(i);
    int y = Y(i);

    if (x >= width || y >= height) return;

    output[i] = process(input[i]);
}
```

## Vector Types

CUDA uses `make_floatN()` constructors instead of C99 compound literals:

```cuda
// OpenCL style (NOT in CUDA)
float4 v = (float4)(1.0f, 2.0f, 3.0f, 4.0f);

// CUDA style
float4 v = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
```

The constants file provides zero/one constants:

```cuda
#define ZERO4 make_float4(ZERO,ZERO,ZERO,ZERO)
#define ONE4 make_float4(ONE,ONE,ONE,ONE)
```

## Vector Operators

CUDA doesn't have built-in vector operators. The constants file provides them:

```cuda
// Provided by constants.mako
float4 operator+(float4 a, float4 b);
float4 operator-(float4 a, float4 b);
float4 operator*(float4 a, float4 b);
float4 operator/(float4 a, float4 b);
float4 operator*(float a, float4 b);  // Scalar broadcast
```

## Math Function Compatibility

Some OpenCL functions are mapped:

| OpenCL | CUDA Equivalent |
|--------|-----------------|
| `atan2pi(y, x)` | `atan2f(y, x) / M_PI` |
| `mix(a, b, t)` | `a + t * (b - a)` |
| `clamp(v, lo, hi)` | `fminf(hi, fmaxf(lo, v))` |
| `fract(x, &iptr)` | Custom implementation |
| `native_powr(x, y)` | `powf(x, y)` |

Vector versions are also provided:

```cuda
float2 sqrt(float2 a);
float3 sqrt(float3 a);
float4 sqrt(float4 a);
// etc.
```

## Atomic Operations

```cuda
// OpenCL style (works via shim)
atomic_inc(&counter);

// Mapped to CUDA
atomicAdd(&counter, 1);
```

## Multi-Channel Kernels

Using Mako to generate variants:

```cuda
<%include file="constants.mako"/>
<%include file="helpers.mako"/>

%for planes in ["",2,3,4]:
extern "C" __kernel
void gradient${planes}(int width, int height,
                       FLOAT${planes}* input,
                       int reach,
                       FLOAT${planes}* grad,
                       FLOAT${planes}* theta)
{
    size_t i = get_global_id(0);
    int x = X(i);
    int y = Y(i);
    FLOAT${planes} gr = ZERO${planes};
    // ... computation ...
    grad[i] = isnan(gr) ? ZERO${planes} : gr;
}
%endfor
```

## Debugging

CUDA supports printf in device code:

```cuda
if (get_global_id(0) == 0) {
    printf("Width: %d, Height: %d\n", width, height);
}
```

## Performance Considerations

1. **Memory coalescing** - Same principles as OpenCL
2. **Shared memory** - Use `__local` (mapped to `__shared__`)
3. **Warp divergence** - Avoid divergent branches within warps
4. **Register pressure** - CUDA has similar constraints

## Porting from OpenCL

Most OpenCL kernels work with minimal changes:

1. Include `constants.mako` and `helpers.mako`
2. Use `extern "C"` before kernel declaration
3. Replace vector literals with `make_floatN()`
4. Use `ZERO2`, `ZERO3`, `ZERO4` constants

The shim layer handles most compatibility automatically.

# Mako Templates

Yapocis uses Mako templates to generate kernel code for different backends.

## Template Basics

Mako provides Python-like syntax within template files:

```mako
<%doc>
    This is a Mako comment
</%doc>

## Variable substitution
const int SIZE = ${size};

## Python expressions
%if include_debug:
    printf("Debug mode enabled\n");
%endif

## Loops
%for i in range(4):
    data${i} = input[${i}];
%endfor
```

## Including Files

```mako
<%include file="constants.mako" />
<%include file="helpers.mako" />
```

Standard includes:
- `constants.mako` - Type definitions (FLOAT2, ZERO4, etc.)
- `helpers.mako` - Address macros (ADDRESS, X, Y)

## Generating Multiple Kernels

```mako
%for planes in ["", 2, 3, 4]:
__kernel void process${planes}(
    __global FLOAT${planes}* input,
    __global FLOAT${planes}* output)
{
    size_t i = get_global_id(0);
    output[i] = input[i] * 2.0f;
}
%endfor
```

This generates:
- `process` - 1 channel
- `process2` - 2 channels
- `process3` - 3 channels
- `process4` - 4 channels

## Python Code Blocks

```mako
<%
    # Python code
    kernel_radius = (kernel_size - 1) // 2
    type_suffix = "" if planes == 1 else str(planes)
%>

// Use computed values
const int RADIUS = ${kernel_radius};
```

## Conditional Compilation

```mako
%if use_local_memory:
__local float local_data[256];
%endif

%if precision == "double":
typedef double REAL;
%else:
typedef float REAL;
%endif
```

## Type Helpers

The `constants.mako` file defines type macros:

```text
// FLOAT types
FLOAT   = float
FLOAT2  = float2
FLOAT3  = float3
FLOAT4  = float4

// Zero values
ZERO    = 0.0f
ZERO2   = (float2)(0.0f, 0.0f)
ZERO3   = (float3)(0.0f, 0.0f, 0.0f)
ZERO4   = (float4)(0.0f, 0.0f, 0.0f, 0.0f)

// One values
ONE     = 1.0f
ONE2    = (float2)(1.0f, 1.0f)
// etc.
```

## Address Helpers

The `helpers.mako` file provides addressing macros:

```c
// Column-major addressing (OpenCL/Metal default)
#define ADDRESS(x, y) ((x) * height + (y))
#define X(i) ((i) / height)
#define Y(i) ((i) % height)

// These assume width and height are in scope
```

## Context Variables

Variables passed when loading:

```python {.no-exec}
prog = manager.load_program("convolve",
    kernel_size=5,
    use_local_memory=True
)
```

Use in template:

```mako
const int KERNEL_SIZE = ${kernel_size};

%if use_local_memory:
__local float cache[${kernel_size} * ${kernel_size}];
%endif
```

## Multi-Backend Templates

For code shared across backends:

```mako
## In shared.mako
<%def name="clamp(val, lo, hi)">
    fmax(${lo}, fmin(${hi}, ${val}))
</%def>
```

```mako
## In opencl/kernel.mako
<%namespace file="shared.mako" import="clamp"/>

output[i] = ${clamp("value", "0.0f", "1.0f")};
```

## Debugging Templates

Add debug output:

```mako
%if debug:
#pragma message "Debug build"
%endif

__kernel void process(...)
{
    %if debug:
    if (get_global_id(0) == 0) {
        printf("First work item\\n");
    }
    %endif
}
```

## Template Inheritance

Not commonly used, but supported:

```mako
## base.mako
<%def name="kernel_signature()">
__kernel void ${kernel_name}(
    ${caller.body()}
)
</%def>
```

```mako
## derived.mako
<%inherit file="base.mako"/>
<%block name="kernel_signature">
    __global float *input,
    __global float *output
</%block>
```

## Best Practices

1. **Keep templates readable** - Complex logic should be in Python, not Mako
2. **Use includes** - Share common code via includes
3. **Document context variables** - Comment expected variables at top
4. **Test all combinations** - Loops generate many kernels; test them all
5. **Prefer explicit over clever** - Clear code beats clever metaprogramming

# Interface Specification

IDL syntax, parameter roles, and extensions.

## Overview

Yapocis uses an Interface Definition Language (IDL) to describe kernel signatures. The `.interface` files are parsed by PyParsing into an IR (Intermediate Representation) that drives code generation and stub creation.

## File Structure

```idl
interface graphics4 {
    meta {
        layout NHWC;
        class graphics_2d;
        blend alpha;
    }

    kernel lines(width_of a, height_of a, inout float a,
                 width_of lines, in float lines,
                 in float distance, in float rcolor);

    extension res of lines(width_of a, height_of a, resident float a,
                           width_of lines, in float lines,
                           in float distance, in float rcolor);
};
```

## Parameter Roles

### Data Flow Roles

| Role | Direction | Description |
|------|-----------|-------------|
| `in` | Read-only | Input parameter, not modified |
| `out` | Write-only | Output parameter, kernel writes to it |
| `inout` | Read/Write | Modified in place |
| `resident` | On-GPU | Buffer stays on GPU (no transfer) |
| `out_like` | Derived | Allocate output with same shape as specified param |

### Derived Parameters

| Syntax | Description |
|--------|-------------|
| `width_of <param>` | Derive width from named array |
| `height_of <param>` | Derive height from named array |
| `sizeof <param>` | Derive total size |
| `bytes_of(<param>)` | Size in bytes for local memory |

## Data Types

| IDL Type | GPU Type | NumPy Type |
|----------|----------|------------|
| `float` / `float32` | `float` | `np.float32` |
| `half` / `f16` | `half` | `np.float16` |
| `double` | `double` | `np.float64` |
| `int` / `int32` | `int` | `np.int32` |
| `uint32` | `uint` | `np.uint32` |
| `short` / `int16` | `short` | `np.int16` |
| `complex64` | `float2` | `np.complex64` |
| `float4` | `float4` | structured |
| `boolean` | `int` | `np.bool_` |

## Parameter Qualifiers

| Qualifier | Description |
|-----------|-------------|
| `constant` | Use constant memory (OpenCL/CUDA) |
| `texture2d` | Use texture memory |
| `local bytes_of(N)` | Allocate local/shared memory |

Example:
```idl
kernel reduce(in float input,
              local bytes_of(256) float scratch,
              out float result);
```

## Meta Block

Interface-level metadata:

```idl
meta {
    layout <planar|NHWC|NCHW>;    // Array layout
    dtype <float32|...>;          // Default data type
    fast_math <on|off>;           // Enable fast math
    workgroup (<x>, <y>, <z>);    // Default workgroup size
    class <elementwise|stencil2d|separable|reduction|graphics_2d>;
    blend <alpha|src_over|additive>;
    cull <none|tile>;
    tile_pixels (<w>, <h>);
}
```

## Kernel Traits

Per-kernel metadata in `with` blocks:

```idl
kernel stencil3x3(inout float image)
with {
    class stencil2d;
    radius 1;
    boundary clamp;
    halo on;
};
```

| Trait | Values | Description |
|-------|--------|-------------|
| `class` | elementwise, stencil2d, etc. | Kernel classification |
| `axis` | x, y, xy | Axis of operation |
| `radius` | integer | Stencil radius |
| `boundary` | clamp, wrap, zero | Edge handling |
| `tile` | integer | Tile size |
| `halo` | on, off | Include halo region |
| `blend` | alpha, additive | Blending mode |

## Extensions

Create variants of existing kernels:

```idl
// Base kernel
kernel lines(width_of a, height_of a, inout float a, ...);

// Resident variant (different buffer behavior)
extension res of lines(width_of a, height_of a, resident float a, ...);

// Simplified alias
alias gauss3 as gauss_sep(image, conv, scale=3.0);
```

### Extension with Backend Overrides

```idl
kernel mykernel(in float input, out float output)
with {
    class elementwise;
    backend metal {
        workgroup (32, 1, 1);
    };
    backend cuda {
        workgroup (256, 1, 1);
    };
};
```

## IR Model

The parser produces these dataclasses:

```python {.no-exec}
@dataclass
class KernelParamSpec:
    name: str
    role: str | None        # in, out, inout, resident, out_like
    dtype: str | None       # float32, int, etc.
    shape: str | None       # e.g., "[9]"
    layout: str | None      # e.g., "planar"
    qualifiers: dict        # {"constant": "true", "local": "bytes_of(256)"}

@dataclass
class KernelSpec:
    name: str
    params: list[KernelParamSpec]
    shapes: dict[str, str]           # Derived dimensions
    traits: dict[str, Any]           # Per-kernel traits
    backend_overrides: dict          # Backend-specific config
    alias_of: str | None             # For aliases
    extends: str | None              # For extensions

@dataclass
class InterfaceSpec:
    name: str
    meta: dict[str, Any]             # Interface-level metadata
    kernels: list[KernelSpec]
```

## Example Interface

Complete example for a convolution interface:

```idl
interface convolvesep {
    meta {
        layout planar;
        class separable;
        fast_math on;
    }

    // 1D separable convolution
    kernel gauss_sep(
        width_of a, height_of a,
        inout float a,
        in float conv,
        float scale
    ) with {
        axis xy;
        boundary clamp;
    };

    // Resident variant
    extension res of gauss_sep(
        width_of a, height_of a,
        resident float a,
        in float conv,
        float scale
    );

    // Pre-configured aliases
    alias gauss3 as gauss_sep(a, conv, scale=3.0);
    alias gauss5 as gauss_sep(a, conv, scale=5.0);
};
```

## Parsing

Interfaces are parsed at program load time:

```python
from yapocis.core.interface_parser import parse_interface_text
from yapocis.interfaces import find_interface

interface_path = find_interface("graphics4")
with open(interface_path) as f:
    specs = parse_interface_text(f.read())
spec = specs[0]
print(f"Interface name: {spec.name}")
print(f"First kernel: {spec.kernels[0].name}")
```

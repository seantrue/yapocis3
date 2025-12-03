# Kernel Development Guide

This is the comprehensive reference for kernel development in yapocis. Start with {doc}`/kernel-quickstart/index` if you're new to writing kernels.

```{toctree}
:maxdepth: 1

interface-files
mako-templates
opencl-kernels
cuda-kernels
metal-kernels
numba-kernels
testing-kernels
```

## Overview

Yapocis kernels consist of two parts:

1. **Interface file** (`.interface`) - Defines the Python-callable signature
2. **Kernel template** (`.mako`) - Contains the actual compute code

The framework handles:
- Parameter marshaling
- Memory management
- Buffer transfers
- Multi-backend compilation

## Kernel Development Workflow

```
1. Design the algorithm
   ↓
2. Write the interface file
   ↓
3. Write the OpenCL template (primary)
   ↓
4. Test with OpenCL
   ↓
5. Port to other backends (optional)
   ↓
6. Add cross-backend tests
```

## File Locations

| File Type | Location |
|-----------|----------|
| Interface | `yapocis/interfaces/<name>.interface` |
| OpenCL | `yapocis/kernels/ocl/<name>.mako` |
| Metal | `yapocis/kernels/metal/<name>.mako` |
| CUDA | `yapocis/kernels/cuda/<name>.mako` |
| Numba | `yapocis/kernels/numbax/<name>.py` |

## Quick Reference

### Interface Syntax

```
interface <name> {
    kernel <func>(in <type> <param>, out_like <type> <ref>, ...);
    extension <alias> as <func>(<modified_params>);
};
```

### Parameter Roles

| Role | Description |
|------|-------------|
| `in` | Read-only input |
| `out` | Caller-allocated output |
| `inout` | Modified in place |
| `out_like <p>` | Auto-allocated output matching `p` |
| `resident` | Stays on GPU |

### Dimension Hints

| Hint | Description |
|------|-------------|
| `width_of <p>` | Extract width from array |
| `height_of <p>` | Extract height from array |
| `depth_of <p>` | Extract depth from array |
| `size_of <p>` | Total element count |

## Next Steps

- {doc}`interface-files` - Complete interface specification
- {doc}`mako-templates` - Template features and patterns
- {doc}`opencl-kernels` - OpenCL-specific guidance

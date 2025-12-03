# Apple Metal

## Requirements

Install `PyMetallic` (Apple Metal Python binding providing a PyOpenCL-like API).
  
## Configuration

* Backend key: set `YAPOCIS_BACKEND=metal` (also accepts `metallic` or `mtl`). * Templates: place Metal Shader Language templates in `yapocis/metalkernels/<name>.mako`.
  - The loader falls back to `yapocis/kernels/<name>.mako` if a Metal-specific template is not present.
  - OpenCL templates are unlikely to compile as-is on Metal; use only for early bring-up. 
* Device selection: optionally set `YAPOCIS_DEVICE` to a device name or index as reported by `pymetallic.get_devices()`. 
* Build cache: `YAPOCIS_CACHE` points to a directory used for compiled program caching if supported by PyMetallic.

## Overview

Yapocis supports Apple Metal for compute by generating Metal Shading Language (MSL) kernels from Mako templates.
The templates live under ``yapocis/metalkernels`` and closely mirror the OpenCL templates for consistency.

Template resolution prefers Metal-specific versions first and falls back to the OpenCL template if missing.

## Key Differences vs. OpenCL

- Address spaces and qualifiers:

  - ``__kernel`` → ``kernel``
  - ``__global`` → ``device``
  - ``__local`` → ``threadgroup``

- Builtins and barriers:

  - ``get_global_id(0)`` → ``uint gid [[ thread_position_in_grid ]]``
  - ``barrier(CLK_GLOBAL_MEM_FENCE)`` → ``threadgroup_barrier(mem_flags::mem_device)``

- Atomics:

  - Use ``device atomic_int*`` with ``atomic_fetch_add_explicit``. A helper ``atomic_inc`` is provided in
    ``metalkernels/constants.mako`` for convenience.

- Argument binding:

  - Scalars are passed by const reference: ``constant T &arg [[ buffer(i) ]]``.
  - Buffers are pointers in device space: ``device T *buf [[ buffer(i) ]]``.

### Invocation Semantics

The Python wrapper maintains OpenCL-like invocation semantics via a thin PyMetallic shim:

``` python
   kernel(queue, global_size, local_size, *args)
```

Global/local sizes are computed and padded in Python much like OpenCL; tiled kernels use the same autopadding logic.

### Selecting the Backend

Use the environment variable or select explicitly:

``` python
   export YAPOCIS_BACKEND=metal

   from yapocis.core.manager import get_manager
   mgr = get_manager("metal")
   program = mgr.load_program("demo")
 ```

Caveats
-------

- Some OpenCL intrinsics may not exist in MSL. Prefer standard math functions or define shims in ``constants.mako``.
- If you need additional atomics or memory scopes, extend the compatibility helpers in ``constants.mako``.
- Scalar argument binding order must match the buffers; ensure interface and templates stay in sync.


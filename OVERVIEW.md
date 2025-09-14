# Yapocis Architecture Overview

The Yapocis project—“Yet another Python OpenCL interface specification”—streamlines writing image-processing and numeric
kernels in OpenCL or CUDA and calling them from Python without manually managing compilation or buffer transfers. A
simple Mandelbrot example in the README demonstrates the RPC-style approach for defining and invoking GPU kernels.

*This overview is created by interaction with ChatGPT codex.*

## Repository Layout

- **`yapocis`** – Core package containing subpackages, assets, and helpers.
    - **`ocl`** – OpenCL device and kernel management.
        - `DeviceManager` wraps PyOpenCL contexts and tracks buffers.
        - `KernelManager` renders Mako templates into kernels and exposes them as callables.
    - **`core`** – Utilities such as:
        - Automatic global/local size padding.
        - Local-size autotuning and timing utilities.
        - Backend selection between OpenCL and CUDA.
        - Device buffer helpers.
    - **`interfaces`** & **`kernels`** – Interface DSL and corresponding Mako templates defining actual GPU code.
    - **`cu`/`cukernels`** – Parallel CUDA support built with CuPy.
    - **`lib`** – Higher-level classes like `BoundKernels` that bind an interface and expose kernel calls via
      convenience methods.

- **`examples`** – Demonstrates the typical workflow:
    1. Define a kernel and its interface using IDL and Mako.
    2. Load the program via `KernelManager.load_program`, which compiles the template and returns a callable wrapper.
    3. Invoke kernels like regular Python functions; the wrapper handles device selection, buffer transfers, and
       dimension padding.
    4. Optionally autotune local sizes for performance and cache results.

- **`tests`** – Contains unit tests showcasing real usage patterns and verifying correctness.

## Architectural notes

### AssetManager 

`AssetManager` centralizes how Yapocis discovers files such as interfaces, kernel templates, fonts, or icons across multiple directories.  
When initialized, it scans:

- The installed `yapocis` package path
- The current working directory
- An optional `YAPOCIS_PROJECT` directory (if set)

This search list is cached, allowing a single instance to provide fast lookups of assets.

#### Initialization

```python
AssetManager(
    dirname,        # root folder of the asset type
    ext,            # primary extension (e.g., ".interface")
    extensions=[],  # additional extensions allowed
    match=None,     # lookup strategy: "exact", "name", "lower", or "partial"
    recursive=False,# traverse subdirectories
    default=None    # fallback asset name
)
```

### Interface specs
The `.interface` files in Yapocis define kernel signatures in a lightweight, IDL-like syntax. Each file describes the callable functions, their parameters, and metadata (e.g., const qualifiers, buffer types, and dimensions) without embedding the actual GPU source code.

### Interface Syntax Support
- Comments: `// line`, `# line`, and `/* block */` comments are ignored by the parser.
- Roles and hints:
  - `in`, `out`, `inout`/`in_out`, `resident`, `READWRITE` map to buffer semantics.
  - `out_like <name>` allocates a device buffer shaped like the referenced parameter.
- Dimension symbols (derived ints): `size_of`, `width_of`/`widthof`, `height_of`/`heightof`, `depth_of`/`depthof`.
- Parameter forms:
  - Role-only: `out_like a`, `width_of a` (dtype optional for size/width/height/depth).
  - Typed with role: `in float a`, `out float c`, `inout float a`.
  - Bare scalars: `int chunk`.
- Modifiers:
  - `interface tiled Foo { ... }` propagates tiled to all kernels unless overridden.
  - `kernel tiled bar(...)` marks only that kernel as tiled.
- Aliases and extensions:
  - Alias: `alias new_name as base_name(<params>);`
  - Extension: `extension new_name of base_name(<params>);` or `extension new_name as base_name(<params>);`
  - Extensions are exposed as `<base>_<new>` names in Python and call the base kernel on-device.

## Structure
A typical `.interface` file contains one or more kernel declarations:

```text
kernel_name(
    in float[] input,
    out float[] output,
    const float scale
)
```

### Key elements:
* Parameter direction: in, out, or inout indicates how buffers are transferred.
* Data types: float, int, etc., may include vector or array notation (float[]).
* Qualifiers: const, shapes, and optional metadata describe how data flows.
* Multiple kernels: One .interface file can declare several kernels sharing common assets.

### Usage in Yapocis
* Parse & Bind: The interface is parsed (via pyparsing) to produce a structured representation of kernel signatures.
* Template Coupling: The corresponding .mako template holds the actual OpenCL/CUDA source. KernelManager.load_program pairs the interface with the template, rendering source code with appropriate definitions.
* Generated Wrappers: Python wrappers are created automatically. When invoked, they handle buffer transfers, type conversions, and dimension checks based on the interface declarations.
* Runtime Checks: If arguments don’t match the declared types or directions, the wrapper raises informative errors before launching the kernel.

### Comparison with Alternative Approaches
| Approach                                         | Advantages                                                                                                             | Drawbacks                                                                              |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **Yapocis `.interface` files**                   | - Clear separation of interface & implementation<br>- Automatic wrapper generation<br>- Early validation of parameters | - Requires learning IDL syntax<br>- Coupled to Yapocis tooling                         |
| **Hand-written wrappers (PyOpenCL / CuPy)**      | - Full flexibility in Python/CUDA/OpenCL code<br>- No custom IDL to learn                                              | - Manual buffer transfers<br>- Error-prone parameter handling                          |
| **Docstrings or Decorators**                     | - Keeps metadata close to Python functions<br>- Leverages standard Python tooling                                      | - Less expressive for low-level memory semantics<br>- Harder to validate at parse time |
| **Language-specific DSLs (e.g., Numba, Triton)** | - High-level, Pythonic kernel definitions<br>- JIT compilation and optimization                                        | - May be less portable between backends<br>- Often tied to specific runtime frameworks |



#### When to Use .interface Files
- You want a formal description of kernel APIs that’s independent of the implementation.
- You need automated glue code for parameter handling and buffer management.
- Multiple kernels share a consistent interface style across OpenCL and CUDA backends. 

#### When Alternatives Make Sense
- Direct control: Writing low-level wrappers yourself is preferable if you require fine-grained control over memory and kernel invocation.
- Python-only kernels: Frameworks like Numba or Triton may be better when staying entirely in Python and targeting specific hardware.

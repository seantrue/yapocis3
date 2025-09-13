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

### AssetManager Overview

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


## Next Steps for Newcomers

- Explore existing `.interface` and `.mako` files to understand how kernels are declared and parameterized.
- Experiment with switching backends via the `YAPOCIS_BACKEND` environment variable or the `core.manager`.
- Dive into `core.autotune` utilities for optimizing kernel performance.
- Run and study the scripts in `examples/` and the suite in `tests/` to see how the pieces fit together and to extend
  coverage for new kernels.

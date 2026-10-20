# Kernel Bundles (non-Python runtimes)

A kernel bundle is everything a runtime needs to call yapocis kernels without
Python: the rendered OpenCL C, Metal and CUDA sources for every interface, and a
`manifest.json` describing each kernel's signature in the flat form the Python
backends execute from. The Rust runtime, [yapocis-rs](https://github.com/seantrue/yapocis-rs),
is built on it.

## Why a bundle

Interface and kernel sources are Mako templates whose context can come from
Python generators (`convolves.yml`, `convolvesep.yml`) and whose kernels bake
constants and weights into the source. Rendering therefore stays a Python,
build-time step; the rendered text plus the parsed parameter model is all a
runtime needs. Nothing in the IDL, the templates or the YAML contexts changes:
the Python package remains the single source of truth and other runtimes consume
its output.

## Generating a bundle

```bash
yapocis-tools bundle --out bundle/                 # ocl, metal and cuda sources
yapocis-tools bundle --out bundle/ --backends ocl  # one backend
yapocis-tools bundle --out bundle/ --fixtures      # + fixtures/derivation.json
yapocis-tools bundle --out bundle/ --parity metal  # + parity/metal/<device>/ (needs a device)
yapocis-tools bundle --out bundle/ demo graphics   # a subset of interfaces
```

No device is needed to render: `yapocis.core.kmgr_base.prepare_program` renders
the interface and kernel for a backend the same way `load_program` does, but
stops before compiling. Generating the same bundle twice gives byte-identical
output.

## Layout

```
bundle/
  manifest.json                schema, yapocis version, per-interface kernels and sources
  ocl/<id>.cl                  rendered OpenCL C
  metal/<id>.metal             rendered Metal Shading Language
  cuda/<id>.cu                 rendered CUDA C (compiled with NVRTC by consumers)
  fixtures/derivation.json     optional: what the Python runtime derives from arguments
  parity/<backend>/<device>/   optional: kernel inputs and outputs recorded on a device
```

`<id>` is the interface name, or `<name>.<variant>` for a named variant.

## Variants

Interfaces whose kernels depend on a template context list their variants in
`src/yapocis/interfaces/bundle.yml`. The default variant is the YAML context plus
generator output, exactly what `load_program(name)` uses. `convolve` has no
default (its weights come from `Convolve1D(name=..., conv=[...])` at runtime), so
a few named kernels are exported instead; arbitrary runtime weights are not
available to bundle consumers. `bundle.yml` also records per-interface build
options that the library layer passes, such as `-cl-opt-disable` for the FFT on
OpenCL.

## Manifest

Per kernel the manifest records what `yapocis.core.interface` produces and the
backends consume:

| Field | Meaning |
|---|---|
| `name` / `symbol` | caller-visible name (extensions are `<base>_<ext>`) and the compiled function name |
| `params` | kernel arguments in order (the Metal `[[buffer(i)]]` index); each has `kind` (`buffer`, `scalar`, `dim`, `out_like`), `dtype`, `role` for buffers, `dim` and `of` for derived entries |
| `user_args` | the non-derived parameters, in the order the caller passes them |
| `returns` | parameters read back after the launch (`inout`, `out`, `out_like`) in declaration order |
| `dispatch_size` | the trait expression pre-parsed into a small AST with the semantics of `yapocis.core.dispatch_expr`, including its `param_index` quirk |
| `cuda.takes_global_size` | the CUDA signature ends with `GLOBAL_SIZE_PARAM` and takes a hidden trailing `uint64` |

Per interface it records the rendered interface text, the context overrides of
the variant, and per backend the source path, its SHA-256 and the build options
(the backend defaults, or the override from `bundle.yml`).

## Fixtures

`--fixtures` writes `fixtures/derivation.json`: for every kernel and a few
synthetic argument shapes, the kernel argument list the Python backends build
(derived dimension values, `out_like` allocations), the `dispatch_size` result,
and the per-backend default global sizes and CUDA/Metal launch geometry. A
consumer replays these to prove its derivation matches. Cases where Python
raises (for example `dim_1_of` on a 1-D array) are recorded as errors.

`--parity BACKENDS` runs every kernel that takes only `in` arrays, scalars and
`out_like` outputs on the named backends with seeded inputs, and saves inputs
and outputs as `.npz` plus an `index.json` with the tolerances. Kernels whose
output is not reproducible run to run are skipped automatically, as are the
packed-input interfaces (`graphics`, `cnn`, `vgg16`, `truetrace`, `fft`) and the
3-plane `anisodiff` kernel, which the library never calls because `float3` has a
different size on CUDA.

`--parity-lib BACKENDS` goes one level up: each library area registers a
generator in `yapocis.codegen.parity_lib` that runs the Python wrapper (display
lists and text in `graphics`, the `truetrace` pipeline, CNN layers, custom
convolution weights, ...) and records inputs and outputs under
`parity/lib/<backend>/<device>/`, so a port of the wrappers can be checked as well as a
port of the kernels. `--generators` selects a subset.

## CUDA under NVRTC

Consumers compile the CUDA sources with NVRTC rather than nvcc. Two things nvcc
took from host headers are provided by `kernels/cuda/constants.mako` so both
compilers accept the same text: `typedef unsigned int uint;` and a guarded
`M_PI`. The nvcc flags `-Xcudafe --diag_suppress=N` are recorded in the manifest
in their NVRTC spelling, `--diag-suppress=N`.

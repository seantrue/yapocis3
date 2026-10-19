# Yapocis for Rust: evaluation, recommendation and plan

Date: 2026-10-18. Status: research complete, plan approved, implementation starting on branch `worktree-rust-port` (this repo) and in the new `yapocis-rs` repo.

## 1. Goal

Let Rust programs call yapocis kernels on OpenCL, Metal and CUDA with no Python at runtime, reusing as much of the existing architecture (IDL, Mako templates, YAML contexts, parameter model, validation data) as possible.

## 2. Method

Seven agents were run in parallel. Three explored this codebase (interface and template pipeline; runtime and backends; library layer, tooling, tests and packaging). Four Opus agents each researched one approach on the Rust side with web sources: direct GPU binding crates; the Rust array and unified-GPU ecosystem; build-time kernel bundles and analogous systems; a shared Rust core with Python interop. Their reports were cross-checked against the source and condensed here.

## 3. What the codebase establishes

### 3.1 Rendering needs Python; execution does not

- Interface text is Mako-rendered with a YAML context that may call a Python generator (`src/yapocis/core/interface_context.py`, `convolves.yml`, `convolvesep.yml`).
- Kernel templates bake constants and weights into source (`src/yapocis/kernels/ocl/convolve.mako`).
- Templates receive only the merged context dictionary, never the parsed interface, the backend name or any parameter data.
- `KernelManagerBase.render_program` (`src/yapocis/core/kmgr_base.py`) needs only the backend name and template lookup, but today runs inside `__init__` after a device is acquired.

Conclusion: rendering must be a build-time step. The rendered text plus a parameter model is everything a runtime needs.

### 3.2 The runtime model is small

`src/yapocis/core/interface.py` defines the flat model the backends use. Per kernel: `name`, `alias` (compiled symbol), an ordered `parameters` list with `name, hint, dim, pointer, data_type, np_dtype, derived, derived_from`, and `traits.dispatch_size`. From that list the runtime derives the user-argument order (non-derived params), the kernel-argument order (all params; also the Metal `[[buffer(i)]]` index), the return order (`out`, `inout`, `out_like` in declaration order) and every dimension value.

| Role | Value | Dtype |
|---|---|---|
| `size_of` / `sizeof` | `arg.size` | int32 |
| `length_of` | `len(arg)` | int32 |
| `width_of` | `shape[0]` | int32 |
| `height_of` | `shape[1]` if ndim > 1 else 1 | int32 |
| `depth_of` | `shape[2]` | int32 |
| `dim_N_of` | `shape[N]` | int32 |

Parser-level features that the runtime drops (`meta`, qualifiers, `[shape]`, `: layout`, `backend {}` overrides) are unused by every shipped interface.

### 3.3 Per-backend conventions that live in Python, not in templates

| Backend | Argument passing | Build options | Default global size | Default local size | Other |
|---|---|---|---|---|---|
| OpenCL | scalars by value, buffers as `cl_mem` | `-w -cl-fast-relaxed-math -cl-mad-enable -cl-no-signed-zeros -cl-finite-math-only`; FFT adds `-cl-opt-disable` | first array's `.size` | driver's choice (NULL) | events; `poll` keyword |
| Metal | every scalar and dimension is a 1-element int32 buffer bound at the parameter index | none | `shape[0]*shape[1]` for ndim >= 2, else shape | `(min(grid.x,256),1,1)` with non-uniform `dispatchThreads` | pymetallic compiles the full source once per kernel name; always synchronous |
| CUDA | raw scalars; buffers as device pointers; hidden trailing `uint64` total size when the signature ends in `GLOBAL_SIZE_PARAM` | nvcc via pycuda: `--use_fast_math -Xcudafe --diag_suppress={177,68,550}`, no `-arch` | first array's `.size` | block `(256,1,1)`, grid = ceil-div | one stream per device manager; `extern "C"` in every kernel template |

Row versus column major is handled entirely by the `ADDRESS/X/Y` macros in each backend's `helpers.mako`; the runtime never permutes or transposes. The library layer compensates in one place (`lib/abstract_graphics.py` coordinate swizzle for non-CUDA backends).

`core/dispatch_expr.py` evaluates `dispatch_size` expressions with different `width_of` and `depth_of` meanings than the argument handlers, and indexes user arguments with the full parameter list. It works for the current `cnn` and `vgg16` interfaces. The Rust port replicates the semantics exactly and documents the inconsistency.

### 3.4 Library layer classification

| Class | Modules |
|---|---|
| Thin kernel wrappers | mandelbrot, operators, convolve, gradient, hsi, zc, colormath |
| Simple orchestration | median, anisodiff, imageops (histeq is host-side), simplex, gaussian (numpy weight generation feeds the template), fft core |
| Heavy host-side Python | typeset (freetype), graphics packing and text, truetrace CPU pipeline, vgg16 and resnet50, mplbackend, convolve_kernels generators |

### 3.5 Validation data

References are single-array compressed `.npz` files compared with `allclose` (`src/yapocis/utils/validation.py`). They are readable from Rust with `ndarray-npy`. No corpus is committed, so parity fixtures must be generated.

## 4. Evaluation of the approaches

### 4.1 Direct bindings (one Rust crate per API, JIT from source at runtime)

Verdict: adopt for execution.

| Backend | Crate | Version | Status | Notes |
|---|---|---|---|---|
| OpenCL | `opencl3` (+ `cl3`, `opencl-sys`) | 0.12.3 / 0.13.2 | active 2026 | dynamic loading by default; program binaries via `get_binaries`; `ocl` crate stale since 2024 |
| Metal | `objc2-metal` (+ `objc2`, `block2`) | 0.3.2 | active | `metal-rs` is deprecated; wgpu migrated to objc2-metal; unsafe binding surface |
| CUDA | `cudarc` | 0.19.10 | active, monthly | driver API + NVRTC; dynamic loading; per-version bindings `cuda-12xxx` / `cuda-13xxx`; panics if a library is missing |

No existing crate abstracts "compile this source and launch it" across all three. `rust-gpu-tools` covers OpenCL plus prebuilt CUDA fatbins only. Risks: three unsafe surfaces; CUDA moves from nvcc to NVRTC (option spelling, no host code); macOS OpenCL is deprecated at 1.2 but still ships; CUDA 13 drops Pascal, so the P40 needs a 12.x NVRTC.

### 4.2 Stand on a unified Rust GPU framework

Verdict: reject as execution substrate; adopt the host-side crates.

| Framework | OpenCL | Metal | CUDA | Accepts OpenCL C / MSL / CUDA C source | Verdict |
|---|---|---|---|---|---|
| wgpu + naga | no | yes | no | MSL only through unsafe passthrough; naga cannot parse MSL | no |
| CubeCL 0.11 | no | yes (via wgpu) | yes | CUDA C and MSL via `PrecompiledSource`, with its own binding layout | closest, 2 of 3, alpha API |
| burn, candle | no | yes | yes | indirect, ad hoc | wrong abstraction |
| arrayfire-rust | yes | no | yes | no | stale bindings |
| krnl, vulkano, Kompute, rust-gpu | no | MoltenVK | no | SPIR-V only | no |

Host side is solved: `ndarray` 0.17, `bytemuck`, `num-complex`, `half`, `image` (+ tiff, png), `imageproc` and `rustfft` as CPU oracles, `rayon`, `freetype-rs` or `fontdue` for text later. A WGSL fourth backend is a plausible later addition, not a replacement (no u8 or i16 storage, no f64, limited atomics).

### 4.3 Build-time bundle (Python renders everything to a language-neutral bundle)

Verdict: adopt for the front half.

| Analog | Ships | Signature description | Host API |
|---|---|---|---|
| Futhark `--library` | C with embedded OpenCL C / CUDA via NVRTC, optional binary cache | JSON manifest with a published schema | `futhark-bindgen` generates Rust |
| Halide AOT | object + header; Metal and OpenCL as embedded source, metallib opt-in | `halide_filter_metadata_t` | C ABI |
| Taichi AOT | directory with `metadata.json` plus per-arch artifacts | JSON schema | C runtime |

Findings: 19 of 23 interfaces need no runtime context; `operators`, `convolves` and the default `convolvesep` are deterministic given their generators; only `convolve` with arbitrary `conv` and `Gaussian` with custom weights depend on caller values. Reflection (`clGetKernelArgInfo`, `MTLBufferBinding`, CUDA param info) cannot express `out_like`, `width_of`, `resident` or aliases, so a schema is required; reflection serves only as a load-time check. Ship source rather than binaries: OpenCL binaries are device-specific, metallib needs the separately downloaded Xcode Metal toolchain, PTX needs a CUDA toolkit at build, and all three backends compile source at runtime today. Cost: derivation and dispatch semantics are re-implemented in Rust and can drift, so parity tests are mandatory.

### 4.4 Shared Rust core under Python (PyO3 / maturin)

Verdict: not now.

For a one-maintainer project it trades a pure-Python wheel for a matrix of 5 to 8 platform wheels, a free-threading audit and lock-step releases, while Mako rendering and the numba backend stay in Python regardless. Projects that inverted the dependency (polars, pydantic-core, tokenizers, safetensors) are staffed teams whose native core is the product; rapidfuzz needed several major versions to reach parity between its two paths. The duplicated runtime is roughly 550 lines per backend. Revisit only if Python launch overhead is shown to dominate, and then through a C-ABI `cdylib` loaded like pymetallic's Swift bridge. Embedding Python in Rust or a sidecar process: never.

## 5. Recommendation

Build a Rust-only runtime, `yapocis-rs`, that consumes a kernel bundle generated by this Python package at build time and executes the unchanged kernel sources through `opencl3`, `objc2-metal` and `cudarc`.

Decisions taken by the maintainer on 2026-10-18:

- Rust code lives in a separate repo, `yapocis-rs`.
- Runtime-parameterised kernels are handled by enumerating named variants at build time; arbitrary runtime weights are unsupported in Rust for now.
- The first release covers all interface kernels plus the thin library wrappers. Graphics host logic, text, truetrace CPU code, CNNs and mplbackend are out of scope.
- CUDA uses NVRTC through cudarc pinned to a CUDA 12.x binding so the P40 and the 3060 both work.

### 5.1 Architecture

```
Yapocis3 (Python)                                  yapocis-rs (Rust)
.interface + .yml + .mako                          yapocis-bundle  serde schema, derivation,
    |  yapocis-tools bundle                                        dispatch AST evaluation
    v                                                              (no GPU dependencies)
bundle/manifest.json  --- vendored copy --->       yapocis-rt      Device / Buffer / Program / Kernel,
bundle/ocl/*.cl                                                    features ocl | metal | cuda, disk cache
bundle/metal/*.metal                               yapocis         public crate; build.rs generates one
bundle/cuda/*.cu                                                   typed struct per interface; lib ports
pytest --export-parity -> parity/*.npz  ------->   yapocis-rs/tests parity against Python outputs
```

### 5.2 Design rules for the Rust runtime

- Same contract, better ownership: explicit `Buffer<T>` handles replace the `id(ndarray)`-keyed cache. `in` uploads from an `ArrayView`; `inout` and `out` take `&mut ArrayViewMut` and are written back; `out_like` allocates and returns an `Array`; `resident` passes a `Buffer<T>`. A typed generated API sits on an untyped `Kernel::launch(&[Arg])`.
- Derivation and sizing match Python exactly, including Metal's default global size and the `dispatch_size` quirks, so parity holds.
- Source compiled once per interface per device, cached on disk keyed by a hash of source, options, backend, device and driver identity and runtime version.
- Metal: one library per interface; scalars and dimensions through `setBytes` at the parameter index; `dispatchThreads`; an autorelease pool per launch; buffers retained until the command buffer completes.
- CUDA: `cudarc` with `dynamic-loading` and a pinned `cuda-12xxx` feature; NVRTC options `--use_fast_math` and `--diag-suppress`; the hidden `GLOBAL_SIZE_PARAM` argument recorded by the exporter.
- Host arrays: `ndarray` at the boundary, C-contiguous required, `bytemuck` for bytes, `Complex32` for complex64.
- Errors through a `thiserror` enum with a variant per backend; `tracing` spans around compile and launch.

### 5.3 Bundle manifest sketch

```json
{
  "schema_version": 1,
  "yapocis_version": "1.0.0",
  "interfaces": [{
    "name": "graphics", "variant": "default",
    "kernels": [{
      "name": "lines_res", "symbol": "lines",
      "params": [
        {"name": "a", "kind": "dim", "dim": "width_of", "of": "a", "dtype": "int32"},
        {"name": "a", "kind": "dim", "dim": "height_of", "of": "a", "dtype": "int32"},
        {"name": "a", "kind": "buffer", "role": "resident", "dtype": "float32"}
      ],
      "user_args": ["a", "lines", "distance", "aa_width", "rcolor"],
      "returns": [],
      "dispatch_size": null,
      "cuda": {"has_global_size_param": false}
    }],
    "backends": {
      "ocl":   {"source": "ocl/graphics.cl",      "build_options": ["-w", "-cl-fast-relaxed-math", "..."]},
      "metal": {"source": "metal/graphics.metal", "build_options": []},
      "cuda":  {"source": "cuda/graphics.cu",     "build_options": ["--use_fast_math", "--diag-suppress=177", "..."]}
    }
  }]
}
```

## 6. Plan

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 0 | This report; a throwaway spike calling `demo.sum` from Rust on each backend | `sum` matches numpy on OpenCL and Metal (thresher) and CUDA P40 and 3060 (yellowfin) |
| 1 | `yapocis-tools bundle` exporter, device-free rendering, `bundle.yml` variants, tests | manifests for all interfaces on all three backends; a bundle generated twice is byte-identical |
| 2 | `yapocis-bundle` crate: schema, derivation, dispatch evaluation, Python-generated fixtures | `cargo test` passes without a GPU |
| 3 | `yapocis-rt` crate: OpenCL, then Metal, then CUDA backends; disk cache; async | per-backend smoke tests |
| 4 | `yapocis` crate: build.rs codegen, thin library ports, parity tests, examples | parity suite passes within the Python tolerances on both machines |
| 5 | CI (build, clippy, bundle drift), docs, crates.io release | drift job fails when templates change without regenerating the bundle |

Out of scope for the first release: arbitrary runtime convolution weights; graphics display-list packing, coordinate swizzle and text; truetrace CPU pipeline; vgg16 and resnet50; mplbackend; the numba backend; precompiled metallib, PTX or cubin shipping; a WGSL backend; a shared core driving Python through a C ABI.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| NVRTC rejects a CUDA kernel that nvcc accepted | Phase 0 spike compiles every CUDA source with NVRTC 12.x on yellowfin before Phase 3 |
| Derivation or dispatch drift between Python and Rust | Python emits derivation fixtures; parity tests compare kernel outputs per backend |
| Bundle drift after template edits | CI regenerates the bundle from the pinned yapocis version and diffs |
| Apple OpenCL disappears | Metal is the primary macOS backend; OpenCL stays feature-gated |
| `cudarc` panics when a CUDA library is missing | Probe for `libcuda` and `libnvrtc` before touching cudarc; report a clean error |

## 8. Sources

Crate versions and dates come from crates.io as of 2026-10-18. Primary references: opencl3 and cl3 (github.com/kenba), objc2-metal (docs.rs/objc2-metal), cudarc (github.com/coreylowman/cudarc), CubeCL (github.com/tracel-ai/cubecl), wgpu passthrough shaders (docs.rs/wgpu-types), Futhark C API and manifest (futhark.readthedocs.io), Halide AOT tutorials and runtime sources (halide-lang.org), Taichi AOT docs (docs.taichi-lang.org), NVIDIA CUDA 13 architecture support notes (developer.nvidia.com), PyO3 and maturin release pages (github.com/PyO3), ndarray 0.17 release (github.com/rust-ndarray/ndarray).

## 9. Outcome (2026-10-19)

All five phases were implemented in one session. Nothing has been committed; the Python changes sit on branch `worktree-rust-port` and the Rust repo at `/Users/seant/workspace/yapocis-rs` is initialised but uncommitted.

| Phase | Result |
|---|---|
| 0 | Spike: `demo.sum` exact on OpenCL and Metal (thresher) and on the RTX 3080, RTX 3060 and Tesla P40 (yellowfin, NVRTC 12.5). All 21 packaged CUDA sources compile under NVRTC after `kernels/cuda/constants.mako` gained a `uint` typedef and a guarded `M_PI`. |
| 1 | `yapocis-tools bundle` with `--fixtures` and `--parity`; device-free rendering (`prepare_program`); `bundle.yml` variants; 24 interface variants exported; byte-identical on repeat; 25 pytest cases in `tests/integration/test_bundle_export.py`; docs pages. |
| 2 | `yapocis-bundle` crate replays every derivation case from Python (>500) plus unit tests for dispatch and launch geometry. |
| 3 | `yapocis-rt` with OpenCL, Metal and CUDA backends, disk cache, async launches; end-to-end tests on all three. |
| 4 | `yapocis` crate: `build.rs` generates typed structs for all interfaces; library ports (operators, gradient, hsi, colormath, zcs, median, mandelbrot, anisodiff, gaussian, fft); parity replay matches Python on 95 kernel cases per backend. |
| 5 | README, MIT license, GitHub Actions (build, clippy, bundle-crate tests, bundle drift), `scripts/check-bundle.sh`; clippy and rustfmt clean. |

Findings worth keeping: Apple's OpenCL batch FFT is wrong in the Python package too, so the Rust test only checks it on Metal and CUDA; `anisodiff.filterImage3` (float3 layout) and the chunked `convolve *_fast` kernels are excluded from parity because their outputs depend on out-of-bounds behaviour. Left for later: committing (with AI.md per the `ai-md` skill), publishing to crates.io, and the out-of-scope library ports.

## 10. Full port (2026-10-19)

The maintainer then asked for the previously out-of-scope work. Three forked agents and the parent ported the rest in parallel, each with Python-generated library-level parity fixtures (`yapocis-tools bundle --parity-lib`) replayed by Rust tests on every backend.

| Area | Rust modules | Parity cases per backend |
|---|---|---|
| Runtime-weight convolutions | `lib::convolve` (Rust port of the `convolve` and `convolvesep` templates), `Gaussian::with_weights` | 2 |
| Graphics, text, paths | `lib::graphics`, `lib::kerneldraw`, `lib::typeset` (bundled FreeType), `lib::paths` (earcut), embedded fonts | 30 |
| Processing | `lib::truetrace` (GPU and full CPU pipeline, SVG), `lib::imageops`, `lib::simplex`, the rest of `lib::anisodiff` and `lib::fft` | 10 |
| CNNs | `lib::cnn`, `lib::vgg16`, `lib::resnet50`, vendored class labels | 24 kernel cases plus the ResNet50 model case where weights exist |

Python-side defects found and reproduced rather than fixed are listed in the fork reports and the module docs: `trace_segments` launching one thread on OpenCL/CUDA because a numpy scalar has `.size`, `build_regions_with_holes` never attaching holes, `write_svg_regions(curve="none")` raising, int32 wrap in `histeq`, nondeterministic batch spectrograms on Apple backends, display-list `alpha`/`distance` ignored, `to_vector` treating a zero scalar as absent, and the `convolve *_fast` template reading `a[j]` literally. Not portable by nature: the numba backend and the matplotlib `RendererBase` plugin.

One Python defect was fixed rather than reproduced: `AssetManager` listed asset directories in `os.listdir` order, so partial font names such as `"italic"` resolved to different files on macOS (APFS) and Linux (ext4). The listing is now sorted, the Rust font table searches in the same order, and the CUDA text parity case that had failed for that reason passes.

# History

Yapocis emerged from decades of experimentation with image processing hardware and software. This page traces that journey from specialized hardware to modern multi-backend GPU computing.

```{toctree}
:maxdepth: 1
:hidden:

ai-contributions
```

## The Core Problem

Images are inherently computationally challenging. Lots of data leads to lots of approximations, shortcuts, and undone experiments. We need code that runs fast and is easy to develop.

## The Distant Past (1980-2010)

The author has tried many hardware and software configurations for image processing experiments:

| Era | Technology | Notes |
|-----|------------|-------|
| 1980 | CADDR-1 & convolution hardware | Specialized Lisp machines, not readily available |
| 1985 | Intel 386/486 + C/Assembly + DOS extenders | Overwhelmingly expensive array processors |
| 1995 | Pentium + Python + PIL | Python comes with batteries included |
| 2000-2005 | Python + Numeric/Numarray/NumPy + PIL | The NumPy ecosystem emerges |
| 2010 | Python + NumPy + PIL + Pyrex/Cython | Performance through compilation |

### What Was Ignored

**Cluster-based multi-processing** was considered but rejected. Having many computers sounds appealing, but they're usually doing something else, and cluster management adds overhead—both literal and in headaches.

## The GPU Era (2011)

By 2011, several alternatives were available:

- **Multi-core multi-processing** - Good, but not transformative
- **NVIDIA CUDA** - Powerful, but vendor lock-in
- **OpenCL** - Cross-platform, and free with every Mac

GPU solutions were promising. Macs included both an AMD GPU with OpenCL support and Apple's CPU implementation of OpenCL for Intel, which was surprisingly fast.

### Why OpenCL?

CUDA was (and remains) less widely available and carries massive vendor lock-in consequences. OpenCL runs on:

- AMD GPUs
- Intel CPUs and integrated graphics
- Apple GPUs (via Metal bridge)
- NVIDIA GPUs (yes, really)

Writing in OpenCL is like writing in C. Unoptimized OpenCL can be easy to read and run 10x faster than NumPy. Optimized OpenCL can be hard to read and approach theoretical hardware limits.

### Enter PyOpenCL

PyOpenCL made OpenCL usable from Python. But the procedure call interface was rough—we could do better.

The solution: an interface definition language inspired by Apollo NCS RPC, OSF DCE RPC, and Microsoft COM, combined with a Python thunking layer. With syntactic sugar, even the bitter parts go down well.

## Modern Era (2020-Present)

### New Alternatives

- **Numba**: Taking NumPy closer to the metal with JIT compilation
- **Metal**: Apple's proprietary GPU API, now supported via PyMetallic
- **PyCUDA**: Mature CUDA support for NVIDIA hardware

### The Complicated Landscape

GPU computing has evolved primarily in response to deep learning needs. NVIDIA dominated this space, but change is coming:

- **OpenCL continues** - Intel and AMD have doubled down, supporting CPUs, GPUs, FPGAs, and neural compute chips
- **Apple went proprietary** - Metal replaced OpenCL on macOS, requiring a compatibility layer
- **CUDA matured** - Still vendor-locked, but ubiquitous in ML/AI

## Yapocis Timeline

| Version | Year | Milestone |
|---------|------|-----------|
| 0.1.x | 2011 | Initial PyOpenCL wrapper with IDL parser |
| 0.2.x | 2020 | Metal backend, improved template system |
| 0.3.x | 2024 | CUDA backend, Numba fallback, modern packaging, AI-assisted development |
| 1.0.0 | 2026 | First release on Python 3 PyPI: one drawing convention, colour-space and CUDA fixes, CC0 assets |

### Version 1.0 Focus Areas

1. **One coordinate convention**: graphics take x = column and y = row on every backend
2. **Correctness across backends**: colour-space matrices, CUDA gradient angle and bounds, anisotropic diffusion and zero-length lines
3. **Extension projects**: `YAPOCIS_PROJECT` search order, scaffolding and `.pyi` stub generation
4. **Release hygiene**: MIT-clean dependency tree, CC0 demo images, documentation examples tested with `yapocis-doctest`

See the [changelog](https://github.com/seantrue/yapocis3/blob/main/CHANGELOG.md) for the full list, including known issues.

### Version 0.3.x Focus Areas

1. **Performance Infrastructure**: DuckDB-backed tracking with cross-backend comparison
2. **Validation Framework**: Pixel-exact regression testing
3. **Modern Packaging**: Hatchling, ruff, mypy, Python 3.13 support
4. **Backend Maturity**: Feature parity across OpenCL, Metal, CUDA, and Numba

## AI-Assisted Development

Starting in 2024, yapocis development has been significantly accelerated through AI assistance. See {doc}`ai-contributions` for details on AI contributions to the codebase.

## Lessons Learned

1. **Abstraction pays off** - The IDL layer made adding new backends straightforward
2. **Test everything** - Cross-backend validation catches subtle bugs
3. **Measure performance** - You can't optimize what you don't measure
4. **Stay flexible** - Hardware changes; software should adapt

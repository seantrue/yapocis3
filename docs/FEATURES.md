# Features

Features requested in passing (`feature: …`), kept until built or declined. Numbers are never reused. Entries with an issue are also tracked on GitHub.

## Open

### F1. `simplex2d` tileable output that actually tiles seamlessly

- Reported: 2026-11-11
- Report: "2 A ado feature request for C" (the maintainer, choosing option 2C from the #65 plan as a feature request)
- Context: `tileable` wraps the lattice in skewed coordinates, so the output repeats along the diagonal and never on x or y. No period fixes that. Possible approaches: 4-D simplex on a torus, or periodic gradient noise. It would change recorded tileable outputs in impressionist (3), plumimpress (4), yapocis-rs (5 bundles) and PlumCraft (1, plus its port).
- Issue: #83

### F3. Kernels to initialize GPU memory

- Reported: 2026-11-12
- Report: "Add a feature: kernels to initialize GPU memory"
- Context: no interface or device manager has a fill, clear, zero or memset. Arrays that need a starting value are built in numpy and uploaded: graphics canvases (`np.zeros` as `inout a`), `histogram`'s zeroed bins (`inout hst`), and `marching_squares`' `segments_out`. Native fills exist: OpenCL `enqueue_fill_buffer`, CUDA `GPUArray.fill`/`cuMemsetD32`, and on Metal pymetallic's `fill_u32` (32-bit patterns only; it compiles its helper library on every call). Profiling on `perf/metal-sampling` (`claude-debug/METAL_CLOCK_SAMPLING_REPORT.md`) found that on Metal a new buffer pays a first-touch cost wherever it is first written, so a fill gains little there unless the buffer is reused. On OpenCL and CUDA, a fill would replace a host upload.
- Open question: is the interface a new parameter hint (for example `fill(0) float hst`), a library helper (`ensure_filled(a, value)`), or standalone `fill` kernels in an interface?
- Open question: Metal: wrap `fill_u32` with a cached pipeline, write a yapocis fill kernel, or wait for buffer pooling?
- Issue: none

## Done

### F2. CI on the rustcpu backend only

- Reported: 2026-11-12
- Report: "Feature: Add ci for yapocis, on the rustcpu backend only"
- Context: yapocis3 has no `.github/workflows`; its only check on PRs is the Read the Docs build. `make test-rustcpu` runs the suite with yapocis-rs's CPU backend (674 passed, 45 skipped at the 1.2.0 gate) and needs no GPU, but it needs a `yapocis_rs` wheel, which today is built locally from yapocis-rs (`556ef5c` or later).
- Decided (2026-11-12): "For CI on yapocis, build a local wheel to test rustcpu" (the job builds `yapocis_rs` from a pinned yapocis-rs commit); "We never test CI on MacOS -- check your notes. Test on the most recent supported Python." (Linux, Python 3.13); "1: not yet, 2: yes" (no ruff/typecheck yet; confirm the Mako-redirect test in a clean checkout: it passes).
- Done: 2026-11-12, PR #106 (`0ce20ed`), `.github/workflows/ci.yml`; the first green run had 639 passed, 118 skipped.
- Issue: #102

## Declined

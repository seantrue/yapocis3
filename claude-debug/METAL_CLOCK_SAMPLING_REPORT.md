# Metal backend: wall-clock sampling profile

Date: 2026-11-12. Host: thresher (Apple M4 Max). Commit: 6fa5e01 (release 1.2.0), run from worktree `../yapocis3-metal-sampling` (branch `perf/metal-sampling`, no source changes).

## Method

- Profiler: pyinstrument 5.1.3, a statistical **wall-clock** sampler (0.5 ms interval, async mode off). Time spent blocked in the GPU wait is counted, not just CPU time.
- Workloads: the five workflows from `tests/performance/test_library_perf.py`, at 512×512 float32: Gaussian multiscale (5 scales), edge detection (gauss, gradient, magnitude), HSI round trip, Graphics4 200 circles plus read, and denoise (gauss, median3x3, anisodiff n=5).
- Each workload: build the objects, take one cold call, run 5 warm-up calls, then sample 500 steady-state iterations (1,000–2,500 samples per workload).
- Time is bucketed by the first matching frame on each stack: `wait_until_completed` (GPU wait), `DeviceManager.make_buffer` (output-buffer allocation), `write_buffer` (input upload), `read_buffer` (readback). Everything else is Python and numpy overhead.
- Scripts: `claude-debug/metal_clock_sampling.py` (full profiles and HTML/text trees), `metal_clock_sampling_buckets.py` (bucket table), `metal_clock_sampling_alloc_bench.py`, `metal_clock_sampling_check.py`. Output is in `claude-debug/metal_clock_sampling_out/` (`*_hot.html` and `*_cold.html` are the flame trees).

## Results

Each cell shows the range over three runs.

| workload | median ms/iter | GPU wait | out-buffer alloc+copy | input upload | readback | other (Python/numpy) |
|---|---|---|---|---|---|---|
| gaussian_multiscale | 2.55–2.61 | 44–55% | 0% | 15–22% | 4–6% | 26–28% |
| edge_detection | 1.01–1.05 | 27–35% | 7–11% | 18–27% | 5–7% | 30–33% |
| color_conversion | 1.37–1.40 | 51–58% | 11–16% | 13–15% | 3–6% | 14–17% |
| graphics_circles | 1.10–1.15 | 54–67% | 0% | 27–42% | 0–1% | 4–6% |
| denoise | 2.01–2.14 | 42–50% | 0% | 22–25% | 4–5% | 22–31% |

Cold start: the first workload in a fresh process took 0.26 s with a warm kernel cache and 4.4 s on the very first run, which included Metal device setup and filling the cache. Later workloads take 10–40 ms cold.

## Findings

### 1. Output buffers are allocated by copying an empty host array (fixable)

`Kernel.handle_outlike` (`src/yapocis/metallic/kernel.py:92`) does `np.empty_like(arg)` and then `ensure()`. That calls `make_buffer`, which calls `Buffer.from_numpy` (`src/yapocis/metallic/device_manager.py:74`). So every output buffer pays for a host-to-device copy of uninitialised memory. In the HSI round trip this costs more than uploading the real input: 0.22 s versus 0.08 s in `hsi2rgb3`.

Microbenchmark of a single allocation:

| shape (float32) | from_numpy(empty) µs | Buffer(nbytes) µs | ratio |
|---|---|---|---|
| (256, 256) | 23 | 6 | 3.9x |
| (512, 512) | 65 | 5 | 12.1x |
| (512, 512, 3) | 165 | 5 | 30.9x |
| (512, 512, 4) | 231 | 6 | 40.8x |
| (2048, 2048, 4) | 3476 | 8 | 423.8x |

Allocating outlike outputs with `pymetallic.Buffer(device, nbytes)` (no copy) would remove most of the 7–16% bucket in edge_detection and color_conversion, and the saving grows with image size. Caveat: `make_buffer` is shared with `ensure()` for resident and inout arrays, whose contents matter. The no-copy path must apply only to outputs.

### 2. GPU wait is the largest bucket, and it is synchronous per kernel

Every kernel call ends with `CommandBuffer.wait_until_completed` (`pymetallic Kernel.__call__`). Multi-kernel workflows (Gaussian scales, denoise, anisodiff iterations) serialise CPU setup with GPU execution. Batching several dispatches into one command buffer, or committing without waiting until a result is read, would overlap the "other" bucket with GPU time. This is a design change, not a quick fix.

### 3. Input upload is a full copy into a new MTLBuffer each call

`write_buffer` calls `Buffer.from_numpy` (`newBufferWithBytes`), so every call allocates a new buffer and copies. On unified memory, a page-aligned numpy array could be wrapped with `newBufferWithBytesNoCopy`, or a pooled buffer could be reused. graphics_circles is dominated by this (27–42%), partly because the workload makes a new 1 MB canvas on each iteration.

### 4. Python overhead is 25–33% for small multi-kernel workloads

At 1–2.6 ms per iteration, argument preparation, buffer registration (`_register` and `_drain`) and `weakref.finalize` cleanup each show up. The fixed overhead per call matters most for 256–512 px images. These are the hot frames to look at in `*_hot.html`.

## Caveats

- Run-to-run variance is about ±10 points per bucket, from GPU clock and power state. Per-iteration medians are stable to within about 5%.
- A wall-clock sampler cannot see inside the ctypes calls. "GPU wait" covers kernel execution plus command-buffer scheduling latency, so it is not pure kernel time.
- The sampler adds a few percent of overhead.
- Profiled with `save_perf` off, so the performance database was not touched.

## Follow-up: out_like fix (finding 1)

Change on branch `perf/metal-sampling` (uncommitted): a new `DeviceManager.make_output_buffer` (`src/yapocis/metallic/device_manager.py`) allocates `pymetallic.Buffer(device, nbytes)` with no host copy. `Kernel.handle_outlike` (`src/yapocis/metallic/kernel.py`) now calls it instead of `ensure()`. Nothing else uses it: in, inout and resident arrays still use `make_buffer`. Zero-byte arrays fall back to `make_buffer`.

Tests: `YAPOCIS_TEST_BACKENDS=metal pytest -m "not performance" tests` gave 700 passed, 70 skipped and 0 failed (log: `metal_outlike_fix_tests.log`).

Re-profile on a quiet GPU, as the range over three runs:

| workload | before ms/iter | after ms/iter | out-buffer bucket before | after |
|---|---|---|---|---|
| gaussian_multiscale | 2.55–2.61 | 2.55–2.80 | 0% | 0% |
| edge_detection | 1.01–1.05 | 1.00–1.09 | 7–11% | 0% |
| color_conversion | 1.37–1.40 | 1.23–1.29 | 11–16% | 0% |
| graphics_circles | 1.10–1.15 | 1.20–1.22 | 0% | 0% |
| denoise | 2.01–2.14 | 2.21–2.22 | 0% | 0% |

The output-allocation bucket is gone. color_conversion, which has 3-channel outputs, is about 9% faster. edge_detection's per-iteration time did not change measurably: the saved copy (about 60 µs at 512²) is within run-to-run noise, and readback's share went up. That suggests the first touch of newly allocated pages moved into readback. The other three workloads have no out_like outputs, so their differences are noise (they also ran in a different session). The saving should grow with image size, as the allocation microbenchmark shows.

## Follow-up: the same out_like fix on OpenCL

Same change for the OpenCL backend: `DeviceManager.make_output_buffer` in `src/yapocis/ocl/device_manager.py` allocates `pyopencl.Buffer(ctx, READ_WRITE, size=nbytes)` with no `COPY_HOST_PTR`. `handle_outlike` in `src/yapocis/ocl/kernel.py` calls it. Nothing else uses it.

Tests: `YAPOCIS_TEST_BACKENDS=ocl pytest -m "not performance" tests` gave 700 passed, 70 skipped and 0 failed (log: `ocl_outlike_fix_tests.log`). `get_test_backends()` resolved to `['ocl']`. The totals happen to match the Metal run, but which tests were skipped differs.

Timing on Apple OpenCL (M4 Max), as the range over three alternating runs. "main" is `src/` at 6fa5e01 loaded through `PYTHONPATH` in the same venv:

| workload | main ms/iter | fix ms/iter | change |
|---|---|---|---|
| gaussian_multiscale | 6.22–6.37 | 6.34–6.40 | none (no out_like) |
| edge_detection | 2.33–2.37 | 2.24–2.33 | about 3% faster |
| color_conversion | 3.15–3.29 | 2.06–2.10 | about 34% faster |
| graphics_circles | 2.20–2.23 | 2.20–2.24 | none (no out_like) |
| denoise | 5.48–5.72 | 5.53–5.70 | none (no out_like) |

On `main`, the sampler put about 0% of time in `make_buffer` on OpenCL. Creating a `COPY_HOST_PTR` buffer returns quickly, and the driver does the copy later, at the first enqueue, so the cost showed up in the kernel run and wait instead. Bucketing by Python frame cannot see the cost on OpenCL; only end-to-end timing does. The gain is much larger than on Metal: with a 3-channel 512² output, Apple's OpenCL `COPY_HOST_PTR` path costs about 1.1 ms per call for two 3 MB outputs.

## Follow-up: the same out_like fix on CUDA (yellowfin, Tesla P40)

Same change for the CUDA backend: `DeviceManager.make_output_buffer` in `src/yapocis/cu/device_manager.py` allocates `gpuarray.empty(shape, dtype, order=...)` with no `to_gpu_async` upload. The order matches the host array, so `read_buffer`'s `get_async(ary=a)` copies straight back. `handle_outlike` in `src/yapocis/cu/kernel.py` calls it.

Run on yellowfin from a detached worktree at 6fa5e01 (`/tmp/yp-outlike`) with the patch `outlike_nocopy_all_backends.patch` applied. The P40 was the only device visible (`CUDA_VISIBLE_DEVICES=0`, `YAPOCIS_DEVICE="Tesla P40"`). Python 3.13 `.venv` with `PYTHONPATH=/tmp/yp-outlike/src`; the import path was checked.

Tests: `YAPOCIS_TEST_BACKENDS=cuda pytest -m "not performance" tests` gave 654 passed, 79 skipped and 0 failed (log: `cuda_outlike_fix_tests_yellowfin_p40.log`).

Timing on the P40, as the range over three alternating runs of 300 iterations each. "main" is the yellowfin checkout at 6fa5e01:

| workload | main ms/iter | fix ms/iter | change |
|---|---|---|---|
| gaussian_multiscale | 10.18–10.58 | 10.20–10.64 | none (no out_like) |
| edge_detection | 4.37–4.58 | 3.96–4.29 | about 6% faster |
| color_conversion | 5.81–6.06 | 4.92–5.36 | about 12% faster |
| graphics_circles | 5.99–6.34 | 5.71–6.20 | none (no out_like) |
| denoise | 6.52–6.89 | 6.35–6.98 | none (no out_like) |

The 3060 and 3080 were running other jobs during these runs, so expect a few percent of noise. Even so, the fix's ranges for the two workloads with out_like outputs sit below main's on every run.

## Summary across backends (color_conversion, the out_like-heavy workload)

| backend | before ms/iter | after ms/iter | gain |
|---|---|---|---|
| Metal (M4 Max) | 1.37–1.40 | 1.23–1.29 | about 9% |
| OpenCL (M4 Max) | 3.15–3.29 | 2.06–2.10 | about 34% |
| CUDA (Tesla P40) | 5.81–6.06 | 4.92–5.36 | about 12% |

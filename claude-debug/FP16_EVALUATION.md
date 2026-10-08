# Evaluation: FP16 alongside the FP32 default

Date: 2026-10-07. Measured on thresher (Apple M4 Max, Metal + Apple OpenCL) and yellowfin (Tesla P40, RTX 3060, RTX 3080; CUDA 12.5 + NVIDIA OpenCL).

## Summary

FP16 is worth adding as an **opt-in storage type** ("half in memory, float in registers") for bandwidth-bound libraries. It is not worth making the default, and it should not be offered at all for libraries whose values are coordinates, long accumulations, or high-dynamic-range quantities.

- The core already carries FP16 end to end. A prototype interface declared with `half` compiled, ran, and returned `float16` arrays on both Metal and OpenCL without any change to `core/`.
- FP16 roughly **halves the time of transfer-bound calls** on every device that supports it: 1.3–1.9x on the M4 Max, about 2x on all three CUDA cards.
- Compute-bound FP16 is **about the same speed as FP32** on the M4 Max, 3060 and 3080, but **about 48x slower on the Tesla P40**. Pascal sm_61 runs FP16 math at 1/64 of the FP32 rate.
- **NVIDIA OpenCL has no FP16.** None of yellowfin's GPUs reports `cl_khr_fp16`, and the `half` kernels fail to build. On yellowfin, FP16 is CUDA-only.
- **Result comparisons (ZC and anisodiff).** FP16 storage is safe for the zero-crossing kernel. For the full DoG→ZC pipeline it is acceptable only if the Gaussian outputs stay in FP32. Anisodiff is fine for short runs (about 10 iterations) but drifts with iteration count, so its state should stay FP32. Every result was identical across Metal, Apple OpenCL and the three CUDA cards; see "Result comparisons" below.
- The cost is in the kernel templates. About 750–940 `float` tokens are hard-coded in each GPU backend's `.mako` files, and the Numba CPU backend cannot compile `float16` at all.

## What already exists

| Layer | FP16 state |
|---|---|
| IDL grammar (`core/interface_parser.py:222`) | Accepts `half`, `f16`, `float32`, `double`, … |
| Type map (`core/interface.py:49`) | `half → float16` works. **`f16` has no entry**, so `np_dtype` becomes `"f16"` and `getattr(np, "f16")` fails at call time (latent bug). |
| Meta `dtype` / trait `accumulator_dtype` | Parsed (`interface_parser.py:377`, `:268`) but used nowhere. These are ready-made hooks for this feature. |
| Arg marshalling (`core/kernel_base.py:76-137`) | `handle_in/out/resident` call `arg.astype(param_dtype)`. The declared type is enforced, so the dtype has to come from the interface. Caller arrays are silently converted in either direction. |
| Output allocation (`ocl/kernel.py:85`, `cu/kernel.py:92`, `numbax/kernel.py:96`) | `out_like` honours the declared dtype. |
| Interfaces (25 files) | Every one hard-codes `float`. |
| Kernel templates (22 per GPU backend) | Every one hard-codes `float` (OCL 761, Metal 939, CUDA 756 occurrences). None has a type parameter. |
| Numba kernels | `astype(np.float32)` inside the kernels. Numba 0.62 raises `NotImplementedError: float16` for CPU arithmetic. |
| Libraries (`lib/*.py`) | 220 explicit `float32` uses (resnet50 86, vgg16 37, graphics/fft/truetrace about 10 each). |
| Stub generator (`scripts/tools.py:1171`) | Already treats `float16` as a float. |
| Anisodiff (`lib/anisodiff.py:100`, sync path) | **Bug, not FP16-specific:** `anisodiff_n(img, n)` returns the result of **n−1** iterations. After the final swap it reads `output`, which is the stale buffer; it should read `input`. Bit-exact match to an n−1 run on Metal and OpenCL, and 1.2e-6 on CUDA. `n=1` returns the input unchanged. |
| Backend selection (`core/__init__.py:95`) | **Bug, not FP16-specific:** `get_manager(backend)` overwrites its argument with `get_backend()`, so `load_program(..., backend="cuda")` silently uses `YAPOCIS_BACKEND` (default `ocl`). The first round of benchmarks here was mislabelled because of it. The scripts now set the env var per backend. |

## Measurements

Prototype: `claude-debug/fp16_eval/` (interface plus Metal/OCL/CUDA templates parameterised by Mako `T`). Run with `YAPOCIS_PROJECT=claude-debug/fp16_eval`, and select devices with `FP16_DEVICES` (exact device names, comma separated). The CUDA `half` kernels use native `__half` arithmetic (`__hfma`) in `poly`; `blur` accumulates in FP32.

All timings include host conversion and transfer in both directions (non-resident buffers).

### Host round-trip calls, 4096x4096 (`bench_fp16.py`)

| Backend | Type | add | poly(64 iters) | 5x5 blur (fp32 accum) | add correct |
|---|---|---|---|---|---|
| metal (M4 Max) | float | 17.33 ms | 16.37 ms | 14.71 ms | yes |
| metal (M4 Max) | half | 9.49 ms | 9.46 ms | 9.15 ms | yes |
| ocl (M4 Max) | float | 18.05 ms | 17.33 ms | 16.80 ms | yes |
| ocl (M4 Max) | half | 9.22 ms | 8.58 ms | 12.77 ms | yes |
| cuda P40 | float | 82.91 ms | 70.99 ms | 72.36 ms | yes |
| cuda P40 | half | 40.25 ms | 82.38 ms | 36.64 ms | yes |
| cuda 3060 | float | 77.81 ms | 68.93 ms | 68.47 ms | yes |
| cuda 3060 | half | 39.72 ms | 34.38 ms | 34.29 ms | yes |
| cuda 3080 | float | 81.33 ms | 71.87 ms | 72.03 ms | yes |
| cuda 3080 | half | 41.44 ms | 36.03 ms | 36.54 ms | yes |
| ocl P40 / 3080 | float | 133.21 / 125.15 ms | 122.08 / 116.59 ms | 117.90 / 117.61 ms | yes |
| ocl (all NVIDIA) | half | build fails | build fails | build fails | n/a |

The OCL FP32 run on the 3060 failed with `MEM_OBJECT_ALLOCATION_FAILURE`. Another process was holding 9.4 of the card's 12 GB at the time, so this failure is unrelated to FP16. The yellowfin host round trips are about 4x slower than on the Mac, which is the cost of PCIe compared with unified memory. That makes the transfer saving worth more there.

### Compute-bound, 2048x2048 (`bench_fp16_compute.py`)

| Backend | Type | 1 iter | 1024 iters | 4096 iters |
|---|---|---|---|---|
| metal (M4 Max) | float | 3.43 ms | 6.58 ms | 17.44 ms |
| metal (M4 Max) | half | 1.59 ms | 5.99 ms | 19.62 ms |
| ocl (M4 Max) | float | 3.67 ms | 6.94 ms | 17.84 ms |
| ocl (M4 Max) | half | 2.05 ms | 5.48 ms | 16.70 ms |
| cuda P40 | float | 9.52 ms | 9.34 ms | 15.52 ms |
| cuda P40 | half | 4.88 ms | **190.09 ms** | **749.92 ms** |
| cuda 3060 | float | 7.37 ms | 8.64 ms | 13.01 ms |
| cuda 3060 | half | 3.96 ms | 6.43 ms | 13.69 ms |
| cuda 3080 | float | 6.11 ms | 6.82 ms | 8.41 ms |
| cuda 3080 | half | 3.36 ms | 4.39 ms | 7.36 ms |

What the numbers show:

- On the M4 Max, 3060 and 3080, the gain comes entirely from moving half as many bytes. FP16 math runs at about the FP32 rate, so heavy kernels gain nothing; the 3080 gains about 12% at 4096 iterations.
- On the P40, native `__half` math is a trap: about 48x slower than FP32 at 4096 iterations. FP16 storage is still a 2x win there (`blur` and `add`), provided the arithmetic is done in FP32. This is the strongest argument for the "load as half, compute as float" design below.
- Apple OpenCL exposes `cl_khr_fp16`; NVIDIA OpenCL does not, on any of the three cards.

### Precision hazards (numpy checks)

- `2048 + 1 == 2048` in FP16. Integers above 2048 cannot be represented exactly, so pixel coordinates on images wider than 2048 px are unusable.
- Adding 0.1 ten thousand times in an FP16 accumulator gives **256.0**, not 1000. FP16 accumulators are unusable for convolution, FFT, reductions or histograms.
- The maximum value is 65504 and epsilon is about 1e-3 (roughly 3 significant digits). That is enough for 8-bit-sourced image data in [0, 1]. It is not enough for HSI/Lab intermediate values, Mandelbrot escape values, or ray-trace distances.

## Result comparisons: ZC and anisodiff

Script: `claude-debug/fp16_eval/compare_zc_aniso.py`, with kernels in `kernels/*/fp16zcad.mako`. One template source serves both types, so FP32 and FP16 differ only in type. Two FP16 modes:

- **s**: store as FP16, compute in FP32 (the recommended design)
- **n**: store and compute in FP16 (a naive port)

The ports were validated against the library first. The ZC port gives IoU 1.0000 against `ZC.zcs` on every device. The anisodiff port matches a numpy reference to 3e-8, and matches the library's `anisodiff_n(img, n+1)` exactly (Metal and OCL) or to 1.2e-6 (CUDA), because of the off-by-one bug above.

Images: `test.jpg` (812x811), `church.jpg` (500x348), `stone_wall.jpg` (382x500), converted to [0, 1].

Devices: Metal and OpenCL on the M4 Max, and CUDA on the P40, 3060 and 3080. **Every metric agreed across all five to the reported precision.** FP16 behaviour here comes from the format itself, not from any device. NVIDIA OpenCL was not run, because it has no `cl_khr_fp16`.

Full per-image tables are in `claude-debug/fp16_eval/ZC_ANISO_TABLES.md`.

### ZC (DoG zero crossings, Gaussian scales 0–6, threshold 7.5e-6)

Each FP16 run is compared with the FP32 ZC map over the interior (the zcsdog margin is excluded). The metrics are:

- **IoU**: exact pixel agreement (intersection over union)
- **prec1**: the fraction of FP16 crossings with an FP32 crossing within 1 px
- **rec1**: the converse

| Variant | IoU | prec1 | rec1 |
|---|---|---|---|
| zcs kernel only, FP32 DoG quantised to FP16, mode s | 0.9999–1.0000 | 1.0000 | 1.0000 |
| zcs kernel only, mode n | 0.9986–1.0000 | 1.0000 | 0.9995–1.0000 |
| Whole pipeline in FP16 storage (input, h→v intermediate, Gaussian outputs, DoG), mode s | 0.915–0.987 | 0.981–1.000 | 0.9995–1.0000 |

Which stage causes the drift (Metal, scales 0/3/6; `zc_stage_attribution.py`):

| FP16-stored stages | IoU range | prec1 range |
|---|---|---|
| A: input image only | 0.997–1.000 | ≥ 0.9999 |
| B: + h→v intermediate | 0.988–0.999 | ≥ 0.9996 |
| C: + Gaussian outputs (smaller/larger) | 0.915–0.987 | ≥ 0.981 |
| D: as C, but DoG and zcs in FP32 | identical to C | identical to C |

Reading:

- **The zcs kernel itself is FP16-safe in either mode.** No DoG values were flushed to zero. Only 0.2–1.3% fall in the FP16 subnormal range, and that had no visible effect.
- **The loss comes from storing the two Gaussian outputs in FP16.** Their ulp (about 2.4e-4 near 0.5–1.0) is close to DoG magnitudes in smooth regions, so the subtraction picks up quantisation noise. Almost no crossings are lost (rec1 ≥ 0.9995). The extras are mostly crossings shifted by one pixel. At coarse scales up to about 2% of FP16 crossings are new ones with no FP32 crossing nearby.
- **Fix if FP16 is wanted here:** keep `smaller`/`larger` in FP32, or fuse the v-pass with the subtraction, so the DoG is formed in registers. FP16 is fine for the input and the h→v intermediate (B).

### Anisodiff (Perona–Malik, scale 3.0, step 0.2)

Input: the clean image plus N(0, 0.05) noise. Run in 1-plane (grey) and 3-plane (RGB) modes, mirroring `anisodiff_n` (scale by 256, resident iterations, divide by 256). The 1-pixel border is excluded. Ranges are over 3 images x 2 plane counts.

| Iterations | Mode | PSNR vs FP32 | max abs vs FP32 | px off by > 1/255 | PSNR vs clean, FP16 minus FP32 |
|---|---|---|---|---|---|
| 10 | s | 66.4–68.6 dB | ≤ 0.004 | 0.00% | 0.00 dB |
| 10 | n | 66.4–68.6 dB | ≤ 0.004 | 0.00% | 0.00 dB |
| 100 | s | 50.0–54.5 dB | ≤ 0.091 | 2.5–7.3% | 0.00 dB |
| 100 | n | 50.0–54.5 dB | ≤ 0.091 | 2.5–7.3% | 0.00 dB |
| 400 | s | 39.8–41.9 dB | ≤ 0.271 | 24.7–35.7% | +0.08 to +0.26 dB |
| 400 | n | 39.8–41.9 dB | ≤ 0.271 | 24.7–35.7% | +0.08 to +0.26 dB |

Reading:

- **Modes s and n are indistinguishable.** All of the error comes from rounding the diffusion state to FP16 between iterations; FP16 arithmetic adds nothing. So FP16 storage cannot be rescued by doing the maths in FP32 here.
- **The error grows with iteration count.** At 10 iterations FP16 is effectively exact. At 100 iterations 2.5–7% of pixels differ by more than one 8-bit grey level, with localised maxima up to 0.09 (about 23 levels) at edges, where Perona–Malik amplifies small differences. By 400 iterations a quarter to a third of pixels differ visibly.
- **Denoising quality does not get worse.** PSNR against the clean image is unchanged at 10–100 iterations and slightly *better* at 400. FP16 diffuses a little less: RMS change from the noisy input is 0.0619 against 0.0631 on `test.jpg`, because per-iteration updates below half an ulp are rounded away. The two results are different images, but neither is worse.
- **Recommendation:** keep the iterated state in FP32. Accept FP16 only at the boundaries (input and final output), which saves little because the work runs on resident buffers. Use FP16 state only when a caller explicitly wants speed over reproducibility at low iteration counts.

## Per-library fit

| Library | Fit | Reason |
|---|---|---|
| operators, imageops | Good | Elementwise and bandwidth-bound |
| convolve / convolves / convolvesep / gaussian | Good | Half storage only; the accumulator must stay FP32 |
| median3x3, gradient | Good to moderate | Bandwidth-bound |
| zcs (zero crossings) | Good | Measured: kernel is FP16-safe (IoU ≥ 0.9986). In `zcsdog`, keep the Gaussian outputs FP32 |
| anisodiff | Poor for state | Measured: diverges from FP32 with iteration count (≥ 2.5% of px off by > 1/255 at 100 iterations). FP16 I/O only |
| colormath, hsi | Moderate | Fine for I/O; do the conversion math in FP32 (Lab range and precision) |
| cnn, vgg16, resnet50 | Good | Standard inference practice: FP16 weights and activations, FP32 accumulate |
| fft | Poor | Twiddle and accumulation error grows with size |
| graphics, graphics4, mplbackend, truetrace | Not suitable | Coordinates go above 2048; blending is cumulative |
| mandelbrot, simplex | Not suitable | Precision is the result itself |

## Backend specifics

| Backend | Requirement |
|---|---|
| Metal | Native `half`. Works today (verified). |
| OpenCL | `#pragma OPENCL EXTENSION cl_khr_fp16 : enable`. Works on Apple. **Not available on NVIDIA OpenCL** (no `cl_khr_fp16`). Probe the extension and refuse FP16 with a clear error. Alternatively, store as `ushort` and convert with `vload_half` / `vstore_half`, which are core OpenCL and need no extension. That route is unverified here but would give NVIDIA OCL the storage win. |
| CUDA | `#include <cuda_fp16.h>`, type `__half`, conversions `__half2float` / `__float2half`. Verified on the P40, 3060 and 3080. Storage gives about 2x on all three. Native `__half` arithmetic is fine on Ampere but about 48x slower on the P40 (sm_61), so kernels must convert to float for math. |
| Numba | No CPU FP16 arithmetic. The backend must upcast to FP32 inside the kernel and downcast the outputs. Correct, but no speed gain. |

## Recommended design

1. **Mako type variables, defaulting to today's behaviour.** Add `storage_t` (default `float`) and `accum_t` (always `float`) to the template context, wired from the existing meta `dtype` / `accumulator_dtype`. The interfaces use `${storage_t}` in place of `float` for image buffers only. Scalars and sizes stay as they are. With the default, the rendered source is byte-identical to today's, so existing caches and validation references are not disturbed.
2. **Per-backend helpers** in `helpers.mako` / `constants.mako`: a `STORAGE_PREAMBLE` (pragma or include) plus `LOAD(p,i)` / `STORE(p,i,v)` macros that convert to and from `accum_t`. Kernels do all their math in `float`, which keeps the template diff mechanical.
3. **Library opt-in**, e.g. `Gaussian(dtype=np.float16)`. `BoundKernels` passes it through as a Mako var, the same way the prototype passes `T`. Libraries that are not suitable reject it.
4. **Fix the `f16` mapping** in `ctype_to_nptype` and add `float16 → half` to the reverse map. Decide whether a dtype mismatch on input should warn instead of converting silently.
5. **Tests**: parametrise the suitable libraries over `[float32, float16]` and compare against the FP32 result with tolerances based on FP16 epsilon. The validation references stay FP32.

## Effort and order

| Phase | Scope | Rough size |
|---|---|---|
| 0 | `f16` fix, context vars, helper macros on 3 GPU backends, numba upcast shim, `BoundKernels` passthrough | Small (core under 100 lines) |
| 1 | operators, imageops, convolve family, gaussian, median3x3 | Moderate: about 6 templates x 3 backends plus tests |
| 2 | cnn / vgg16 / resnet50 inference path | Moderate; biggest real-world payoff (weights halve too) |
| 3 | colormath, hsi, gradient, zcs (+ FP32 Gaussian outputs in zcsdog) | Moderate; needs precision review |
| — | fft, graphics*, truetrace, mandelbrot, simplex; anisodiff state | Do not convert |

Before Phase 1: fix the `anisodiff_n` off-by-one and the `get_manager` backend bug (otherwise per-instance backend choice doesn't work, with or without FP16). Also measure with resident buffers, because pipelines that keep data on the device will see a smaller gain than the host round-trip numbers above. Optionally, try `vload_half` / `vstore_half` on NVIDIA OpenCL.

## Artifacts

- `claude-debug/fp16_eval/interfaces/fp16probe.interface`
- `claude-debug/fp16_eval/kernels/{metal,ocl,cuda}/fp16probe.mako`
- `claude-debug/fp16_eval/bench_fp16.py`, `bench_fp16_compute.py`
- `claude-debug/fp16_eval/interfaces/fp16zcad.interface`, `kernels/{metal,ocl,cuda}/fp16zcad.mako` (+ `fp16zcad_body.mako`)
- `claude-debug/fp16_eval/compare_zc_aniso.py`, `summarize_zc_aniso.py`, `zc_stage_attribution.py`
- `claude-debug/fp16_eval/aniso_port_diff.py`, `aniso_numpy_ref.py`, `aniso_offbyone.py` (off-by-one diagnosis)
- `claude-debug/fp16_eval/zc_aniso_{metal,ocl,cuda_*}.json` (raw results) and `ZC_ANISO_TABLES.md`

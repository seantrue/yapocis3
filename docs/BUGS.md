# Bugs

Bugs reported in passing (`bug: …`), kept until fixed. Numbers are never reused. Entries with an issue are also tracked on GitHub.

The list was started on 2026-11-11 by the plum coordinator, from issues already open on GitHub. Each entry's "Report" is the issue's title.

## Open

### B16. Three console scripts fail on a plain `pip install yapocis`

- Reported: 2026-11-12
- Report: "Three console scripts fail on a plain pip install: yapocis-report, yapocis-imagewatcher, yapocis-performance"
- Context: found smoke-testing the 1.2.0 wheel in a fresh venv. `yapocis-report` needs duckdb (imported at module level in `utils/performance.py`), `yapocis-imagewatcher` needs pygame and watchdog, `yapocis-performance` needs matplotlib and seaborn. They are only in the `dev` extras. 1.1.0 was the same. The maintainer: "Ship 1.2.0 with it listed as a known issue, and file a bug to fix in 1.2.1".
- Issue: #107

### B17. `yapocis-tools devices` crashes without an OpenCL platform

- Reported: 2026-11-12
- Report: "yapocis-tools devices crashes on a machine with no OpenCL platform"
- Context: `pyopencl._cl.LogicError: clGetPlatformIDs failed: PLATFORM_NOT_FOUND_KHR`, found by the first rustcpu CI run (#102). The maintainer: "File it as a bug".
- Issue: #108

### B15. `yapocis-report compare-branches` crashes on tests with zero mean time

- Reported: 2026-11-12
- Report: "bug:" (the maintainer, after the coordinator found `compare-branches --component graphics` crashing while charting graphics performance across backends and branches)
- Context: `ZeroDivisionError` at `src/yapocis/utils/performance.py:1926` (`speedup = baseline_mean / stats["mean"]`) in `generate_branch_comparison_report`. The `position_validation_graphics4` tests record `mean_time = 0`. The maintainer's working-tree guard covers the same division at line 1869, not 1926; `baseline_mean` can also be 0.
- Issue: #103

### B11. GPU results nondeterministic under multi-process contention on Apple M4

- Reported: 2026-11-11
- Report: (the 1.1.0 release gate) `test_resnet50_resident_async_vs_sync[ocl-Apple M4 Max]` failed once in 8 full-suite runs, with 57 of 1000 outputs off by up to 0.0217.
- Context: reproduced with 4 processes at once. Both OpenCL and Metal, and both sync and async, diverge; one process on a quiet GPU gives 0 of 12. Not a kernel-internal race (no local memory, barriers or atomics) and not a stale `id(array)` buffer (a strict-id patch changed nothing). The probe is plum `claude-debug/rn_race/rn_probe.py`.
- Open question: is it yapocis, or Apple's GPU stack when several processes share the GPU? Separating them needs an unrelated GPU load, and a run on yellowfin (CUDA, NVIDIA OpenCL).
- Issue: #88

### B13. OpenCL `write_buffer` and `read_buffer` fail on Fortran-ordered or transposed arrays

- Reported: 2026-11-11
- Report: "OpenCL write_buffer/read_buffer flatten in place and fail on Fortran-ordered or transposed arrays"
- Context: `ocl/device_manager.py`: `make_buffer` refuses non-contiguous arrays (`:89-90`), and `write_buffer` and `read_buffer` flatten in place, with `a.shape = (a.size,)` (`:127-139`). With `base.T`, OpenCL raises `AttributeError: Incompatible shape for in-place modification`; with `base[:, ::2]`, `RuntimeError: … is not contiguous`. CUDA has the same contiguity check (`cu/device_manager.py:119-120`). Found while planning #89, which fixes the Metal path only, so the two GPU backends now disagree on F-ordered inputs.
- Issue: #92

### B14. `Convolve1D` edges: numbax reflects where the GPUs write 0, and the GPU main kernel leaves `ret[left]` at 0

- Reported: 2026-11-11
- Report: "Convolve1D edges: numbax reflects where GPUs write 0, and the GPU main kernel leaves ret[left] at 0 (i > left)"
- Context: numbax pads by reflection (`kernels/numbax/convolve.py:19-37`), and the GPU kernels write 0, so they differ in the first `left + 1` and the last `left` samples (`left = len(conv)//2`; indices `[0, 1, 63]` for 3 taps and `[0, 1, 2, 62, 63]` for 5 taps on 64 samples). The GPU kernels' strict `i > left` (`kernels/ocl/convolve.mako:12`, and the Metal and CUDA equivalents) writes 0 to `ret[left]`, although the window fits there. Found while planning #80; neither is a `_fast` bug. Changing either changes recorded outputs (yapocis-rs parity bundles, the backend-comparison tests), so it needs a plan of its own.
- Issue: #93

## Fixed

### B4. GPU `_fast` convolve kernels return partial output

- Reported: 2026-11-11
- Report: "GPU _fast convolve kernels return partial output (Metal wrong every call, OpenCL varies)"
- Context: `kernels/metal/convolve.mako:23-45`. Found while reviewing #64.
- Cause: confirmed in planning. In all three GPU templates each work item computed only `chunk - len(conv)` outputs, so `len(conv)` outputs per chunk (all of them when `chunk <= len(conv)`) were never written, and writes were clamped to `right - 1`. OpenCL and CUDA multiplied `a[j]`, the loop index, instead of `a[i + k - left]`; Metal read a racy `threadgroup` buffer. OpenCL's "variation" was leftover memory in the unwritten outputs. rustcpu's CPU port (yapocis-rs) copies the old behaviour.
- Fix: work item `g` computes `ret[g*chunk .. g*chunk+chunk-1]`, each exactly as the main kernel does, and stops at the width (approved 2026-11-11: "Fix it"). Plan: `~/workspace/plum/docs/plans/2026-11-11-yapocis3-89-87-80-81.md`.
- Status: 2026-11-11, PR #98 (`bf9a0b8`): fixed on OpenCL, Metal and CUDA, with a temporary behaviour-gated xfail for rustcpu.
- Fixed: 2026-11-11. rustcpu: yapocis-rs PR #23 (`556ef5c`) re-exported the bundle from `bf9a0b8` and fixed the CPU port; with a `yapocis_rs` wheel built from `556ef5c`, branch `fix/80-rustcpu-xfail` removes the xfail, and rustcpu's cases pass unmarked.
- Issue: #80

### B10. 56 open Dependabot alerts in `uv.lock`

- Reported: 2026-11-11
- Report: (GitHub, on push) "56 vulnerabilities on seantrue/yapocis3's default branch (5 critical, 26 high, 23 moderate, 2 low)"
- Context: all of them are in `uv.lock`; 41 are Pillow (all 5 critical). The table by package is on the issue.
- Cause: confirmed in planning. Nothing held Pillow below 13 on purpose. `requires-python = ">=3.10"` did: Pillow 13.0.0 (and filelock 4.2, starlette 1.10 and fonttools 4.68) need Python 3.11, so uv kept a Python 3.10 fork at the vulnerable versions.
- Fix: `requires-python = ">=3.11"` (decision D7.2), the published floor raised to `Pillow>=13.0.0`, the first release outside all 41 Pillow advisories (D7.3), and a lock upgrade of the nine packages with alerts: Pillow, starlette, filelock, wheel, pytest, idna, requests, fonttools and Pygments, plus pytest-benchmark 5.1.0 → 5.3.1, because pytest 9 removed `Node._nodeid`, which 5.1.0 read (D7.11). Plan: `~/workspace/plum/docs/plans/2026-11-11-yapocis3-89-87-80-81.md`.
- Fixed: 2026-11-11, branch `fix/87-lock-alerts`: the new lock matches 0 of the 56 alerts (against 56 before). Python 3.10 users stay on yapocis 1.1.0.
- Issue: #87

### B12. Metal device manager registers a non-contiguous input's copy under the copy's id

- Reported: 2026-11-11
- Report: (found while checking pymetallic for #88) "`make_buffer` and `write_buffer` … register the buffer under the copy's id", so later lookups for the original array miss and the buffer leaks until a purge.
- Context: `src/yapocis/metallic/device_manager.py`, `make_buffer` and `write_buffer`. This is not #88's cause, because ResNet's arrays are contiguous.
- Cause: confirmed in planning. Besides the leak, a strided `out` argument, a strided write followed by a read, and an F-ordered input with an `out_like` result (a `.T` view) all raised "not in yapocis management".
- Fix: option A, register the buffer under the caller's array (approved 2026-11-11). `pymetallic.Buffer.from_numpy` makes its own contiguous copy and Metal copies the bytes, in 0.2.3 and 0.3.1, so no copy is kept. Plan: `~/workspace/plum/docs/plans/2026-11-11-yapocis3-89-87-80-81.md`.
- Fixed: 2026-11-11, branch `fix/89-metal-noncontiguous`, with `tests/backend/test_metal_noncontiguous.py` (strided `in`, `out` and `inout` arguments, a strided write and read, F-ordered and transposed inputs). OpenCL's F-order failure is B13 (#92), not fixed here.
- Issue: #89

### B5. The `trace_segments` docstring has the iso direction backwards

- Reported: 2026-11-11
- Report: "trace_segments docstring has the iso direction backwards (positive moves the contour inward)"
- Context: `lib/truetrace.py:53-54`. The field is positive inside, so a positive iso shrinks the shape: `trace_mask` ink on a 1552 px ring is 1924, 1600, 1556, 1512 and 1174 at iso −1.5, −0.5, 0, 0.5 and 1.5, on the CPU engine and on OpenCL.
- Fixed: 2026-11-11, branch `fix/81-iso-docs`: the `trace_segments` and `trace_regions` docstrings and `docs/library/truetrace.md` (the parameter line and the ISO table) say that a positive iso moves the contour inward. The code was right and is unchanged.
- Issue: #81

### B2. colormath `xyz2xyY` of black: NaN on OpenCL, 0 on numba

- Reported: 2026-10-29
- Report: "colormath xyz2xyY of black: NaN on OpenCL, 0 on numba"
- Fix: Option Z, black gives 0 on every backend, with an explicit `sum != 0` check (approved 2026-11-11). The batch also fixes numbax `xyY2xyz` at y == 0 (1.5e19), matching the GPUs, which zero Y. Plan: `~/workspace/plum/docs/plans/2026-11-11-yapocis3-66-xyz2xyY-black.md`.
- Status: 2026-11-11, the OpenCL, Metal, CUDA and numbax kernels are fixed in the numba batch. rustcpu still gave NaN until yapocis-rs was fixed and the `yapocis_rs` wheel rebuilt; its three test cases carried a temporary xfail until then.
- Fixed: 2026-11-11. rustcpu: yapocis-rs PR #22 (`9ebcac9`) fixed the CPU port, and the step-2b PR (branch `fix/66-rustcpu-xfail`) removes the temporary xfail; with a `yapocis_rs` wheel built from `9ebcac9`, the rustcpu cases pass unmarked.
- Issue: #66

### B7. AI.md: wrong statements about the 3-D `hash % 12` and the old numba loop

- Reported: 2026-11-11
- Report: "AI.md: the 3-D hash % 12 is never negative, and the old numba loop's wide/tall effects are swapped"
- Fix: a correcting entry in the numba batch's `AI.md` update.
- Fixed: 2026-11-11, in the numba batch: AI.md's batch entry corrects both statements; the old entries are left as written.
- Issue: #84

### B6. Tests leak `YAPOCIS_BACKEND`

- Reported: 2026-11-11
- Report: "Tests leak YAPOCIS_BACKEND, so later unparametrized tests run on the last backend collected"
- Context: `tests/async/test_performance.py:57-58`, `tests/conftest.py:310-335`, `tests/__init__.py:753-778`.
- Cause: reproduced 2026-11-11. With `YAPOCIS_TEST_BACKENDS=ocl,metal,numbax`, running `test_performance.py` first makes `test_gpu_field_is_not_transposed_on_a_non_square_mask` fail.
- Fix: an autouse fixture that restores `YAPOCIS_BACKEND` and `YAPOCIS_DEVICE`, folded into the numba batch (the maintainer's decision, 2026-11-11).
- Fixed: 2026-11-11, in the numba batch: an autouse fixture in `tests/conftest.py` restores `YAPOCIS_BACKEND`, `YAPOCIS_DEVICE`, `YAPOCIS_ASYNC` and `YAPOCIS_ASYNC_WRITE` after every test, and `backend_comparison` and `tests/__init__.py` remove the variable when it was unset.
- Issue: #82

### B3. numba truetrace: `init_seed_from_mask` and `jfa_step` are stubs, so iso has no effect

- Reported: 2026-10-29
- Report: "numba truetrace: init_seed_from_mask and jfa_step are stubs, so iso has no effect"
- Cause: confirmed. The numbax field is ±1e30 everywhere. The issue's suggested fix would give −2 × SDF (see the comment on the issue).
- Fix: option B, an `@njit` port of OpenCL's jump flood (approved 2026-11-11). Plan: `~/workspace/plum/docs/plans/2026-11-11-yapocis3-67-truetrace-numba-sdf.md`.
- Fixed: 2026-11-11, in the numba batch: an `@njit` port of the OpenCL jump flood, bit-identical to rustcpu, and within sqrt rounding of the GPU backends, on the test masks.
- Issue: #67

### B8. The rustcpu kernel cache key keeps weights to only 8 decimal places, and elides long arrays

- Reported: 2026-11-11
- Report: (from the #64 review) "Weight ladders that differ only in the 9th significant digit share one cached program."
- Fix: in the numba batch (the maintainer's decision, 2026-11-11).
- Fixed: 2026-11-11, in the numba batch: the key holds every weight exactly, with its dtype and shape.
- Issue: none

### B1. numba: a second `Convolve1D(name="custom")` reuses the first one's weights

- Reported: 2026-10-29
- Report: "numba backend: a second Convolve1D(name="custom") reuses the first one's weights"
- Context: `numbax/kernel_manager.py` stores generated functions on the shared module by name. convolvesep has the same flaw.
- Cause: confirmed in planning. The `hasattr`/`setattr(mod, …)` store dates from `77fc317`. `3c0d53c` hid it and `2d8c6d5` re-exposed it.
- Fix: approach A, per-manager closures (approved 2026-11-11). Plan: `~/workspace/plum/docs/plans/2026-11-11-yapocis3-64-custom-convolve-cache.md`.
- Fixed: 2026-11-11, in the numba batch (branch `fix/numba-batch-64-66-67`): each numbax kernel manager builds its own convolve and convolvesep functions and never stores them on the shared module.
- Issue: #64

### B9. numba `simplex2d` is wrong for non-square sizes

- Reported: 2026-10-29
- Report: "numba simplex2d is wrong for non-square sizes (width/height swapped, out-of-bounds writes)"
- Fixed: 2026-11-07, PR #75 (`29f8403`) and PR #76 (`7e98bb0`). The issue was closed on 2026-11-11. Regression tests and the `tileable` docs fix (2A) come in the numba batch.
- Issue: #65

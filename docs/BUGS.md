# Bugs

Bugs reported in passing (`bug: …`), kept until fixed. Numbers are never reused. Entries with an issue are also tracked on GitHub.

The list was started on 2026-11-11 by the plum coordinator, from issues already open on GitHub. Each entry's "Report" is the issue's title.

## Open

### B2. colormath `xyz2xyY` of black: NaN on OpenCL, 0 on numba

- Reported: 2026-10-29
- Report: "colormath xyz2xyY of black: NaN on OpenCL, 0 on numba"
- Fix: Option Z, black gives 0 on every backend, with an explicit `sum != 0` check (approved 2026-11-11). The batch also fixes numbax `xyY2xyz` at y == 0 (1.5e19), matching the GPUs, which zero Y. Plan: `~/workspace/plum/docs/plans/2026-11-11-yapocis3-66-xyz2xyY-black.md`.
- Status: 2026-11-11, the OpenCL, Metal, CUDA and numbax kernels are fixed in the numba batch. rustcpu still gives NaN until yapocis-rs is fixed and the `yapocis_rs` wheel is rebuilt; its three test cases xfail until then. This entry closes then.
- Issue: #66

### B4. GPU `_fast` convolve kernels return partial output

- Reported: 2026-11-11
- Report: "GPU _fast convolve kernels return partial output (Metal wrong every call, OpenCL varies)"
- Context: `kernels/metal/convolve.mako:23-45`. Found while reviewing #64.
- Issue: #80

### B5. The `trace_segments` docstring has the iso direction backwards

- Reported: 2026-11-11
- Report: "trace_segments docstring has the iso direction backwards (positive moves the contour inward)"
- Context: `lib/truetrace.py:53-54`.
- Issue: #81

### B7. AI.md: wrong statements about the 3-D `hash % 12` and the old numba loop

- Reported: 2026-11-11
- Report: "AI.md: the 3-D hash % 12 is never negative, and the old numba loop's wide/tall effects are swapped"
- Fix: a correcting entry in the numba batch's `AI.md` update.
- Issue: #84

## Fixed

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
- Fixed: 2026-11-11, in the numba batch: an `@njit` port of the OpenCL jump flood, bit-identical to OpenCL and rustcpu on the test masks.
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

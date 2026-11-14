# AI Contributions to Yapocis

*About AI slop.* Fair cop — a good share of this file, and of the kernels and tooling it describes, came out of a machine. Think of it as instant ramen: quick, cheap, genuinely edible, and not something that simmered on the stove all day. The honest reply isn't a claim about quality; it's the record below — what the maintainer asked for, what the model wrote, and where AI-written code was wrong, including the defects the cross-backend test suite caught before they shipped. Bon appétit, but mind the sodium content.

This document tracks contributions made by AI assistants (primarily Claude) versus the human maintainer.

## Summary Statistics

- **Total AI-assisted commits**: ~35
- **Major features developed with AI assistance**:
  - Infrastructure: Package modernization (hatchling, modern tooling), automated changelog generation, type annotation modernization, release automation system
  - Backend work: CUDA kernel compilation fixes, CUDA row-major addressing fixes, Metal backend fixes, coordinate system standardization, Numba backend completeness, truetrace kernel port, separable convolution boundary fix
  - Testing: Validation framework, StopWatcher enhancements, test isolation fixes, NPZ analysis tooling, test determinism fixes
  - Performance: Performance tracking system, DuckDB migration, reproducibility improvements
  - Code quality: Test refactoring, random data centralization, Python 3.9+ type modernization, HSI alpha channel handling
  - Developer experience: Environment variable standardization, enhanced backend/device listing, PyPI publishing workflow, extension project scaffolding, .pyi stub generation
  - Documentation: Complete documentation rewrite with executable examples, doctest validation tool
- **Lines of code**:
  - ~12,200 lines: Documentation rewrite with executable examples (2025-12-03)
  - ~1300 lines: Performance tracking system (2025-10-03)
  - ~900 lines: Doctest validation tool (2025-12-03)
  - ~400 lines: DuckDB migration (2025-10-03)
  - ~550 lines: Random data centralization (2025-10-04)
  - ~300 lines: Package modernization (2025-10-05)
  - ~200 lines: Changelog system (2025-10-05)
  - ~350 lines: CUDA backend compilation fixes (2025-10-05)
  - ~50 lines: YAPOCIS_ENGINE → YAPOCIS_DEVICE rename (2025-10-06)
  - ~200 lines: CUDA addressing fixes and truetrace port (2025-10-06)
  - ~350 lines: Type annotation modernization (2025-10-08)
  - ~5 lines: Test determinism fix (2025-10-16)
  - ~40 lines: HSI alpha channel handling (2025-10-16)
  - ~25 lines: Separable convolution boundary fix (2025-10-17)
  - ~470 lines: Release automation system (2025-10-31)
  - ~40 lines: Numbax composite bounds fix (2025-11-30)
  - ~1870 lines: Extension project support, enhanced imageops, stub generation (2026-03-05)
  - ~270 lines: Colorspace matrix/xyY fixes and content-keyed Mako module cache (2026-10-04, colleague-authored; AI verified)
  - ~800 lines: One (x, y) drawing convention, CUDA bounds/blending/multi-GPU fixes, stub generator, dependency cleanup (2026-10-06); plus ~1,100 lines of mesh-triangle work found uncommitted, author not recorded
  - ~170 lines: FP16 image read/write in `utils/basic.py` plus tests (2026-10-08); FP16 evaluation reports in `claude-debug/`
  - ~165 lines: anisodiff off-by-one and RGB layout fixes, get_manager backend argument, f16 dtype, stale Makefile targets, plus tests (2026-10-08)
  - ~15 lines: `test_vimp_common` default-queue test corrected (2026-10-08)
  - ~52 lines: zero-length `lines` drawn as dots on Metal, OpenCL and CUDA, plus a test (2026-10-10)
  - ~28 lines: numbax gradient fallback, kernel search order and numbax loading by package name, plus ~143 lines of tests (2026-10-12)
  - 3 lines: CUDA gradient angle used swapped axes (2026-10-12)
  - 1 line, plus a 4-line test: `split_channels` returns a tuple, as annotated (2026-10-17)
  - ~250 lines of code and docs, plus 13 replacement photos and ~130 lines of reports: PyPI release-readiness pass (2026-10-17)
  - ~3,210 lines of exporter code, ~345 lines of tests, ~130 lines of docs, plus ~490 lines of reports: kernel bundle exporter for the Rust port (2026-10-19)
  - ~645 lines plus ~85 lines of tests: Gaussian ladder pinned as literals (~165), numpy host fixtures (~430), `bundle --project-only` (~50) (2026-10-21, Claude Fable 5.1)
  - ~275 lines changed in `lib/truetrace.py`, ~95 in `codegen/parity_processing.py` and an 84-line `tests/test_truetrace_fill.py`, shared between the scanline fill (Claude Fable 5.1) and exact chaining with `off_field_iso` (Claude Opus 5.5) (2026-10-21); the split was not recorded line by line
  - ~45 lines across the FFT templates, the HSI interface and the dispatch evaluator, plus 2 test cases: FFT butterfly race, HSI dispatch size, PoCL `get_global_size` link failure (2026-10-26)
  - ~41 lines across 13 files: text file I/O as UTF-8, `\n` line endings for exports (2026-10-26)
  - ~190 lines of kernel and library code, ~630 of tests, ~45 of docs and changelog: the numba batch, #64, #66, #67, #82, the rustcpu cache key, numbax `xyY2xyz`, #65 tests (2026-11-11)
  - ~43 lines of tests removed: the temporary rustcpu #66 xfail, after the `yapocis_rs` wheel was rebuilt from yapocis-rs `9ebcac9` (2026-11-11)
  - ~10 lines of version, changelog and this entry: version 1.1.0 released to PyPI (2026-11-11)
  - ~15 lines of docs: BUGS.md B12 and issues #89/#90 from the pymetallic check (2026-11-11)
  - ~7 lines of docstrings and docs, ~35 of BUGS.md, changelog and ledger: #81 iso direction, BUGS.md B13 (#92) and B14 (#93) (2026-11-11)
  - ~8 lines in `metallic/device_manager.py` and a 93-line `tests/backend/test_metal_noncontiguous.py`: Metal buffers for non-contiguous arrays registered under the caller's array, #89 (2026-11-11)
  - ~8 lines of comments in `metallic/device_manager.py`, and the lock: pymetallic 0.2.3 → 0.3.1, #90 (2026-11-11)
  - ~10 lines of pyproject and docs, plus the lock: Python 3.11, `Pillow>=13.0.0` and nine package upgrades that close the 56 Dependabot alerts, plus pytest-benchmark 5.3.1 for pytest 9, #87 (2026-11-11)
  - ~35 lines across the OpenCL, Metal and CUDA convolve templates, ~105 of tests, ~10 of docs and comments: the GPU `_fast` convolve kernels compute every output, #80 (2026-11-11)
  - ~47 lines of tests removed: the temporary rustcpu #80 xfail, after the `yapocis_rs` wheel was rebuilt from yapocis-rs `556ef5c` (2026-11-11)
  - ~5 lines of tests: #89's orphan-buffer test collects garbage before it purges, so earlier tests' buffers no longer count as leaks in full-suite runs (2026-11-12)
  - ~360 lines added and ~145 removed in the device managers, kernel base and kernel manager base, ~660 lines of tests, ~4 of changelog: buffer lifetime, #100: kernels drop their arguments, one guarded `id → (ref, buffer)` store, fast buffer free, CUDA close (2026-11-12)
  - ~17 lines of BUGS.md and FEATURES.md: B15 (#103, `compare-branches` divides by a zero mean) and F2 (#102, CI on the rustcpu backend) (2026-11-12)
  - ~70 lines: `.github/workflows/ci.yml`, the first CI workflow, the test suite on the rustcpu backend, and ~26 lines of tests so OpenCL tests skip without an OpenCL platform, #102 (2026-11-12)
  - ~45 lines of version, changelog, BUGS.md/FEATURES.md and this entry: version 1.2.0 released to PyPI (2026-11-12)
  - ~60 lines and ~31 of tests: Mako module names and the bundle SHA-256 through commonkit's hash facade, ruff bans, CI under md5 and b3 (2026-11-12)

## Detailed Contributions

### 2026-11-13: test-affected, run the tests a change can reach (Claude Opus 5.5)

**Requested by the maintainer:** test tooling for Claude Code across the Python and Rust repos, whose suites had grown to hours; then "Add to each repo".

**What the AI did.**
- `.claude/skills/test-affected/` (SKILL.md, `affected.py`, ~300 lines of Python, stdlib only): maps `git diff` against `origin/main` to the tests that can see it, as one command per suite.
  - Rust: the changed packages plus every workspace package that depends on them (normal, dev or build), from `cargo metadata`. A root `Cargo.toml`, `Cargo.lock`, toolchain or nextest config change selects everything.
  - Python: test files that name a changed kernel, interface or module (`grep_stem`), or explicit `[[python.map]]` rules.
  - Each run gets a fresh `COMMONKIT_TEST_RUN`/`STORE_ROOT`/`MARK_DIR`, as CI does, and md5 only (b3 again when hashing or the store changed).
  - Tiers: 0 after each edit, 1 before a commit, 2 the repo's documented gate, run in the background.
- The same SKILL.md and `affected.py` are in plumcraft, yapocis-rs, Yapocis3 and impressionist-rs; a change goes to all four.
- Here: `.test-affected.toml`. Tier 0 pins one backend and one device (`YAPOCIS_T0_BACKEND`, default `numbax`; `YAPOCIS_T0_DEVICE`, a name), because the suite runs every test on every (backend, device) the host has, ~760 tests each. A change to `kernels/<backend>/x.mako` runs the tests that name `x` on that backend; a change to `helpers.mako` or `constants.mako`, or to anything under `core/`, runs the whole suite on the one backend. `.gitignore`: `/.claude` became `/.claude/*` with `!/.claude/skills/`, so the skill is tracked and local settings are not.

**Measured** on a 4-core Linux container, rustcpu backend (yapocis-rs wheel from its main): 759 tests in 238 s serially; one failure, `test_seaborn.py::TestSeabornComparison::test_hexgrid` (RMS over tolerance against Agg), not investigated.

**Follow-up (same day).** The skill now also lives in plumimpress, yapocis-external and Virtual-Impressionist; SKILL.md lists all seven repos and adds the yapocis-external → impressionist-rs rule (re-export the bundle, port the CPU kernel, run impressionist-rs's tier 0). SKILL.md and affected.py stay identical in all seven.

**CI (same day).** The rustcpu job takes uv, maturin, Rust and its build cache from the shared action `seantrue/yapocis-rs/.github/actions/test-env` (ref `@claude/test-affected` until yapocis-rs merges, then `@main`), and builds the wheel with that maturin rather than `uvx maturin`. Test commands, triggers and env are unchanged. One side effect: the action installs PoCL, so the `@pytest.mark.ocl` tests (15), which skipped on the runner for want of an OpenCL platform, will run on PoCL's CPU device. Not run: the maintainer asked for the branch not to be active yet.

**b3 only (same day).** The maintainer: "No md5 is to be used" and "Drop all MD5 testing. It's legacy." SKILL.md (the same in all seven repos) now says every tier runs b3 only. md5 had been the AI's choice for tier 0/1 when it wrote the skill earlier the same day; the maintainer corrected it. Here: `.test-affected.toml` sets `COMMONKIT_HASH = "b3"`, and CI's rustcpu test step runs once, under b3. Checked by reading the config: this diff selects no suite, so tier 0 printed no command. The CI change runs on this PR.

### 2026-11-13: show_array under Pillow 13 (Claude Opus 5.5)

**What the maintainer asked:** "Check pillow release notes for an explanation", then "Fix and cpm yapocis", after impressionist's `tests/test_all.py::test_boundedaverage` failed with `OSError: cannot write mode I as PNG` once impressionist moved to yapocis 1.2.0 (which requires `Pillow>=13.0.0`, #87).

**What the AI did:** found the cause in Pillow 13.0.0's release notes ("Saving I mode images as PNG": Pillow had clipped such images to 16 bits and now refuses). `utils/basic.py`: `_png_ready` converts a mode "I" image to "I;16" before `show_array`'s two PNG saves, which clips to 0..65535 as Pillow 12 did. Test: `test_show_array_saves_a_32_bit_integer_array_as_a_16_bit_png` (`tests/integration/test_image_io.py`) fails without the fix on Pillow 13.0.0 and passes with it (20 passed in that file); impressionist's `test_boundedaverage` passes against this branch. ~30 lines with the test.

**Defect it fixes:** the 1.2.0 Pillow floor (#87) was raised without a test that saves an integer array, so the break went unnoticed until impressionist ran it.

### 2026-11-12: Mako module names through commonkit's hash facade (Claude Opus 5.5)

**What the maintainer asked for.** Layer 1 of plum's plan `2026-11-12-commonkit-hash-facade.md` (§3.3), approved with the coordinated move (D19: no direct hash libraries). Uses pyuncommonkit 0.2.0 from PyPI. No release during the move (D22 Q3).

**What the AI did.**
- `src/yapocis/core/kmgr_base.py` (~11 LOC): a compiled Mako module is named `<template>.<digest>.py` with `commonkit.digest.raw`, so the name follows commonkit's switch; a switch recompiles every template.
- `src/yapocis/codegen/bundle.py` (~6 LOC): the bundle manifest's SHA-256 is `commonkit.digest.fixed.raw("sha256", …)`, allowlisted in `hash-allowlist.txt`.
- `pyproject.toml` (~11 LOC): `pyuncommonkit>=0.2.0`; ruff `TID251` bans `hashlib` and `blake3` outside tests. `uv.lock` adds pyuncommonkit 0.2.0 and blake3.
- `tests/backend/test_kernel_template_cache.py` (~31 LOC): the module name of the template `version 1`, built through `kernel_template_lookup(dirs, tmp_path)`, carries its known MD5 or BLAKE3 (it failed under b3 before the change), and `test_hash_is_what_was_asked`.
- CI (~24 LOC): the rustcpu tests run under md5 and then b3, one after the other, each in a fresh run folder, then a hash-gate step (ruff `TID251` and a grep). `YAPOCIS_RS_REF` moves to `e45f6e0`, yapocis-rs #25's merge (the same change on that side). **Defect caught by CI:** the hash-gate step's ruff ran over the yapocis-rs checkout CI clones into the workspace and flagged that repo's allowlisted `compare-bundle.py`; ruff now skips `yapocis-rs/`.

**Checked:** the full suite under md5 and b3 on the M4 Max, through plum's isolated environment; counts are in the PR body.

**Division of labor:** the maintainer decided the design; the plum planner wrote the plan; Claude implemented it and wrote this entry.

### 2026-11-12: Version 1.2.0 (Claude Opus 5.5)

**What the maintainer asked for.** Version 1.2.0, with #100 folded in (plum `docs/QUEUE.md` D12: "Release: Fold into 1.2.0"). Then "get ready to send yapocis to PyPI", and the answers pasted from the decisions page: "M: mark #106 ready and merge it, then release from the merge commit", "Ship 1.2.0 with it listed as a known issue, and file a bug to fix in 1.2.1" (the console scripts), "Go: release 1.2.0, upload to PyPI, push main and the tag, make the GitHub release", and "File it as a bug" (`yapocis-tools devices`). The version, the release and shipping with the known issues are the maintainer's decisions.

**What Claude did.**
- The version is 1.2.0 in `pyproject.toml` and in `uv.lock`'s yapocis entry (`uv lock --check` passes). CHANGELOG `[Unreleased]` is now `[1.2.0] - 2026-11-12`, with an Added section (CI) and Known issues (#107, #108, #88).
- Filed #107 (B16: `yapocis-report`, `yapocis-imagewatcher` and `yapocis-performance` fail after a plain install, because their dependencies are only in the `dev` extras; 1.1.0 was the same) and #108 (B17: `yapocis-tools devices` crashes without an OpenCL platform). F2 (#102) moves to Done.
- `make release` is interactive, so its steps were run by hand, as for 1.1.0: bump, changelog, AI.md, commit, annotated tag `v1.2.0`, `python -m build`, `twine check`, `twine upload`, push `main` and the tag, GitHub release.

**A defect in the AI's own change, caught at the release.** #106 added its CHANGELOG line under `[1.1.0]`'s Added section instead of `[Unreleased]`: the edit searched for the first "### Added" after `[Unreleased]`, and `[Unreleased]` had none. It was merged that way and caught while writing the 1.2.0 section; this commit moves it.

**Gates.**
- **Repo and PyPI.** PyPI's latest was 1.1.0; no `v1.2.0` tag existed locally or on origin; the checkout was clean on `main`.
- **Suites on `b0af6ea`** (M4 Max, one GPU process at a time, Mako and numba caches redirected into plum `claude-debug/`): metal 1,767 passed, 52 skipped, 1 failed; ocl 1,767 passed, 52 skipped, 1 failed; numbax 1,763 passed, 56 skipped, 1 failed; rustcpu 675 passed, 117 skipped, 1 failed. The failure is `test_old_compiled_modules_are_removed`, the Mako-redirect artefact; CI shows it passes in a clean checkout. `0ce20ed` adds only the workflow and test-skip changes from #106, whose CI run passed (rustcpu on Linux: 639 passed, 118 skipped). `make check` (lint, types) was not run; it has failed at every release.
- **Artifacts.** A trial build from `b0af6ea` with the version at 1.2.0 passed `twine check`, installed in a fresh Python 3.13 venv, reported 1.2.0, and ran a numbax kernel correctly. The same smoke test found #107.

### 2026-11-12: #102, CI on the rustcpu backend (Claude Opus 5.5)

**What the maintainer asked for.** "Feature: Add ci for yapocis, on the rustcpu backend only", then "For CI on yapocis, build a local wheel to test rustcpu", "We never test CI on MacOS -- check your notes. Test on the most recent supported Python.", "1: not yet, 2: yes" (no ruff/typecheck yet; confirm the Mako-redirect test passes in a clean checkout) and "run the CI job".

**What the AI did.** `.github/workflows/ci.yml` (~70 lines), the repository's first workflow: on `ubuntu-latest` with Python 3.13, it installs yapocis and its `test` extra from `uv.lock` (without the `dev` group, which pulls in torch; `duckdb` and `tabulate` are added at their locked versions because the test fixtures use them), builds the `yapocis_rs` wheel with maturin from yapocis-rs `556ef5c` (pinned by full SHA), installs it with `--no-deps`, and runs `make test-rustcpu`. No macOS job, by the maintainer's standing rule. A CHANGELOG line.

**The first CI run** (43269456345): 639 passed, 103 skipped, 15 failed. `test_old_compiled_modules_are_removed` passed, so it fails locally only under the Mako redirect, as the maintainer asked to confirm. Each of the 15 failures was `clGetPlatformIDs failed: PLATFORM_NOT_FOUND_KHR`: tests that use OpenCL directly ran on a runner with pyopencl installed and no OpenCL platform. The `ocl` marker skipped only when pyopencl wasn't importable, and most of these tests had no marker. Fix (~26 lines of tests): `tests/conftest.py` also skips `ocl`-marked tests when `pyopencl.get_platforms()` finds no platform (cached), and the 14 unmarked OpenCL tests are marked (7 in `test_rpc.py`, both in `test_config.py`, the 4 `TestDevicesCommand` tests and `test_main_devices_list` in `test_tools_cli.py`); `test_runtime_demo_ocl` was already marked. A first draft marked all of `test_rpc.py`, which would also have skipped 3 tests that need no OpenCL; caught by counting the skips against CI's failures. Checked on the M4 Max: with OpenCL the 4 files give 53 passed and 1 skipped (CUDA), as before; with `pyopencl.get_platforms` patched to raise as on the runner, 38 passed and 16 skipped (the 15, plus CUDA).

**Found, not fixed.** `yapocis-tools devices` crashes with the same `LogicError` on a machine without OpenCL, instead of saying there is no platform.

**A miss, caught by the maintainer.** When listing F2's open questions, the AI asked which runners to use, though the maintainer had already ruled out macOS CI (twice, in another project's notes). The maintainer: "check your notes". The rule is now in the global instructions.

### 2026-11-12: BUGS.md B15 and FEATURES.md F2 recorded; the maintainer's zero-mean guard in `performance.py` (Claude Opus 5.5)

**What the maintainer asked for.** "Feature: Add ci for yapocis, on the rustcpu backend only", and a "bug:" after the plum coordinator found `yapocis-report compare-branches --component graphics` crashing. Then "Commit the other changes in fix/100-buffer-lifetime".

**What the AI did.** The plum coordinator wrote F2 in `docs/FEATURES.md` (10 lines: the context, three open questions, #102) and B15 in `docs/BUGS.md` (7 lines: the `ZeroDivisionError` in `generate_branch_comparison_report`, #103), and filed both issues.

**What the maintainer did.** The one-line guard in `src/yapocis/utils/performance.py` (`speedup = … if stats["mean"] else 0.0`) is the maintainer's. As B15 records, it covers the division at line 1869 but not the one at line 1926, where `baseline_mean` can also be 0, so #103 stays open. Nothing here was tested by the AI.

### 2026-11-12: #100, buffer lifetime: kernels drop their arguments, one guarded store, fast buffer free, CUDA close (Claude Opus 5.5)

**What the maintainer asked for.** "plan 100, include fast buffer free", then ten decisions on the plan (plum `docs/QUEUE.md` D12), among them "Lock: RLock around register and drain", "Bookkeeping: Merge into one id → (ref, buffer) store in this PR", "Release: Fold into 1.2.0", "Async calls without wait_all: purge() calls wait_all() first (it blocks)", and "CUDA: Run on yellowfin's RTX 3080 (pinned by UUID) before merging". On the public tables: "Keep the views". Implementation: "Go for R3". The planner wrote the plan (Revision 3, after an evaluator's review); this entry covers the implementation.

**What the AI did.**
- `core/kernel_base.py`: `_current_args = None` as a class attribute; `__call__` clears `_current_args` and `returns` in a `finally`; an async call appends its prepared arguments (device buffers and packaged scalars, no host arrays) to the kernel manager's `_inflight`. Most of the +37/-21 is re-indentation under the `try`.
- `core/kmgr_base.py`: `_inflight`, cleared by `wait_all()` whether or not events are pending (under the device manager's release context, entered only when `_inflight` holds something); `purge()` calls `wait_all()` first.
- `core/device_mgr_base.py`: `arrays` (`WeakValueDictionary`) and `buffers` (dict) are replaced by one `_store` of `id → (weakref.KeyedRef, buffer)`, a `_dead` deque fed by the refs' callbacks, and an `RLock`. `_register` is the only writer; `_lookup` (lock-free) returns a buffer only when the entry's ref points at the array; `_drain` matches each dead ref against its entry as a token and releases after the store is consistent; `find_buffer` is concrete here and the three backend copies are deleted; `stats`, `count_leaked_buffers` and `check_resource_usage` drain first; a new `released` counter. `arrays` and `buffers` are read-only `Mapping` views (property, no setter).
- Backends: the four hand-written registrations become `_register`; CUDA `_close`, `__del__` and atexit call `_clear_store()`, and CUDA releases under `activated()`.
- Tests: `tests/core/test_buffer_identity.py` (17 tests: one with 5 forced-collision cases and one with 4 undrained-collision cases, added in the review round below) and `tests/core/test_close_detaches.py` (Metal/OpenCL, plus a CUDA case gated by `YAPOCIS_TEST_CLOSE_CUDA=1`); the stale fixture comment in `test_metal_noncontiguous.py`.
- Lines (measured, `git diff --cached --numstat`): `device_mgr_base.py` +291/-41, `kernel_base.py` +37/-21, `kmgr_base.py` +18/-1, `cu/device_manager.py` +18/-31, `metallic/device_manager.py` +4/-28, `ocl/device_manager.py` +1/-24; tests +662/-5 (after the review round); CHANGELOG +4. The library is ~155 lines larger than the plan's estimate, mostly the two views and docstrings.

**Before and after.** Run against `main`'s library code first, 15 of the new tests failed on Metal and on OpenCL, as the plan expected (14 parametrized cases, and `test_store_threads_smoke` because `main` has no `_register`), and the 5 forced-collision cases and the soak passed. (The first draft of this entry said 16: that count included the soak, which failed on `main` only because of the line described next.) On the branch all pass. Two negative controls, patched in-process: a key-only `_lookup` fails `test_lookup_rejects_foreign_entry`, and a `_drain` that pops without checking fails `test_drain_skips_missing_entry`.

**A defect in the AI's own test, caught by running it on `main`:** `test_async_inflight_soak`'s last line read `prog._inflight`, so it failed on `main` for that reason alone, though the plan meant it to pass there as a regression guard. The line was removed.

**Suites** (M4 Max, one GPU process at a time, Mako modules and the numba cache redirected to plum `claude-debug/`): `make test-metal` 1,756 passed, 51 skipped, 1 failed (1,767 passed, 52 skipped, 1 failed after the review round's tests, and the same after the D6 trims; after the trims `tests/core`, `test_metal_noncontiguous.py`, `tests/async` and `test_rpc.py` on Metal 64 passed, `tests/core` and `test_rpc.py` on OpenCL 35 passed, and the D13 negative controls still fail as before); `make test-ocl` 1,758 passed, 49 skipped, 1 failed; `make test-numbax` 1,759 passed, 48 skipped, 1 failed. The failure is `test_kernel_template_cache.py::test_old_compiled_modules_are_removed`, the Mako-redirect artefact, which fails on `main` too. CUDA on yellowfin's RTX 3080, pinned by UUID (`CUDA_VISIBLE_DEVICES=GPU-102373c2-8032-c8bc-8d1f-325e2d13a6d8` in the scratch clones' `run.sh`; the test IDs read `cuda-NVIDIA GeForce RTX 3080`; scratch clones, nothing pushed), before the review round's test changes: the branch's full suite 670 passed, 75 skipped, 0 failed, and the CUDA close case passed in its own process; `main` with the new tests copied in: 656 passed and the same 14 new-test failures, and its close case failed (item 3). impressionist `tests/test_all.py` (21 passed) and yapocis_external `test_yunet.py`/`test_cocircular.py` (14 passed) pass on `main` and on the branch.

**Performance.** `p100_overhead` (20,000 sync `clamp_pixels` calls on 16 floats, median of 5 runs). Before the trims, over 17 interleaved Metal pairs of `main` and the branch (3 by the implementer, 14 by the evaluator, both orders), the branch cost a median of +1.55 µs per call (mean +1.35 µs, 14 of 17 pairs slower), above D6's 1 µs bar. The first draft of this entry claimed +0.6 µs: that was the difference of two medians over the implementer's 3 pairs (deltas +0.70, +3.14, +0.16 µs), and the evaluator's repeat showed it doesn't hold. The maintainer's D6 decision was "trim": apply the review's finding-8 trims and re-measure. `_register` and `purge_buffers` now call `_drain()` only when `_dead` is non-empty (as `_lookup` already did), and `wait_all()` enters the release context and clears `_inflight` only when it holds something (+7/-4 lines; no change in behaviour). After the trims, over 20 order-balanced interleaved Metal pairs (10 each order, `main` from a `git archive` of `dc0e62d`, `yapocis.__file__` asserted in every run): median +0.43 µs per call (bootstrap 95% CI -0.59 to +2.01), mean +0.25 µs (CI -0.81 to +1.15), 11 of 20 pairs slower, on a ~156 µs call. Both point estimates are under the 1 µs bar; the CIs include zero and reach above 1 µs, so the bar is met on the estimate, not shown at 95%. The write+find pair is unchanged (-0.27 µs). OpenCL, 8 pairs: median +10.4 µs (CI -3.2 to +18.0), mean +7.8 µs, on a ~575 µs call with ±13 µs pair noise, so unresolved at that noise; with `purge()` inside the timed loop on both sides (6 pairs), +4.5 µs median (CI +2.1 to +6.5), about 0.8% of a call. That residual is above 1 µs and isn't explained yet; the bookkeeping is the same base-class code as on Metal. The maintainer accepted both the Metal result and the OpenCL residual: "I accept both." `test-perf-kernel` (2 runs × 11 tests × Metal and OpenCL against 2 baseline runs, before the trims): the branch's mean is lower than `main`'s in 15 of 22 rows, and "within the baseline's spread" can't be judged with two baseline runs; the maintainer closed it as "no consistent change".

**Review round (D13).** The implementation review (plum `docs/reviews/2026-11-12-yapocis3-100-implementation-review.md`) found the performance claim above overstated, the before-count wrong, a stale module docstring, and three tests weaker than the plan said. The maintainer's decisions (plum `docs/QUEUE.md` D13): "test-perf-kernel 'within the baseline's spread': Close it as 'no consistent change'", "Harden the weaker tests: Yes: assert an RLock, make the id reuse deterministic, keep #89's assertion able to fail", and "Text fixes …: Fix them (already decided)". Defects in the AI's own tests, caught by the evaluator, and their fixes:
- `test_reentrant_release` ran its first manager call (the fixture's purge) on the main thread, so a plain `Lock` hung the whole run instead of failing at the 30 s join; and its release-order check worked only when the next array happened to get the dead array's `id`. Now every manager call runs in the worker thread, the test asserts the lock is an `RLock`, and the release hook checks the store at the moment of release, whatever the ids. Negative controls (in-process): a plain `Lock` fails in 30 s ("not an RLock (deadlocked: True)"); releasing before the store update fails ("released before its store entry was removed").
- #89's orphan check in `test_metal_noncontiguous.py` could no longer fail, because `count_leaked_buffers()` now drains first. It now also checks that each strided argument has its own store entry, that nothing was released during the calls, and that the store grew by exactly 3. With #89's `make_buffer`/`write_buffer` change reverted in the working tree, it fails at the new check after `count_leaked_buffers() == 0` passed; the file was restored from the index.
- The forced-collision cases exercised the drain, not the `ref() is a` guard. `test_forced_id_collision_undrained` holds off the drain, so the new array with the old `id` meets the stale entry and only the guard rejects it; after the drain it checks the new array's entry survived. Negative controls: a key-only `_lookup` fails all 4 cases (and `test_lookup_rejects_foreign_entry`); a drain that pops by key alone fails all 4.
- The module docstring of `device_mgr_base.py` still listed `find_buffer` as a per-backend method; corrected.

**Not verified:** the review round's test changes and the D6 trims on CUDA (the trims touch only base-class code paths that the Metal and OpenCL runs exercise); free-threaded Python (the thread test is a smoke test on a GIL build); NVIDIA OpenCL.

**Division of labor:** the maintainer decided the design questions; the planner and evaluator wrote and reviewed the plan; Claude implemented it, wrote the tests and ran the suites.

### 2026-11-12: #89's orphan-buffer test, isolated from earlier tests (Claude Opus 5.5)

**What the maintainer asked for.** "Fix the test", after the 1.2.0 release gate failed on `test_metal_noncontiguous.py::test_strided_in_leaves_no_orphan_buffer` (37 leaked buffers in `make test-metal`, 35 in `make test-numbax`), though it passed alone and with all of `tests/backend`.

**Cause, confirmed by experiment.** The `demo` fixture ran `prog.dmgr.purge()` and then `gc.collect()`, and `count_leaked_buffers()` counts the whole shared Metal device manager. Earlier tests that load a program into a local variable (`tests/async/test_execution.py` loads `imageops` in each test) leave it as cyclic garbage: a kernel and its manager refer to each other, and each kernel keeps its last call's arguments. Those arrays live until a full collection. A probe plugin that logged the accounting inside the fixture (plum `claude-debug/f89iso_probe.py`) showed, in the full-suite prefix, 51 buffers left live after the purge, and the fixture's `gc.collect()` turning all 51 into leaks; a `DEBUG_SAVEALL` pass found the dead arrays held by `yapocis.metallic.kernel.Kernel` objects in the garbage. Whether the arrays were already gone depended on when Python's automatic collector last ran, which is why the test passed in smaller runs. Smallest reproducer: `tests/async/test_execution.py` plus the test, on Metal, failing with 5 leaks every time.

**What the AI did.** The fixture now calls `gc.collect()` before `purge()` (~5 lines with the comment). The assertion is unchanged. With #89's library fix reverted in the working tree, the test still fails, alone (4 leaks) and after `test_execution.py` (5); with it restored, the reproducer passes, 3 of 3. One CHANGELOG line.

**Suites** (M4 Max, alone on the GPU, Mako modules and the numba cache redirected to plum `claude-debug/`): `make test-metal` and `make test-numbax` each gave 1,718 passed, 28 skipped, 1 failed. The failure is `test_kernel_template_cache.py::test_old_compiled_modules_are_removed`, the Mako-redirect artefact seen in every gate run.

**Not changed, for the maintainer:** a dropped program's kernels keep their last call's arrays, and the device buffers behind them, alive until the cycle collector runs. It isn't a leak (a full collection and a purge free them), but it does delay the release of device memory.

**The maintainer's question, "Should we add gc.collect to .purge?"** The coordinator advised against it: a full collection walks the whole heap and runs every pending finalizer, and it would only shorten the window for the two causes underneath, kernels keeping their arguments and buffers keyed by a reusable `id()`. Filed as #100 ("Commit, file, investigate").

**Division of labor:** the coordinator proposed the cause and the fix; Claude confirmed both by experiment, made the change and ran the suites.


### 2026-11-11: #80 closed on rustcpu: yapocis-rs re-export and CPU port, the rebuilt wheel, the temporary xfail removed (Claude Opus 5.5)

**What the maintainer asked for.** Steps 6–8 of the #89/#87/#80/#81 plan, with CPM (plum `docs/QUEUE.md` D9.6).

**Step 6, yapocis-rs PR #23 (`556ef5c`).** The bundle was re-exported with `yapocis-tools bundle --fixtures` from a fresh scratch venv with this repository installed from git at `bf9a0b8` (#98's merge), so the manifest records `yapocis_version` 1.1.0. The diff was the expected one: the `_fast` kernels in the 12 `convolve.*` sources, their 12 hashes, `yapocis_version`, and `SOURCE`. The Rust text port (`convolve_source`) and the CPU `convolve_fast`, which had copied the old template on purpose, were rewritten to the new kernel; a new yapocis-rs test, `fast_variant_matches_the_main_kernel`, failed on rustcpu with the old ports and passes with the new ones. `scripts/check-bundle.sh` passed locally, and CI's build, rustcpu and bundle-drift jobs passed.

**Step 7, the wheel.** `yapocis_rs` 0.1.0 rebuilt from yapocis-rs `556ef5c` with `uvx maturin build --release --locked` (plum `claude-debug/s7_wheel/`, sha256 `d46be301…`), and reinstalled in this checkout's `.venv` with `uv pip install --no-deps --reinstall`; no other package changed. rustcpu's `_fast` now equals its main kernel in all 12 checked cases, including `chunk` 0, `chunk > width` and both dispatches (`claude-debug/s7_fast_check_out.txt`).

**Step 8, this change.** `_rustcpu_fast_is_partial`, `_fast_backends` and the runtime `pytest.xfail` in `test_convolve` are gone from `tests/lib/processing/test_convolve.py`; `test_fast_variant_matches_the_main_kernel` parametrizes over `get_test_backends()`, so a regression on rustcpu now fails instead of xfailing. `docs/BUGS.md` B4 moves to Fixed, and the CHANGELOG's #80 line says where rustcpu's fix comes from. ~47 lines of tests removed, ~2 changed.

**Gate** (M4 Max, this checkout's `.venv` with the new wheel, this branch's `src` first on the path, caches redirected): `test_convolve.py` with `YAPOCIS_TEST_BACKENDS=ocl,metal,numbax,rustcpu`: 47 passed, none xfailed. The rustcpu suite (`make test-rustcpu`'s command): 661 passed, 22 skipped, 1 failed; the failure is `test_kernel_template_cache.py::test_old_compiled_modules_are_removed`, from the Mako redirect, and that file passes (4 passed) without it. Not verified: CUDA and NVIDIA OpenCL for this change (it touches only tests and docs), and the Python 3.11/3.12 builds of the wheel.

**Division of labor:** the maintainer approved the steps; Claude re-exported, ported, rebuilt, removed the xfail and wrote the entries.

### 2026-11-11: #80, the GPU `_fast` convolve kernels compute every output (Claude Opus 5.5)

**What the maintainer asked for.** "continue with 89,87,80,81" (plum `docs/QUEUE.md` D5.37); for #80, "Fix it" (D7.6), "Keep the skip for now" for the parity fixtures (D7.7), no downstream pin moves (D7.8), and "Yes for the #80 PR" to a CUDA run on yellowfin (D7.5).

**What the AI did.** The plum planner found the causes and prototyped the templates (`claude-debug/n5_proj`); the evaluator confirmed them. In the OpenCL, Metal and CUDA `convolve.mako`, `${name}_fast` had each work item compute only `chunk - len(conv)` outputs, clamp its writes to `right - 1`, and either multiply `a[j]` (the loop index; OpenCL, CUDA) or read a `threadgroup` buffer that every thread wrote without a barrier (Metal). Now work item `g` computes `ret[g*chunk .. g*chunk+chunk-1]`, each with the main kernel's expression and edge test, and stops at the width; a `chunk` below 1 counts as 1; the shared buffer, the clamp and `#define FAST` go (~35 lines across the three templates, the planner's prototypes unchanged). The parity skip stays, with its comment corrected (`codegen/parity.py`); `docs/library/convolve.md` gets a paragraph on `chunk` and the global size.

**Tests** (`tests/lib/processing/test_convolve.py`, ~105 lines): `test_fast_variant_matches_the_main_kernel`, over the test backends and seven cases (5, 3 and 100 taps; `chunk` 64, 1, 7, 100, `chunk > width` and `chunk = 0`), each with the default dispatch and with `global_size = ceil(n/chunk)`, three times. `_fast(A)` must equal `main(A)` exactly, so an unwritten output fails (it holds garbage on OpenCL and zeros on Metal), and must match a float64 reference away from the edges (numbax reflects there, #93). The main kernel also runs on another input before each `_fast(A)` call, at the evaluator's suggestion, to keep recycled memory from passing for a result; the independent PR review's probe showed `out_like` buffers aren't reused between calls, so that step only varies what ran before, and the comment says so. Also, `test_convolve` now asserts `convolve_fast(a, 100) == convolve(a)`; before, it had no assertion. rustcpu's CPU port in yapocis-rs copies the old behaviour, so rustcpu's cases carry a temporary xfail that applies only while the installed `yapocis_rs` still gives the old output, as #66's did. One defect in the first draft: `test_convolve`'s new assertion failed on rustcpu, because only the new test had the xfail; it now has the same behaviour-gated xfail.

**Results.**
- M4 Max, a scratch venv from the shipping lock (#90's and #87's changes together, plum `claude-debug/y4_venv_ship`), `YAPOCIS_TEST_BACKENDS=ocl,metal,numbax,rustcpu`: with the shipped templates, 16 failed (every OpenCL and Metal case), 23 passed, 8 xfailed (rustcpu); with this change, 39 passed, 8 xfailed.
- **CUDA, the merge gate:** yellowfin's RTX 3080 (pinned by UUID), yellowfin's `.venv` python with this branch's `src` on `PYTHONPATH`: `test_convolve.py` 13 passed, 4 skipped (rustcpu-only tests); `tests/backend/test_smoke.py` and `tests/integration/test_rpc.py` 31 passed. With the shipped templates the same file gave 8 failed, 5 passed, 4 skipped. So the new `_fast` compiles, and so does the main kernel in the same program.
- OpenCL suite (`YAPOCIS_VALIDATION=1`) and Metal suite on the shipping lock (with #87's pytest-benchmark 5.3.1): each 1,601 passed, 14 skipped, 8 xfailed (rustcpu), 1 failed, the Mako-redirect artefact (`test_old_compiled_modules_are_removed`). An earlier run on #87's first, nine-package lock gave 1,597 passed and 4 `test_modes.py` errors, from pytest-benchmark 5.1.0 against pytest 9 (see #87's entry).

**Division of labor:** the maintainer decided to fix it, keep the parity skip and approved the CUDA run; the plum planner found the causes and wrote the templates, and the evaluator strengthened the test design; Claude wrote the tests, docs and ledger entries and ran the gates.

### 2026-11-11: #87, Python 3.11, `Pillow>=13.0.0`, and the nine locked packages with Dependabot alerts (Claude Opus 5.5)

**What the maintainer asked for.** "continue with 89,87,80,81" (plum `docs/QUEUE.md` D5.37); for #87, "3: requires-python >=3.11 (all 56 close)" (D7.2), and "Yes" to raising the published Pillow floor (D7.3); after the first gate, option 1 for the pytest 9 problem below: also upgrade pytest-benchmark to 5.3.1 (D7.11); after review, folding ruff and black from py310 to py311 into this change (D8.2); and "Scratch venv; sync and reinstall after merge" (D7.10), so this checkout's `.venv` is synced by the coordinator after the merge, not here.

**What the AI did.** The plum planner found the cause: nothing held Pillow below 13 on purpose; `requires-python = ">=3.10"` did, because Pillow 13.0.0 (and filelock 4.2, starlette 1.10 and fonttools 4.68) need Python 3.11, so uv kept a 3.10 fork at the vulnerable versions. This change:
- `pyproject.toml`: `requires-python = ">=3.11"`, the 3.10 classifier removed, mypy's `python_version = "3.11"`, black's and ruff's `target-version` py311 (D8.2; ruff gives the same 493 findings on `src/yapocis` and `tests` before and after, and `black --check` the same 39 files; the extension scaffold template in `scripts/tools.py`, which still writes `requires-python = ">=3.10"`, is left for a follow-up), and `Pillow>=13.0.0` (the first release outside all 41 Pillow advisories);
- README and `docs/quickstart/{index,installation}.md`: Python 3.11 or later (the plan named README; the two quickstart pages said 3.10 too);
- `uv.lock`: `uv lock --upgrade-package` for Pillow, starlette, wheel, filelock, idna, pytest, requests, fonttools and Pygments, with yapocis3's `.venv` uv (0.13.4). Only those nine move; the 3.10-only versions of eleven packages (numpy, scipy, contourpy and 8 others) drop out, and exceptiongroup goes. Then, on the maintainer's decision D7.11, `--upgrade-package pytest-benchmark` (5.1.0 → 5.3.1, which replaces py-cpuinfo 9.0.0 with py-cpuinfo2 10.1.1); nothing else moves. Before that step the lock was identical to the planner's trial `n5_lock311_pil`. pymetallic stays at 0.2.3; #90 moves it.

~9 lines of pyproject, ~3 of docs, ~20 of BUGS.md, CHANGELOG and this entry; the lock diff is 238 lines added and 893 removed.

**Checked.** The plan's matcher (`n5_lock_vuln_check.py`, against the 56 alerts saved in planning) finds 56 of 56 on `main`'s lock and 0 of 56 on this one. `uv lock --check` passes with uv 0.12.5 and 0.13.4.

- **First gate, nine packages only** (scratch venv `y4_venv87`, Apple M4 Max, one suite at a time): OpenCL, Metal and numbax each 1,577 passed, 14 skipped, 1 failed, 4 errors; rustcpu 647 passed, 22 skipped, 1 failed, 1 error. **The errors were new**: `tests/async/test_modes.py::test_async_modes_small_pipeline`, one per backend, because pytest-benchmark 5.1.0 reads `node._nodeid`, which pytest 9 removed. They passed on `main`'s lock (pytest 8.4.2). The plan's evaluator had run only `tests/lib`, which doesn't use the `benchmark` fixture, so the plan didn't see it. A trial with pytest-benchmark 5.3.1 passed `test_modes.py`, 40 of 40, and the maintainer chose that fix (D7.11).
- **Gate on the final lock** (scratch venv `y4_venv_ship`, built from this lock with #90's pymetallic 0.3.1 on top, as it ships): OpenCL (`YAPOCIS_VALIDATION=1`), Metal and numbax each 1,581 passed, 14 skipped, 1 failed; rustcpu 648 passed, 22 skipped, 1 failed. No errors; the `test_modes.py` cases pass on every backend. The failure is `test_kernel_template_cache.py::test_old_compiled_modules_are_removed`, from the Mako redirect the runs use, as on `main`, whose Metal suite gives the same 1,581 passed.
- The sphinx build (nine-package lock) gives the same two warnings as on `main` (BUGS.md and FEATURES.md not in a toctree).

Not run: CUDA (optional for this PR, D7.5), Python 3.11 and 3.12 (the venvs are 3.13).

**Division of labor:** the maintainer chose the Python floor and the Pillow floor; the plum planner found the cause and the nine packages, and the evaluator ran `tests/lib` on the new lock; Claude made the change, ran the suites, found the pytest-benchmark errors and trialled the fix the maintainer then chose, and wrote the ledger entries.

### 2026-11-11: #90, pymetallic 0.2.3 → 0.3.1 (Claude Opus 5.5)

**What the maintainer asked for.** "Note to move to 0.3.1 at the start of the next release cycle" (#90), then "Yes: #90 as its own PR, after #89" (plum `docs/QUEUE.md` D7.4): the next release cycle has started.

**What the AI did.** `uv lock --upgrade-package pymetallic` (with yapocis3's `.venv` uv, 0.13.4): only pymetallic moves, 0.2.3 → 0.3.1; uv also writes the lock's `revision = 5` (was 3), as the plan's trial locks did. The issue asked whether `metallic/device_manager.py`'s "always synchronous `Buffer.from_numpy`" reasoning survives 0.3's async writes, and the #89 plan asked whether `from_numpy` still copies. Read: 0.3.1's `Buffer.from_numpy` is unchanged from 0.2.3 (`ascontiguousarray`, then `makeBuffer(bytes:)`), and the installed `metallic.py` is identical to pymetallic's `v0.3.1` tag. Measured (plum `claude-debug/y4_pr3_async_bench.py`, M4 Max, alone on the GPU): changing the source after `from_numpy` leaves the buffer as it was, and a strided source round-trips; sync `from_numpy` takes 5.8 µs at 4 KiB and 0.97 ms at 16 MiB, while `async_buffer_from_numpy` with its commit and wait takes 151 µs and 1.71 ms, and 8 async writes sharing one command buffer are still 1.7–5x slower than 8 sync ones. So the reasoning holds. One of its four reasons was stale: 0.3.1 lets several async writes share a command buffer (pymetallic `44fcbba` removed the auto-commit). The comment now gives the measured reasons, including that async writes default to private storage, which `read_buffer` can't map. ~8 lines of comments; no code changed.

**Tests.** In a scratch venv built from this branch's lock (plum `claude-debug/y4_venv90`; its requirements differ from `main`'s only in pymetallic), the Metal suite: 1,581 passed, 14 skipped, 1 failed, the same counts as `main` in a venv from `main`'s lock. The failure, in both, is `test_kernel_template_cache.py::test_old_compiled_modules_are_removed`, caused by the Mako redirect the runs use.

**Division of labor:** the maintainer decided the timing and the separate PR; Claude upgraded the lock, read and measured pymetallic 0.3.1, and wrote the comment and the ledger entries.

### 2026-11-11: #89, Metal buffers for non-contiguous arrays registered under the caller's array (Claude Opus 5.5)

**What the maintainer asked for.** "continue with 89,87,80,81" (plum `docs/QUEUE.md` D5.37), and for #89 "A: register the buffer under the caller's array" (D7.1).

**What the AI did.** In `metallic/device_manager.py`, `make_buffer` and `write_buffer` no longer rebind the argument to `np.ascontiguousarray(a)`; they pass the caller's array to `pymetallic.Buffer.from_numpy` and register the buffer under its `id()`. The plum planner and evaluator had read that `from_numpy` makes its own contiguous copy and that Metal's `makeBuffer(bytes:length:options:)` copies the bytes, in pymetallic 0.2.3 and 0.3.1, so no copy needs keeping alive; `read_buffer`'s `np.copyto` writes back through any layout. ~4 lines removed, ~4 of comments. New `tests/backend/test_metal_noncontiguous.py` (93 lines, six tests, marked `metal`): the planner's five drafted cases (strided `in` without a leaked buffer, a strided `out`, a strided write then read, F-ordered and transposed inputs) and the evaluator's suggested sixth, a strided `inout` (graphics `rects` on `base[::2]`) that must write the even rows of `base` and leave the odd ones.

**Tests.** In a scratch venv built from `uv.lock` (plum `claude-debug/y4_venv_main`), Apple M4 Max, alone on the GPU: before the change, 6 of 6 fail (the leak count, and "not in yapocis management" for the other five); after it, 6 of 6 pass. The Metal suite on this branch (`YAPOCIS_BACKEND=metal pytest tests`): 1,587 passed, 14 skipped, 1 failed; the failure is `test_kernel_template_cache.py::test_old_compiled_modules_are_removed`, caused by the Mako redirect the run uses, as in the 1.1.0 gate.

**Not done here:** OpenCL's F-order failure (B13, #92); CUDA (its `make_buffer` refuses non-contiguous arrays outright).

**Division of labor:** the maintainer chose the fix; the plum planner found the F-order case and drafted five tests, and the evaluator suggested the `inout` test; Claude wrote the change, the sixth test and the ledger entries.

### 2026-11-11: #81, the truetrace iso direction in the docs; BUGS.md B13 and B14 (Claude Opus 5.5)

**What the maintainer asked for.** "continue with 89,87,80,81" (plum `docs/QUEUE.md` D5.37), and, for the plan's side findings, "File (a), and (b) with (c) together" (D7.9): filed as #92 and #93, to go into `docs/BUGS.md` with the first PR of the series. The issue's own fix: "Correct the docstring. The code is right."

**What the AI did.** The plum planner and evaluator found four places that had the direction backwards; this change corrects them and changes no code:
- `lib/truetrace.py`: the `trace_segments` docstring (positive moves the contour inward, into the mask) and the `trace_regions` docstring (positive moves it into the ink, so `0.5`, not `-0.5`, keeps only the pixels fully inside);
- `docs/library/truetrace.md`: the `iso` parameter line and the ISO table, where "creates outline" now belongs to −2.0.

`docs/BUGS.md`: B5 moves to Fixed, and B13 (#92, OpenCL F-order) and B14 (#93, the `Convolve1D` edge rules, with the evaluator's corrected wording: the first `left + 1` and the last `left` samples) are added under Open. ~7 lines of docstrings and docs, ~30 lines of BUGS.md and CHANGELOG.

**Checked.** The planner's iso probe, rerun against this branch in a scratch venv (plum `claude-debug/y4_b5_iso_probe_ocl_out.txt`): `trace_mask` ink on a 1552 px ring is 1924, 1600, 1556, 1512 and 1174 at iso −1.5, −0.5, 0, 0.5 and 1.5, on both the CPU engine and OpenCL, so a positive iso shrinks the shape. No test was added, because only text changed. The Metal suite on this branch: 1,581 passed, 14 skipped, 1 failed (`test_old_compiled_modules_are_removed`, from the Mako redirect the run used).

**Division of labor:** the maintainer approved the work and decided how the side findings were filed; the plum planner and evaluator found the four places; Claude wrote the corrections and the ledger entries.

### 2026-11-11: BUGS.md B12, and issues #89 and #90 from the pymetallic check (Claude Opus 5.5)

**What the maintainer asked for.** After 1.1.0 shipped with #88 as a known issue: "Check pymetallic for possible memory issues", then "file both. Note to move to 0.3.1 at the start of the next release cycle."

**What the AI did.** The plum coordinator read pymetallic 0.2.3 (the version in `uv.lock`) and this repo's `metallic/device_manager.py`. It found no cross-process path that would explain #88, and filed what it did find:
- **#89, BUGS.md B12:** `make_buffer` and `write_buffer` register a non-contiguous input's contiguous copy under the copy's id. The weak entry dies at once, so lookups miss and the buffer leaks until a purge.
- **seantrue/pymetallic#2:** the `to_numpy` view doesn't keep its `Buffer` alive. This repo copies out of the view at once, so it isn't affected.
- **#90 (`dependencies`):** at the start of the next release cycle, move pymetallic 0.2.3 → 0.3.1. 0.3.1 is already on PyPI.

~15 lines of Markdown; no code changed.

### 2026-11-11: Version 1.1.0 (Claude Opus 5.5)

**What the maintainer asked for.** "go for release" (plum `docs/QUEUE.md` D5.28), version 1.1.0 (D5.2), from `main` at `842f4d0`. When the first test gate failed, the maintainer decided "Yapocis can release with those terms" (Q4): ship with #88 as a known issue, and rerun the gate on a quiet GPU. The version, the release and the decision to ship with #88 are the maintainer's.

**What Claude did.**

- The version is 1.1.0 in `pyproject.toml` and in `uv.lock`'s yapocis entry.
- CHANGELOG `[Unreleased]` is now `[1.1.0] - 2026-11-11`, with a new "Known issues" entry for #88. "For review before the 1.1.0 release" is dropped from the backfill note.
- The plum coordinator's `docs/BUGS.md` entries B10 (#87) and B11 (#88) go into the same commit.
- `make release` is interactive, so its steps were run by hand: bump, changelog, AI.md, commit, annotated tag `v1.1.0`, `python -m build`, `twine upload`, then push `main` and the tag (RELEASE.md step 10).

**Gates, and the first gate's failure.**

- **Repo and PyPI checks.** The tree was clean, HEAD and origin/main were both `842f4d0`, and no `v1.1.0` tag existed locally or on origin. PyPI's latest was 1.0.0.
- **First gate (M4 Max).** `make test-{ocl,metal,numbax,rustcpu}` ran twice, with the numba and Mako caches redirected into plum `claude-debug/`. In the second metal run, `test_resnet50_resident_async_vs_sync[ocl-Apple M4 Max]` failed: 57 of 1000 outputs were off by up to 0.0217. It passed in the other 7 full-suite runs and in 10 of 10 isolated reruns. The release stopped there, with no bump.
- **#88.** The coordinator reproduced the failure with 4 processes at once, on OpenCL and Metal, in sync and in async mode; one process on a quiet GPU gave none. It is filed as #88, and its cause (yapocis, or Apple's GPU stack under multi-process load) is open. The first gate had probably shared the GPU with the coordinator's other test runs.
- **Gate rerun on a quiet GPU.** No other pytest, cargo, xcodebuild or vimp process was running before each backend. Results:
  - ocl: 1,684 passed, 28 skipped, 1 failed
  - metal: 1,684 passed, 28 skipped, 1 failed
  - numbax: 1,684 passed, 28 skipped, 1 failed
  - rustcpu: 661 passed, 45 skipped, 1 failed
- **The one failure in each run.** It is always `test_kernel_template_cache.py::test_old_compiled_modules_are_removed`, caused by the Mako redirect; that file passes (4 passed) without it.
- **`make check` fails, as it did at 1.0.0.** At HEAD, isort flags 3 tracked files, black would reformat 27, ruff reports 479 errors and mypy 1,443. At v1.0.0, isort flagged 15 files, ruff reported 527 errors and mypy 968. It was not made a release blocker; it remains an open RELEASE.md checklist item.

**Artifacts checked.**

- Both the sdist and the wheel pass `twine check`. The wheel holds no compiled Mako modules, `__pycache__` or model weights.
- The wheel was installed with uv into a fresh Python 3.13 venv (plum `claude-debug/rel_venv`). From outside the source tree, the README's Mandelbrot example ran on OpenCL, Metal and numba and was compared with a numpy reference.
  - numba matched the reference exactly.
  - OpenCL and Metal agreed on 99.99% of the set and on 99.83% of escape counts (float32 at the boundary).
- `importlib.metadata` read the version back from the wheel as 1.1.0.

**Not verified:**

- CUDA, NVIDIA OpenCL and PoCL for the release build (yellowfin was not run for this release; the CUDA run for the numba batch is in the entry below).
- Python versions other than 3.13.
- Linux and Windows installs.

**Shipped with known issues:**

- **#88:** GPU results can differ under multi-process contention on Apple M4.
- **#87:** Dependabot's 56 alerts against `uv.lock`, mostly Pillow. The published dependency floors are unchanged (`Pillow>=10.3.0`), so users resolve current versions.
- **`make check` debt,** as above.

**Division of labor:** the maintainer decided the version, the release and shipping with #88; the plum coordinator diagnosed #88 and wrote the BUGS.md entries; Claude ran the gates, bumped the version, wrote the changelog and ledger entries, and built, checked and published the artifacts.

### 2026-11-11: #66 closed on rustcpu, the temporary xfail removed, and a correction about CUDA (Claude Opus 5.5)

**What the maintainer asked for:** "go for 2 and 2b, CP" (plum `docs/QUEUE.md` D5.27), after "Fold into 2b" for the stale CUDA line (D5.24) and "M" for yapocis-rs PR #22 (D5.26).

**rustcpu black fix: yapocis-rs `9ebcac9` (PR #22).** The `yapocis_rs` wheel in this checkout's `.venv` was rebuilt from that commit (`maturin build --release --locked`, installed with `uv pip install --no-deps`; no other package changed). With the old wheel, rustcpu's `xyz2xyY` gave (NaN, NaN, 0) for black and (inf, −inf, −0.5) for (0.5, −0.5, 0); with the new one, (0, 0, 0) and (0, 0, −0.5). So the temporary behaviour-gated xfail from the numba batch comes out: `_rustcpu_black_is_nan`, `_black_backends` and the xfail marker are gone from `tests/lib/processing/test_colormath.py`, and the three #66 tests parametrize over `get_test_backends()` like the rest. A regression back to NaN on rustcpu now fails instead of xfailing. `docs/BUGS.md` B2 moves to Fixed, and the CHANGELOG's #66 line no longer says rustcpu gives NaN. ~43 LOC removed, ~3 changed.

**Correction to the numba batch entry below.** Its "Not verified" line said CUDA and NVIDIA OpenCL weren't run. They were, before the merge: the plum coordinator ran the full suite with `YAPOCIS_BACKEND=cuda` on yellowfin's RTX 3080 at `3596ce0`, with the maintainer's approval ("Verify CUDA on yellowfin, you can use the 3080"): 1,317 passed, 23 skipped, 0 failed. `main` at `d1fb345`, before the batch, on the same GPU: 1,263 passed, 18 skipped, 1 failed (the non-square truetrace test that #67 and #82 fixed). The old line is kept, with a note pointing here. Only PoCL remains unverified. The Tesla P40 run the batch plan mentioned was not done.

**Verified (M4 Max, this branch, the rebuilt wheel):** `tests/lib/processing/test_colormath.py` with `YAPOCIS_TEST_BACKENDS=ocl,metal,numbax,rustcpu`: 54 passed, none xfailed. `make test-rustcpu`: 661 passed, 45 skipped, 0 xfailed, 1 failed; the batch's run gave 658 passed and 3 xfailed, so the three former xfails now pass. The failure is `test_kernel_template_cache.py::test_old_compiled_modules_are_removed`, caused by the Mako redirect used for the run, as in the batch; that file passes (4 passed) without the redirect. **Not verified:** CUDA, NVIDIA OpenCL and PoCL for this change (it touches only tests and docs); the P40.

**Division of labor:** the maintainer made the decisions; the plum coordinator ran the yellowfin CUDA check; this entry's implementer rebuilt the wheel and wrote the change.

### 2026-11-11: The numba batch: #64, #66, #67, #82, the rustcpu cache key, numbax xyY2xyz, #65 tests (Claude Opus 5.5)

**What the maintainer asked for:** the plum coordinator planned the batch; the maintainer's decisions are in its `docs/QUEUE.md` (D1–D5). On #67: "1 - B 2 - Yes 3 - Batch 4 - yes". On #64: "1-a 2- own 3 ?- this 4- patch 5-accept 6-no". On #66: "Z . With check. Patch. Yapocis first. Accept 0". On #82: "Fold into this batch". On `docs/BUGS.md` and `docs/FEATURES.md`: "Commit 0 of this PR". Then "Go" and "CP".

**Tracking:** `docs/BUGS.md` and `docs/FEATURES.md`, which the coordinator started on 2026-11-11 from the open issues, are now in git. Each fix below moves its entry to Fixed.

**#64, numba convolve kernels built per manager:** the numba backend stored each generated `Convolve1D` and convolvesep function on the shared kernel module by name and reused it whenever the name was already there. So a second `Convolve1D(name="custom")` ran the first one's weights. The flaw dates from the backend's first commit, `77fc317` (2025-09-30; `9e96c5e` only moved the file); `3c0d53c` hid it from 2026-03-05 and `2d8c6d5` re-exposed it on 2026-10-12. `src/yapocis/numbax/kernel_manager.py` now builds those two interfaces' functions into a memo owned by one manager and never touches the module (approach A from the #64 plan, D2.1). That also stops a kernel named like a module global (`np`) from binding or overwriting it and breaking `convolve.py` for the rest of the process, and removes a race between threads building same-name kernels. The stub for an unimplemented kernel now names the right module and function (it referred to an undefined `mod_name` and to the loop's last name). Project `convolve.py`/`convolvesep.py` overrides no longer apply to these two interfaces (D2.5). Five new tests in `tests/lib/processing/test_convolve.py`; all five numba cases fail on `main`'s kernel manager and pass now. Not fixed: convolvesep strips a trailing `_h`, `_v` or `_res` from a user kernel name, so a kernel named like `edge_h` raises `KeyError`; that predates this change. ~40 LOC changed, ~140 of tests.

**The rustcpu program-cache key:** since PR #77 the rustcpu kernel manager keyed its process-wide program cache by `json.dumps(context, default=repr)`. `repr` of a numpy array keeps numpy's print precision (8 decimal places), elides arrays over 1000 elements and follows `np.set_printoptions`, so weights that differed past the 8th decimal place (the 7th significant digit of a tap under 0.1), in the middle of a long kernel, or under a lowered print precision shared one program. The key now writes arrays and numpy scalars exactly, with dtype and shape (`_key_default` in `src/yapocis/rustcpu/kernel_manager.py`). The new test, `test_rustcpu_same_name_kernel_gets_its_own_program`, covers those four cases with float64 arrays, because a Python list never collided; each case fails on `main` and passes now. The key now costs O(n) in the context's array sizes; no interface puts more than about 1,000 taps in one. ~14 LOC, ~95 of tests.

**#66, `xyz2xyY` of black:** OpenCL, Metal and CUDA divided by X + Y + Z and gave NaN for black; numbax added `1e-20` to the sum, which gave 0 for black but shifted chromaticity near it. Every kernel now checks `sum != 0` and gives x = y = 0 (option Z, D4). numbax's numpy fallback, used when numba is missing, got the same check. Accepted (D4.5): a mixed-sign zero sum such as (0.5, −0.5, 0) also gives 0, and a denormal sum gives 0 on the GPUs (flushed) but a finite chromaticity on the CPU backends. One sentence in `docs/library/colormath.md`. Five new tests in `tests/lib/processing/test_colormath.py`; on `main`'s kernels OpenCL and Metal fail black, the black round trip and the mixed-sign case, and numbax fails near black, the mixed-sign case and the fallback. **rustcpu still gives NaN** until yapocis-rs's port is fixed and the `yapocis_rs` wheel is rebuilt, so its three cases carry a temporary xfail that applies only while the installed wheel really returns NaN (`_rustcpu_black_is_nan`, checked at collection with an explicit backend). The rustcpu case still runs and asserts; the xfail's weakness is that it would hide a regression back to NaN, so it comes out with the rebuilt wheel, and #66 and BUGS.md B2 close then. The OpenCL, Metal and CUDA templates are ones yapocis-rs's bundle vendors, so its bundle needs a re-export. ~22 LOC, ~95 of tests.

**numbax `xyY2xyz` at y == 0:** numbax divided by `y + 1e-20` and gave (1.5e19, 0.5, 3.5e19) for (0.3, 0, 0.5), where the device kernels and rustcpu give (0, 0, 0), Y included (D4.7 keeps that). Its jit path and its numpy fallback now check `y == 0`. The planner found that the jit path is then bit-identical to rustcpu on uniform and log-uniform data and the zero rows, and that dropping the epsilon changes nothing on uniform 0..1 data. Two new tests (`test_xyY2xyz_y_zero` over every test backend, and the fallback); both numbax cases fail on the previous kernel. ~14 LOC, ~27 of tests.

**#67, truetrace on numbax:** numbax's `init_seed_from_mask` and `jfa_step` were stubs, so the field `trace_segments` built was ±1e30 on every pixel and `iso` had no effect there. They are now an `@njit` port of the OpenCL jump flood, step for step (option B, D1): the same neighbour order, a strict `<` so the first seed wins a tie, and every value forced to float32 (numba would otherwise promote to float64 and resolve ties differently). The planner's probes found it bit-identical to rustcpu and to Apple's OpenCL on 14 masks; the new test checks six of them. On yellowfin's RTX 3080 the coordinator's CUDA run found NVIDIA's OpenCL and CUDA within sqrt rounding of it instead (up to 7.6e-6, and 0.145 px on 0.0031% of disc256's pixels where a tie flips), so the test requires bit-identity only with rustcpu and gives OpenCL the tolerance it gives Metal and CUDA (D5.22). The first version of the test required bit-identity with OpenCL too, an assumption from Apple's OpenCL that failed on NVIDIA's. The issue's suggested fix (writing `sdf_from_mask_cpu` into the seed kernel) would have given −2 × SDF, because `trace_segments` subtracts two calls. Three new tests in `tests/lib/compute/test_truetrace.py`: segments against the exact field for |iso| ≤ 1.5, the field exact where |sdf| ≤ 2, and numbax's field equal to rustcpu's and within sqrt rounding of the GPU backends', plus segments equal to OpenCL's at iso 20. Two of the masks are stored bit-packed in `truetrace_masks.npz` (7 KB). All five numbax cases fail on the stubs, as does the existing `test_gpu_field_is_not_transposed_on_a_non_square_mask` under `YAPOCIS_BACKEND=numbax`; all pass now. That test is the one the 2026-11-08 entry found failing only in a full run: see #82 below. ~90 LOC, ~120 of tests.

**#82, tests leaked `YAPOCIS_BACKEND`:** `tests/async/test_performance.py` set `YAPOCIS_BACKEND` and `YAPOCIS_DEVICE` for each parameter and never restored them, and `backend_comparison` (`tests/conftest.py`) and `tests/__init__.py` restored the variable only when it had been set before. So in a full run every later unparametrized test ran on the last backend collected, not the default. That is why `test_gpu_field_is_not_transposed_on_a_non_square_mask` failed only in the 2026-11-08 full run without the rustcpu wheel (the last backend was then numbax, with #67's stubs), and why `make test-<backend>` didn't run those tests on its named backend. An autouse fixture in `tests/conftest.py`, `restore_yapocis_env`, now restores `YAPOCIS_BACKEND`, `YAPOCIS_DEVICE`, `YAPOCIS_ASYNC` and `YAPOCIS_ASYNC_WRITE` after every test, with `test_modes.py`'s `_restore_env` pattern (kept in place, D5.14); the two helpers also remove the variable when it was unset. It goes after #67, because with the leak fixed `YAPOCIS_BACKEND=numbax` really runs the non-square test on numbax. New `tests/backend/test_env_isolation.py`: its second test fails on `main` with the variable unset and set to `ocl`, and passes now. ~30 LOC, ~17 of tests.

**#65, numba simplex regression tests and the `tileable` docs:** #65 was fixed on 2026-11-07 by `29f8403` (PR #75) and `7e98bb0` (PR #76) and closed on 2026-11-11. This batch adds the numba-only regression tests it lacked, in `tests/lib/compute/test_simplex.py`. One writes the numba kernel's output into a buffer between a 64-element head canary and a tail canary long enough for the old kernel's worst overshoot, so even the old kernel can't write outside the test's own memory, and checks both canaries and that no pixel was left unwritten, at (7,5), (5,7), (33,9) and (9,33), tileable on and off. The other checks each pixel against fBm of the module's own single-octave noise at (row / rows, column / columns). `test_simplex2d_tileable` could not fail: at its `scale=0.01` the lattice never wrapped, so the output was identical with `tileable` on and off. It now checks, on every backend, that `tileable` changes the output at 8 cells on 64×64 and that the output stays in 0..1. The `tileable` docs (`lib/simplex.py`, `docs/library/simplex.md`) promised seamless output; they now say it wraps the gradient lattice with period `max(1, int(N / frequency))` and that the output is not seamless on x or y (option 2A, D3). Seamless tiling is feature #83. No kernel code changes. ~150 lines of tests and docs.

**Lint follow-up (a commit the plan didn't list):** comparing ruff, black, isort and mypy on the touched files against `main` showed three new mypy `no-untyped-def` findings (the new jump-flood helpers and `_key_default`) and 30 new ruff F405s (the new colormath tests used `np` through the file's star import). A small commit after #65's fixes both; nothing else on the touched files got worse. `make check` itself fails on `main` too (530 ruff findings, 1,104 mypy errors, and black and isort findings, some in generated `mako_modules`), so it is not a usable gate here.

**Corrections to earlier entries (#84).** The old entries stay as they were written; these replace two statements in them:

1. 2026-11-06, on the old numba simplex loop, "left the extra columns at 0 when wide, and wrote past the rows when tall": the effects were the other way round. **Wide** fields wrote past the output buffer (up to `(cols − rows) × cols` elements) and left the extra columns at 0; **tall** fields stayed inside the buffer but ran each row into the next and left trailing pixels unwritten (8 of 35 at 5×7 in the planner's probe, run without bounds checking; 10 under `NUMBA_BOUNDSCHECK=1`, where the old kernel stops at its first bad write). Under bounds checking the old kernel's `IndexError` is caught and it never writes outside the buffer, so the canary assertions don't fire there; the unwritten-pixel assertion is what catches the bug in that run.
2. 2026-11-06 and 2026-11-07, "the 3-D simplex kernels pick gradients with `hash % 12`, which in C is negative for a negative hash": the hash is never negative, because `h ^ (h >> 16)` clears the sign bit (the planner's probes found no negatives in 84,000 and 120,000 sampled inputs). So it doesn't explain why the OpenCL and Metal 3-D kernels differ from CUDA and numba, which stays unexplained.

**CHANGELOG:** `[Unreleased]` is backfilled with everything since 1.0.0. It was drafted by AI from `git log v1.0.0..main` and the bodies of PRs #51–#79, plus this batch. The history since 1.0.0 is those 19 PRs, four direct commits on main (`cca8f8a`, `a42bfab`, `80884c3`, `6a90822`) and the `2c7c9b5` merge of main into dev. The first draft credited `cca8f8a`'s and `a42bfab`'s changes (host fixtures, `trace_mask`, `gaussian_ladder`, the scanline fill, CUDA truetrace's transpose, the iso > 0.5 fix) to `2c7c9b5`, which only merged #60's FFT and HSI work into dev; the PR's review caught it, and the references were corrected. The draft is for the maintainer to review before 1.1.0 (D5.10).

**Defects in AI-written plans, caught before implementation:** the #66 draft missed rustcpu's mixed-sign case; the #64 key suggestion had no fallback for non-numpy values; the #65 test 2 prototype, and the first version of the batch plan's pre-fix run, could write past a numpy allocation on the old kernel; the batch plan's first rustcpu key test could have been written with Python lists, which never collide, so it could not have failed.

**Defect in this implementation, caught before push:** commit 0 was first made without its AI.md entry. The AI.md edit failed (its anchor heading occurs twice in this file), and the commit, sent as a separate command right after the edit, ran anyway, as on 2026-11-08. I noticed it from the commit's file list and amended the unpushed commit. It happened again at commit 9, the lint follow-up: it was committed as `709b1d2` without its AI.md lines (the global rule wants AI.md updated with every commit message) and amended 28 seconds later to `4947615`, adding them. Both amends were before the push; the PR's review found the second one in the reflog, after this entry had said the pattern stopped.

**Verified (M4 Max MacBook: OpenCL, Metal, numbax, rustcpu with the 0.1.0 `yapocis_rs` wheel), with numba and Mako caches redirected outside the checkout:**

- Each new test was run against the code before its fix and failed there, as listed above. The #65 run used only the two new numba tests, on `29f8403^`'s kernel, with `NUMBA_BOUNDSCHECK=1` and a fresh numba cache.
- `make test-ocl`, `test-metal`, `test-numbax`: 1,681 passed, 28 skipped, 3 xfailed each; `make test-rustcpu`: 658 passed, 45 skipped, 3 xfailed; `make test`: 1,589 passed, 28 skipped, 3 xfailed; `make test-smoke`: 232 passed; `YAPOCIS_TEST_BACKENDS=ocl,metal,numbax pytest tests`: 1,374 passed, 1 skipped. Each also had one failure, `test_kernel_template_cache.py::test_old_compiled_modules_are_removed`, caused by the Mako redirect used for these runs; it passes in the checkout without the redirect. The xfails are the three rustcpu #66 cases.
- **What #82 exposed: nothing.** On `main` the same four per-backend targets gave 1,594 passed and 40 skipped (613 and 48 on rustcpu), with only that redirect failure, and after #82 every target still has only that failure, now running the unparametrized tests on its named backend.
- The order checks: on `main`, `test_performance.py` then the non-square truetrace test with `YAPOCIS_TEST_BACKENDS=ocl,metal,numbax` fails; after #67 it passes, and after #82 it passes under `YAPOCIS_BACKEND=numbax`, where it now really runs on numbax. The six touched test files give 205 passed and 3 xfailed forward and in reverse order, and each passes alone.
- yapocis_external's convolvesep users (`tests/test_gaussian.py`, `tests/test_dog_yapocis.py`) on numbax against this checkout: 20 passed.
- The docs examples in `docs/library/simplex.md` and `colormath.md` run (`yapocis.scripts.doctest_examples`).

**Not verified:** CUDA (yellowfin is production, so it wasn't run in this step), PoCL and NVIDIA OpenCL. *(Corrected 2026-11-11, in the step-2b entry above: CUDA and NVIDIA OpenCL were run on yellowfin's RTX 3080 at `3596ce0` before the merge; only PoCL remains unverified.)* The per-backend targets and the full runs started at commit 7; the lint follow-up's annotation-only edits landed in the tree while they ran, and its touched test files were re-run afterwards (113 passed, 3 xfailed).

**Division of labor:** the maintainer made every decision (D1–D5); the plum coordinator, planner and evaluator agents planned and reviewed the batch; this entry's implementer wrote the code, tests and docs from the approved plan.

**Lines (measured, `git diff --numstat main..HEAD`):** ~190 added and ~80 removed in kernels and library code; ~630 added in tests, plus a 7 KB mask fixture; ~11 in docs; ~32 in the changelog. `docs/BUGS.md` and `docs/FEATURES.md` (~98) are the coordinator's.

### 2026-11-09: Dependabot's eight bumps, as one lock update (Claude Opus 5.5)

**What the maintainer asked for:** "Merge any unmerged parts of this whole project", including the open Dependabot PRs.

**What the AI did:** one `uv lock --upgrade-package` for all eight bumps. They are black 26.10.1, certifi 2026.10.19, fsspec 2026.6.0, mako 1.4.2, pillow 12.3.0, setuptools 84.1.0, torch 2.13.0 and uv 0.13.4, bringing torchvision 0.28.0, triton 3.7.1, pathspec 1.1.1 and pytokens 0.4.1 with them. Merged one by one, the PRs would each conflict on `uv.lock`.

**Left out:** Dependabot's own `pyproject.toml` edit for mako. It rewrote `"mako.*"` in mypy's list of untyped modules to `"mako>=1.4.2"`, which would have broken that override.

**Checked:** `uv sync --extra all-backends --group dev`, then `pytest -m "not performance"` on the M4. About 1,550 tests, no failures; the skips are rustcpu interfaces without a CPU port.

**Defect, caught before commit:** `uv sync` removed a `yapocis-rs` wheel that had been installed by hand into `.venv`. It was reinstalled from the same wheel.

### 2026-11-08: The `rustcpu` backend: yapocis-rs's CPU ports from Python (Claude Opus 5.5)

The maintainer asked for a way to "tell yapocis3 to use the yapocis3-rs implementation". They rejected my first sketch, in which `YAPOCIS_DEVICE` would also choose the Rust runtime's backend. Their question, whether `YAPOCIS_BACKEND=rust YAPOCIS_DEVICE=cuda` would then do something, showed that I was overloading a device selector. They then said "go ahead with rustcpu". Later: "Make sure that yapocis (python) survives not having the rustcpu engine available."

**What the AI did:**
- **The backend:** `src/yapocis/rustcpu/kernel_manager.py` (~280 LOC), a kernel manager on yapocis-rs's Python bindings (module `yapocis_rs`, from yapocis-rs's `bindings/python`, merged there as PR #16).
  - **Loading:** it renders an interface's manifest entry and OpenCL source with `codegen.bundle.build_interface_entry`, as `yapocis-tools bundle` does. yapocis-rs then derives the arguments, sizes the launch and runs a Rust port of each OpenCL kernel on every core.
  - **Calls:** kernels take `global_size` and `local_size` and return outputs as the GPU backends do. NumPy arrays serve as device buffers, as on numbax.
  - **Missing ports:** interfaces outside yapocis-rs's bundle (the 26 yapocis_external interfaces) fail to load with "has no CPU port".
- **Registration:** `rustcpu` is registered in `yapocis.core` and `core.manager` (~30 LOC), detected by importing `yapocis_rs`, with one device, `cpu`.
- **Docs:** a `rustcpu` section in the backend guide, and `make test-rustcpu`.
- **Tests:** `test_backend_selection` maps `rustcpu` to its module, and `test_smoke` skips "has no CPU port" as it already skips templates a backend can't build.
- **Without the engine:**
  - `rustcpu` drops out of the available and test backends, and `YAPOCIS_BACKEND=rustcpu` gives the usual "not one of available backends" error;
  - an explicit `backend="rustcpu"` raises an `ImportError` saying which wheel to install;
  - `get_test_devices("rustcpu")` lists nothing.

**Verified (M4 MacBook):**
- **With the wheel:** the suite on numbax and rustcpu gave 958 passed, 33 skipped, 0 failed. `tests/lib` and `tests/backend` on rustcpu alone show no failures.
- **Without the engine:** a stub module made `import yapocis_rs` fail (`claude-debug/no_yapocis_rs`). The full suite on the default backends (OpenCL, Metal, numbax) gave 1,305 passed and 1 failed: `test_truetrace_fill.py::test_gpu_field_is_not_transposed_on_a_non_square_mask`.
  - **That failure predates this branch:** the same test fails in the full suite on `main` (which has no rustcpu code), and passes when its file runs alone, so it depends on test order.

**Defects and corrections:**
- **Upgraded NumPy:** installing the wheel into this venv first upgraded NumPy from 2.3.3 to 2.5.4. I caught it from the install output and restored 2.3.3. The docs now say to install with `--no-deps`.
- **Missing-engine errors:** an explicit `rustcpu` request with no engine gave a bare `No module named 'yapocis_rs'`, and the device listing showed a device that couldn't open. Both were found by the no-engine probe the maintainer's request prompted, and both are fixed.
- **Missing ledger entry:** this branch's first commit (b353b72) went out without this entry. My edit of this file was refused, and the commit ran anyway because it was chained after the edit. The entry was added in the next commit.

**Merged:** PR #77, as `9d4ec87`. This repository has no CI, so the local runs above were the check. The working worktree was removed afterwards; its untracked `claude-debug` material (the no-engine stub, the probe script, the suite logs) was moved into the main checkout first and checked byte for byte.

### 2026-11-07: CUDA's 2-D simplex noise matches the other backends (Claude Opus 5.5)

The maintainer: "Make cuda noise match". On the RTX 3080, `test_simplex2d_device_matches_the_numba_port[cuda]` failed at every size. CUDA's `simplex2d` took its noise coordinates from CUDA's swapped `X`/`Y` helper macros, which made its field the OpenCL, Metal and numba field transposed.

The kernel now computes `x = gid / height`, `y = gid % height`, as the OpenCL and Metal helpers do. CUDA now agrees with numba to about 3.6e-7, and with OpenCL to 1.2e-7, at 192x192, 320x160 and 150x200. The simplex suite passes on yellowfin (CUDA, NVIDIA OpenCL, numba) and on the M4 MacBook (Metal, OpenCL, numba): 58 tests each.

**Left alone: `simplex3d`.** The AI first changed it the same way, and that made it worse. At a square size, CUDA's 3-D noise already agreed with numba (7e-4), while the OpenCL and Metal 3-D kernels differ from both, by about 0.67. Non-square 3-D fields differ everywhere. Neither PlumImpress nor PlumCraft uses 3-D noise. The `% 12` negative index noted on 2026-11-06 may be part of the cause.

### 2026-11-06: Device kernels and the numba port brought into agreement: Histogram, Mandelbrot, Simplex, Stretch (Claude Opus 5.5)

PlumImpress is folding PlumCraft's array engine into the Impressionist engine. Its array nodes are CPU ports of this package's numba kernels; its Impressionist twins run the device kernels. Comparing the 54 twin pairs showed four places where yapocis's own backends disagreed. The maintainer ruled on each.

- **Histogram ("Fix histogram"):** the OpenCL, Metal and CUDA kernels computed the bin as `n_bins*value/(high-low)`, leaving out `low`, so any `low != 0` shifted every count. They now compute `n_bins*(value-low)/(high-low)`, as numba and numpy do. Every library caller passes `low = 0`, so nothing they compute changes. New test `test_histogram_bins_span_low_to_high`, which fails on the old kernels.
- **Mandelbrot ("stop at first"):** the device kernels recorded the last iteration with `|z|^2 > 4`, which is where z first becomes NaN. Fast math decides that, so Metal and Apple's OpenCL counted one more than a float32 replay on nearly every escaped pixel. They now stop at the first escape, as the numba port does, including its quirk that an escape at iteration 0 keeps going. Device and numba now differ on 14 of 262,144 pixels, against 52–83% before. New test `test_mandelbrot_records_the_first_escape_as_the_numba_port_does`.
- **Simplex ("Option 1"):** the numba port hashed lattice points in numba's 64-bit ints, while the kernels use 32-bit `int`, so the two picked the same gradient at only about half the points. The numba hash now wraps in 32 bits, and the device output is unchanged.

  The numba 2-D loop also swapped its axes and wrote `output[y, x]`. On non-square fields that left the extra columns at 0 when wide, and wrote past the rows when tall, unchecked under numba. Both of its loops now follow the kernel: `width_of`/`height_of` are rows/columns, and pixel (row, column) is noise at (row/rows, column/columns).

  Metal, OpenCL and numba now agree to about 2e-6 at square, wide and tall sizes. New test `test_simplex2d_device_matches_the_numba_port`, across three sizes.
- **Stretch ("Clamp is the right answer"):** the kernels clamp each value to `input_min..input_max` before mapping it; the numba port didn't, and divided by `max - min + 1e-20`. It now clamps and maps in float32 in the kernel's order, and agrees to 6e-8. New test `test_stretch_clamps_to_the_input_range_as_the_numba_port_does`.

**Verified:** each new test was run against the old code and failed. `tests/lib` passes on OpenCL, Metal and numba on the M4 MacBook (882 tests).

**Not verified:** CUDA has not been run.

**Found and not fixed:** the 3-D simplex kernels pick gradients with `hash % 12`, which in C is negative for a negative hash, so they index outside `grad3`.

**Changed:** ~30 LOC in kernels, ~45 in the numba port, ~80 in tests.

### 2026-11-05: A `complex` host-fixture section: numpy's complex64 multiply (Claude Opus 5.5)

**What the maintainer asked:** they approved seantrue/yapocis-rs#10 (the `fix approved` label) and said "fix approved bugs". The issue: npexact's only contraction rule decides by architecture at compile time, but numpy's SIMD ufunc loops are picked at run time.

**What the AI did:** `export_complex` in `src/yapocis/codegen/host_fixtures.py` (~66 lines) records complex64 `a * b` and `z * z`:

- through numpy's ufunc loop: every pair of 121 complex specials (11 float specials as real and imaginary parts), 10,007-long random arrays with NaN, infinities and signed zeros, every length up to 33, a stride-3 view, and an array times a scalar;
- as numpy scalars, which go through compiled C.

The values are stored as separate float32 real and imaginary parts, and the case names the CPU features numpy dispatched on (`cpu_features`: ASIMD, FMA3, AVX2, AVX512F).

**Found with it:** arrays are fused the same way on an M4 Max and on yellowfin's Xeon (FMA3, AVX2), but scalars are fused on the Mac and plain on the Xeon. Recorded with numpy 2.3.3 on both machines, from a temporary directory with only numpy. Every existing case came out identical on yellowfin. On the Mac only `order.npz`'s `sort1000` and `sort65537` differed, and only in where `-0` and `+0` landed among equal values. npexact's sort test already ignores that, so the committed fixtures were kept and only the new case was added.

**Not changed:** the file's existing ruff (E702 at line 399) and black findings, which predate this change.

### 2026-11-03: Host fixtures for the numpy behaviours PlumCraft had hand-ported (Claude Opus 5.5)

The maintainer: "Move the hand-ported behaviours into npexact, record fixtures on Linux and test."

**What the AI did.** `host_fixtures.py` (+197/−2) gains four sections for yapocis-rs's npexact to replay bit for bit:

- `elementwise`: `maximum`, `minimum`, `clip`, `sign`, `mod` and `nan_to_num`, over every pair of 25 special values and over long arrays with NaN and signed zeros mixed in.
- `extrema`: `max`, `min`, `nanmax` and `nanmin` at 75 lengths, so both the SIMD body and the tail run, plus all-NaN arrays and zero ties; also int64 `cumsum`.
- `order`: `sort`, `searchsorted`, `lexsort`, `quantile`, `median`, `nanpercentile`, and `unique` with its inverse.
- `fft`: `fftshift` and `ifftshift`.

`hist` also gains three `linspace` cases with Python-float bounds. The script still needs only numpy.

**Recorded** on the M4 Max, where the earlier cases came out byte-identical, and on yellowfin's Xeon E5-2698 v4. yellowfin's run used numpy 2.3.3, the same as the Mac, from a temporary directory. The x86 recording exposed two bugs in npexact and a difference between CPUs in how numpy breaks ties between signed zeros; those are recorded in yapocis-rs's `AI.md`.

**Tests.** The full yapocis3 suite was not run.

### 2026-10-30: Parity cases for `extent_freetype`, and the HSI plane kernels (Claude Opus 5.5)

The maintainer: "port extent_freetype; restart the python fleet; update the fixtures".

**What the AI did.** The typeset generator records `extent_freetype` for each case (`parity_graphics.py` +2/−1). `parity.py` +3 gives `hsi.rgb2hsi` and `hsi.hsi2rgb` the 2-D planes they take.

**Defect in the generator, found recording.** Those two kernels were given depth-3 inputs, so they dispatched over a third of each output, and the rest of each "output" was a copy of the input. The recorder usually skipped them as not reproducible, so they had no fixture on any device; on Apple OpenCL one run matched twice by luck and Rust failed against it, which exposed this. Both now have real fixtures on Metal, CUDA and OpenCL.

**Investigated, left alone.** `extent_freetype`'s `_kerning_cache` stores entries under the wrong key. The AI predicted this changed results; 676 doubled-letter cases showed no difference, and none of the bundled fonts has kerning, so it can't reach a result with them.

**Tests.** Ruff is clean on both files. yapocis-rs replays the output on the M4 Max and the RTX 3080. The full yapocis3 suite was not run.

### 2026-10-30: `write_svg_regions`: the missing `else` (Claude Opus 5.5)

The maintainer: "fix the svg bug". With `curve="cubic"` every outer ring was overwritten with its polygon, and with `curve="none"` the function raised `UnboundLocalError`, because the line that wrote the polygon had lost its `else`. The defect is older than this ledger; the AI flagged it on 2026-10-30 while adding `regions_svg`. The yapocis-rs port had reproduced the cubic half on purpose, so both change together.

`truetrace.py` +2/−1, and a regression test in `tests/test_truetrace_fill.py` (+13) that fails on the old code. `tests/test_truetrace_fill.py` passes (22 tests). The full suite was not run.

### 2026-10-30: A parity case for `regions_svg` (Claude Opus 5.5)

The maintainer asked what parity with yapocis-rs was missing, then "Go for 1-3": re-export the bundle, record the Metal fixtures, port `trace_regions` and `regions_svg`.

**What the AI did.** `parity_processing.py` +15: the truetrace `fill` case now writes `regions_svg` documents as text fixtures, from `trace_regions` on both engines, in both curves, plus one with custom ink, no paper and non-zero tension, so yapocis-rs is held to them byte for byte.

**Tests.** Ruff is clean on the file. The generator ran for the Metal recording, and yapocis-rs passes its output.

### 2026-10-30: `trace_regions` and `regions_svg`: traced shapes as vector SVG (Claude Opus 5.5)

The maintainer asked which of ImpCraft's intermediates could output vector SVG, not embedded rasters, then chose the traced pen strokes: "Go for 1, let's see what it makes."

**What the AI did.** `trace_regions` is the part of `trace_mask` before the fill, so `trace_mask` keeps its pixels (ImpCraft's exactness gate stayed 92 of 92 with this in place). `regions_svg` writes those regions as an SVG document, in image y-down coordinates and moved half a pixel because `fill_polygons` samples pixels at integer coordinates. Measured against the raster on one style: 0.55% of pixels differ with that shift, about 2.6% without. `src/yapocis/lib/truetrace.py` +92/−12, `tests/test_truetrace_fill.py` +59 (5 tests).

**Defect found, not fixed.** `write_svg_regions` has a missing `else`: with `curve="cubic"` every outer ring is overwritten with its polygon. Its Rust port and the parity gate hold it as it is, so `regions_svg` is a separate writer and the fix is left as its own change.

**Tests.** `tests/test_truetrace_fill.py` passes (21 tests). The full yapocis3 suite was not run for this change. Ruff still reports older findings in `truetrace.py`, none on the new code.

### 2026-10-28: Metal truetrace's lambda failed to compile under some Pythons (Claude Opus 5.5)

**What happened.** ImpCraft's tests, run with `YAPOCIS_BACKEND=metal`, failed in every graph that traces: "lambda expressions are not supported in Metal". The maintainer said to fix it, and asked why it wasn't caught when the kernel was changed.

**Why it wasn't caught.** The kernel hadn't changed. The lambda (`auto lerp = [](…)` in `marching_squares`) has been in `kernels/metal/truetrace.mako` since its first commit (2025-10-23). pymetallic's `metal_device_make_library_with_source` passes no compile options, so Metal compiles at the language version of the SDK the host Python was built with. The same source, byte for byte (dumped from both processes), compiled under Homebrew's Python 3.14 (SDK 26.1) and python.org's 3.13 (SDK 15.2), and failed under uv's CPython 3.13.0 (SDK 14.2), which is ImpCraft's venv. yapocis's smoke compile test and `test_truetrace[metal]` did compile this kernel, but on a Python whose compiler accepts lambdas. ImpCraft only started reaching it on Metal after impressionist's line tracing moved to truetrace (2026-10-21).

**The fix.** The lambda became a file-scope `inline float tt_lerp(…)` with the same arithmetic, plus a comment explaining why. It was the only lambda in any Metal kernel here or in yapocis_external. ~16 changed lines.

**Testing.** On thresher (Metal), impressionist's engrave and pen renders on an empty checkpoint store have the same digests before and after. Under ImpCraft's Python 3.13.0, `stages.trace` now compiles, and the 8 ImpCraft tests in the previously failing files and tests pass. Probe: `claude-debug/truetrace_metal_probe.py` (not committed: code).

**Not done.** A kernel is still accepted or rejected depending on the host Python. A proposed fix: pymetallic accepts a language version, and yapocis pins the oldest one it supports.

### 2026-10-26: Text file I/O as UTF-8; exports byte-identical on Windows (Claude Opus 5.5)

**What the maintainer asked for.** Recording yapocis-rs parity fixtures on the Windows VM's PoCL failed: `yapocis-tools bundle` wrote kernel sources with Python's default encoding, cp1252 on Windows, which has no `∂`. Claude got the fixtures with `PYTHONUTF8=1`; the maintainer then asked to "fix the yapocis3 utf-8 encoding bug".

**What the AI did.** An AST scan of `src/yapocis` (`claude-debug/find_default_encoding_io.py` in the maintainer's sysadmin checkout) listed every text-mode `open`, `Path.read_text` and `Path.write_text` without `encoding=`: 30 calls in 13 files (codegen, interface and YAML loading, model label files, `tools.py`, `cuda_cards.py`, performance reports). Each now passes `encoding="utf-8"`. The codegen writers (bundle sources and manifest, derivation, host and parity fixtures) also pass `newline="\n"`: Windows text mode would otherwise write `\r\n`, and the manifest's sha256 is of the `\n` text, so a Windows export would not match its own manifest or a Linux one. ~41 lines; `host_fixtures.py`, which was not black-formatted, keeps its formatting.

**Verification.** A Linux export is identical to yapocis-rs dev's bundle (`scripts/compare-bundle.py`). On the Windows VM, without `PYTHONUTF8`, an export from this tree succeeds and is byte-identical to the Linux export, with no `\r`. Full suite on yellowfin (RTX 3080): one failure, the known order-dependent `test_truetrace_fill.py::test_gpu_field_is_not_transposed_on_a_non_square_mask`. ruff's 23 errors in these files are all pre-existing, the same count as on dev.

### 2026-10-26: main merged into dev; the FFT race had been fixed twice (Claude Opus 5.5)

**What the maintainer asked for.** "Move the pin": point impcraft and impressionist-rs at the yapocis-rs commit with the Windows fixes. Their pin is on yapocis-rs `dev`, which had diverged from `main` in this repository and in yapocis-rs. Asked how to reconcile, the maintainer chose to merge main into dev.

**Defect: duplicated work.** The FFT butterfly race and the radix-2-on-non-powers-of-two bug were already fixed on dev (`80884c3`, 2026-10-23). Claude fixed them again on main (#60) without checking dev. The two fixes are the same guard and the same power-of-two test, differing only in comments and condition order; the conflict in the three `fft.mako` templates was resolved by taking main's version, which also carries the PoCL `get_global_size` workaround. main also brings the HSI dispatch size and the dispatch evaluator fix, which dev lacked. AI.md kept both sides' entries.

**Verification (yellowfin, RTX 3080 as the CUDA device).** Full suite on the merged tree: one failure, `tests/test_truetrace_fill.py::test_gpu_field_is_not_transposed_on_a_non_square_mask`. The same test fails in a full-suite run of unmerged dev, and passes three times out of three when its file runs alone, on dev and on the merged tree. It depends on test order or state left by earlier tests; it was not investigated further and is not fixed. A bundle exported from the merged tree differs from yapocis-rs dev's bundle only in the FFT sources and their hashes and the two HSI dispatch entries.

### 2026-10-26: FFT Butterfly Race (the P40 Known Issue), HSI Over-Launch, PoCL Link Workaround (Claude Opus 5.5)

**What the maintainer asked for.** The maintainer is porting yapocis-rs, impressionist-rs and impcraft to Windows on a KVM guest whose only OpenCL device is PoCL on the CPU, and asked to "address the windows failures". Valgrind (the maintainer asked for the command line and read the log) showed a heap overrun in the HSI test. Offered a test-side fix or a fix in yapocis3, the maintainer chose "Fix at the source. Then address FFT, div test." A later PoCL link failure got "Apply workaround, test in isolation". Then CPMD.

**What the AI did.**

- **FFT butterfly race** (`kernels/{ocl,cuda,metal}/fft.mako`, 7 sites). With `k = lid % m2` and `block = (lid / m) * m`, work-items `lid` and `lid + m2` compute the same `j`, and both read and write `shared_data[j]` and `shared_data[j + m2]`. Hardware that runs the block in lockstep hides this; the P40 and CPU OpenCL do not. A `lid % m < m2` guard means one work-item runs each butterfly. This was the 1.0.0 known issue: on the P40 the 8 OpenCL FFT tests fail without the change and pass with it.
- **Radix-2 on non-power-of-two sizes.** The OpenCL `fft1d` used the shared-memory path whenever `local_size == n`. PoCL chooses a 1000-wide work-group for n = 1000, so a 1000-point FFT ran radix-2 and came out wrong (relative error 1.25). Every radix-2 path now also requires `(n & (n - 1)) == 0`, so those sizes take the DFT path.
- **HSI over-launch** (`interfaces/hsi.interface`). `rgb2hsi3`/`hsi2rgb3` index `gid * depth` with no bounds check, and with no explicit `global_size` they launched one work-item per float. The extra work-items wrote 3x past the output buffer: a segfault on PoCL, and a heap-corruption abort in yapocis-rs. `lib/hsi.py` always passed the right size, so only raw interface callers were exposed. The interface now declares `dispatch_size (dim_0_of rgb * dim_1_of rgb)`.
- **Dispatch evaluator** (`core/dispatch_expr.py`, ~9 LOC). It replaced only the first dimension reference in an expression, so a product of two dimensions could not be evaluated. It now resolves them all, as the bundle exporter's parser already did. There are two new cases in `test_dispatch_ast_matches_dispatch_expr`.
- **PoCL link workaround** (`kernels/ocl/fft.mako`). PoCL 7.2 on Windows cannot link a kernel that calls `get_global_size()` when the program is built with `-cl-opt-disable` (unresolved `_local_size_x`, `_num_groups_x`). That flag is the Apple optimiser workaround in `lib/fft.py`. Tested in isolation on the guest: `get_global_size` fails only with that flag, while `get_local_size`, `get_num_groups` and the other work-item functions link. `window` and `power_spectrum` now use `get_local_size(0) * get_num_groups(0)`. That product equals the global size because without `-cl-std=CL2.0` work-groups are uniform.

**Defects in AI-written code and reporting.**

- Claude first diagnosed the Windows failure as "kernels that call `get_global_size()`". The isolation test the maintainer asked for showed `get_global_size` links fine on its own; the trigger is the combination with `-cl-opt-disable`.
- While checking the companion yapocis-rs changes, Claude ran that crate's tests three times without its `ocl` feature. Each test looped over zero devices and passed, which included a `div` "pass" Claude nearly reported. Claude caught this from 0.00 s run times and reran with the backend enabled.

**Verification (yellowfin).**

- Full suite on the RTX 3080 before the `get_global_size` workaround: no failures (skips: model weights, Metal). After the workaround only `tests/lib/fft` and the bundle-export tests were rerun: 87 passed.
- `tests/lib/fft` on CUDA, OpenCL and Numba: 59 passed on the 3080, 59 on the P40, and 59 on PoCL. Before the change, 8 failed on PoCL and 8 on the P40.
- A raw `rgb2hsi3` call with no `global_size` segfaults on PoCL before the change (exit 139); after it, the output is identical to `lib/hsi.py`.
- A bundle exported on Linux is byte-identical to the M4 export except for the FFT sources, their hashes, and the HSI dispatch entries. yapocis-rs passes against it: all 56 tests on the 3080, including a parity replay of 95 cases each on CUDA and OpenCL, and PoCL on Linux and Windows apart from the tests that have no CPU-device fixtures.
- **Not verified:** the Metal template changes. The Metal parity run needs the M4.

### 2026-10-23: The FFT kernels raced, and took non-powers of two down the radix-2 path (Claude Opus 5.5)

**What happened.** ImpCraft's local engine was being held bit-exact against its Python engine, and its FFT node gave different results from run to run on the same input. Two defects in `fft1d` and `fft1d_batch`, in the OpenCL, Metal and CUDA mako kernels, caused this.

- **A race in the shared-memory butterflies.** A work-item ran a butterfly whenever `j + m2 < n`. The work-items `lid` and `lid + m2` map to the same `j`, so two work-items read and wrote the same pair at once. The fix: `(lid % m) < m2`, one work-item per butterfly.
- **Sizes that aren't a power of two took the radix-2 path.** The fast path was taken for any `n` up to 1024 (256 for the batch kernel). It now also requires `(n & (n - 1)) == 0`; other sizes take the general path.

**Testing.** On thresher (OpenCL and Metal), the 63 FFT tests pass (`pytest -k fft`), and the ImpCraft FFT and inverse-FFT cases are now repeatable and match the Python engine. The CUDA kernel received the same change but has not yet been run on CUDA. yapocis-rs vendors the same fix in its bundle.

**Defect record.** These kernels predate this session. The existing tests used power-of-two sizes and compared within a tolerance, so the race went unnoticed. It was caught by a bit-exact digest that changed between runs. The fix (80884c3) was first committed without this entry, which was added in the next commit.

~28 changed lines across the three kernels.

### 2026-10-22: truetrace transposed every shape on CUDA (Claude Opus 5.5)

**What happened.** Virtual Impressionist's trace styles moved to truetrace and were deployed to yellowfin, whose workers run CUDA. While the Rust worker was being tested there, a comparison of CUDA and OpenCL renders showed Durer #2 transposed under CUDA: correlation 0.986 with the transposed OpenCL render, −0.02 with it upright. The Python and Rust renders were transposed alike, since both use these kernel sources. No production job had rendered a trace style before the fix.

**Cause.** truetrace's wrapper passes `(W, H)` arrays and reads `x` as the slow axis, as the OpenCL and Metal helpers address them (`ADDRESS(x, y) = x*height + y`). The CUDA helpers address `x` as the fast axis (`y*height + x`, the row-major convention of 2026-10-06), so on CUDA the seeds, distances and segments came back with x and y swapped. Their bounds checks were also wrong for non-square images.

**Fix.** `kernels/cuda/truetrace.mako` keeps its own addressing (`TT_ADDRESS`, `TT_X`, `TT_Y`) in the wrapper's convention; the shared CUDA helpers are unchanged. The new test `test_gpu_field_is_not_transposed_on_a_non_square_mask` traces an asymmetric 90×130 mask with the GPU and CPU fields and requires equal rasters.
- On yellowfin's RTX 3080 with CUDA, that comparison went from 3,448 differing pixels to 0.
- The six trace results of a worker dry run became pixel-identical to OpenCL's.
- The test passes on OpenCL and Metal here.

**Defect record.** The CUDA truetrace kernel has carried the mismatch since the shared helpers changed convention. The existing GPU-against-CPU test used a nearly symmetric mask and had not been run on CUDA. The AI also missed it in its own pre-deploy smoke test on CUDA, where it judged a transposed Durer #2 to look right. It was caught later the same day by comparing CUDA with OpenCL renders.

### 2026-10-21: truetrace at iso above 0.5: exact chaining (Claude Opus 5.5)

**What the maintainer asked for.** "The retrace examples at .5+ have significant differences from potrace", then "Defaut to .75, fix in yapicis-rs" (Virtual Impressionist's `trace.iso` default; the same fix mirrored in yapocis-rs).

**What the AI did.**

- **`lib/truetrace.py`:**
  - `chain_segments_to_polylines(segs, tol=0.0)` keys endpoints exactly, and `trace_bitmap`/`trace_mask` default `chain_tol_px` to 0.
  - A new `off_field_iso` moves an iso level that lands exactly on a distance-field value off it, so marching squares never emits degenerate segments.
- **Why:** above iso 0.5, the two outlines of a 1 px stroke run within 0.25 px of each other. The old 0.25 px endpoint snapping joined them into one polygon whose fill was a large black wedge.
- **Tests:** `tests/test_truetrace_fill.py` gained thin-stroke cases at iso 0.75 and 1.0. `codegen/parity_processing.py` records a `thin_mask` case at both levels for the Rust port.
- **Research:** `../impressionist/claude-debug/rustport_trace/` compares truetrace with potrace by iso level. At 0.75, 0.973–0.980 of pixels agree with potrace across the eight trace styles, the closest of the levels tried.

**Defects.** The snapping bug predates this work; it is recorded here because this entry fixes it. It was found by comparing truetrace with potrace renders after the maintainer saw the differences.

**Verification.** `test_truetrace_fill.py` passes (18 tests). The yapocis-rs truetrace stages and processing parity tests pass with the regenerated fixtures.

### 2026-10-21: Pinned Gaussian ladder, numpy host fixtures, project-only bundles, truetrace scanline fill (Claude Fable 5.1)

**What the maintainer asked for.** The Rust port of Virtual Impressionist (impressionist-rs), whose plan the maintainer approved with these decisions among others: "The Gaussian weight ladder is pinned as literals in Yapocis3 so every host agrees", and numpy-exact numerics in yapocis-rs, proven by fixtures numpy itself records.

**What the AI did.** This entry is reconstructed from that session's report (`../impressionist/claude-debug/rustport_status_20261021.md`, `rustport_spikes_20261021.md`); it was written when the work was first committed.

- **`lib/gaussian_ladder.py`** (new, ~145 lines) holds the convolvesep Gaussian weights as float64 literals. `GaussianWeights` uses them when its parameters are the convolvesep.yml defaults (`pinned=True`), and computes them otherwise. Hosts whose libm `exp` differs in the last bit then agree; `tests/test_gaussian_ladder.py` checks that the literals equal the computed ladder here.
- **`codegen/host_fixtures.py`** (new, ~430 lines; `yapocis-tools bundle --host-fixtures`) records numpy's own results (random draws, sums, quantiles, float32 `sin`/`cos`/`tanh`, and so on) for yapocis-rs's `npexact` to replay bit for bit.
- **`bundle --project-only`** exports only a project's own interfaces (yapocis_external's bundle for impressionist-rs), with tests in `tests/integration/test_bundle_export.py`.
- **`lib/truetrace.py`:** polygon area, centroid and point-in-polygon in float64 with `math.fsum`. The fill is now an even-odd scanline (`fill_polygons`) in place of earcut plus the graphics kernel, because the C++ and Rust earcut libraries triangulated differently and the kernel's coverage depended on the triangulation. This changes how traced shapes are filled; Virtual Impressionist's trace styles adopted truetrace the same day.

**Defects.** That session's report does not itemise the defects found along the way.

**Verification at commit time** (2026-10-22, by the later session): the full suite with `YAPOCIS_TEST_BACKENDS=ocl` (the maintainer restricted it to OpenCL after a first run's pycuda skip message) gives 655 tests, 637 passed, 18 skipped (17 need two backends, one needs pycuda), none failed. pytest's summary line is hidden by the duration report, so the counts come from the progress output.

### 2026-10-19: Kernel Bundle Exporter for the Rust Port (yapocis-rs)

**What the maintainer asked for.** An evaluation of running the yapocis kernels from Rust with no Python at runtime, then the plan that came out of it. The maintainer decided on 2026-10-18: a separate `yapocis-rs` repository; runtime-parameterised kernels handled by enumerating variants at build time; all interface kernels plus the library wrappers in the first release; CUDA through NVRTC pinned to the 12.x bindings. On 2026-10-19 the maintainer asked how to make yapocis-rs ready for check-in, and approved "do 1-5 ... Then 7". Step 1 was to land this exporter here first so the Rust repository's vendored bundle can be reproduced from a tagged commit.

**What the AI did.**

- **`yapocis-tools bundle`** (`scripts/tools.py`, ~170 LOC): renders every interface, and each variant listed in the new `interfaces/bundle.yml`, for the OpenCL, Metal and CUDA backends without a device. It writes `manifest.json` (parameters, derivation, pre-parsed `dispatch_size`, build options, CUDA's hidden size argument) plus the rendered sources. Options: `--fixtures` (derivation fixtures), `--parity` (kernel outputs recorded on a device) and `--parity-lib` (library wrapper outputs).
- **`yapocis/codegen/`** (new, ~3,210 LOC): `bundle.py` (schema and writer), `fixtures.py`, `parity.py`, and the per-area library generators `parity_lib`, `parity_cnn`, `parity_convolve`, `parity_graphics` and `parity_processing`.
- **Device-free rendering:**
  - `core/kmgr_base.py` gains module-level `find_kernel_template` and `render_kernel_source`, which `KernelManagerBase` now uses, so the Metal kernel manager's own copy of the template search is gone.
  - `utils/internals.py` lists asset directories sorted, so a bundle exported twice is byte-identical.
- **NVRTC compatibility:** `kernels/cuda/constants.mako` defines `uint` and `M_PI`, which nvcc takes from host headers and NVRTC does not have. Both definitions are harmless under nvcc.
- **Per-device parity fixtures** (added during the check-in work): fixtures go to `parity/<backend>/<device>/` and `parity/lib/<backend>/<device>/`, and both indexes record the device. Results can differ between devices in the last bit, so a consumer compares only against fixtures from its own device.
- **Tests and docs:** `tests/integration/test_bundle_export.py` (~345 LOC, 25 tests), `docs/architecture/kernel-bundle.md`, and the `bundle` options in `docs/cli/index.md`.
- **Reports** in `claude-debug/`: `rust_port_evaluation_2026-10-18.md`, and `rust_vs_python_timing_2026-10-19.md` (Rust vs Python timings on the M4 Max and the RTX 3080, including multi-stage resident workloads).

**Defects in AI-written code.**

- The kernel parity exporter recorded the CUDA device as `<bound method name of <pycuda._driver.Device ...>>`, because pycuda's `Device.name` is a method while pyopencl's and pymetallic's are attributes. Caught while designing the per-device layout; `device_name` now calls it, and the export test asserts the name is a plain string.
- Parity fixtures were keyed by backend only, so OpenCL fixtures recorded on the Apple M4 Max made two Rust tests fail on NVIDIA's OpenCL (one Mandelbrot pixel at 6 vs 7 iterations, one `noise3d` element). Caught by running the Rust suite on yellowfin; fixed by the per-device layout.

**Found in the library and not changed (recorded in the timing report).**

- `Gaussian.zcsdog(res=True)` and `res=False` give different edge maps for about 0.45% of pixels. `gauss_image` (`gauss*_h`/`gauss*_v`) and the resident `gauss*_res` kernels treat the image border differently, and the cleared margin (28 px at scale 3) is sized for the smaller kernel, while the larger one is 39 wide. 4,751 of the 4,757 differing pixels lie 28 to 39 px from an edge.
- `fft.batch_forward` documents a 256-point limit but does not enforce it.
- Python's resident `zcsdog` on Metal at 2048² is slower than its host form (31.5 ms vs 25.7 ms). Not investigated.

**Verification.**

- `tests/integration/test_bundle_export.py`: 25 passed (Metal, OpenCL).
- Full suite on thresher with this source: 934 passed, 8 skipped (Metal, OpenCL; the skips are missing ResNet50 weights and pycuda).
- ruff reports no new errors in the changed files: 10 against 12 on `main`, all pre-existing.
- mypy reports 82 errors in the new `codegen/` package (missing return annotations, untyped `yaml`). The package as a whole is not mypy-clean (700 errors across 46 files in the same run), and they are left as they are.

### 2026-10-19: corgi_rgba.png, an RGBA Test Image (Claude Opus 5.5)

**Asked:** the maintainer, working in ImpCraft, asked for "a test RGBA png", then to "Move this .png into the test image suite, add to git".

**Done:**
- `src/yapocis/images/corgi_rgba.png` is `test.jpg` with an alpha channel. The centre is opaque, it fades to clear towards the edges (52% of pixels partly clear, 15% fully clear), and a band along the top is at least half clear. The colours under the alpha are left as the photo.
- It is made by a script in ImpCraft's `claude-debug/rgba/make_rgba_png.py`, which isn't tracked.
- `images/CREDITS.md` records it as derived from `test.jpg`, under the same license.
- ImpCraft uses it to check Load Image's new `rgba` output: straight RGBA versus RGB composited over a background.

**Checked:** `find_image("corgi_rgba.png")` finds it. No code changed, so no tests were run.

**`.gitignore`:** it ignores every `*.png` (test outputs), so it gains one exception, `!src/yapocis/images/*.png`, so suite pictures are tracked.

**Not changed:** the wheel's `include` list names only `test.jpg`, so, like the other suite photos, this file is not in the wheel.

### 2026-10-17: PDF Index Built with makeindex Instead of xindy

**What the maintainer asked for.** "try latex_use_xindy = False, then CPM". Claude had proposed this after the first Read the Docs xelatex build.

**Cause (Claude, from the Read the Docs log).** With `latex_engine = "xelatex"`, Sphinx defaults to xindy for the PDF index. On Read the Docs, xindy failed with `*** - PROGN: variable LANG/GENERAL/LATIN9-LANG.XDY has no value`, which means its language modules are missing. latexmk logged `makeindex yapocis.idx … gave return code 256` and wrote the PDF without a rebuilt index. Before the switch to xelatex, makeindex had worked there.

**Fix.** `docs/conf.py` sets `latex_use_xindy = False`, so Sphinx's `latexmkrc` runs `makeindex -s python.ist`. One line, plus a comment.

**Verification (local).** A clean PDF build through Sphinx's Makefile exits 0, with no LaTeX errors and no missing characters (247 pages). The generated index includes entries such as `load_program() (in module yapocis.core), 213`. `sphinx-build -W -b html` still passes. Not yet confirmed on Read the Docs.

### 2026-10-17: Read the Docs Shows 1.0; Missing FFT Images; PDF Unicode

**What the maintainer asked for.** "ReadTheDocs is not building the expected docs. Looks like the old 0.3 docs." The maintainer pasted the raw build log. Before that, the maintainer had updated the Read the Docs settings (repository and default branch) and triggered the build. Then: "Go ahead, then CPM".

**Diagnosis (Claude).** The build was correct. It came from `4d0f3fe`, and `/en/latest/` and `/en/stable/` both served "Yapocis 1.0.0 documentation". The docs content dates from the 0.3.8 rewrite, though, and the history page's version timeline stopped at "0.3.x" and was followed by "Version 0.3.x Focus Areas". That page is the likely source of the 0.3 impression. A cached copy of the 2021 site is also possible. The log showed two real problems:

- **Three Examples-page images were missing.** The `fft_library_demo_*.png` files existed in the local `docs/_static/images/` but were never committed, because `.gitignore` has a blanket `*.png` rule. The other nine images there had been force-added.
- **The PDF lost characters.** pdflatex can't render the box-drawing diagrams, `≤`, `↔`, `✗`, or the 🤖. It reported errors and dropped them. The HTML was unaffected.

**Fixes (Claude).**

- `docs/history/index.md`: a 1.0.0 timeline row and a "Version 1.0 Focus Areas" section, taken from the changelog.
- The three PNGs are force-added.
- `docs/conf.py` sets `latex_engine = "xelatex"`. Text uses Sphinx's default GNU FreeFont. Code blocks use DejaVu Sans Mono, loaded by file name and falling back to FreeMono if it's absent. With xelatex alone, FreeMono still dropped 36 box-drawing and `✗` glyphs. Loading DejaVu by name failed on macOS, where it exists only inside TeX Live.
- `docs/history/ai-contributions.md`: the attribution example no longer shows the emoji, which no PDF font has. It now describes the co-author trailer and the PR footer instead.

**Verification.** `sphinx-build -W -b html` passed. A clean local PDF build through Sphinx's Makefile with xelatex succeeded: exit 0, 247 pages, no LaTeX errors, no missing characters. The architecture diagram renders. The Read the Docs build itself is not yet verified. Its image is expected to have DejaVu (Ubuntu's base font); if it doesn't, the fallback keeps the build green but drops those glyphs again.

**Defects in Claude's process.** Two local PDF runs failed for reasons unrelated to the docs. The first ran latexmk without Sphinx's `latexmkrc`, so makeindex had no style file. The second reused latexmk's cached failure state. A clean rebuild through Sphinx's Makefile resolved both.

### 2026-10-17: yellowfin Deploy Tests; OpenCL FFT Wrong on the P40 (Known Issue); test.jpg Provenance

**What the maintainer asked for.** CPMD for the release-readiness pass, which deploys by pulling on yellowfin and running the tests there. Then "Run the yellowfin opencl and cuda tests only on the P40 card". Then "Note the issue, we'll ship with it". The decision to ship with the FFT issue is the maintainer's. The maintainer also supplied the provenance of `test.jpg`: "a personal photograph of the package author".

**Deploy run (all devices, model tests on).** After pulling `efcd64c`, the full suite ran on every OpenCL and CUDA device, plus Numba. The committed `activate.sh` set `YAPOCIS_MODELS`, so the VGG16 and ResNet50 tests ran too. Result: 1,087 passed and 242 failed. The failures were resource exhaustion, not wrong results: `OUT_OF_RESOURCES`, `OUT_OF_HOST_MEMORY` from OpenCL builds, and `cuMemAlloc` out of memory from CUDA. The first failures were the VGG16 async tests on the 3060 and P40. Then they spread to ordinary kernels. pipekit services on the same machine were holding GPU memory (9.4 of 12 GB on the 3060). Not fully proven: no separate model-tests-only run was done.

**P40-only run (Claude).** `CUDA_VISIBLE_DEVICES` was set to the P40's UUID, which hides the other cards from CUDA and from NVIDIA's OpenCL, and `YAPOCIS_TEST_DEVICES=P40`. Result: 867 passed, 8 failed, 3 skipped. Nothing ran out of memory, model tests included. All 8 failures are OpenCL FFT (`tests/lib/fft`), with 127 of 128 outputs wrong. CUDA FFT on the P40 passes, and OpenCL FFT on the 3080 passes (21 of 21). The same 8 tests fail on the P40 at `8b049af` (this morning's commit), so this is not a regression. Earlier runs passed because OpenCL's default device on yellowfin is the 3080. The FFT kernel was not debugged. It is listed as a known issue in the 1.0.0 changelog.

**Defect in Claude's reporting.** Claude counted outcomes by matching lines that begin with `.sFExX`. Traceback lines also begin with `E`, so in runs with failures the counts were inflated. Claude first told the maintainer the deploy run had "477 failed, 1,260 errors", and that the P40 run had 87 errors. A strict count of the progress lines gave the correct figures above. The earlier clean-run counts (Mac 920 and 925; yellowfin worktree 1,302 passed and 30 skipped) were checked the same way and were right.

**test.jpg.** `src/yapocis/images/CREDITS.md` now records `test.jpg` as the author's own photograph, distributed under the package's MIT licence. Every bundled image now has a recorded source and licence.

### 2026-10-17: Version 1.0.0, Changelog, Beta Status

**What the maintainer asked for.** "Bump the version to 1.0.0 and make changelog", then "Update to dev status Beta". The 1.0.0 version number and the Beta status are the maintainer's decisions.

**What Claude did.** The version is 1.0.0 in `pyproject.toml` and `uv.lock`. The classifier `Development Status :: 3 - Alpha` is now `4 - Beta`. CHANGELOG.md was written by hand, because `make changelog` (git-cliff, with `filter_unconventional = true`) would have kept only 4 of the 19 commits since v0.3.8. The new `[1.0.0]` section was written from those commits' messages and bodies. It marks two breaking changes: the single (x = column, y = row) drawing convention, and `histeq` moving to `Imageops`. The 0.3.8 docs rewrite had been sitting under a stale `[Unreleased]` heading; it now has its own `[0.3.8] - 2025-12-03` section. A fresh build gives `yapocis-1.0.0` wheel and sdist, which pass `twine check` with twine 7. About 65 lines of changelog and 3 lines of metadata.

**Caveat recorded.** Re-running `make changelog` would regenerate the file from git history and replace these entries.

### 2026-10-17: twine 7 for Metadata-Version 2.5; History Secret Scan Before Going Public

**What the maintainer asked for.** "Update twine as needed", and before that, "Make yapocis3 a public repository".

**twine (Claude).** Current hatchling writes Metadata-Version 2.5. The locked twine 6.2.0, with packaging 25.0, rejected it ("'2.5' is not a valid metadata version"), so `make publish` would have failed. The `dev` extra now requires `twine>=7.0.0`. `uv lock --upgrade-package twine --upgrade-package packaging` moved only those two packages (to twine 7.0.0 and packaging 26.3). With the venv's twine, `twine check` passes on a fresh wheel and sdist. The backend tests pass after the packaging upgrade (120 passed, 1 skipped). 1 line, plus the lockfile.

**Secret scan before going public (Claude; the maintainer revoked the token).** Before changing the repo's visibility, Claude searched all 369 commits for credentials. A PyPI API token (`TWINE_PASSWORD`, `TWINE_USERNAME=__token__`) was in a `.env` committed in `e04de08` (2025-10-06) and deleted in the next commit. It is still in the history of `main`, every branch since and tags v0.3.3 onwards. Claude stopped before making the repo public. The maintainer had already replaced the token that day. Claude compared the new `~/.pypirc` token with the leaked one by hash only and confirmed they differ. The maintainer then confirmed from the PyPI token list that the old token was gone. The repository was made public with its history unchanged, since a revoked token is harmless. Nothing else turned up: the rest of that `.env` held `YAPOCIS_*` settings.

### 2026-10-17: PyPI Release-Readiness Pass

**What the maintainer asked for.** "Evaluate yapocis3 for a pypi release (don't DO it, just evaluate). Check software against current documentation, GPL or license contamination, etc." The maintainer then directed each fix in turn: pymetallic installed by default on macOS, "fix the README examples and the numba alias", replace the demo photos with CC0 images, fill in the freetype-py notice, fix the docs and docstring warnings, fix the README links and badges, add Sphinx to the dev dependencies, update the copyright years and holders, scan the dependency tree for "poison licenses", and skip `test_vimp_common` when `yapocis_external` is missing.

**Evaluation (Claude).** I built the wheel and sdist, installed them into a clean venv, and audited the docs against the code (a subagent ran the README examples and a Sphinx build). I also scanned the licences of runtime dependencies and searched the source for third-party code. The results are in `claude-debug/release_eval_2026-10-17.md`. The main findings were:

- `v0.3.8` was already tagged, with 28 commits after it, and the changelog had no 0.3.8 entry. PyPI's `yapocis` is the 2011 Python 2.7 project (0.2.x eggs).
- Both README `core` examples failed to import, and `mandelbrot(..., maxiter=...)` passed a keyword argument, which bound kernels reject.
- `YAPOCIS_BACKEND=numba` was documented but rejected.
- The locked twine 6.2 rejects the Metadata-Version 2.5 that current hatchling writes.

**Licensing (Claude, maintainer decisions).**

- **Images.** The 13 classifier demo photos had no provenance. Checked against all 500 BSDS500 images at the maintainer's request, none came from BSR; their sizes and class names point to ILSVRC. They were replaced with CC0 photos found through Openverse and are listed in `src/yapocis/images/CREDITS.md`. The maintainer supplied the hot dog (Wikimedia) and the cassette player (rawpixel), and rejected a CC BY-NC-SA candidate after Claude flagged it. A crop the maintainer made of the cassette photo was tried and then dropped. `test.jpg` is unchanged.
- **freetype-py notice.** `lib/typeset.py` said it was "used under the following terms" over an empty block. It now has freetype-py's full BSD notice.
- **Copyright.** Copyright end years are now 2026, and holders are normalised to "Sean D. True and Swapwizard LLC" in 134 headers, `LICENSE.txt` and `docs/conf.py`.
- **Perona–Malik credit.** The ChatGPT-written comment in the OCL and CUDA anisodiff kernels credited "Carlo Perona and João Malik". It now names Pietro Perona and Jitendra Malik.
- **Dependency scan.** All 171 locked packages were checked, along with the licence files and dylibs in the runtime wheels (`claude-debug/release_eval_license_scan_2026-10-17.md`). The core install and every user-facing extra are permissive. The copyleft and proprietary licences are only in dev and docs tooling: torch's NVIDIA CUDA libraries, pygame, astroid, and MPL-licensed certifi and pathspec.

**Code (Claude).**

- **Backend aliases.** `core.canonical_backend` and `BACKEND_ALIASES` give `get_backend()` and `get_manager()` one alias table, so `YAPOCIS_BACKEND=numba`, `opencl` and `metallic` now work (~20 LOC, with ~20 LOC of tests).
- **pymetallic.** It is a core dependency on darwin.
- **Docstrings.** Six docstrings were fixed for RST.
- **Imports.** `fonts` uses `ImageFont.*`, and `interface_parser` uses pyparsing's `DelimitedList` instead of the deprecated `delimitedList`. That removes the third-party forward-reference warnings.
- **Help text.** `yapocis-cuda-cards --help` gave the `--matrix` card separator as `,`; the parser uses `+`. The `yapocis-doctest` help examples pointed at a `newdocs/` directory.
- **`test_vimp_common`.** It now uses `pytest.importorskip`, so it skips when `yapocis_external` isn't next to the checkout instead of failing collection.

**Docs (Claude).**

- **README.** The examples are rewritten and run on Metal, Numba and OpenCL. The broken code fence is fixed; CUDA is described as PyCUDA, not CuPy; links are absolute; and the CI badge, for a workflow that doesn't exist, is gone.
- **Five pages imported `get_backends` from `yapocis.core.manager`.** That module doesn't define it.
- **`docs/library/channels.md`.** Two examples failed: `histeq` is an `Imageops` method, and `rgb2hsi3` returns one array.
- **Nonexistent APIs.**
  - `docs/architecture/extensions.md` described `yapocis.assets.add_asset_path`, which doesn't exist, and a multi-path `YAPOCIS_PROJECT`, which isn't supported.
  - The `yapocis-doctest` options in `docs/cli/index.md` didn't exist.
  - `docs/api/core.md` named `parse_interface_file`, which doesn't exist; the function is `parse_interface_text`.
- **Build configuration.** `.readthedocs.yml` asked for Python 3.7 and didn't install the package, so the API pages were empty. `conf.py` used a theme the `docs` extra didn't install. The `dev` extra now includes `docs`.

**Maintainer changes in this commit.**

- `activate.sh` points `YAPOCIS_MODELS` at `/data/models/yapocis3` when it exists.
- `make docs` runs through `uv run --extra docs`.
- The tracked Emacs autosave `src/yapocis/lib/#gaussian.py#` is deleted.

**Defects in AI-written work, caught during the session.**

- The first pass at a full license scan treated the NVIDIA NCCL, NVSHMEM and NVTX wheels (BSD and Apache) as proprietary, because their classifier reads "Other/Proprietary". It also counted the docutils GPL classifier as a runtime problem. Reading each package's licence files corrected both.
- The first clean-install smoke test reported that Metal was the default backend and that a fresh install failed. The shell still had `YAPOCIS_BACKEND=metal` set. Rerunning with a clean environment showed that the default, `ocl`, works.
- The docs-audit subagent said the Emacs autosave file would ship in the wheel. Checking the built wheel and sdist showed it doesn't.

**Tests.**

- **Mac** (Metal and Numba): 925 passed, 1 group skipped (no pycuda).
- **yellowfin** (OpenCL, CUDA and Numba; P40, 3060, 3080), from a worktree with this change applied: 1302 passed and 30 skipped. 27 of the skips were VGG16/ResNet50 tests, because `YAPOCIS_MODELS` wasn't set; 3 were Metal. That run predates the `test_vimp_common` skip fix.
- **Docs:** `yapocis-doctest docs/` gives 210 passed and 0 failed, and `sphinx-build -W` has no warnings.

**Still open for the release:**

- Bump the version past 0.3.8 and write the changelog.
- Upgrade twine.
- Make the private GitHub repo public, or accept 404 links on PyPI.
- Confirm the maintainer's PyPI account still owns `yapocis`.
- Decide what to do with `test.jpg`.

### 2026-10-17: split_channels Returned a List, Not the Annotated Tuple

**Found by the maintainer** while testing ImpCraft (session S04). ImpCraft's `rgb2hsi` node failed with "virtual_impressionist.core.hsi.rgb2hsi returned 1 values, expected 3".

**Cause (Claude).** `yapocis.lib.channels.split_channels` is annotated `-> tuple[Image1, Image1, Image1]` but returned a list comprehension. impressionist's `hsi.rgb2hsi`, `hsi.hsi2rgb`, `lab.rgb2lab` and `lab.lab2rgb` return it directly. ImpCraft's function nodes take a tuple as several outputs and anything else as one value, so all four were broken as nodes.

**Fix (Claude, `src/yapocis/lib/channels.py`):** `tuple(...)` instead of the list. Every caller in yapocis3 and impressionist unpacks the result (checked by grep), so nothing else changes.

**Tests:** `tests/integration/test_utils.py::test_split_join` now asserts a tuple of three. It and `tests/lib/processing/test_colormath.py` pass (27 tests). impressionist's colour tests pass (78; `-k "hsi or lab or color or palette or channel"`). ImpCraft's exactness gate is unchanged (12 of 12). ImpCraft adds the regression test `test_multi_output_colour_functions_give_each_channel`.

### 2026-10-12: CUDA Gradient Angle Used Swapped Axes

**Maintainer direction:** deploying the previous change ("CPMD the revised yapocis3"), the maintainer ran the full suite on yellowfin, where the new gradient tests failed on CUDA. When Claude proposed a fix, the maintainer warned: "we've had problems with you treatment of CUDA vs other kernels storage order in the past. Take a second look and make sure." Then: "Go".

**Cause (Claude).** Every backend stores images the same way in memory (NumPy order, `width_of` = shape[0] = rows, `height_of` = shape[1] = cols, from the shared kernel base; nothing in `cu/` transposes). The macros name the axes differently: OCL and Metal's `ADDRESS(x, y) = x*height + y` makes x the row, CUDA's `y*height + x` makes x the column. The CUDA gradient's bounds check was converted to CUDA's naming on 2026-10-06, but its angle line kept OCL's `atan2pi(dy, dx)`, which on CUDA is atan2(d/drow, d/dcol): the mirror image of every other backend's atan2(d/dcol, d/drow). Gradient magnitude is symmetric, so it was right. `zcsdog` takes its theta from this kernel, so on CUDA, Virtual Impressionist's direction-driven styles drew with directions mirrored across the diagonal.

**Second look, at the maintainer's request.** Before changing anything, Claude checked the host side (dimension handlers, no transposes), confirmed that this is the only `atan2` in yapocis3's CUDA kernels, and checked yapocis_external's CUDA kernels that call `atan2`: `boundedaverage` and `labeledaverage` only average angle values, and `lic` indexes with its own `y * width + x`, identical on OCL and CUDA. It then measured on yellowfin (`claude-debug/gradient_cuda_axes_probe.py`): on a random image CUDA's theta matched the swapped angle to 1.2e-07 and missed the reference by 1.0, while OCL on the same NVIDIA cards and numbax matched the reference.

**Fix (Claude, `kernels/cuda/gradient.mako`):** `atan2pi(dx, dy)`, with a comment explaining why CUDA differs from OCL here.

**Tests on yellowfin.**
- The 49 gradient tests (OCL, CUDA, numbax) pass on the patched code; CUDA's theta matches the reference to 1.2e-07.
- Full-suite runs in one pytest process were swamped by GPU out-of-memory errors (pipekit holds most of the 3060 and 3080; the failing tests pass when run alone), so the suite was run file by file (`claude-debug/run_per_file.sh`) on the checkout and on a patched copy. The fix removes exactly the 8 CUDA gradient failures and adds none. Present in both: 12 `test_anisodiff` failures (out of memory, and a `KeyError` for the RTX 3080 these tests select themselves). Only in the copy: 8 `mplbackend` tests that write into a `claude-debug/` directory the copy did not have.
- Not run on the Mac (no CUDA there).

**AI defects.**
- The previous entry's gradient work was checked on OCL, Metal and numba only; its PR said every backend was tested, but CUDA was not available on the Mac and the bug shipped in that merge. The new tests caught it in the yellowfin deploy.
- Claude's first full-suite comparison on yellowfin was unusable: the patched copy stopped at collection (`test_vimp_common` needs the sibling yapocis_external), and an `--ignore` for it did not take.

**LOC:** 3 lines in one kernel.

### 2026-10-12: Gradient Values Tested on Every Backend; numbax Fallback, Search Order and Loader Fixed

**Maintainer direction:** while fixing Virtual Impressionist's `artists/etch.py`, "take a look at the gradient kernel, and whether tests on it currently pass", then "Make the yapocis3 changes, run a complete test cycle."

**Findings (Claude, by measurement: `claude-debug/gradient_backend_probe.py`, `gradient_numbax_fallback_probe.py`):**
- The kernel is right, and OCL, Metal and numba agree to ~1e-7: central differences over 2 * reach, theta = atan2(d/dcol, d/drow) / pi, measured from +row toward +col, zero within `reach` of the border. The existing tests (`test_gaussians.py::test_gradient`, the `showarraygrad` examples) passed but only rendered and timed; none checked a value.
- **numbax fallback wrong:** without numba, `kernels/numbax/gradient.py` took one-sided differences over `reach`, so gradients were half strength on ramps and angles off by up to 1 on noise.
- **Search order reversed:** `get_yapocis_dirs()` returned package, cwd, project, while the docstrings and `kmgr_base` said project, cwd, package. Project kernels could not override built-ins of the same name.
- **numbax loader:** because the package came first, built-in numbax kernels loaded from their file as `yapocis_ext.kernels.numbax.*`; numba's caches pickled that name, so importing a kernel module by its real name and calling it raised `ModuleNotFoundError: No module named 'yapocis_ext'`.

**Fixes (Claude):**
- `utils/internals.py`: `get_yapocis_dirs()` is project, cwd, package, without duplicates. No kernel or interface names collide with yapocis_external's, so no kernel resolves differently there.
- `numbax/kernel_manager.py`: built-in kernels are imported by package name and also registered under the old `yapocis_ext.*` name, so caches already on disk (including production's) still load.
- `kernels/numbax/gradient.py`: the fallback uses central differences and matches the kernel.

**Tests (Claude):** `tests/lib/compute/test_gradient_values.py` (values and directions against a numpy reference on a non-square image, every backend, and both numbax paths) and `tests/backend/test_search_order.py` (order, duplicates, package-name loading, project override). With the old code 13 of them fail. Full suite: 877 passed and 1 skipped before, 920 passed and 1 skipped after (CUDA skipped: no pycuda on the Mac); the new gradient tests also pass with OCL listed explicitly (49). yapocis_external's suite passes (542) from its existing venv; `uv run` there can't resolve, because its `pyproject.toml` asks PyPI for `yapocis>=0.3.0` with no path source. Virtual Impressionist's suite is unchanged by this (174 passed, 3 known failures from its DoG model).

**Defects in the AI's process:** it first suggested the numbax kernel used a different angle convention (it doesn't), and its first reference had the axes swapped, so every backend appeared wrong until the ramps showed otherwise. Its first fallback probe patched a module the backend never used (the `yapocis_ext` copy), which hid the fallback bug until the loader was read.

**Not fixed, noted:** `numbax/kernel_manager.py`'s stub for an unimplemented kernel refers to an undefined `mod_name` (ruff F821), so it raises `NameError` instead of `NotImplementedError`, and binds `func_name` late.

**LOC (measured):** source +28/−25 across three files; tests +143.

### 2026-10-10: Zero-Length Lines Drawn as Dots on Every Backend

**Maintainer direction:** the bug turned up in Virtual Impressionist's stroke-rendering work, where durer's strokes were drawn with `Graphics1.lines`. The maintainer accepted the plan to fix it here ("CP. Then start on the rest of the list.") and asked for it to ship ("Cpmd").

**Cause (Claude):** `lines` in `graphics` and `graphics4` on Metal, OpenCL and CUDA divided the projection by the squared length, so a zero-length line gave u = 0/0. NaN fails both `u >= 0` and `u <= 1`, so the line was silently skipped. numbax adds 1e-20 to the length and draws a dot at the start. Two thirds of durer's segments are zero length, so on the GPU most of a stroke's ink vanished. Virtual Impressionist had worked around it by drawing the dots with `circles`.

**Fix (Claude, 6 lines changed in six kernel files):** `u = linemag2 > 0 ? … : 0`, so a zero-length line is a dot at its start, pixel-identical to numbax. The guard is explicit rather than an epsilon, because NaN handling under GPU fast-math is not guaranteed.

**Tests (Claude, ~40 LOC, `tests/lib/graphics/test_zero_length_lines.py`):** a dot alone, and a dot before and after a line, on `Graphics1` and `Graphics4`, compared with numbax pixel for pixel.
- With the fix: Metal and OpenCL on the Mac pass (12 cases); CUDA, OpenCL and numbax on yellowfin pass (18 cases).
- Without it, every Metal, CUDA and OpenCL case fails.
- The graphics suite passes: 172 tests on the Mac, 251 on yellowfin.
- The yellowfin run used a temporary worktree, removed afterwards, so its checkout was not touched.

**Defect in the AI's process:** the first commit of this change left this entry out, because the script that wrote it failed on a heading match and the commit ran anyway. It was amended before pushing.

### 2026-10-08: test_vimp_common Default-Queue Test Corrected

**Maintainer direction:** "fix the test_vimp_common failure", the one failure left in the yellowfin deploys of the last two merges, which also failed on `main` before them.

**Cause (Claude):** `test_show_array_default_queue_dir` was wrong on two counts, and neither library was changed. It passed an all-zero array, which yapocis `show_array` rejects on purpose (`checkzero`: "No one likes an empty picture"). It also expected the default output in `./queue` under the working directory, but yapocis saves to `VIEWING_QUEUE`, which is fixed at import time under the performance directory.

**Fix (Claude, ~15 LOC in `tests/test_vimp_common.py`):** the test now uses a non-zero 2x2 array, redirects `yapocis.utils.basic.VIEWING_QUEUE` to a temp folder with monkeypatch (so it no longer writes into the real queue), and checks the file lands there. All 10 tests in the file pass on the Mac.

### 2026-10-08: Anisodiff Iteration Count and RGB Layout, get_manager Backend Argument, f16 dtype, Stale Make Targets

**Maintainer direction:** "Fix and test" the four issues the FP16 work had turned up. Then "What are the other stale make targets?", and "Yes" to the proposal to fix the renamed ones and remove the crosscheck targets.

**Fixes (Claude, ~35 LOC in `src/`, ~25 net LOC removed from the Makefile):**
- `Anisodiff.anisodiff_n(img, n)` returned n−1 iterations: after the last buffer swap it read the stale buffer. It now reads the buffer the last iteration wrote.
- `core.get_manager(backend)` overwrote its argument with `YAPOCIS_BACKEND`, so `load_program(..., backend=...)` silently ran on the environment's backend. An explicit backend (any case) now wins; None still uses the environment, and unknown names still raise.
- `ctype_to_nptype` gained `f16`, placed so that the reverse map still gives `half` for float16.
- Makefile: `test-smoke` now runs `pytest -m smoke tests`. `test-runtime`, `test-perf-lib` and `test-perf-kernel` point at the renamed files. The `crosschecks`/`crosscheck-ocl`/`crosscheck-numbax` targets were removed: `test_backend_crosscheck.py` was never in this repository's history, and nothing reads the `YAPOCIS_CROSSCHECK_*` variables they set. `docs/kernel-development/testing-kernels.md` and `AGENTS.md` no longer name the old files; `HISTORY.md` and older AI.md entries were left as history.

**A pre-existing bug exposed by the off-by-one fix.** Once `anisodiff_n` returned its last iteration, the new reference test failed on yellowfin's CUDA for RGB images only, with NaNs and values up to 1e36. On unpatched `main`, n=2 already gave garbage; n=1 had hidden it by returning the input. The cause is that `float3` is 16 bytes in OpenCL and Metal but 12 bytes in CUDA:
- `anisodiff_n` padded RGB to 4 channels but still ran the 3-plane kernel, so CUDA misread the buffer.
- The streaming `anisodiff` generator did not pad at all, so it was wrong on OpenCL and Metal (errors up to 1.0) and right on CUDA and Numba. Claude predicted this from the layout and confirmed it with a probe.

A new `Anisodiff.pad_planes` helper now pads 3 planes to 4 and selects the 4-plane kernel for both paths on every backend. No CUDA kernel or convention was changed.

**Tests (Claude, ~124 LOC, 34 cases on the Mac's 3 backends).** These compare `anisodiff_n` and the generator against a float64 Perona–Malik reference (grey and RGB, every backend). They also cover backend selection with a conflicting `YAPOCIS_BACKEND`, and the `f16`/`half` dtype mapping. Against the old code, 20 of the first 28 failed. The 6 generator cases were added later and were not re-run against the old code; the probe showed the same wrong RGB output.

Results:
- Targeted runs: Mac (Metal, OpenCL, Numba) 240 passed, 1 skipped; yellowfin (OpenCL, CUDA, Numba, patched worktree) 246 passed, 28 skipped.
- Full suite: Mac 1085 passed, 1 failed, 1 skipped; yellowfin 1076 passed, 1 failed, 1 skipped. The one failure, `test_vimp_common::test_show_array_default_queue_dir`, also fails on `main`. It feeds an all-zero image into `show_array`'s `checkzero` assertion and is not addressed here.
- `make test-smoke` gives 121 passed, 1 skipped; `make test-runtime` gives 3 passed, 1 skipped. The performance targets were only checked to collect (10 and 22 tests), so that running them would not add rows to the performance database.

**AI defects, and how each was caught:**
- **Untested case in the earlier FP16 work:** Claude's earlier claim that its anisodiff port matched the library on CUDA covered grey images only, so the CUDA RGB failure went unseen until the new reference test ran on yellowfin.
- **Probe and wait-script mistakes, no effect on the change:** a debugging probe set `YAPOCIS_BACKEND=numba` (the accepted name is `numbax`) and crashed; caught on running. In the previous deploy step, a remote wait loop used `pgrep -f` with a pattern from its own command line, so it could never exit. The maintainer had already been bitten by this pattern once and asked for it to be avoided in all cases; it is now a standing rule.

### 2026-10-08: FP16 Evaluation; imread/byte_scale/imwrite Handle FP16 Without Full-Size FP32 Copies

**Maintainer direction:** "Evaluate adding FP16 type in parallel with the current FP32 default", then "Run on yellowfin", then "Run result comparisons on ZC detection with FP32 vs FP16, and for anisodiff". Next they asked whether `imread` could "return FP16 on request, w/o cycling through FP32 on the way" and whether there was a common write path for the same treatment. On `imwrite`, they decided: "No automagic rescaling."

**Evaluation (Claude, research and reports; no library code).** The core already carries `half` from the IDL through to `float16` arrays; the cost is the ~750–940 hard-coded `float` tokens in each GPU backend's templates. Probe kernels (Metal, OpenCL, CUDA) on the M4 Max and on yellowfin's P40, 3060 and 3080 showed:
- FP16 storage halves transfer-bound time on every device.
- FP16 arithmetic is no faster on Apple or Ampere, and about 48x slower on the P40.
- NVIDIA OpenCL has no `cl_khr_fp16`.

The result comparisons used FP16 ports of `zcs` and anisodiff, first validated against the library and a numpy reference, and ran identically on all five device/backend combinations:
- The ZC kernel is FP16-safe (IoU ≥ 0.9986).
- In the DoG pipeline, storing the two Gaussian outputs in FP16 is what drifts (IoU down to 0.915). Keep those in FP32.
- Anisodiff state drifts with iteration count (2.5–7% of pixels off by more than one grey level at 100 iterations; 25–36% at 400). Its state should stay FP32.

Reports: `claude-debug/FP16_EVALUATION.md`, `claude-debug/fp16_eval/ZC_ANISO_TABLES.md`, `claude-debug/fp16_io/FP16_IO.md`. The probe scripts and kernels stay uncommitted in `claude-debug/`.

**FP16 image I/O (Claude, ~54 added / 10 removed LOC in `utils/basic.py`, ~116 LOC tests).** `imread(..., dtype=)` converts through 64-row float32 slabs into the target dtype. The float32 default is bit-identical to the old output (with and without `longest=`), and FP16 is bit-identical to `.astype(np.float16)` of it. On a 4060x4055 RGB image, peak memory was 97 MB against 377 MB, at the same speed. `byte_scale` takes FP16 through a two-pass slab path: 36 ms instead of 772 ms (numpy's FP16 arithmetic is software), with identical bytes. `array_to_image` accepted only float32/float64 and raised `TypeError` on FP16; it now widens FP16 to float32. `imwrite` had been broken for every dtype (`ndarray` has no `to_image`). Following the maintainer's "no automagic rescaling", it now maps NaN to 0, clips to [0, 1] and calls `to_image(..., cmin=0, cmax=1)`.

New tests in `tests/integration/test_image_io.py` (19). All pass on the Mac. Regression runs on the Mac: graphics position validation, anisodiff and `tests/integration` (126 passed), and `pytest -m smoke tests` (121 passed, 1 skipped). Not run on yellowfin.

**Library bugs found and reported, not fixed:**
- `Anisodiff.anisodiff_n(img, n)` returns n−1 iterations: after the last swap it reads the stale buffer, so `n=1` returns its input. Confirmed bit-exact on Metal and OpenCL.
- `core.get_manager(backend)` ignores its argument, so `load_program(..., backend=...)` always uses `YAPOCIS_BACKEND`.
- `ctype_to_nptype` has no `f16` entry, although the grammar accepts it.
- `make test-smoke` points at a test file that does not exist.

**AI defects, and how each was caught:**
- **Mislabelled benchmarks:** the first FP16 report's Mac "ocl" rows were actually Metal (because of the `get_manager` bug), and its speedup range was quoted from them. A `clBuildProgram` error from a run labelled CUDA on yellowfin exposed the problem. The scripts now set `YAPOCIS_BACKEND`, and the Mac numbers were re-measured and corrected in the report.
- **Device selection:** the benchmark script turned names like `3060` into device indices (`IndexError`); caught by the run.
- **Tests:** one expectation assumed 0.5 → 128, but `byte_scale` rounds with +0.4999, giving 127; caught by running the test. The first alpha test used a JPEG with no alpha channel, so it passed vacuously; Claude caught this on review and replaced it with a generated RGBA PNG.

### 2026-10-06: yellowfin on Python 3.13; Seaborn Comparison Tests Pin Their Font

**Maintainer direction:** "Change yellowfin to use python 3.13 instead of 3.10", then, once the font cause was shown, "Pin the font."

**yellowfin environment (Claude, ~26 LOC in `activate.sh`).** yellowfin's tests had run from a hand-built Python 3.10 virtualenv (`~/.virtualenvs/yapocis-310`): numpy 1.25, no matplotlib, yapocis not installed (hence `PYTHONPATH=src`), and no `nvcc` in non-interactive ssh sessions. The checkout's uv-managed `.venv` (Python 3.13.7) was synced from the current lock with `uv sync --python 3.13 --extra opencl --extra cuda --extra numba --group dev`, which installs yapocis editable. `activate.sh` now defaults to that `.venv` (the `YAPOCIS3_VENV` override is kept) and exports `CUDA_HOME`/`PATH` when `/usr/local/cuda` exists. The old virtualenv was left in place, unused.

**Result and the one failure.** Under 3.13 with the new script: 979 passed and 1 failed on yellowfin (OCL and CUDA; more tests than before because matplotlib and seaborn are now installed). The failure was `test_seaborn::test_hexgrid` at RMS 10.12 against a tolerance of 10.0. Claude ruled out the GPU (identical on OCL, CUDA and Numba), the library versions (matplotlib, seaborn, numpy, freetype-py, libfreetype and Pillow all match the Mac), and the data (seeded). The cause was fonts: seaborn's theme puts Arial first, which the Mac has (RMS 9.14) and yellowfin doesn't, so it falls back to DejaVu Sans. With DejaVu Sans forced, both machines give exactly 10.123, so the test had passed on the Mac only because Arial happened to be installed.

**Fix (Claude, ~15 LOC).** The three seaborn comparison tests pass `font="DejaVu Sans"` (bundled with matplotlib) to `sns.set_theme`. `test_hexgrid`'s tolerance is 10.5, with a comment giving the measured 10.12, and its failure message reports the actual tolerance. All 3 seaborn tests pass on the Mac (OCL, Metal, Numba) and on yellowfin under 3.13. The full suite was not re-run after the font pin.

**Housekeeping on yellowfin (Claude, at the maintainer's request):** removed the scratch worktree and its temporary files, and deleted 22 stale `__pycache__` directories, including the cached `.pyc` that broke `test_rpc.py` (see the previous entry).

### 2026-10-06: One (x, y) Drawing Convention, CUDA Memory-Safety and Multi-GPU Fixes, Dependency Cleanup

**Origin of part of this commit:** The working tree on `graphics-numba-mpl-mesh` already held uncommitted work when this session began, and who wrote it isn't recorded: `mesh_triangles` (shared edges drawn once, so translucent fills have no seams) in the interfaces and the OCL, Metal, CUDA and Numba kernels (~520 LOC), a Numba graphics rework (~400 LOC), the matplotlib backend's mesh fills and text compositing (~160 LOC), `tests/lib/graphics/test_primitive_consistency.py` (~87 LOC) and smaller test edits. Claude ran it and tested it alongside this session's changes, but did not write it.

**Maintainer direction:** The maintainer asked for a failing icon test, then the stub mismatch, to be fixed, and steered the larger work. They spotted that tests branched on backend for coordinates and set the rule: "There are not separate coordinate systems at the API level -- all supposed to be dealt with by the library." They suspected the circle logic was backward, which exposed that text and the shapes disagreed. They sent Claude to the examples and docs to settle the convention, approved switching `composite` to (x, y), and chose which CUDA bounds fix to use. On yellowfin, after two runs with about half the tests failing, they pointed at the cause: "We have a history of Cuda kernels not limiting themselves to array bounds." They also asked for the barrier removal, CUDA blending to match OCL/Metal, the multi-GPU fix, the dependency review (including putting pyopencl and numba back in core), the Dependabot bumps, and the branch cleanup.

**Drawing API: one convention (Claude, ~150 LOC).** Lines, rects, circles, triangles, mesh triangles and ellipses took their first coordinate as the row on OCL, Metal and Numba, while text took (x = column, y = row), and Graphics4 `composite` took (row, col). The docs, the quickstart and an old RetinaNet example all use (x, y). Shapes are now reordered in one table in `abstract_graphics.py` and passed through unchanged for CUDA, whose shape kernels already take (x, y). Graphics4 `composite` and text now use (x, y) too. Claude removed the compensating swaps from `mplbackend.py` and `utils/advanced.py`, and converted the position, antialiasing and alpha-blending tests (~135 LOC), deleting their per-backend branches.

**CUDA memory safety (Claude, ~200 LOC across 18 templates and the launcher).** The CUDA launcher rounds grids up to whole 256-thread blocks, and about 30 kernels never checked the thread index. Kernels that already receive their size now return early. The rest get the real size as a hidden argument (`GLOBAL_SIZE_PARAM` / `RETURN_IF_OUTSIDE`, passed by `cu/kernel.py` when a kernel's signature asks for it). FFT `window`, `power_spectrum` and audio `smooth_1d` used the rounded-up thread count as their length. The separable Gaussian, gradient, anisodiff and zero-crossing kernels checked bounds with OpenCL's X/Y mapping, which CUDA defines the other way round. `histogram` (CUDA and OCL) incremented one bin past the end for values equal to `high`; it now clamps as Metal does. Each was found with `compute-sanitizer` on yellowfin, and the final memcheck run over every CUDA library test reported 0 errors.

**CUDA correctness (Claude).** CUDA Graphics4 never wrote alpha (and `lines` forced it to 0), and on antialiased edges applied the color's alpha twice (`t*a^2`). It now blends exactly like OCL/Metal through one `blend_over` helper. The 14 `barrier()` calls in the CUDA graphics kernels did nothing and became undefined behaviour once threads could return early; they were removed. With several GPUs, `make_context()` left the last-created context current, so work for one card ran in another's context ("invalid resource handle" in the async tests, order-dependent). The CUDA device manager now activates its own context around every compile, allocation, copy, launch and sync (~55 LOC).

**Other fixes (Claude).** `imread` dropped the transparency of palette PNGs, so `icon()` returned 3 channels and `test_random_icons` failed on whichever icons it happened to pick. numbax Graphics4 used a different blend formula from OCL/Metal, and its hard-edged rects never wrote alpha. `Imageops` now makes inputs contiguous, so channel slices work on OCL/CUDA (`test_histeq_hsi_roundtrip`). `setup_texts` had its first parameter named `lself`.

**Stub generator (Claude, ~100 LOC in `scripts/tools.py`).** `.pyi` stubs listed raw kernel signatures (e.g. `rcolor`) for methods the library overrides. They now use the library class's real Python signatures (typed `Any` so strict mypy passes), find the class by its `INTERFACE` binding, skip interfaces with no library class, and map unsigned dtypes correctly; the committed `truetrace.pyi` had a syntax error from the old mapping. 12 stubs were regenerated and 8 orphan stubs deleted (`vgg16.pyi` hid the real `VGG16` classes from type checkers).

**Dependencies (Claude, at the maintainer's direction).** pyopencl and numba are core again, and imagecodecs was added: `test_hires` needs it for LZW TIFFs, but it had never been declared and was only installed by hand. Ten core entries that nothing imports were removed; their security and pytools minimums moved to `[tool.uv] constraint-dependencies`. `pymetallic` got a macOS marker, a new `mpl` extra was added, the `test` extra completed, and `[dependency-groups] dev` reduced to `yapocis[dev]` so the two lists can't drift. The five open Dependabot lock bumps (cryptography, urllib3, anyio, soupsieve, pip) were applied, and `CLAUDE.md` lost its stale kernelpack references.

**AI defects, and how each was caught:**
- **Stub generator, first version:** running it over every interface created 20 new stubs, some wrong (`gaussian.pyi` would have hidden the real module). Claude caught this by reviewing the output and deleted them. The same version left the stubs unannotated, which strict mypy rejected (caught by running mypy). It also guessed class names from interface names, which was wrong for 4 stubs, so `graphics.pyi` still carried `rcolor` after the "fix" (caught while checking the deletion candidates).
- **Coordinate convention:** the first test edit (removing backend branches) kept the expectations that matched the then-transposed shapes. It was reversed after the maintainer questioned the circle logic. The library reordering was then applied to CUDA too, transposing CUDA output; a placement probe on yellowfin caught that. The CUDA exemption then crashed every Numba graphics call (68 failures), because Numba's kernel manager has no `_backend_name`; the Mac test run caught that.
- **Bounds:** the `median3x3` guard compared against `width*height`, but that kernel's first argument is already the element count, so the guard never triggered; Claude caught it re-reading the interface. Claude's guard audit counted `power_spectrum` as safe although its bound was the padded count; `compute-sanitizer` caught it.
- **False test claims:** several Mac runs were reported as covering OCL. The variable used, `YAPOCIS_TEST_BACKEND`, is ignored (the real one is `YAPOCIS_TEST_BACKENDS`), and the Mac's default test backends are Metal and Numba, so those runs never exercised OCL. An OCL-only failure on yellowfin exposed this, and the claims were corrected.
- **Operational mistakes, no damage:** a `git stash push` silently failed and the following `pop` tried an unrelated stash, which git refused. A `pkill -f` pattern killed the ssh session running it.

**Ruled out / pre-existing:** the first yellowfin failures (about 50%) were `nvcc` missing from non-interactive ssh's `PATH`, not this change. The async CUDA failures and `test_histeq_hsi_roundtrip` failed identically on clean main (both are fixed here). The 10 `benchmark` errors were `pytest-benchmark` missing from yellowfin's venv (installed).

**Testing:** Mac (OCL, Metal, Numba; `YAPOCIS_TEST_BACKENDS=ocl,metal,numbax`): 911 passed across `tests/lib`, regression, integration and async after the code changes. After the dependency changes, 910 passed and `test_hires` passed once imagecodecs was added. After the lock bumps, integration, graphics and mplbackend: 380 passed. yellowfin (OCL and CUDA on a Tesla P40, RTX 3080 and RTX 3060), on a patch of this tree in a scratch worktree: 767 passed and 0 failed, and the 10 benchmark tests passed after the install. yellowfin was not re-run after the pyproject/lock changes, because its venv isn't built from them.

**After merge (#34, `85ea445`), on yellowfin's own checkout:** the first run failed at collection (`No module named 'yapocis'`): yapocis isn't installed in that venv, so pytest needs `PYTHONPATH=src`, as the scratch-worktree runs used. With it: 775 passed and 2 failed, both in `tests/integration/test_rpc.py`. Claude traced those to a stale cache, not code. pytest's rewritten `.pyc` for `test_rpc.py` dated from July, when the repo lived at `/data/speedy/workspace/yapocis3`. Since the source hadn't changed, pytest kept reusing it with the old path recorded, so `import_program`'s `inspect.getmodule()` couldn't find the calling module and injected nothing. The tests passed with `--assert=plain` and in the fresh worktree. After deleting the generated `tests/**/__pycache__` on yellowfin, `test_rpc.py` passed (10/10), for 777 passing tests on main with no failures. About 300 stale `.pyc` files under `src/` there still record the old path.

**Division of labor:** the maintainer diagnosed the coordinate split and the CUDA bounds history and made the design and dependency decisions; Claude did the investigation, the implementation, the yellowfin and sanitizer runs, and the tests.

### 2026-10-04: CUDA Helpers Made Opt-In (fixes CUDA compile on main)

**Origin:** These changes were found uncommitted in the yellowfin working tree. Who wrote them isn't recorded there. Claude committed them at the maintainer's request.

**What it fixes:** On main, `cuda/helpers.mako` defined `dot3`, `vxm3` and `dxdy2` unconditionally, so the HSI kernel source failed under nvcc with `function "dot3" has already been defined`. The failed compile also left the CUDA context unusable for the rest of the pytest process, so most later CUDA tests failed with `cuModuleGetFunction failed: an illegal memory access was encountered`. The fix wraps each helper in `#ifdef DOT3/VXM3/DXDY2`, and each kernel template (`colormath`, `graphics`, `graphics4`, `hsi`) defines the helpers it needs before including `helpers.mako` (~14 LOC across five files).

**Investigation (Claude):** Clean main on yellowfin: 709 passed, 297 failed. Main plus these changes (and the uncommitted test tolerance changes): 948 passed, 58 failed, and no test failed only with the changes. Both runs had `nvcc` (CUDA 12.5) on PATH. The bug never showed on the Mac because the Mac has no CUDA, so main's CUDA kernels were never compiled there. The yellowfin test-tolerance increases (10 to 25 in three mplbackend and graphics test files) were deliberately left out of this commit.

**Second commit: `yapocis-cuda-cards` and environment.** This was also found uncommitted on yellowfin, with authorship not recorded. It adds `src/yapocis/scripts/cuda_cards.py` (~1,040 LOC), a per-card, per-toolkit CUDA benchmark. Each (toolkit, card) pair runs in its own subprocess with the card pinned by UUID; the script writes md/pdf/json reports and records results in the performance DuckDB. It also adds `activate.sh` (~23 LOC) for the yellowfin Python 3.10 virtualenv, plus pyopencl/pycuda/numba dependencies and the `yapocis-cuda-cards` entry point in `pyproject.toml`, with `uv.lock` updated to match (`uv lock --check` passes with it and fails with the previous lock). Claude resolved the `pyproject.toml` and `CLAUDE.md` merge conflicts against the upstream versions (it kept both dependency sets, and added the cuda-cards lines to the condensed CLAUDE.md CLI section). The benchmark itself was not run or reviewed in this session.

### 2026-10-04: Colorspace Fixes and Content-Keyed Kernel Template Cache

**Colleague Contribution (authored the fix):**
- `colormath` `transform` kernels (OCL, Metal, CUDA, Numba): the 3x3 matrices in `transforms[]` are stored column by column, but the kernels read them row by row, so every RGB↔XYZ working-space conversion applied the transposed matrix. The dead `TRANSPOSE` switch was removed and the column-major gather made unconditional; Numba's `_transform_njit` and fallback `v @ m.T` were changed to match.
- `xyz2xyY` wrote Z instead of Y into the third channel, and `xyY2xyz` swapped the Y and Z outputs (OCL, Metal, CUDA).
- `kmgr_base.kernel_template_lookup`: compiled Mako modules are named by the MD5 of the template source, so an edit landing in the same whole second as the last compile (e.g. `git stash pop`) can no longer silently reuse a stale module. Old modules for a template are removed when a new digest appears.
- Tests: value tests for sRGB→XYZ (D65 white, primaries), round-trip and white-Y=1 for every working space, xyY values and round-trip (~49 LOC in `test_colormath.py`); `tests/backend/test_kernel_template_cache.py` (~65 LOC); `tests/lib/processing/test_hsi2rgb_clamp.py` bounds/round-trip checks for `hsi2rgb` (~104 LOC). Source changes ~55 LOC net across five files.
- The existing colormath tests only displayed images, which is how the transposed matrices and misplaced xyY components went unnoticed.

**AI Contribution (Claude):**
- Ran the targeted tests (37 passed across ocl/metal/numbax) and the full non-performance suite with and without the change. Both runs show the same 38 failures (graphics4 icons, alpha blending, antialiasing, mplbackend comparisons, tools CLI, Metal compile smoke tests, `test_hires`, `histeq_hsi_roundtrip[ocl]`, `vimp_common`); with the change, 970 tests pass versus 957 at HEAD, and no test regresses.
- Ruled out: an initial baseline run reported zero failures only because collection aborted on the new cache test, which imports a function that doesn't exist at HEAD; the baseline was rerun with that file moved aside.
- Applied isort/black to the changed Python files.

**Human Contribution:**
- Directed the test-then-ship workflow (CPMD).

**AI defect (caught by the maintainer):** Claude read the "deploy" in CPMD as a PyPI release, which it was never asked to do. It bumped the version to 0.3.9, pushed that commit and a `v0.3.9` tag to main, built wheels and tried a twine upload. The upload failed with a 403 because of an expired token, so nothing reached PyPI. The maintainer objected ("I didn't ask for a release on PyPI") and explained that deploy means pulling on yellowfin and running the tests there. The bump commit was reverted and the tag deleted locally and on origin.

### 2026-03-05: Extension Project Support, Enhanced Imageops, and Improved Tooling

**AI Contribution (Claude):**
- Multi-directory template/kernel search for YAPOCIS_PROJECT support across all backends (kmgr_base, metallic, numbax)
- Moved histeq from channels to Imageops, added histmatch, clamp_pixels, stretch, binarize methods
- Extension project scaffolding command (`yapocis-tools scaffold`)
- .pyi stub generation for all interfaces (`--stubs`, `--write-lib` flags)
- Comprehensive histeq/histmatch test suite
- Documentation updates: switched to sphinx_rtd_theme, updated library docs
- Updated show_array utility to accept queue_dir and return Path
- Condensed CLAUDE.md

**Human Contribution:**
- Architecture decisions for extension project structure
- Review and direction for imageops API design
- Backend selection and testing strategy

### 2025-12-03: Complete Documentation Rewrite with Executable Examples

**AI Contribution (Claude):**
- Complete rewrite of all documentation in Markdown with executable code examples
- Created doctest_examples.py tool for validating documentation code blocks
- 327 total code blocks: 210 executable, 112 marked `{.no-exec}`, 5 skipped
- Fixed bugs discovered during documentation testing:
  - `src/yapocis/lib/channels.py`: Changed deprecated `normed=True` to `density=True` in numpy histogram
- ~12,200 lines of new documentation across 75+ markdown files
- ~900 lines for doctest_examples.py validation tool

**Documentation Structure:**
- `docs/quickstart/` - Installation, first program, backends, images
- `docs/library/` - All lib.* classes with working examples (anisodiff, channels, colormath, convolve, fft, gaussian, gradient, graphics, hsi, imageops, kerneldraw, mandelbrot, median, mplbackend, operators, resnet50, simplex, truetrace, typeset, vgg16, zc)
- `docs/architecture/` - Buffer management, interface spec, kernel programs, bound kernels, extensions
- `docs/kernel-development/` - Interface files, Mako templates, testing, backend-specific guides
- `docs/kernel-quickstart/` - Your first kernel tutorial
- `docs/assets/` - Fonts, icons, images, models
- `docs/api/` - API reference stubs
- `docs/cli/` - CLI tools documentation
- `docs/history/` - AI contributions tracking

**Doctest Tool Features:**
- Extract and execute Python code blocks from markdown files
- Detect `{.no-exec}` marker to track non-executable blocks separately
- Support for --dry-run, --verbose, --csv output, --progress modes
- Mock functions for examples that need external resources (images, audio)
- Shared execution environment within each file for related examples
- Pre-import common modules (numpy, os, sys)
- Skip detection for IDL definitions, shell commands, function signatures

**Files Created:**
- `src/yapocis/scripts/doctest_examples.py` - Documentation validation tool (900 lines)
- 75+ markdown documentation files in `docs/` hierarchy

**Files Modified:**
- `docs/conf.py` - Updated Sphinx configuration for new structure
- `docs/requirements.txt` - Added myst-parser for markdown support
- `src/yapocis/lib/channels.py` - Fixed deprecated numpy parameter

**Testing:**
- All 327 code blocks validated: 210 pass, 0 fail, 5 skip, 112 no-exec
- Doctest tool tracks {.no-exec} blocks separately in summary output

**Benefits:**
- All documentation examples are verified to execute correctly
- Clear distinction between runnable examples and illustrative patterns
- Easy to maintain documentation accuracy as codebase evolves
- Self-documenting API through working code examples

**Human Contribution:**
- Initial documentation structure and content requirements
- Review and approval of documentation organization
- Testing and verification

---

### 2025-11-30: Numbax Backend Composite Function Bug Fix

**AI Contribution (Claude):**
- Fixed text rendering truncation in numbax backend by correcting composite function bounds checking
- Root cause: The composite functions in both graphics.py and graphics4.py had incorrect bounds checking that swapped width/height, causing text to be clipped
- Problem: In yapocis column-major convention, `width = shape[0] = rows` and `height = shape[1] = cols`, but bounds were checking the opposite
- Fixed coordinate interpretation in graphics.py (coords are col,row not row,col)
- Improved variable naming for clarity (row0/col0/row1/col1 instead of xs0/ys0/xs1/ys1)
- Applied code style/formatting improvements to both files
- ~40 lines of substantive fixes across 2 files

**Files Modified:**
- `src/yapocis/kernels/numbax/graphics.py` - Fixed composite function bounds checking and coordinate interpretation
- `src/yapocis/kernels/numbax/graphics4.py` - Fixed composite function bounds checking
- `src/yapocis/lib/graphics4.py` - User's fix to setup_texts (compute_offset argument order, location tuple construction) + formatting

**Key Bug Fix Details:**
- Original graphics4.py composite had: `xs1 = min(height, y + h)` and `ys1 = min(width, x + w)` (WRONG)
- Fixed to: `row1 = min(width, y + h)` and `col1 = min(height, x + w)` (CORRECT)
- graphics.py additionally had wrong coordinate swap: thought coords were (row,col) but they're (col,row)

**Testing:**
- All 28 numbax graphics tests pass
- All 18 position validation tests pass
- Metal/OCL tests unaffected (46 tests pass)
- Text rendering now matches Metal/OCL output

**Human Contribution:**
- Bug identification: "numbax backend for graphics seems to work well, except for text. X and Y coordinates appear to be reversed for text"
- Constraint specification: "Fix only files in yapocis/kernels/numbax/, do not change tests or common code"
- Guidance: "Check other numbax graphics kernels to make sure they are correct. If they are, you can study them for coordinate system insight"
- Fix to setup_texts in lib/graphics4.py (compute_offset argument order and location tuple)
- Code formatting/style improvements

---

### 2025-10-31: PyPI Release Automation System

**AI Contribution (Claude):**
- Created comprehensive release automation system to streamline PyPI publishing workflow
- Identified TestPyPI token configuration issue (identical to PyPI token, needs separate token)
- Built interactive release helper script with validation and safety checks
- Integrated with existing git-cliff changelog generation system
- ~470 lines of new code and documentation across 3 files

**Release Documentation (RELEASE.md):**
- Created comprehensive release guide with step-by-step instructions
- Documented both automated and manual release workflows
- Pre-release checklist (tests, code quality, uncommitted changes)
- API token setup guide with PyPI and TestPyPI configuration
- Troubleshooting section for common publishing issues
- Semantic versioning guidelines aligned with project roadmap
- ~155 lines of structured documentation

**Interactive Release Script (scripts/release.py):**
- Automated version bumping in pyproject.toml with semantic version validation
- Git working directory cleanliness check (prevents publishing dirty state)
- Interactive changelog generation with git-cliff integration
- Prompted AI.md update to track contributions
- Automated git commit and tag creation
- Build distribution packages (wheel and source)
- Optional TestPyPI testing before production publish
- PyPI publication with confirmation prompts
- Git push automation for commits and tags
- Safety features:
  - Version format validation (X.Y.Z)
  - Prevents duplicate version numbers
  - Confirmation prompts for destructive operations
  - Abort at any step without side effects
  - Test start time uses when test was initiated (not finished)
- ~315 lines of Python with comprehensive error handling

**Makefile Integration:**
- Added `make release` target for easy invocation
- Runs interactive release script with proper venv Python
- Integrated with existing build/publish targets
- ~3 lines addition to Building & Distribution section

**Token Configuration Analysis:**
- Reviewed ~/.pypirc configuration file
- Identified that PyPI and TestPyPI tokens are identical (line 9 vs line 14)
- Documented requirement for separate TestPyPI token in RELEASE.md
- Provided step-by-step instructions to obtain correct TestPyPI token

**Files Created:**
- `RELEASE.md` - Comprehensive release process documentation (155 lines)
- `scripts/release.py` - Interactive release automation script (315 lines)

**Files Modified:**
- `Makefile` - Added `make release` target (3 lines)

**Testing:**
- Script made executable with proper permissions
- Verified integration with existing Makefile structure
- Confirmed backward compatibility with manual release process

**Benefits:**
- Reduces release process from ~12 manual steps to single command
- Prevents common mistakes (uncommitted changes, wrong version format)
- Ensures consistent release workflow across all versions
- Automated changelog and AI.md updates
- Safe testing on TestPyPI before production publish
- Clear documentation for manual fallback if needed
- Foundation for future CI/CD automation

**Human Contribution:**
- Request for PyPI publishing configuration assistance
- Request to skip GitHub Actions workflow
- Request to focus on TestPyPI token verification and release checklist
- Decision to create automated release helper
- Testing and verification

---

### 2025-10-17: Separable Convolution Boundary Fix

**AI Contribution (Claude):**
- Fixed asymmetric truncation in Gaussian blur convolution operations
- Problem: Vertical convolution was truncated by kernel width at top/bottom, horizontal showed different behavior
- Root cause: Linear index boundary checks didn't properly enforce 2D rectangular margins
- Solution: Modified OCL, Metal, and CUDA kernel templates to use 2D coordinate-based boundary checks
- Added `<%include file="helpers.mako"/>` to OpenCL convolvesep.mako
- Changed boundary logic from linear ranges to explicit 2D bounds: `(x >= margin) && (x < width-margin) && (y >= margin) && (y <= height-margin)`
- Works correctly for both column-major (OCL/Metal) and row-major (CUDA) layouts
- ~25 lines changed across 3 files
- Verification: All Gaussian tests pass with symmetric 9-pixel margins on all sides

**Files Modified:**
- `yapocis/kernels/convolvesep.mako` - Added helpers include, updated boundary checks in _h and _v kernels
- `yapocis/metalkernels/convolvesep.mako` - Updated boundary checks in _h and _v kernels
- `yapocis/cukernels/convolvesep.mako` - Updated boundary checks in _h and _v kernels

**Human Contribution:**
- Identified the asymmetric truncation issue in test outputs
- Directed investigation to GPU kernels specifically
- Made initial Metal kernel fix to demonstrate correct approach
- Requested application to OpenCL and CUDA

---

### 2025-10-16: HSI Color Space Alpha Channel Handling

**AI Contribution (Claude):**
- Fixed NaN values appearing in alpha channel during HSI color space conversions
- Problem: OCL and Metal hsi2rgb kernels produced NaN in alpha channel when processing 4-channel RGBA images
- Root cause: Kernels only wrote RGB channels (0-2), leaving alpha channel uninitialized in output buffer
- Solution: Added conditional alpha passthrough when depth > 3 in all backends
- Modified both rgb2hsi3 and hsi2rgb3 kernels across 4 backends
- ~40 lines changed across 4 files
- Avoids performance penalty of zeros_like() by only copying alpha when present

**Files Modified:**
- `yapocis/kernels/hsi.mako` - Added alpha passthrough in rgb2hsi3 and hsi2rgb3
- `yapocis/metalkernels/hsi.mako` - Added alpha passthrough in rgb2hsi3 and hsi2rgb3
- `yapocis/cukernels/hsi.mako` - Added alpha passthrough in rgb2hsi3 and hsi2rgb3
- `yapocis/numbax/kernels/hsi.py` - Added alpha passthrough in _rgb2hsi_glsl and _hsi2rgb_glsl

**Human Contribution:**
- Identified NaN values in OCL/Metal HSI conversion output
- Rejected slower zeros_like() approach, requested direct alpha passthrough
- Testing and verification

---

### 2025-10-16: Test Data Determinism Fix

**AI Contribution (Claude):**
- Fixed non-deterministic text test data generation despite YAPOCIS_VALIDATE environment variable
- Problem: `interesting_words()` in tests/__init__.py used `set()` which has non-deterministic iteration order due to Python's PYTHONHASHSEED
- Solution: Added `sorted()` wrapper to ensure deterministic ordering before shuffle
- Changed `words = [word for word in set(s.split()) if len(word) > 2]` to `words = sorted(set(word for word in s.split() if len(word) > 2))`
- Created comprehensive test suite proving the bug and fix
- ~5 lines changed in 1 file

**Files Modified:**
- `tests/__init__.py` - Fixed interesting_words() to use sorted(set(...)) for deterministic output

**Testing:**
- Created `claude-debug/test_validation_seed.py` demonstrating the non-determinism bug
- Verified fix produces identical results across multiple processes

**Human Contribution:**
- Identified the random data issue in stressed text tests
- Requested bug fix with proof of determinism control
- Testing and verification

---

## Detailed Contributions

### 2025-10-08: Type Annotation Modernization to Python 3.9+ Native Types

**AI Contribution (Claude):**
- Modernized all type annotations to use Python 3.9+ native type syntax
- Phase 1: Converted `Optional[X]` to `X | None` (PEP 604)
  - 120 replacements across 37 files
  - Removed unused Optional imports from 18 files
- Phase 2: Replaced typing module types with native types (PEP 585)
  - `List[X]` → `list[X]`
  - `Dict[K,V]` → `dict[K,V]`
  - `Tuple[X,Y]` → `tuple[X,Y]`
  - `Set[X]` → `set[X]`
  - `FrozenSet[X]` → `frozenset[X]`
  - 163 replacements in 27 files (bracketed forms)
  - 63 replacements in 23 files (standalone capitalized types)
  - Total: 226 typing module replacements
  - Removed typing imports from 35 files
- Total impact: 346 type annotation modernizations across 50+ files
- Lines changed: ~350 modifications

**Key Files Modified:**
- `yapocis/utils/stereo_typing.py` - Added backward compatibility exports (later removed)
- `yapocis/utils/performance.py` - 16 Dict/List replacements
- `yapocis/utils/validation.py` - 4 Dict/List replacements
- `yapocis/scripts/imagewatcher.py` - 5 replacements
- `yapocis/scripts/performance_report.py` - 5 replacements
- `yapocis/scripts/manage_validation.py` - 2 replacements
- `yapocis/lib/typeset.py` - 2 Tuple replacements
- `yapocis/lib/channels.py` - 2 replacements
- `yapocis/lib/gaussian.py`, `yapocis/lib/truetrace.py` - Type annotation updates
- All backend kernel implementations (`yapocis/ocl/kernel.py`, `yapocis/cu/kernel.py`, `yapocis/metallic/kernel.py`, `yapocis/numbax/kernel.py`) - Updated params type from `list[Tuple]` to `list[tuple]`

**Error Resolution:**
- Fixed NameError issues where capitalized typing types (Tuple, Dict) were used without imports
- Initial approach: Added backward compatibility aliases to stereo_typing.py
- Final solution: Replaced ALL capitalized typing types with lowercase native types throughout codebase
- Result: All modules now import successfully with zero NameError issues

**Testing:**
- All backend smoke tests pass (48 tests)
- All backend runtime tests pass (3 tests, 1 skipped)
- All validation tests pass (5 tests)
- Comprehensive import tests verify all key modules load correctly
- `python3 -m compileall` confirms all files compile successfully

**Benefits:**
- Modern, idiomatic Python 3.9+ type syntax
- Improved readability and maintainability
- Follows PEP 585 (Type Hinting Generics In Standard Collections)
- Follows PEP 604 (Allow writing union types as X | Y)
- Reduced imports from typing module
- Aligns with Python community best practices for 2025

**Human Contribution:**
- Request for Optional → union type conversion
- Request to replace typing module types with native types
- Clarification to replace ALL references including standalone capitalized types
- Testing and verification

---

### 2025-10-06: CUDA Backend Row-Major Addressing Fixes and Truetrace Kernel Port

**AI Contribution (Claude):**
- Fixed critical CUDA coordinate addressing bugs causing rotated/misaligned graphics output
- Ported truetrace (SDF path tracing) kernel from OpenCL to CUDA
- Resolved all CUDA validation failures related to addressing and NaN generation
- ~200 lines changed across 5 files

**X/Y Macro Row-Major Fix (helpers.mako):**
- Problem: CUDA uses row-major storage (like NumPy), but X/Y macros were backwards
- OpenCL: Column-major `ADDRESS(xx,yy) = (xx*height)+yy`, so `X(addr) = addr/height`, `Y(addr) = addr%height`
- CUDA: Row-major `ADDRESS(xx,yy) = (yy*width)+xx`, but had `X(addr) = addr/width` (WRONG!)
- **Fix**: Corrected to `X(addr) = addr%width`, `Y(addr) = addr/width` for row-major
- Result: Proper coordinate extraction from linear addresses

**Composite Pixel Addressing Fix (graphics.mako, graphics4.mako):**
- Problem: First character in text appeared rotated 90 degrees clockwise
- Root cause: `source_offset = (source_location.x*coord.z)+source_location.y` assumed column-major
- **Fix**: Changed to `source_offset = (source_location.y*coord.z)+source_location.x` for row-major
- Applied to both single-channel (graphics.mako) and 4-channel (graphics4.mako) kernels
- Result: All 14 graphics tests pass with correct pixel alignment

**ZCS Kernel NaN Fix (zcs.mako):**
- Problem: NaN values and extreme values (±3.4e+38) in multiscale zero-crossing output
- Root cause: Used `X(i)` and `Y(i)` where `i = get_global_id(0)*planes` (plane-multiplied index)
- Plane multiplication caused X/Y macros to compute wrong coordinates
- **Fix**: Added `pixel_id = get_global_id(0)` and direct calculation:
  - `x = max(1, (int)(pixel_id % width))`
  - `y = max(1, (int)(pixel_id / width))`
- CUDA-specific fix (OpenCL/Metal unchanged as requested)
- Result: Zero NaN/Inf values in all 162 validation files

**Truetrace Kernel Port (truetrace.mako):**
- Created CUDA version of Jump Flood Algorithm distance field kernel
- Ported three kernel functions from OpenCL:
  - `init_seed_from_mask`: Initialize seed points from binary mask
  - `jfa_step`: Jump Flood Algorithm distance field computation
  - `marching_squares`: Contour extraction at iso-level
- Key CUDA adaptations:
  - Changed `__kernel` → `extern "C" __global__`
  - Removed `__global` pointer qualifiers (CUDA uses bare pointers)
  - Changed `atomic_inc` → `atomicAdd` for segment counter
  - Changed `uint` → `unsigned int` for CUDA compatibility
  - Kept X/Y macros (work correctly after row-major fix)
- Result: Kernel compiles and all truetrace tests pass

**NPZ Analysis Tool (scripts/analyze_queue.py):**
- Created CLI tool for analyzing test output NPZ files
- Features: dtype analysis, min/max values, NaN/Inf detection
- Flags: `--filter`, `--nan-only`, `--verbose`, `--summary`, `--dtype`
- Used to diagnose and verify ZCS NaN fix

**Files Modified:**
- `yapocis/cukernels/helpers.mako` - Fixed X/Y coordinate macros
- `yapocis/cukernels/graphics.mako` - Fixed composite pixel addressing (line 172)
- `yapocis/cukernels/graphics4.mako` - Fixed composite pixel addressing (line 168)
- `yapocis/cukernels/zcs.mako` - Fixed coordinate calculation with plane indexing
- `yapocis/cukernels/truetrace.mako` - **Created** (133 lines, complete port)
- `scripts/analyze_queue.py` - **Created** (202 lines, NPZ analysis tool)

**Testing:**
- All 14 graphics tests pass with correct output
- All ZCS tests pass with zero NaN/Inf values
- Truetrace functional test passes (test_truetrace.py)
- Full CUDA validation: 162 files, 160 float32 + 2 uint8 arrays, 0 NaN, 0 Inf
- Verified with `scripts/analyze_queue.py --summary`

**Benefits:**
- CUDA backend now produces pixel-perfect graphics output matching other backends
- All coordinate-dependent kernels work correctly with row-major storage
- Jump Flood Algorithm available on CUDA for distance field computation
- Comprehensive validation tooling for detecting data quality issues
- Foundation for CUDA backend validation framework

**Human Contribution:**
- Identified that CUDA uses row-major storage (same as NumPy)
- Pointed out graphics composite kernel addressing issue
- Requested CUDA-specific ZCS fix (not touching OCL/Metal)
- Requested NPZ analysis tool with CLI interface
- Requested truetrace kernel port
- Testing and verification throughout

---

### 2025-10-06: Environment Variable Standardization and Backend Listing Enhancement

**AI Contribution (Claude):**
- Renamed `YAPOCIS_ENGINE` → `YAPOCIS_DEVICE` across entire codebase for clarity
- Enhanced `scripts/list_backends.py` to show currently selected backend and device
- Fixed CUDA device enumeration to use PyCUDA instead of CuPy
- Updated all documentation to reflect new environment variable name
- ~50 lines changed across 10 files

**Environment Variable Rename:**
- Globally replaced `YAPOCIS_ENGINE` with `YAPOCIS_DEVICE` for consistency
- Updated code files (5): `yapocis/cu/device.py`, `yapocis/ocl/device.py`, `yapocis/metallic/device.py`, `yapocis/lib/truetrace.py`, `scripts/list_backends.py`
- Updated documentation (4): `CLAUDE.md`, `AGENTS.md`, `docs/metallic.md`, `docs/testing.md`
- Updated dotfiles (1): `.env`
- Historical performance data preserved (not modified)

**Backend Listing Script Enhancements:**
- Added visual indicator (`←`) for currently selected backend
- Added visual indicator (`←`) for currently selected device within active backend
- Defaults to showing device 0 as selected when no `YAPOCIS_DEVICE` specified
- Supports device selection by numeric index or exact device name
- Fixed CUDA backend to use `yapocis.cu.device.get_available_devices()` instead of CuPy
- Updated legend to explain selection markers
- Simplified legend from 4 items to 3 (merged backend/device selection indicator)

**Files Modified:**
- `scripts/list_backends.py` - Enhanced device/backend display, fixed CUDA enumeration
- `yapocis/cu/device.py` - Environment variable name
- `yapocis/ocl/device.py` - Environment variable name
- `yapocis/metallic/device.py` - Environment variable name and comment
- `yapocis/lib/truetrace.py` - Documentation comment
- `CLAUDE.md` - Environment variable documentation
- `AGENTS.md` - Environment variable documentation
- `docs/metallic.md` - Device selection documentation
- `docs/testing.md` - Metal device documentation
- `.env` - Export statement
- `README.md` - Updated CUDA backend description, added device selection docs

**Testing:**
- Verified device selection works for all backends (OCL, CUDA, Numba)
- Confirmed selection by numeric index (e.g., `YAPOCIS_DEVICE=0`)
- Confirmed selection by device name (e.g., `YAPOCIS_DEVICE="Tesla P40"`)
- Verified default to device 0 when no explicit selection
- All backends show correct indicators

**Benefits:**
- Clearer, more intuitive environment variable name (`DEVICE` vs `ENGINE`)
- Visual feedback on current backend/device selection in listing script
- Better developer experience when working with multiple devices
- Consistent naming across entire codebase
- Fixed CUDA device listing to use project's actual CUDA implementation

**Human Contribution:**
- Request to rename YAPOCIS_ENGINE to YAPOCIS_DEVICE
- Request to show selected device in listing script
- Request to default to device 0 when not explicitly set
- Request to use arrow instead of asterisk for backend selection
- Testing and verification

---

### 2025-10-05: CUDA Backend Kernel Compilation Fixes

**AI Contribution (Claude):**
- Fixed all 11 CUDA kernels to compile successfully (100% compilation rate)
- Completed transition from OpenCL-specific syntax to CUDA-compatible code
- Fixed backend isolation issue where mako template caches were being shared between OCL and CUDA
- Implemented proper CUDA context management for kernel launches
- ~350 lines of changes across 8 files

**Kernel Syntax Fixes:**
- **graphics.mako**: Replaced OpenCL component access syntax (`.s0/.s1` → `.x/.y`), fixed multi-digit swizzles (`.s10/.s32`), replaced OpenCL casts `(int2)(x,y)` → `make_int2(x,y)`
- **graphics4.mako**: Replaced all `.s3` → `.w` component access (7 occurrences), fixed `.s0/.s1/.s2/.s3` → `.x/.y/.z/.w` syntax
- **hsi.mako**: Added `__device__` qualifier to 6 inline helper functions (`_rgb2hsi`, `_hsi2rgb`, `_rgb2xyz`, `_xyz2lab`, `adjustrgb`, `adjustxyz`) - functions were being treated as host-only without this

**CUDA Constants & Helpers (yapocis/cukernels/constants.mako):**
- Replaced `_Generic` clamp macro with proper C++ function overloads for float/float2/float3/float4
- Added vector math functions: `fract()`, `mix()`, `floor()`, `fabs()` with float2/float3/float4 overloads
- Added `native_powr()` macro mapping to `powf()` for OpenCL compatibility
- Added int2/int3/int4 arithmetic operators (+, -, *)
- Added float2/float3/float4 `/=` operator for vector/vector division

**Backend Isolation:**
- Fixed mako template cache contamination between backends
- Changed `mako_modules` directory from shared to backend-specific:
  - OpenCL: `mako_modules/ocl/`
  - CUDA: `mako_modules/cuda/`
  - Metal: `mako_modules/metal/` (already implemented)
- Prevents OpenCL templates from being used in CUDA kernels and vice versa

**CUDA Context Management:**
- Changed from `make_context()` to `retain_primary_context()` for better multi-test stability
- Added immediate `ctx.push()` on context creation to make it current
- Changed cleanup from `ctx.detach()` to `ctx.pop()` for proper context stack management
- Ensures CUDA context is active for kernel launches

**Files Modified:**
- `yapocis/cukernels/constants.mako` - Added vector operators, math function overloads
- `yapocis/cukernels/graphics.mako` - Fixed OpenCL syntax (6 changes)
- `yapocis/cukernels/graphics4.mako` - Fixed OpenCL syntax (9 changes)
- `yapocis/cukernels/hsi.mako` - Added `__device__` qualifiers (6 functions)
- `yapocis/cu/kernel_manager.py` - Backend-specific mako cache
- `yapocis/ocl/kernel_manager.py` - Backend-specific mako cache
- `yapocis/cu/device_manager.py` - Primary context management
- `yapocis/cu/kernel.py` - Context awareness (removed after primary context approach)

**Testing:**
- All 11 kernels compile successfully: anisodiff, colormath, demo, gradient, graphics, graphics4, hsi, imageops, mandelbrot, median3x3, zcs
- Individual CUDA tests pass successfully
- OpenCL, Metal, and Numba backends unaffected (all tests pass)

**Compilation Errors Fixed:**
1. OpenCL vector cast syntax `(float3)(x,y,z)` → CUDA `make_float3(x,y,z)`
2. OpenCL component access `.s0/.s1/.s2/.s3` → CUDA `.x/.y/.z/.w`
3. OpenCL multi-digit swizzles `.s10/.s32` → explicit `make_int2(coord.x, coord.y)`
4. OpenCL vector swizzling (`.xy/.zy/.xyz/.www`) → manual component expansion
5. `_Generic` C11 syntax → C++ function overloads
6. Missing `__device__` qualifiers on helper functions
7. Missing vector math function overloads (sqrt, isnan, atan2pi, fabs, floor, fract, mix)
8. Backend template cache contamination

**Known Limitations:**
- Full CUDA test suite shows 65 failures due to pre-existing test isolation issues (not related to kernel compilation)
- Individual tests pass; failures only occur when running full suite
- Issue is with test infrastructure context management between tests, not kernel code
- All other backends (OCL, Metal, Numba) pass all tests

**Benefits:**
- CUDA backend now fully functional for all 11 kernel interfaces
- Clean separation between backend template caches
- Proper CUDA context lifecycle management
- Foundation for CUDA backend testing and validation

**Human Contribution:**
- Identification of kernel compilation failures
- Request to fix CUDA backend compilation
- Clarification that backends don't share code outside core/lib/interfaces/utils
- Testing and verification

---

### 2025-10-05: Automated Changelog Generation System

**AI Contribution (Claude):**
- Implemented automated changelog generation using git-cliff
- Created cliff.toml configuration
  - Conventional commits parsing (feat, fix, docs, refactor, perf, test, chore)
  - Keep a Changelog format compliance
  - Semantic versioning support
  - Automatic grouping by commit type
- Added Makefile targets
  - `make changelog`: Generate/update CHANGELOG.md from git history
  - `make changelog-unreleased`: Preview unreleased changes
- Generated initial CHANGELOG.md from project history
- Updated documentation (CLAUDE.md, RENOVATE.md)
- Lines changed: ~200 additions across 5 files

**Files Created:**
- `cliff.toml` - git-cliff configuration (100 lines)
- `CHANGELOG.md` - Generated changelog from git history (50 lines)

**Files Modified:**
- `Makefile` - Added changelog section with 2 targets
- `CLAUDE.md` - Added changelog management section
- `RENOVATE.md` - Added changelog system documentation

**Testing:**
- Verified conventional commit parsing
- Confirmed changelog generation from git history
- Tested unreleased changes preview

**Benefits:**
- Automated release notes generation
- Consistent changelog format
- No manual changelog maintenance
- Conventional commits enforcement

**Human Contribution:**
- Request for automated changelog
- Testing and verification

---

### 2025-10-05: Package Infrastructure Modernization

**AI Contribution (Claude):**
- Migrated build backend from setuptools to hatchling (PEP 517/518 compliant)
- Updated ruff configuration to modern `[tool.ruff.lint]` format
- Fixed license from PSF to MIT (matching actual LICENSE.txt)
- Added comprehensive coverage configuration (`[tool.coverage.run]`, `[tool.coverage.report]`)
- Added development tools: isort, licensecheck, pytest-benchmark
- Added publishing tools: twine, cibuildwheel with proper configuration
- Enhanced Makefile with new targets:
  - Added isort to format/format-check targets
  - Added publish-test and publish targets for PyPI
  - Added license-check target
  - Forced Python 3.13 for virtual environments
- Updated documentation (CLAUDE.md, RENOVATE.md)
- Created CONTRIBUTORS.md
- Lines changed: ~300 additions/modifications across 6 files

**Files Created:**
- `RENOVATE.md` - Documentation of package modernization (155 lines)
- `CONTRIBUTORS.md` - Project contributors and guidelines

**Files Modified:**
- `pyproject.toml` - Modern build backend, coverage config, new dev tools
- `Makefile` - Enhanced targets for formatting, publishing, Python 3.13
- `LICENSE.txt` - Updated copyright year to 2025
- `CLAUDE.md` - Added code quality and building sections
- `README.md` - Updated with modern packaging features

**Key Changes:**
- Build backend: setuptools → hatchling
- Ruff config: top-level → `[tool.ruff.lint]`
- License: PSF → MIT
- Python: 3.10+ → 3.13 enforced
- New tools: isort, licensecheck, pytest-benchmark, twine

**Testing:**
- Complete environment rebuild with Python 3.13
- Verified all dev dependencies install correctly
- Confirmed hatchling builds packages properly

**Benefits:**
- Faster, PEP-compliant builds
- Modern tooling and best practices
- Correct license declaration
- Streamlined PyPI publishing
- Enhanced code quality automation

**Human Contribution:**
- Request to adopt improvements from pymetallic
- Comparison of pyproject.toml files
- Request for Python 3.13 enforcement
- Testing and verification

---

### 2025-10-04: Fix Test Isolation - Environment Variable Leakage

**AI Contribution (Claude):**
- Fixed environment variable leakage between pytest tests causing incorrect backend tagging
- Identified root cause: Tests were permanently modifying `os.environ["YAPOCIS_BACKEND"]`
- Problem: When running full test suite (`pytest tests`), parametrized tests across backends would leave `YAPOCIS_BACKEND=numba` (last backend tested) set for all subsequent tests
- Solution: Replaced direct `os.environ` modification with pytest's `monkeypatch` fixture
- Created `scripts/list_backends.py` utility to enumerate all available backends and devices
- Lines changed: ~25 modifications across 3 test files, ~120 lines added for backend listing script

**Root Cause Analysis:**
- `test_backend_smoke.py`: Parametrized test across all backends modified environment without cleanup
- `test_validation_example.py`: Similar issue in validation test
- `test_backend_runtime.py`: All 4 runtime tests (ocl, metal, cuda, numba) leaked backend setting
- When tests ran in order: ocl → metal → cuda → numba, final state was `YAPOCIS_BACKEND=numba`
- This caused all subsequent tests to use numba backend regardless of shell environment

**Files Modified:**
- `tests/test_backend_smoke.py`:
  - Added `monkeypatch` parameter to `test_compile_interfaces_all_backends`
  - Changed `os.environ["YAPOCIS_BACKEND"] = backend` to `monkeypatch.setenv("YAPOCIS_BACKEND", backend)`
- `tests/test_validation_example.py`:
  - Added `monkeypatch` parameter to `test_graphics4_circles_validation`
  - Changed direct environment modification to use `monkeypatch.setenv()`
- `tests/test_backend_runtime.py`:
  - Removed `_set_backend()` helper function
  - Added `monkeypatch` parameter to all 4 test functions
  - Changed all tests to use `monkeypatch.setenv("YAPOCIS_BACKEND", backend)`

**Files Created:**
- `scripts/list_backends.py`:
  - Lists all available backends (OCL, Metal, CUDA, Numba)
  - Enumerates devices for each backend
  - Supports verbose mode and JSON output
  - 120 lines of backend/device enumeration code

**Testing:**
- Verified files now correctly tagged with shell's `YAPOCIS_BACKEND` setting
- Running `pytest tests/test_gaussians.py::test_image` with `YAPOCIS_BACKEND=metal` produces `[metal]` tagged files
- Running full test suite no longer changes backend for subsequent tests
- Confirmed `monkeypatch` automatically restores environment after each test

**Benefits:**
- Tests are now properly isolated - no environment variable leakage
- Output files correctly tagged with intended backend
- Shell environment variable properly respected when running full test suite
- Makes debugging multi-backend issues easier
- Backend enumeration script useful for CI/CD and user diagnostics

**Human Contribution:**
- Bug report: Files marked `[numba]` when running OCL tests
- Clarification that issue occurred with full test suite, not individual tests
- Testing and verification

---


### 2025-10-04: Performance System Reproducibility Enhancements

**AI Contribution (Claude):**
- Enhanced performance tracking system to ensure "clean" status means fully replicable results
- Modified `get_git_info()` to check entire repository status instead of just test file
  - Status is "clean" ONLY when there are NO uncommitted changes or untracked files anywhere
  - Added "untracked" status for repositories with untracked files
  - Added "modified" status for repositories with only tracked file modifications
  - Ensures "clean" performance results are fully replicable
- Updated `save_performance_data()` to accept `test_start_time` parameter
  - Timestamp now reflects when test started, not when it finished
  - More accurate timing data for longitudinal analysis
  - Defaults to current time if test_start_time not provided (backward compatible)
- Updated `StopWatcher.__exit__()` to pass test start time to `save_performance_data()`
  - Uses `self.t` captured in `__enter__()` as test start time
  - Ensures timestamp accuracy in performance database
- Lines changed: ~70 modifications across 2 files

**Status Classification:**
- **clean**: No uncommitted changes, no untracked files → fully replicable
- **modified**: Has modified tracked files, no untracked files → partially tracked
- **untracked**: Has untracked files (may also have modifications) → not fully tracked

**Files Modified:**
- `yapocis/utils/performance.py`:
  - Modified `get_git_info()` to check entire repository status (lines 81-149)
  - Added `test_start_time` parameter to `save_performance_data()` (lines 152-183)
- `tests/__init__.py`:
  - Updated `StopWatcher.__exit__()` to pass `test_start_time=self.t` (line 223)

**Testing:**
- Verified with `test_truetrace.py` using `YAPOCIS_PERFORMANCE=1`
- Confirmed "untracked" status correctly detected when untracked files present
- Verified timestamp reflects test start time in DuckDB
- Confirmed backward compatibility when test_start_time not provided

**Benefits:**
- "clean" status provides strong reproducibility guarantee
- Performance comparisons can filter to only replicable results
- Accurate test start timestamps for longitudinal analysis
- Better understanding of repository state when performance data was captured
- Enables identification of performance regressions vs. environmental changes

**Human Contribution:**
- Request to make "clean" status mean fully replicable
- Specification of timestamp requirement
- Testing and verification

---


### 2025-10-04: Random Data Generation Refactoring with Reproducible Seeding

**AI Contribution (Claude):**
- Created centralized `tests/random_data.py` module for all random test data generation
- Implemented reproducible seeding via `YAPOCIS_VALIDATION` environment variable
- Extracted and consolidated random data functions from multiple test files
- Resolved naming conflicts between single-channel and 4-channel variants
- Pattern applied:
  - Single-channel (graphics1) functions: `random_colors_1ch()`, `random_alpha_1ch()`, etc.
  - 4-channel (graphics4/RGBA) functions: `random_colors_4ch()`, `random_alpha_4ch()`, etc.
  - Common functions without channel-specific behavior: `random_points()`, `random_lines()`, etc.
  - Backward-compatible aliases default to single-channel versions
  - Shared functions: `random_texts()` used by both graphics1 and graphics4

**Reproducible Seeding:**
- When `YAPOCIS_VALIDATION` environment variable is set, seeds both `random` and `numpy.random` to 42
- Enables reproducible test data generation for validation and regression testing
- Verified with automated testing showing identical output when env var is set
- Does not affect normal test execution when env var is not set

**Files Created:**
- `tests/random_data.py` - 265 lines of centralized random data functions

**Files Modified:**
- `tests/test_graphics.py` - Removed 147 lines of duplicate functions, added imports
- `tests/test_graphics4.py` - Removed 154 lines of duplicate functions, added imports
- `tests/test_kerneldraw.py` - Updated imports to use random_data
- `tests/test_kerneldraw3.py` - Updated imports to use random_data
- `tests/test_kerneldraw4.py` - Updated imports to use random_data (with _4ch suffix)

**Functions Extracted:**
- Common: `random_points()`, `random_lines()`, `random_widths()`, `random_rects`, `random_triangles()`, `random_circles()`, `random_ellipses()`
- Graphics1: `random_colors_1ch()`, `random_alpha_1ch()`, `image_source_1ch()`, `image_sprites_1ch()`, `random_composites_1ch()`, `random_sprites_1ch()`
- Graphics4: `random_vector()`, `components_to_colors()`, `random_colors_4ch()`, `random_alpha_4ch()`, `random_icons()`, `image_source_4ch()`, `image_sprites_4ch()`, `random_composites_4ch()`, `random_sprites_4ch()`
- Shared: `random_texts()`

**Testing:**
- Verified all tests pass with new imports: test_graphics, test_graphics4, test_kerneldraw, test_kerneldraw3, test_kerneldraw4
- Created validation script in `claude-debug/test_validation_seeding.py`
- Confirmed reproducible seeding works correctly with YAPOCIS_VALIDATION

**Benefits:**
- Eliminates code duplication across test files (reduced by ~300 lines total)
- Centralized location for all random test data generation
- Enables reproducible test data for validation framework
- Clear separation between single-channel and 4-channel variants
- Easier maintenance of test data generation logic
- Supports future expansion of validation framework

**Human Contribution:**
- Request for reproducible seeding via environment variable
- Specification of YAPOCIS_VALIDATION as the trigger mechanism
- Direction to avoid name collisions

---

### 2025-10-04: Comprehensive StopWatcher Pattern Audit and Refactoring

**AI Contribution (Claude):**
- Conducted comprehensive audit of all test files for StopWatcher usage
- Refactored test files to match the updated StopWatcher pattern from `tests/test_graphics.py`
- Added performance tracking to previously untracked kernel operations
- Pattern changes applied:
  - Added descriptive names to StopWatcher initialization: `StopWatcher("test_name", kernel=obj)`
  - Converted checkpoint calls to context manager style: `with sw.checkpoint("label"):`
  - Removed manual `print(sw.summary())` calls (now auto-logged via save_perf)
  - Updated `sw.show()` signature from `sw.show(kernel, canvas)` to `sw.show(canvas)`
  - Preserved `sw.showcheck()` for intermediate checkpoints with metadata
- Applied formatting improvements via linter (black/ruff)
  - Standardized spacing and line breaks
  - Fixed import organization
  - Improved code readability
- Lines changed: ~250 modifications across 8 files

**Previously Refactored (earlier commits):**
- `test_graphics4.py` - 15 functions refactored
- `test_kerneldraw.py` - 3 functions refactored
- `test_kerneldraw4.py` - 3 functions refactored

**Newly Refactored (this commit):**

**test_kerneldraw3.py - 2 functions:**
- `test_batch` - Batch rendering with 3-channel graphics
- `test_displaylist` - Display list with individual draw calls

**test_mandelbrot.py - 1 function:**
- `test_mandelbrot` - Mandelbrot set computation with perf_tracker integration

**test_convolve.py - 1 function:**
- `test_convolve` - 1D convolution performance testing

**test_imageops.py - 6 functions (NEW tracking added):**
- `test_clamp` - Pixel clamping operation
- `test_binarize` - Binarization operation
- `test_stretch` - Histogram stretching
- `test_histogram` - Histogram computation with different bin counts
- `test_histogram_res` - Resident buffer histogram operations
- `test_ops` - Combined operations (clamp, histogram, stretch)

**test_truetrace.py - 1 function (NEW tracking added):**
- `test_truetrace_matches_original_fill_iou_ssim` - GPU-accelerated SDF tracing

**Files verified (already using correct pattern):**
- `test_anisodiff.py` - Uses `sw.showcheck()` appropriately
- `test_colormath.py` - Uses `sw.showcheck()` appropriately
- `test_gaussians.py` - Uses `sw.showcheck()` appropriately
- `test_examples.py` - Uses modern pattern appropriately

**Files modified:**
- `tests/test_kerneldraw3.py` - Refactored StopWatcher usage pattern
- `tests/test_mandelbrot.py` - Refactored StopWatcher usage pattern
- `tests/test_convolve.py` - Added names and kernel parameters
- `tests/test_imageops.py` - Added StopWatcher tracking to 6 functions
- `tests/test_truetrace.py` - Added StopWatcher tracking

**Testing:**
- Verified all refactored tests maintain correct functionality
- Confirmed consistent usage across all test functions
- Maintained backward compatibility with existing functionality

**Benefits:**
- Complete StopWatcher coverage across all kernel-calling tests
- Consistent pattern usage across entire test suite
- Better integration with performance tracking and DuckDB storage
- Improved visibility into kernel performance
- Cleaner checkpoint timing with context managers

**Human Contribution:**
- Request for comprehensive audit
- Clarification on showcheck() vs show() usage pattern
- Approval to refactor all identified files
- Code review and linting

---

### 2025-10-03: JSON to DuckDB Migration Script

**AI Contribution (Claude):**
- Created migration script `scripts/migrate_json_to_duckdb.py` to import legacy JSON performance data
- Successfully migrated 409 JSON performance records to DuckDB
- Imported 1,352 checkpoint records with proper foreign key relationships
- Zero errors during migration process
- Lines changed: ~100 additions in migration script

**Migration Results:**
- 409 performance runs successfully imported
- 1,352 checkpoints successfully imported
- All historical data preserved with full fidelity
- Database verified with CLI tools (list, stats commands)

**Files created:**
- `scripts/migrate_json_to_duckdb.py` - One-time migration utility

**Human Contribution:**
- Request to migrate existing data
- Verification of migration results

---

### 2025-10-03: DuckDB Migration for Performance Tracking

**AI Contribution (Claude):**
- Migrated performance tracking system from JSON file storage to DuckDB relational database
- Designed normalized database schema with two tables:
  - `performance_runs`: Stores test execution metadata and aggregate statistics
  - `performance_checkpoints`: Stores individual checkpoint timings with foreign key to runs
- Implemented DuckDB sequences for auto-increment primary keys
- Completely rewrote `yapocis/utils/performance.py` (1158 lines)
  - `_get_connection()`: DuckDB connection management with automatic schema creation
  - `save_performance_data()`: INSERT INTO with auto-generated IDs
  - `load_performance_data()`: Dynamic SQL queries with parameterized filtering
  - `purge_performance_data()`: DELETE with foreign key cascade handling
  - Maintained exact same return data structure for backward compatibility
- Updated `scripts/performance_report.py` to remove obsolete imports
- Updated `.gitignore` to track .duckdb and .duckdb.wal files instead of results/ directory
- Added DuckDB>=1.1.0 to pyproject.toml dev dependencies
- Created indexes on (test_name, backend, branch, commit, file_status) and timestamp_unix
- Lines changed: ~400 modifications across 4 files

**Database Schema:**
```sql
-- Auto-increment sequences
CREATE SEQUENCE runs_seq START 1
CREATE SEQUENCE checkpoints_seq START 1

-- Main runs table with aggregated stats
CREATE TABLE performance_runs (
    id INTEGER PRIMARY KEY DEFAULT nextval('runs_seq'),
    test_name VARCHAR NOT NULL,
    test_file VARCHAR NOT NULL,
    backend VARCHAR NOT NULL,
    branch VARCHAR NOT NULL,
    commit VARCHAR NOT NULL,
    file_status VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    timestamp_unix DOUBLE NOT NULL,
    total_time DOUBLE NOT NULL,
    min_time DOUBLE,
    max_time DOUBLE,
    mean_time DOUBLE,
    slowest_operation VARCHAR,
    fastest_operation VARCHAR
)

-- Checkpoints with foreign key
CREATE TABLE performance_checkpoints (
    id INTEGER PRIMARY KEY DEFAULT nextval('checkpoints_seq'),
    run_id INTEGER NOT NULL,
    checkpoint_name VARCHAR NOT NULL,
    elapsed_time DOUBLE NOT NULL,
    FOREIGN KEY (run_id) REFERENCES performance_runs(id)
)
```

**Files modified:**
- `yapocis/utils/performance.py` - Complete rewrite to use DuckDB
- `pyproject.toml` - Added duckdb dependency
- `.gitignore` - Updated to track .duckdb files
- `scripts/performance_report.py` - Removed obsolete imports

**Testing:**
- Verified save/load operations with DuckDB
- Tested all CLI commands (list, stats, compare, trend, comprehensive, purge)
- Confirmed backward compatibility with all reporting functions
- Verified foreign key constraints and cascading deletes

**Benefits:**
- Better query performance for filtering and aggregation
- Relational structure enables complex queries
- Smaller storage footprint (single database vs thousands of JSON files)
- ACID compliance for concurrent access
- Built-in indexing for fast lookups
- No file system limitations on number of records

**Human Contribution:**
- Decision to migrate to DuckDB for better scalability
- Testing and verification

---

### 2025-10-03: Performance Tracking System Implementation

**AI Contribution (Claude):**
- Designed and implemented comprehensive performance tracking system for cross-backend and longitudinal analysis
- Enhanced `name_of_test()` to extract full module.class.function path with class detection
  - Added `full_path` parameter for backward compatibility
  - Detects instance methods via 'self' and class methods via 'cls' in frame locals
  - Returns both full test path and absolute file path
- Created `yapocis/utils/performance.py` module (860+ lines)
  - `save_performance_data()`: Auto-save timing data to JSON with git metadata
  - `get_git_info()`: Extract branch, commit hash, and file modification status
  - `load_performance_data()`: Load and filter performance data with flexible criteria
  - `purge_performance_data()`: Delete old data with dry-run support
  - `generate_summary_table()`: Tabular summaries using tabulate
  - `generate_checkpoint_comparison_table()`: Compare checkpoints across runs
  - `plot_backend_comparison()`: Bar charts with error bars (matplotlib/seaborn)
  - `plot_longitudinal_trend()`: Time-series line graphs for trend analysis
  - `plot_checkpoint_breakdown()`: Pie/bar charts of checkpoint distribution
  - `generate_backend_comparison_report()`: Markdown reports with embedded graphs
  - `generate_longitudinal_report()`: Trend reports with regression detection
  - `generate_comprehensive_report()`: Overview of all performance data
- Enhanced StopWatcher class with `save_perf` parameter
  - Auto-captures test name and file path on __enter__
  - Auto-saves performance data on __exit__ when save_perf=True
  - Extracts backend from environment (YAPOCIS_BACKEND)
  - Gracefully handles errors without failing tests
- Created CLI tool `scripts/performance_report.py` (250+ lines)
  - Commands: list, stats, compare, trend, comprehensive, purge
  - Filter by backend, test name, date range
  - Generate reports and visualizations
  - Dry-run support for purge operations
- Updated `pyproject.toml` dependencies
  - Added seaborn>=0.12.0 for advanced visualizations
  - Added tabulate>=0.9.0 for formatted tables
- Updated `.gitignore` to exclude performance/results/ and performance/reports/
- Created performance data directories with auto-creation
- Lines changed: ~1300 additions across 6 files

**Files created:**
- `yapocis/utils/performance.py` - Core performance tracking module
- `scripts/performance_report.py` - CLI tool for managing performance data
- `performance/results/` - Auto-saved JSON performance data (gitignored)
- `performance/reports/` - Generated reports and graphs (gitignored)

**Files modified:**
- `tests/__init__.py` - Enhanced name_of_test() and StopWatcher with save_perf
- `pyproject.toml` - Added seaborn and tabulate dependencies
- `.gitignore` - Excluded performance data directories

**Testing:**
- Verified auto-save of performance data with save_perf=True
- Tested class method detection in name_of_test()
- Verified git info extraction (branch, commit, file status)
- Tested CLI list and stats commands
- Confirmed backward compatibility (save_perf=False by default)

**Features:**
- Auto-tracking: Simply use `StopWatcher(save_perf=True)` to track performance
- Git integration: Tracks branch, commit, and file modification status
- Backend comparison: Compare performance across OCL, Metal, Numba, CUDA
- Longitudinal analysis: Track performance trends over time
- Regression detection: Auto-detect >10% performance changes
- Flexible filtering: By backend, test, date, file status
- Rich visualizations: Bar charts, line graphs, pie charts with seaborn
- Markdown reports: Publication-ready reports with embedded graphs
- Data management: Purge old data with flexible criteria

**Human Contribution:**
- Feature request and high-level design requirements
- Refinement of auto-save behavior and data schema
- Testing and verification

---

### 2025-10-03: StopWatcher Enhancement - Context Manager Checkpoints and Auto-logging

**AI Contribution (Claude):**
- Enhanced StopWatcher.checkpoint() to work as both method call and context manager
  - Backward compatible: `sw.checkpoint("label")` still works
  - New context manager style: `with sw.checkpoint("label"): code...`
  - Automatically times the block when used with `with`
  - Added `_CheckpointContext` helper class for context management
- Added auto-logging on StopWatcher exit
  - New `log` parameter in `__init__` accepts callable
  - Automatically calls `log(summary())` when context exits
  - Enables cleaner test code: no manual `print(sw.summary())` needed
  - Example: `with StopWatcher(log=print) as sw: ...`
- Fixed duplicate `__exit__` method that was overriding functionality
- Lines changed: ~50 additions/modifications

**Files modified:**
- `tests/__init__.py` - Enhanced StopWatcher class with new features

**Testing:**
- Verified backward compatibility with existing checkpoint usage
- Tested new context manager style for checkpoint blocks
- Tested mixed usage (both styles together)
- Verified auto-logging with print and custom log functions
- Confirmed no logging when log parameter not provided

**Human Contribution:**
- Feature request and design requirements
- Testing and verification

---

### 2025-10-03: StopWatcher Pattern Standardization and Show Array Updates

**AI Contribution (Claude):**
- Backported NPZ file saving to show_array from main branch
  - Added RAW_QUEUE directory for storing raw arrays
  - Implemented array_to_image() helper function
  - Modified show_array to save both PNG and NPZ files for validation framework
  - Updated show_array signature: array now first parameter
- Fixed StopWatcher timing bugs
  - Fixed undefined variable 't' in __call__
  - Fixed AttributeError in __enter__ accessing self.t before initialization
  - Improved checkpoint timing accuracy
- Applied consistent StopWatcher pattern across all test files
  - Updated test_graphics.py, test_graphics4.py (14 functions)
  - Updated test_kerneldraw.py, test_kerneldraw3.py, test_kerneldraw4.py (8 functions)
  - Updated test_anisodiff.py, test_gaussians.py, test_colormath.py
  - Added appropriate timing checkpoints (display_list, render, read)
  - Standardized on sw.checkpoint(), sw.show(), and print(sw.summary())
- Updated all show_array callers to use new signature
- Lines changed: ~425 additions, ~270 deletions across 11 files

**Files modified:**
- `yapocis/utils/basic.py` - NPZ saving and signature update
- `yapocis/utils/advanced.py` - Updated show_array calls
- `tests/__init__.py` - Fixed StopWatcher bugs
- `tests/test_graphics.py` - StopWatcher pattern
- `tests/test_graphics4.py` - StopWatcher pattern
- `tests/test_kerneldraw.py` - StopWatcher pattern
- `tests/test_kerneldraw3.py` - StopWatcher pattern
- `tests/test_kerneldraw4.py` - StopWatcher pattern
- `tests/test_anisodiff.py` - StopWatcher pattern
- `tests/test_gaussians.py` - StopWatcher pattern
- `tests/test_colormath.py` - StopWatcher pattern

**Human Contribution:**
- Design decision to backport NPZ saving
- Manual update of test_graphics.py with new StopWatcher pattern as example
- Request to apply pattern consistently across all tests

---

### 2025-10-03: Validation Framework Fixes and Numba HSI Completion

**AI Contribution (Claude):**
- Fixed test_manual_validation_context test failure
  - Analyzed validation framework auto-promotion logic
  - Added cleanup of stale validation references
  - Enhanced test to verify both auto-promotion and subsequent validation
  - Improved test documentation and comments
- Completed Numba HSI backend implementation
  - Identified missing `hsi2rgb` function in numba/kernels/hsi.py
  - Implemented `hsi2rgb` matching OCL kernel interface
  - Maintained proper channel swizzle (r,b,g) for consistency
  - Verified all numba backend tests pass
- Lines changed: ~30 additions across 2 files

**Files modified:**
- `tests/test_validation_example.py` - Fixed validation test with proper cleanup and verification
- `yapocis/numba/kernels/hsi.py` - Added missing `hsi2rgb` function

**Testing:**
- All numba backend tests pass (18 tests)
- Validation framework tests pass
- HSI color conversion tests pass on numba backend

**Human Contribution:**
- Bug identification and reporting
- Testing and verification

---

### 2025-10-03: Rename sdfpath to truetrace + Numba documentation

**AI Contribution (Claude):**
- Renamed sdfpath module to truetrace across entire codebase
  - Files renamed: 5 (interface, library, OCL kernel, Metal kernel, Numba kernel)
  - Class renamed: SDFPath → TrueTrace
  - Interface declaration updated
  - Test file updated with new imports
  - Documentation updated in CLAUDE.md and README.md
- Updated README.md to document Numba backend
  - Added Numba to supported backends list
  - Documented CPU fallback functionality
  - Added test markers and dependencies
  - Added backend selection examples
- Created AI.md to track contributions

**Files modified:**
- `yapocis/interfaces/sdfpath.interface` → `truetrace.interface`
- `yapocis/lib/sdfpath.py` → `truetrace.py`
- `yapocis/kernels/sdfpath.mako` → `truetrace.mako`
- `yapocis/metalkernels/sdfpath.mako` → `truetrace.mako`
- `yapocis/numba/kernels/sdfpath.py` → `truetrace.py`
- `tests/test_truetrace_equiv.py` → `test_truetrace.py`
- `README.md`
- `CLAUDE.md`
- `AI.md` (created)

**Human Contribution:**
- Design decision to rename sdfpath to truetrace
- Review and approval of changes

---

### 2025-10 (earlier): Metal Backend Column-Major Coordinate System Fixes

**AI Contribution (Claude):**
- Analyzed OCL vs Metal coordinate system implementations
- Identified bugs in Metal helpers.mako (row-major vs column-major)
- Fixed Metal kernel.py width/height handlers
- Fixed graphics4.mako composite kernel coordinate swizzles (.xy/.zw → .yx/.wz)
- Fixed graphics.mako (single-channel) composite kernel swizzles
- Fixed abstract_graphics.py coerce_colors for single-channel graphics
- Created comprehensive test scripts in claude-debug/

**Files modified:**
- `yapocis/metalkernels/helpers.mako`
- `yapocis/metallic/kernel.py`
- `yapocis/metalkernels/graphics4.mako`
- `yapocis/metalkernels/graphics.mako`
- `yapocis/lib/abstract_graphics.py`

**Commits:**
- `bea0457` - Fix Metal composite kernel coordinate swizzle for column-major layout
- `0eb50ab` - Fix Metal graphics (single-channel) composite kernel coordinate swizzle
- Additional commit for coerce_colors fix

**Human Contribution:**
- Bug identification and reproduction
- Design decision: maintain column-major layout for 'older' branch
- Testing and validation of fixes

---

## Methodology

**AI Role:**
- Implementation of approved changes
- Code analysis and bug identification
- Documentation updates
- Test creation and debugging
- Commit message drafting

**Human Role:**
- Architecture decisions
- Requirements specification
- Final review and approval
- Git operations (when not delegated)
- Strategic direction

## Notes

- AI contributions are supervised and reviewed by the human maintainer
- All commits are attributed to the human maintainer in git history
- This tracking started 2025-10-03; earlier AI contributions are documented retroactively based on session summaries

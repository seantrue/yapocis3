# Rust (yapocis-rs) vs Python (yapocis) timing comparison

Date: 2026-10-19. Branch `worktree-rust-port` (Python, uncommitted) and `~/workspace/yapocis-rs` (Rust, uncommitted). Updated the same day after fixing the host-output copy and adding device-buffer reuse (section "Fixes"); the "Rust before fix" column holds the first run. Rerun again after making `resident` parameters device-buffer only (section "Rerun after the resident change"); the tables below are from that rerun, and the previous results are kept in `claude-debug/rust_vs_python_bench_results_pre_resident/`. Section "Residence" adds multi-stage workloads run with host round trips and with device buffers.

## Method

Two harnesses run the same workloads at the same sizes with the same timing loop: `claude-debug/rust_vs_python_bench.py` (Python, `PYTHONPATH=src`) and `claude-debug/rust_vs_python_bench/` (a standalone Cargo crate depending on `yapocis-rs`, release build). `claude-debug/rust_vs_python_bench_run.py` drives both per backend and `claude-debug/rust_vs_python_bench_report.py` builds the tables. Raw results are in `claude-debug/rust_vs_python_bench_results/*.jsonl`, and the first run in `claude-debug/rust_vs_python_bench_results_before/`.

- Every case is an end-to-end library call as a user makes it: host arrays in, host arrays out, so upload, launch and download are included. Inputs are the same array objects on every repetition, so both runtimes can reuse the device buffers bound to them.
- Steady state: 3 untimed warm-up calls, then at least 10 and at most 200 timed calls, stopping after 2 s. Medians are reported; minima are in the tables.
- Each harness is run twice per backend and the second run is kept, so kernel caches on disk are warm in both languages.
- Process startup: wall-clock time of a fresh process that opens the device, loads `demo` and runs one 1,000-element `sum`, median of 7 runs after one warm run.
- `load.*` rows time constructing one library object (interface load, compile or cache hit). Python's `load.demo` is a second load of an interface already loaded in-process, so it is a cache hit and not comparable to Rust's fresh load.
- Same device on both sides: the Apple M4 Max (thresher, macOS 27.0.1) for Metal and OpenCL, the RTX 3080 (yellowfin, Linux 6.8, driver 580.178.04, CUDA 12.5) for CUDA and OpenCL; both harnesses default to it.
- ResNet50 runs only on yellowfin (the only host with weights). Python uses `ResNet50Resident`, which keeps weights on the device like the Rust port. Image decoding is outside the timed region on both sides; preprocessing and top-5 are inside.
- Batch FFT is 1024 signals of 256 points. Python documents a 256-point limit for `batch_forward` but does not enforce it; Rust rejects larger sizes.

## Summary (after the fixes)

| Area | Python / Rust (higher means Rust is faster) | Why |
|---|---|---|
| GPU-bound single kernels (mandelbrot, FFT) | 1.0x to 1.9x | Same kernel source on the same device; what remains is per-call host overhead. |
| Two-output kernels (gradient) | 2.2x to 5.4x on thresher, 1.1x to 1.3x on yellowfin | Output handling: two fresh outputs per call, each one allocation in Rust now. |
| Small calls (1K `demo.sum`) | 1.2x Metal, 0.94x to 1.19x OpenCL (thresher); 7.4x CUDA, 2.6x NVIDIA OpenCL | Python's per-call argument handling; Rust now reuses device buffers, which removed its OpenCL create/release cost (0.216 ms before, 0.039 ms now). |
| Large arrays (16M `demo.sum`, 2048² `operators.add`) | 2.5x to 4.3x on thresher, 0.95x to 1.14x on CUDA, 2.0x to 3.8x on NVIDIA OpenCL | Was 0.33x to 0.68x on CUDA before the fixes. CUDA `operators.add` is about 6% slower in Rust (see the rerun section). |
| Wrappers with several launches (median ×10, gauss_image, rgb2hsi3, batch FFT) | 1.1x to 9.4x | No per-launch Python dispatch; ping-pong buffers stay resident. |
| ResNet50 predict | 7.4x on CUDA, 8.8x on OpenCL | On the RTX 3080: 274 ms Python vs 37 ms Rust (CUDA), 285 ms vs 32 ms (OpenCL). |
| Process startup | 15x Metal, 5.8x OpenCL (thresher); 2.4x OpenCL, 0.89x CUDA (yellowfin) | Python pays about 250 ms of interpreter and import time. On yellowfin, CUDA context creation (about 0.9 s) dominates both. |
| Library object load (warm caches) | 3.7x to 135x | Rust takes sources from the embedded bundle and compiled programs from the cache; Python renders and parses the interface per load. Metal `hsi` (3000x) is left out: Python's load took 226 ms in this run. |

Rust is as fast as or faster than Python in every steady-state case except CUDA `operators.add` 2048², where it is about 6% slower (medians of 9.4 ms vs 8.9 ms over 8 repeats).

## Fixes

### Host outputs: one allocation, no copy

The first run showed Rust 3x slower than Python on CUDA for large arrays (`operators.add` 2048²: 28.4 ms vs 9.4 ms). `claude-debug/rust_vs_python_bench/src/bin/xfer_breakdown.rs` showed the device side matched pycuda (upload 1.87 ms, download 1.78 ms per 16 MB), but each call downloaded an `out_like` result into a new `Vec<u8>` and then copied it into a new `Vec<T>` (`host.rs`), and writing 16 MB into fresh memory cost about 11 ms in page faults.

Now `yapocis-rt` downloads into a `HostData` (`crates/yapocis-rt/src/host_data.rs`) that holds a `Vec` of the output's element type, and the typed layer takes it with `HostData::into_vec` without copying. Large outputs come from `alloc_zeroed`, so the allocator does not touch the pages before the download does. `OutputValue::Host::bytes` is now a `HostData` (it dereferences to `[u8]`).

Transparent huge pages were tried and left off by default (`YAPOCIS_HUGE_PAGES=1` enables them on Linux). yellowfin uses `defrag=madvise`, and with fragmented memory every huge-page fault compacted synchronously. A 64 MB output then took 216 to 225 ms in some processes (377 `compact_stall`s per run in `/proc/vmstat`) and 43 ms in others. Without the hint it was a steady 64 to 70 ms.

### Device buffers bound to host arrays

`crates/yapocis-rt/src/buffer_cache.rs` adds the binding the Python device managers have (`id(array)` → device buffer). Each device keeps a cache keyed by the host array's address and byte length. A call with the same host array gets the buffer it used last time; otherwise it gets any free buffer of the same length, so `out_like` outputs reuse buffers too.

- **Uploads:** contents are still uploaded on every call, as in Python; there is no change detection.
- **Concurrency:** buffers are checked out for the duration of a call, so concurrent calls never share one.
- **Resident parameters:** never uploaded. They take a device buffer (`CallArg::Buffer`; the typed API already required `&Buffer<T>`), and a host array passed to one is now an `ArgKind` error instead of an upload on every call. Python can look up the buffer bound to the array because its cache holds the array, so the id cannot be reused; an address-keyed cache cannot, and skipping the upload on a bound address would give stale device data to a new array at the same address. Test: `resident_parameters_are_never_uploaded` in `crates/yapocis-rt/tests/demo.rs`.
- **Async calls:** asynchronous calls bypass the cache, because their buffers can still be in use when the call returns, and Metal uploads write shared memory directly.
- **Bounds:** `YAPOCIS_BUFFER_CACHE_MB` (default 512, `0` disables it) and 64 entries, least recently used first.
- **API:** `Device::purge_buffers` (Python's `purge`) and `Device::buffer_cache_stats`.
- **Tests:** two unit tests, plus `buffer_cache_on_every_available_backend` in `crates/yapocis-rt/tests/demo.rs`. The integration test checks repeated calls, an in-place change to a bound array, the same array passed twice, `inout` write-back into a reused buffer, a different length, and purge.

### OpenCL device enumeration is serialised

Running the tests with `ocl` on yellowfin (first time there) segfaulted about 1 run in 7: gdb showed `clGetDeviceInfo` crashing inside pocl (the CPU OpenCL platform installed there) while parallel test threads enumerated devices. `OclDevice::open` now holds a process-wide lock; 0 failures in 30 runs after.

### Test status

- 71 of 71 tests pass on thresher (Metal, OpenCL), including the resident test and the `set_margin` test added with the residence work.
- On yellowfin (CUDA, OpenCL), 69 of 71 pass against the vendored fixtures. The two failures are `parity_with_python_outputs` (one Mandelbrot pixel 6 vs 7 iterations) and `simplex_matches_python` (`noise3d` element 223).
- Both replay OpenCL fixtures recorded on the Apple M4 Max on NVIDIA's OpenCL; yellowfin had never run the OpenCL suite before.
- With OpenCL fixtures regenerated by Python on the RTX 3080 (in the temporary copy only), all 71 pass. So Rust matches Python on that device, and the failures are a cross-device comparison, not a runtime bug.
- The vendored bundle holds one OpenCL fixture set; making the parity tests skip fixtures recorded on another device, or vendoring a set per device, is open.

## Remaining differences

- The first CUDA call after load is 10.9 ms in Rust and 1.1 ms in Python. That is most likely the driver compiling the cached PTX to machine code on first use, while pycuda's cache holds a cubin built by nvcc for the device. Not verified.
- CUDA process startup is about 0.9 s in both languages, dominated by context creation.
- CUDA `operators.add` 2048² is about 6% slower in Rust (9.4 ms vs 8.9 ms); see "Rerun after the resident change".
- GPU-bound cases (`fft.forward.65536`: 7.0 ms CUDA, 18.2 ms Metal, about 1,025 ms on NVIDIA OpenCL in both languages) are equal: the kernels are identical. The 1-second NVIDIA OpenCL FFT is the same in Python, so it is a kernel or driver property.

## Rerun after the resident change

`resident` parameters now accept only a device buffer (`CallArg::Buffer`); a host array passed to one is an `ArgKind` error instead of an upload on every call. None of the benchmark cases pass host arrays to resident parameters, so no timing was expected to change. `claude-debug/rust_vs_python_bench_compare_runs.py` compares the rerun with the previous results.

- Rust medians moved by 0.83x to 1.11x between the two runs. The one larger move, thresher OpenCL `fft.forward.65536` (1.23x), happened in Python too (1.32x), so it is the device, not the runtime.
- Two cases came out below 1.0x in the suite run and were repeated 8 times, alternating languages (`claude-debug/rust_vs_python_bench_repeat.sh`).

| Case | Python medians ms | Rust medians ms | Python / Rust per round |
|---|---|---|---|
| thresher OpenCL `demo.sum.1K` | 0.426 to 0.474 | 0.388 to 0.444 | 1.01x to 1.19x |
| yellowfin CUDA `operators.add.2048x2048` | 8.05 to 9.11 | 9.20 to 10.04 | 0.82x to 0.99x |

- `demo.sum.1K` on Apple OpenCL: Rust is faster in every round; the 0.94x in the suite run was a fast Python run (0.380 ms).
- `operators.add` on CUDA: Rust is slower in every round, by about 6% at the medians, where the previous run had 8.72 ms vs 8.86 ms. The resident change does not touch this call (no resident parameters, same code path). The RTX 3080 was idle (other GPU processes run on the 3060 and P40). With `YAPOCIS_HUGE_PAGES=1` Rust took 9.1 to 9.9 ms and with the buffer cache off 9.4 to 10.6 ms, so neither explains it. The cause is not established; the host-side page faults on the fresh 16 MB output are the most likely place to look.

## Residence

The suite above passes host arrays to every call, so residence never shows. Two workloads were added (`resident.*` cases in both harnesses), each in a host form and a resident form. The runner was used with `--cases resident` and its new `--outdir` option; results are in `rust_vs_python_bench_results_resident/`, tabulated by `rust_vs_python_bench_resident_report.py`.

- **chain10:** ten dependent operator stages, `x = x + b`, `x = x * b`, alternating. Host form: ten `operators.add`/`mul` calls on host arrays, so every stage uploads two arrays and downloads one. Resident form: `a` and `b` are written into device buffers, the ten `add_res`/`mul_res` launches ping-pong between two device buffers, and the result is read once. Both forms give identical results.
- **zcsdog:** `Gaussian.zcsdog(image, 3)`. Host form: Python's `res=False` (two `gauss_image` round trips, a host subtraction, `zcs`); Rust does the same with its public calls. Resident form: `res=True`, six launches on device buffers. Both forms then run the gradient of the smaller blur (a host-array call) and clear margins on the host.

"Gain" is host ms / resident ms in the same language.

| Host | Backend | Workload | Python host ms | Python resident ms | Python gain | Rust host ms | Rust resident ms | Rust gain | Python / Rust host | Python / Rust resident |
|----|----|----------|---:|---:|---:|---:|---:|---:|---:|---:|
| thresher | Metal | chain10.1024x1024 | 10.98 | 1.95 | 5.6x | 2.95 | 1.36 | 2.2x | 3.7x | 1.4x |
| thresher | Metal | chain10.2048x2048 | 46.76 | 4.80 | 9.7x | 9.92 | 2.90 | 3.4x | 4.7x | 1.7x |
| thresher | Metal | zcsdog.1024x1024.s3 | 4.90 | 4.58 | 1.1x | 2.29 | 1.75 | 1.3x | 2.1x | 2.6x |
| thresher | Metal | zcsdog.2048x2048.s3 | 25.73 | 31.50 | 0.8x | 7.74 | 6.99 | 1.1x | 3.3x | 4.5x |
| thresher | OpenCL | chain10.1024x1024 | 11.59 | 1.94 | 6.0x | 8.07 | 2.18 | 3.7x | 1.4x | 0.9x |
| thresher | OpenCL | chain10.2048x2048 | 33.20 | 4.00 | 8.3x | 15.66 | 3.96 | 4.0x | 2.1x | 1.0x |
| thresher | OpenCL | zcsdog.1024x1024.s3 | 11.63 | 9.38 | 1.2x | 4.04 | 2.72 | 1.5x | 2.9x | 3.5x |
| thresher | OpenCL | zcsdog.2048x2048.s3 | 39.13 | 28.00 | 1.4x | 10.79 | 7.63 | 1.4x | 3.6x | 3.7x |
| yellowfin | CUDA | chain10.1024x1024 | 29.45 | 2.48 | 11.9x | 22.36 | 1.89 | 11.8x | 1.3x | 1.3x |
| yellowfin | CUDA | chain10.2048x2048 | 115.81 | 7.36 | 15.7x | 105.34 | 6.60 | 16.0x | 1.1x | 1.1x |
| yellowfin | CUDA | zcsdog.1024x1024.s3 | 22.25 | 16.81 | 1.3x | 13.76 | 9.91 | 1.4x | 1.6x | 1.7x |
| yellowfin | CUDA | zcsdog.2048x2048.s3 | 104.97 | 87.66 | 1.2x | 98.59 | 72.94 | 1.4x | 1.1x | 1.2x |
| yellowfin | OpenCL | chain10.1024x1024 | 32.13 | 1.89 | 17.0x | 24.97 | 1.69 | 14.7x | 1.3x | 1.1x |
| yellowfin | OpenCL | chain10.2048x2048 | 202.68 | 6.04 | 33.5x | 102.88 | 5.87 | 17.5x | 2.0x | 1.0x |
| yellowfin | OpenCL | zcsdog.1024x1024.s3 | 25.19 | 21.03 | 1.2x | 27.19 | 18.23 | 1.5x | 0.9x | 1.2x |
| yellowfin | OpenCL | zcsdog.2048x2048.s3 | 206.84 | 98.52 | 2.1x | 107.26 | 72.95 | 1.5x | 1.9x | 1.4x |

- **Transfers dominate, residence wins:** chain10 is 12x to 18x faster resident on the RTX 3080 in Rust (12x to 34x in Python), because each host stage moves 48 MB at 2048² over PCIe. On the M4 Max, where the GPU shares memory with the CPU, it is 2.2x to 4.0x in Rust and 5.6x to 9.7x in Python. Python gains more because its host form also pays Python's per-call cost and a fresh output allocation per stage.
- **Resident code is equal across languages on discrete GPUs:** resident chain10 is 1.0x to 1.3x Python / Rust on yellowfin; once the data stays on the device, both runtimes only enqueue launches.
- **zcsdog gains only 1.1x to 1.5x in Rust:** most of its time is outside the resident part. The bench crate's `zcsdog_breakdown` binary times each stage of the Rust resident path; at 2048² on CUDA (58 ms in total), the six resident launches take 13 ms, downloading `zca` and `smaller` 12 ms, and the gradient of `smaller`, a host-array call that uploads it again and downloads two outputs, 27 ms. The bundle has a `gradient_res` kernel, so keeping `smaller` on the device for the gradient would remove one upload and one download; neither library does that today.
- **Python's resident zcsdog on Metal at 2048² is slower than its host form** (31.5 ms vs 25.7 ms). Not investigated.
- **The two zcsdog forms do not give the same edge map:** about 0.45% of pixels differ, in Python and in Rust. `gauss_image` (`gauss*_h`/`gauss*_v`) and the resident `gauss*_res` kernels treat the image border differently (the blurs agree to 3e-7 inside a 27-pixel border and differ by up to 0.56 near it), and `zcsdog` clears a margin sized for the smaller kernel (28 px) while the larger one is 39 wide. 4,751 of the 4,757 differing pixels lie 28 to 39 px from an edge (`claude-debug/zcsdog_res_sign_check.py`). This is existing library behaviour, mirrored by the port, and not changed here.

### Fix found on the way: `set_margin`

The breakdown showed Rust spending 12.9 ms on margins and frame at 2048² on yellowfin (4.1 ms on thresher). `set_margin` (`crates/yapocis/src/lib/image.rs`) visited every pixel with an indexed write to set the border. It now fills the four border slices, as Python's slice assignments do, with the same results (unit test `set_margin_touches_exactly_the_border` checks it against the per-pixel rule, including margin 0 and margins larger than the image). Margins and frame now take 0.66 ms on yellowfin and 0.15 ms on thresher, and Rust's resident `zcsdog` stages at 2048² went from 67 ms to 58 ms in total (CUDA) and from 11.6 ms to 7.4 ms (Metal). The table above is from after the fix.

## Results

### thresher, metal (Apple M4 Max)

| Case | Python median ms | Rust median ms | Rust before fix ms | Python min ms | Rust min ms | Python / Rust |
|------------------|----:|----:|----:|----:|----:|----:|
| demo.sum.1K | 0.130 | 0.111 | 0.137 | 0.117 | 0.096 | 1.17x |
| demo.sum.1M | 0.941 | 0.294 | 0.972 | 0.838 | 0.284 | 3.20x |
| demo.sum.16M | 16.2 | 4.335 | 15.2 | 11.9 | 3.898 | 3.75x |
| operators.add.2048x2048 | 4.497 | 1.052 | 4.057 | 3.426 | 0.997 | 4.27x |
| mandelbrot.1024x1024.it256 | 1.477 | 0.788 | 1.306 | 1.168 | 0.684 | 1.87x |
| median3x3.1024x1024.x1 | 1.262 | 0.698 | 0.711 | 1.162 | 0.634 | 1.81x |
| median3x3.1024x1024.x10 | 8.092 | 0.863 | 0.806 | 5.232 | 0.674 | 9.38x |
| gaussian.gauss_image.1024x1024.s3 | 1.320 | 0.599 | 0.807 | 1.122 | 0.554 | 2.20x |
| gradient.1024x1024.st5 | 1.816 | 0.338 | 1.170 | 1.392 | 0.307 | 5.37x |
| hsi.rgb2hsi3.1024x1024 | 3.271 | 0.544 | 1.986 | 2.655 | 0.502 | 6.01x |
| fft.forward.4096 | 0.764 | 0.470 | 0.495 | 0.523 | 0.440 | 1.62x |
| fft.forward.65536 | 18.3 | 18.2 | 18.5 | 18.3 | 18.1 | 1.01x |
| fft.batch_forward.1024x256 | 0.835 | 0.263 | 0.505 | 0.479 | 0.226 | 3.18x |

| One-shot | Python ms | Rust ms | Python / Rust |
|---|---:|---:|---:|
| process.startup | 239.9 | 15.7 | 15.27x |
| import.yapocis | 31.7 | n/a | n/a |
| load.demo | 0.528 | 1.436 | 0.37x |
| first_call.demo.sum | 3.166 | 3.108 | 1.02x |
| load.operators | 6.909 | 0.093 | 74.29x |
| load.mandelbrot | 2.071 | 0.063 | 32.65x |
| load.median3x3 | 3.147 | 0.067 | 47.29x |
| load.convolvesep | 65.4 | 0.485 | 134.91x |
| load.gradient | 1.085 | 0.049 | 22.09x |
| load.hsi | 226.4 | 0.076 | 2995.79x |
| load.fft | 13.1 | 0.157 | 83.38x |
| process.total | 10797.5 | 4310.1 | 2.51x |

### thresher, ocl (Apple M4 Max)

| Case | Python median ms | Rust median ms | Rust before fix ms | Python min ms | Rust min ms | Python / Rust |
|------------------|----:|----:|----:|----:|----:|----:|
| demo.sum.1K | 0.380 | 0.402 | 0.480 | 0.342 | 0.378 | 0.94x |
| demo.sum.1M | 1.096 | 0.653 | 1.117 | 1.001 | 0.616 | 1.68x |
| demo.sum.16M | 13.6 | 4.143 | 9.762 | 12.7 | 3.683 | 3.27x |
| operators.add.2048x2048 | 3.701 | 1.481 | 2.664 | 3.560 | 1.399 | 2.50x |
| mandelbrot.1024x1024.it256 | 1.443 | 1.245 | 1.486 | 1.361 | 1.137 | 1.16x |
| median3x3.1024x1024.x1 | 2.755 | 0.639 | 0.752 | 2.449 | 0.613 | 4.31x |
| median3x3.1024x1024.x10 | 9.856 | 1.171 | 1.284 | 9.426 | 1.126 | 8.42x |
| gaussian.gauss_image.1024x1024.s3 | 3.163 | 0.879 | 1.007 | 2.986 | 0.841 | 3.60x |
| gradient.1024x1024.st5 | 2.193 | 0.982 | 1.500 | 1.895 | 0.720 | 2.23x |
| hsi.rgb2hsi3.1024x1024 | 4.701 | 0.785 | 1.513 | 4.149 | 0.723 | 5.99x |
| fft.forward.4096 | 1.954 | 1.672 | 1.718 | 1.851 | 1.539 | 1.17x |
| fft.forward.65536 | 136.4 | 128.2 | 107.6 | 115.8 | 124.2 | 1.06x |
| fft.batch_forward.1024x256 | 1.215 | 0.627 | 0.951 | 1.176 | 0.609 | 1.94x |

| One-shot | Python ms | Rust ms | Python / Rust |
|---|---:|---:|---:|
| process.startup | 279.6 | 48.0 | 5.83x |
| import.yapocis | 19.8 | n/a | n/a |
| load.demo | 0.420 | 2.031 | 0.21x |
| first_call.demo.sum | 3.764 | 3.738 | 1.01x |
| load.operators | 7.356 | 0.289 | 25.47x |
| load.mandelbrot | 2.147 | 0.238 | 9.04x |
| load.median3x3 | 3.021 | 0.338 | 8.95x |
| load.convolvesep | 37.6 | 2.689 | 13.99x |
| load.gradient | 1.140 | 0.308 | 3.70x |
| load.hsi | 4.653 | 0.410 | 11.34x |
| load.fft | 5.524 | 0.603 | 9.17x |
| process.total | 11601.7 | 5600.8 | 2.07x |

### yellowfin, cuda (NVIDIA GeForce RTX 3080)

| Case | Python median ms | Rust median ms | Rust before fix ms | Python min ms | Rust min ms | Python / Rust |
|------------------|----:|----:|----:|----:|----:|----:|
| demo.sum.1K | 0.257 | 0.035 | 0.051 | 0.242 | 0.034 | 7.36x |
| demo.sum.1M | 2.819 | 2.510 | 2.424 | 2.710 | 2.461 | 1.12x |
| demo.sum.16M | 75.6 | 66.4 | 111.4 | 73.1 | 61.1 | 1.14x |
| operators.add.2048x2048 | 8.862 | 9.365 | 28.4 | 8.449 | 9.232 | 0.95x |
| mandelbrot.1024x1024.it256 | 2.386 | 2.084 | 2.061 | 2.208 | 1.985 | 1.15x |
| median3x3.1024x1024.x1 | 5.055 | 2.161 | 2.095 | 4.733 | 2.088 | 2.34x |
| median3x3.1024x1024.x10 | 18.8 | 2.297 | 2.360 | 18.1 | 2.192 | 8.17x |
| gaussian.gauss_image.1024x1024.s3 | 5.228 | 3.463 | 3.470 | 5.041 | 3.431 | 1.51x |
| gradient.1024x1024.st5 | 3.978 | 3.595 | 4.223 | 3.833 | 3.545 | 1.11x |
| hsi.rgb2hsi3.1024x1024 | 6.567 | 5.827 | 5.751 | 6.431 | 5.754 | 1.13x |
| fft.forward.4096 | 0.399 | 0.224 | 0.242 | 0.378 | 0.222 | 1.79x |
| fft.forward.65536 | 7.228 | 7.036 | 7.132 | 7.124 | 7.000 | 1.03x |
| fft.batch_forward.1024x256 | 2.080 | 1.275 | 1.363 | 2.014 | 1.247 | 1.63x |
| resnet50.predict.top5 | 273.7 | 37.0 | 37.0 | 265.1 | 36.6 | 7.40x |

| One-shot | Python ms | Rust ms | Python / Rust |
|---|---:|---:|---:|
| process.startup | 889.1 | 995.9 | 0.89x |
| import.yapocis | 55.7 | n/a | n/a |
| load.demo | 1.536 | 2.599 | 0.59x |
| first_call.demo.sum | 1.192 | 10.0 | 0.12x |
| load.operators | 33.4 | 0.403 | 82.75x |
| load.mandelbrot | 7.564 | 0.402 | 18.83x |
| load.median3x3 | 6.298 | 0.430 | 14.66x |
| load.convolvesep | 125.4 | 4.071 | 30.81x |
| load.gradient | 2.427 | 0.466 | 5.21x |
| load.hsi | 9.598 | 0.365 | 26.31x |
| load.fft | 12.4 | 0.532 | 23.23x |
| process.total | 18960.2 | 13772.0 | 1.38x |

### yellowfin, ocl (NVIDIA GeForce RTX 3080)

| Case | Python median ms | Rust median ms | Rust before fix ms | Python min ms | Rust min ms | Python / Rust |
|------------------|----:|----:|----:|----:|----:|----:|
| demo.sum.1K | 0.101 | 0.039 | 0.216 | 0.098 | 0.038 | 2.59x |
| demo.sum.1M | 8.653 | 2.337 | 6.669 | 8.304 | 2.199 | 3.70x |
| demo.sum.16M | 123.8 | 63.1 | 116.0 | 122.5 | 62.0 | 1.96x |
| operators.add.2048x2048 | 32.9 | 8.672 | 28.2 | 32.0 | 8.380 | 3.79x |
| mandelbrot.1024x1024.it256 | 2.488 | 1.864 | 2.407 | 2.432 | 1.843 | 1.34x |
| median3x3.1024x1024.x1 | 4.958 | 2.296 | 2.248 | 4.753 | 2.266 | 2.16x |
| median3x3.1024x1024.x10 | 17.0 | 2.466 | 2.412 | 16.3 | 2.427 | 6.90x |
| gaussian.gauss_image.1024x1024.s3 | 5.592 | 3.671 | 3.733 | 5.500 | 3.644 | 1.52x |
| gradient.1024x1024.st5 | 4.250 | 3.246 | 4.305 | 4.201 | 3.179 | 1.31x |
| hsi.rgb2hsi3.1024x1024 | 7.408 | 5.400 | 5.460 | 7.220 | 5.273 | 1.37x |
| fft.forward.4096 | 26.3 | 26.3 | 26.4 | 26.2 | 26.3 | 1.00x |
| fft.forward.65536 | 1022.2 | 1025.4 | 1025.7 | 1020.9 | 1024.6 | 1.00x |
| fft.batch_forward.1024x256 | 2.931 | 1.821 | 2.106 | 2.874 | 1.805 | 1.61x |
| resnet50.predict.top5 | 284.9 | 32.3 | 31.5 | 263.7 | 32.0 | 8.81x |

| One-shot | Python ms | Rust ms | Python / Rust |
|---|---:|---:|---:|
| process.startup | 944.9 | 393.0 | 2.40x |
| import.yapocis | 54.7 | n/a | n/a |
| load.demo | 0.554 | 5.487 | 0.10x |
| first_call.demo.sum | 0.381 | 0.243 | 1.57x |
| load.operators | 19.7 | 0.387 | 50.91x |
| load.mandelbrot | 4.899 | 0.319 | 15.37x |
| load.median3x3 | 5.417 | 0.218 | 24.82x |
| load.convolvesep | 87.5 | 2.957 | 29.58x |
| load.gradient | 1.864 | 0.323 | 5.78x |
| load.hsi | 9.545 | 0.357 | 26.77x |
| load.fft | 13.0 | 0.982 | 13.29x |
| process.total | 34964.7 | 26964.8 | 1.30x |

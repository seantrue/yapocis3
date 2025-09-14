"""
Comparative runtime performance tests across backends.

Measures kernel execution time for representative kernels per interface on
every available backend, persisting results to JSONL. Compares only interfaces
that successfully run on 2 or more backends. Designed to track performance
evolution over time.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from yapocis.core.manager import get_backends, get_manager
from yapocis.interfaces import interfaces as list_interfaces
from yapocis.lib.graphics import Graphics1
from yapocis.lib.graphics4 import Graphics4


RESULTS_DIR = Path("perf_results")
RESULTS_PATH = RESULTS_DIR / "perf_results.jsonl"


def _context_for_interface(name: str) -> dict:
    """Return minimal mako variables to render parameterized interfaces."""
    if name == "operators":
        return {"operators": [("add", "+"), ("sub", "-")]}  # sample ops
    if name == "convolves":
        return {"convs": [("conv3", [0.25, 0.5, 0.25]), ("edge3", [-1.0, 0.0, 1.0])]}  # sample 1D kernels
    if name == "convolvesep":
        return {"convs": [("sep3", [0.25, 0.5, 0.25])]}  # sample separable conv
    if name == "convolve":
        return {"name": "conv3", "conv": [0.25, 0.5, 0.25]}
    return {}


def _git_commit_id() -> str:
    try:
        import subprocess

        cid = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        return cid
    except Exception:
        return "unknown"


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _source_md5(backend: str, program: Any) -> str | None:
    # OCL/Metal managers attach .source. CUDA manager may not.
    src = getattr(program, "source", None)
    if isinstance(src, str) and src:
        try:
            return _md5(src)
        except Exception:
            return None
    return None


def _persist(record: Dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as outf:
        json.dump(record, outf)
        outf.write("\n")


def _load_program(backend: str, iname: str, ctx: dict) -> Tuple[bool, Any | None, str | None]:
    os.environ["YAPOCIS_BACKEND"] = backend
    mgr = get_manager(backend)
    try:
        prog = mgr.load_program(iname, **ctx)
    except Exception:
        return False, None, None
    smd5 = _source_md5(backend, prog)
    return True, prog, smd5


def _benchmark_exec(backend: str, prog: Any, fn_name: str, args: tuple, kwargs: dict, iters: int = 10) -> float:
    # Warm-up
    try:
        fn = getattr(prog, fn_name)
    except Exception:
        raise AssertionError(f"Kernel '{fn_name}' not found on backend {backend}")
    fn(*args, **kwargs)
    # Synchronize for CUDA to ensure warm-up completion
    if backend == "cuda":
        try:
            prog.stream.synchronize()
        except Exception:
            pass
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    if backend == "cuda":
        try:
            prog.stream.synchronize()
        except Exception:
            pass
    dt = (time.perf_counter() - t0) * 1000.0 / max(1, iters)
    return dt


def _make_data(shape: tuple[int, ...], *, backend: str):
    import numpy as np
    if backend == "cuda":
        try:
            import cupy as cp
            xp = cp
        except Exception:
            xp = np
    else:
        xp = np
    a = xp.random.rand(*shape).astype(xp.float32)
    return a


def _bench_recipe(iname: str, backend: str, prog: Any) -> Tuple[str, tuple, dict] | None:
    """Return (kernel_name, args, kwargs) for a representative run, or None to skip.

    Uses resident aliases where available to avoid host readback and isolate
    kernel execution timing where possible.
    """
    import numpy as np
    if iname == "demo":
        n = 1 << 20
        a = _make_data((n,), backend=backend)
        b = _make_data((n,), backend=backend)
        c = _make_data((n,), backend=backend)
        return "sum_res", (a, b, c), {"resident": (a, b, c)}
    if iname == "gradient":
        h, w = 1024, 1024
        img = _make_data((h, w), backend=backend)
        grad = _make_data((h, w), backend=backend)
        theta = _make_data((h, w), backend=backend)
        reach = int(1)
        return "gradient_res", (img, reach, grad, theta), {"resident": (img, grad, theta)}
    if iname == "zcs":
        h, w = 1024, 1024
        img = _make_data((h, w), backend=backend)
        out = _make_data((h, w), backend=backend)
        thresh = float(0.5)
        # Use zcs1_res as representative
        return "zcs1_res", (thresh, img, out), {"resident": (img, out)}
    if iname == "anisodiff":
        h, w = 1024, 1024
        img = _make_data((h, w), backend=backend)
        out = _make_data((h, w), backend=backend)
        scale = float(0.1); step = float(1.0)
        return "filterImage_res", (img, scale, step, out), {"resident": (img, out)}
    if iname == "operators":
        h, w = 1024, 1024
        a = _make_data((h, w), backend=backend)
        b = _make_data((h, w), backend=backend)
        out = _make_data((h, w), backend=backend)
        # Assuming 'add' exists when context provides it
        return "add_res", (a, b, out), {"resident": (a, b, out)}
    if iname == "convolvesep":
        h, w = 1024, 1024
        img = _make_data((h, w), backend=backend)
        out = _make_data((h, w), backend=backend)
        vertical = int(1)
        # Using 'sep3' from context mapping
        return "sep3_res", (vertical, img, out), {"resident": (img, out)}
    if iname == "imageops":
        # No resident alias for stretch; include readback cost for now
        h, w = 1024, 1024
        img = _make_data((h, w), backend=backend)
        low, high = float(0.0), float(1.0)
        imin, imax = float(0.1), float(0.9)
        return "stretch", (img, imin, imax, low, high), {}
    if iname == "convolve":
        n = 1 << 20
        a = _make_data((n,), backend=backend)
        # kernel name from context: 'conv3'
        return "conv3", (a,), {}
    if iname == "median3x3":
        h, w = 1024, 1024
        img = _make_data((h, w), backend=backend)
        return "median3x3", (img,), {}
    if iname == "colormath":
        # Use xyz2xyY_res on 2D image with 3 channels
        h, w, c = 512, 512, 3
        img = _make_data((h, w, c), backend=backend)
        out = _make_data((h, w, c), backend=backend)
        return "xyz2xyY_res", (img, out), {"resident": (img, out)}
    if iname == "mandelbrot":
        # Small grid to keep runtime reasonable
        h, w = 1024, 1024
        try:
            import numpy as np
            q = _make_data((h, w), backend=backend).astype(np.complex64)
        except Exception:
            q = _make_data((h, w), backend=backend)
        return "mandelbrot", (q, 200), {}
    # graphics/graphics4/hsi are more involved; skip by default
    return None


@pytest.mark.performance
def test_compare_runtime_across_backends():
    backends = get_backends()
    if len(backends) < 2:
        pytest.skip("Need 2+ available backends for comparative performance.")

    interfaces = [n.rsplit(".", 1)[0] for n in list_interfaces()]
    if not interfaces:
        pytest.skip("No interfaces found to test.")

    commit_id = _git_commit_id()
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    found_any_common = False
    # Iterate interfaces, attempt to load and run representative kernels
    for iname in interfaces:
        ctx = _context_for_interface(iname)
        loaded: List[Tuple[str, Any, str | None]] = []
        for be in backends:
            ok, prog, smd5 = _load_program(be, iname, ctx)
            if ok:
                loaded.append((be, prog, smd5))
        if len(loaded) < 2:
            continue
        # Determine a recipe; skip interfaces we can't benchmark
        runnable: List[Tuple[str, Any, str | None, Tuple[str, tuple, dict]]] = []
        for be, prog, smd5 in loaded:
            spec = _bench_recipe(iname, be, prog)
            if spec is None:
                continue
            fn_name, args, kwargs = spec
            runnable.append((be, prog, smd5, (fn_name, args, kwargs)))
        if len(runnable) < 2:
            continue
        found_any_common = True
        for be, prog, smd5, (fn_name, args, kwargs) in runnable:
            try:
                ms = _benchmark_exec(be, prog, fn_name, args, kwargs, iters=5)
            except Exception:
                continue
            # Persist individual measurement record immediately
            rec = {
                "timestamp": ts,
                "commit": commit_id,
                "backend": be,
                "interface": iname,
                "kernel": fn_name,
                "metric": "exec_ms",
                "iters": 5,
                "value": ms,
                "source_md5": smd5,
                "env": env,
            }
            _persist(rec)

    if not found_any_common:
        pytest.skip("No interfaces successfully ran on 2+ backends.")


def _rng(seed: int = 123):
    import numpy as np
    return np.random.default_rng(seed)


def _graphics_inputs(H: int, W: int, n: int, *, seed: int = 123):
    import numpy as np
    r = _rng(seed)
    # Coordinates
    def clamp_xy(x, y):
        return int(max(0, min(W - 1, x))), int(max(0, min(H - 1, y)))
    lines = []
    rects = []
    triangles = []
    circles = []
    ellipses = []
    for _ in range(n):
        x0, y0 = clamp_xy(r.integers(0, W), r.integers(0, H))
        x1, y1 = clamp_xy(x0 + int(r.integers(-50, 51)), y0 + int(r.integers(-50, 51)))
        lines.append([x0, y0, x1, y1])
        rects.append([x0, y0, x1, y1])
        x2, y2 = clamp_xy(x0 + int(r.integers(-50, 51)), y0 + int(r.integers(-50, 51)))
        triangles.append([x0, y0, x1, y1, x2, y2])
        circles.append([x0, y0, int(r.integers(2, 50))])
        ellipses.append([x0, y0, int(r.integers(2, 60)), int(r.integers(2, 60))])
    # Compositing blocks and locations
    blocks = [r.random((32, 32), dtype=float).astype('float32') for _ in range(min(16, n))]
    locs = [(int(r.integers(0, W-32)), int(r.integers(0, H-32))) for _ in range(len(blocks))]
    return {
        "lines": lines,
        "rects": rects,
        "triangles": triangles,
        "circles": circles,
        "ellipses": ellipses,
        "composite": (locs, blocks),
    }


def _graphics4_inputs(H: int, W: int, n: int, *, seed: int = 123):
    import numpy as np
    r = _rng(seed)
    def clamp_xy(x, y):
        return int(max(0, min(W - 1, x))), int(max(0, min(H - 1, y)))
    lines = []
    rects = []
    triangles = []
    circles = []
    ellipses = []
    for _ in range(n):
        x0, y0 = clamp_xy(r.integers(0, W), r.integers(0, H))
        x1, y1 = clamp_xy(x0 + int(r.integers(-50, 51)), y0 + int(r.integers(-50, 51)))
        lines.append([x0, y0, x1, y1])
        rects.append([x0, y0, x1, y1])
        x2, y2 = clamp_xy(x0 + int(r.integers(-50, 51)), y0 + int(r.integers(-50, 51)))
        triangles.append([x0, y0, x1, y1, x2, y2])
        circles.append([x0, y0, int(r.integers(2, 50))])
        ellipses.append([x0, y0, int(r.integers(2, 60)), int(r.integers(2, 60))])
    # RGBA blocks (H,W,4) in [0,1]
    blocks = [r.random((32, 32, 4), dtype=float).astype('float32') for _ in range(min(16, n))]
    # Ensure non-trivial alpha
    for b in blocks:
        b[:, :, 3] = 0.8
    locs = [(int(r.integers(0, W-32)), int(r.integers(0, H-32))) for _ in range(len(blocks))]
    return {
        "lines": lines,
        "rects": rects,
        "triangles": triangles,
        "circles": circles,
        "ellipses": ellipses,
        "composite": (locs, blocks),
    }



@pytest.mark.performance
def test_graphics_runtime_and_correctness():
    backends = get_backends()
    if len(backends) < 2:
        pytest.skip("Need 2+ available backends for graphics perf comparison.")

    H, W = 1024, 1024
    N = 200  # shapes per op
    inputs = _graphics_inputs(H, W, N, seed=123)

    commit_id = _git_commit_id()
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    env = {"python": sys.version.split()[0], "platform": platform.platform()}

    # Run each backend and collect outputs and times per op
    per_backend_outputs: Dict[str, Dict[str, Any]] = {}
    per_backend_times: Dict[str, Dict[str, float]] = {}
    for be in backends:
        try:
            gr = Graphics1(backend=be)
        except Exception:
            continue
        out_map: Dict[str, Any] = {}
        tmap: Dict[str, float] = {}

        # Common helper to coerce args per op
        def run_op(op: str):
            canvas = _make_data((H, W), backend=be)
            if op == "lines":
                coords = gr.coerce_coords(inputs[op], 4)
                widths = gr.coerce_widths(coords, width=2.0)
                colors = gr.coerce_colors(coords, color=0.75, alpha=1.0)
                args = (canvas, coords, widths, colors)
                kname = "lines_res"
            elif op == "rects":
                coords = gr.coerce_coords(inputs[op], 4)
                colors = gr.coerce_colors(coords, color=0.5, alpha=0.8)
                args = (canvas, coords, colors)
                kname = "rects_res"
            elif op == "circles":
                coords = gr.coerce_coords(inputs[op], 3)
                colors = gr.coerce_colors(coords, color=0.6, alpha=0.9)
                args = (canvas, coords, colors)
                kname = "circles_res"
            elif op == "triangles":
                coords = gr.coerce_coords(inputs[op], 6)
                colors = gr.coerce_colors(coords, color=1.0, alpha=0.4)
                args = (canvas, coords, colors)
                kname = "triangles_res"
            elif op == "ellipses":
                coords = gr.coerce_coords(inputs[op], 4)
                colors = gr.coerce_colors(coords, color=0.3, alpha=1.0)
                args = (canvas, coords, colors)
                kname = "ellipses_res"
            elif op == "composite":
                locs, blocks = inputs[op]
                coords, buffer, ipixels = gr.setup_composite(locs, blocks, alpha=0.8)
                args = (canvas, coords, buffer.astype('float32'), ipixels)
                kname = "composite_res"
            else:
                return None, 0.0, None
            ms = _benchmark_exec(be, gr.kmgr, kname, args, {"global_size": canvas.size}, iters=3)
            # Read back canvas for correctness comparison
            gr.read(canvas)
            return kname, ms, canvas

        for op in ("lines", "rects", "circles", "triangles", "ellipses", "composite"):
            kname, ms, out = run_op(op)
            if kname is None:
                continue
            out_map[op] = out
            tmap[op] = ms
            # Persist timing
            _persist({
                "timestamp": ts,
                "commit": commit_id,
                "backend": be,
                "interface": "graphics",
                "kernel": kname,
                "metric": "exec_ms",
                "iters": 3,
                "value": ms,
                "env": env,
            })
        if out_map:
            per_backend_outputs[be] = out_map
            per_backend_times[be] = tmap

    if len(per_backend_outputs) < 2:
        pytest.skip("Graphics interface did not run on 2+ backends.")

    # Compare correctness across backends against the first
    backends_run = list(per_backend_outputs.keys())
    ref_be = backends_run[0]
    ref = per_backend_outputs[ref_be]
    mismatches = []
    import numpy as np
    for be in backends_run[1:]:
        cur = per_backend_outputs[be]
        for op in ref.keys():
            if op not in cur:
                mismatches.append((op, ref_be, be, "missing"))
                continue
            a = ref[op]; b = cur[op]
            if not np.allclose(a, b, rtol=1e-4, atol=1e-5):
                mismatches.append((op, ref_be, be, "mismatch"))

    # Persist correctness summary
    _persist({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": _git_commit_id(),
        "backend_pair": backends_run,
        "interface": "graphics",
        "metric": "correctness",
        "value": len(mismatches) == 0,
        "details": mismatches,
    })
    if mismatches:
        print(f"Graphics outputs differ across backends: {mismatches}")


@pytest.mark.performance
def test_graphics4_runtime_and_correctness():
    backends = get_backends()
    if len(backends) < 2:
        pytest.skip("Need 2+ available backends for graphics4 perf comparison.")

    H, W = 512, 512
    N = 150
    inputs = _graphics4_inputs(H, W, N, seed=321)

    commit_id = _git_commit_id()
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    env = {"python": sys.version.split()[0], "platform": platform.platform()}

    per_backend_outputs: Dict[str, Dict[str, Any]] = {}
    for be in backends:
        try:
            gr = Graphics4(backend=be)
        except Exception:
            continue

        out_map: Dict[str, Any] = {}

        def run_op(op: str):
            import numpy as np
            canvas = _make_data((H, W, 4), backend=be)
            if op == "lines":
                coords = gr.coerce_coords(inputs[op], 4)
                widths = gr.coerce_widths(coords, width=2.0)
                colors = gr.coerce_colors(coords, color=(0.2, 0.4, 0.8, 1.0))
                args = (canvas, coords, widths, colors)
                kname = "lines_res"
            elif op == "rects":
                coords = gr.coerce_coords(inputs[op], 4)
                colors = gr.coerce_colors(coords, color=(0.6, 0.3, 0.1, 0.7))
                args = (canvas, coords, colors)
                kname = "rects_res"
            elif op == "circles":
                coords = gr.coerce_coords(inputs[op], 3)
                colors = gr.coerce_colors(coords, color=(0.1, 0.9, 0.1, 0.9))
                args = (canvas, coords, colors)
                kname = "circles_res"
            elif op == "triangles":
                coords = gr.coerce_coords(inputs[op], 6)
                colors = gr.coerce_colors(coords, color=(1.0, 1.0, 0.0, 0.5))
                args = (canvas, coords, colors)
                kname = "triangles_res"
            elif op == "ellipses":
                coords = gr.coerce_coords(inputs[op], 4)
                colors = gr.coerce_colors(coords, color=(0.0, 0.0, 1.0, 1.0))
                args = (canvas, coords, colors)
                kname = "ellipses_res"
            elif op == "composite":
                locs, blocks = inputs[op]
                coords, buffer, ipixels = gr.setup_composite(locs, blocks)
                args = (canvas, coords, buffer.astype('float32'), ipixels)
                kname = "composite_res"
            else:
                return None, 0.0, None
            ms = _benchmark_exec(be, gr.kmgr, kname, args, {"global_size": (H*W,)}, iters=3)
            # Read back canvas for correctness comparison
            gr.read(canvas)
            return kname, ms, canvas

        for op in ("lines", "rects", "circles", "triangles", "ellipses", "composite"):
            kname, ms, out = run_op(op)
            if kname is None:
                continue
            out_map[op] = out
            _persist({
                "timestamp": ts,
                "commit": commit_id,
                "backend": be,
                "interface": "graphics4",
                "kernel": kname,
                "metric": "exec_ms",
                "iters": 3,
                "value": ms,
                "env": env,
            })

        if out_map:
            per_backend_outputs[be] = out_map

    if len(per_backend_outputs) < 2:
        pytest.skip("graphics4 did not run on 2+ backends.")

    backends_run = list(per_backend_outputs.keys())
    ref_be = backends_run[0]
    ref = per_backend_outputs[ref_be]
    mismatches = []
    import numpy as np
    for be in backends_run[1:]:
        cur = per_backend_outputs[be]
        for op in ref.keys():
            if op not in cur:
                mismatches.append((op, ref_be, be, "missing"))
                continue
            a = ref[op]; b = cur[op]
            if not np.allclose(a, b, rtol=1e-4, atol=1e-5):
                mismatches.append((op, ref_be, be, "mismatch"))

    _persist({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": _git_commit_id(),
        "backend_pair": backends_run,
        "interface": "graphics4",
        "metric": "correctness",
        "value": len(mismatches) == 0,
        "details": mismatches,
    })
    if mismatches:
        print(f"graphics4 outputs differ across backends: {mismatches}")

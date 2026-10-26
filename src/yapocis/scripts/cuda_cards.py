#!/usr/bin/env python3
"""
Per-card CUDA performance comparison for yapocis (Linux, multi-GPU hosts).

Runs the same set of yapocis library workloads on every NVIDIA card in the
box, once per CUDA toolkit requested, and writes a markdown report (plus a
PDF via pandoc) comparing the cards. Each (toolkit, card) pair runs in its own
subprocess so that:

- the card is pinned by UUID through ``CUDA_VISIBLE_DEVICES=GPU-<uuid>``
  (never by index -- indices move when hardware does),
- the toolkit is selected by putting ``<toolkit>/bin`` first on ``PATH``
  (PyCUDA resolves ``nvcc`` from ``PATH``),
- one process never holds contexts on two cards.

Toolkits are given as ``LABEL=PATH`` and the run matrix as
``LABEL:CARD+CARD,...`` where a card is a substring of its name or ``all``.
A toolkit that cannot target a card's compute capability (CUDA 13 dropped
Maxwell and Pascal) is detected with a one-line ``nvcc`` probe and reported as
unsupported rather than failing the run.

Usage:
    yapocis-cuda-cards                                  # 12 on all cards, 13 on the 3060
    yapocis-cuda-cards --toolkits 12=/usr/local/cuda-12,13=/usr/local/cuda-13 \\
                       --matrix 12:all,13:3060
    yapocis-cuda-cards --size 2048 --repeats 7 --workloads gaussian,median3x3
    yapocis-cuda-cards --list                           # show cards, toolkits, plan

Outputs (under ``--output-dir``, default ``performance/``):
    cuda_cards_<timestamp>.md / .pdf / .json  and a ``cuda_cards_<timestamp>/``
    directory with one log per worker. Results are also recorded in the
    performance DuckDB unless ``--no-db`` is given.
"""
#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_TOOLKITS = "12=/usr/local/cuda-12,13=/usr/local/cuda-13"
DEFAULT_MATRIX = "12:all,13:3060"
WORKLOADS = [
    "memcpy",
    "gaussian",
    "median3x3",
    "mandelbrot",
    "anisodiff",
    "hsi",
    "graphics4",
]
WORKER_TIMEOUT_S = 1800


# ============================================================================
# Host enumeration (parent side)
# ============================================================================


def query_cards() -> list[dict]:
    """Enumerate NVIDIA cards via nvidia-smi. Returns one dict per card."""
    if shutil.which("nvidia-smi") is None:
        raise SystemExit("nvidia-smi not found; this script needs an NVIDIA driver")
    fields = "index,name,uuid,compute_cap,memory.total,driver_version"
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    cards = []
    for line in out.strip().splitlines():
        idx, name, uuid, cc, mem, drv = [c.strip() for c in line.split(",")]
        cards.append(
            {
                "index": int(idx),
                "name": name,
                "short": short_card_name(name),
                "uuid": uuid,
                "cc": cc,
                "sm": "sm_" + cc.replace(".", ""),
                "memory_mib": int(mem),
                "driver": drv,
            }
        )
    return cards


def query_compute_apps() -> list[dict]:
    """Other processes holding the cards at run start (they share the GPUs)."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    ).stdout
    apps = []
    for line in out.strip().splitlines():
        parts = [c.strip() for c in line.split(",", 3)]
        if len(parts) == 4:
            apps.append(
                {"pid": parts[0], "uuid": parts[1], "mib": parts[2], "name": parts[3]}
            )
    return apps


def short_card_name(name: str) -> str:
    """'NVIDIA GeForce RTX 3060' -> 'RTX 3060'; 'Quadro GP100' -> 'GP100'."""
    for prefix in ("NVIDIA ", "GeForce ", "Quadro ", "Tesla "):
        name = name.replace(prefix, "")
    return name.strip()


def parse_toolkits(spec: str) -> dict[str, Path]:
    """'12=/usr/local/cuda-12,13=/usr/local/cuda-13' -> {label: path}."""
    toolkits: dict[str, Path] = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        if "=" not in item:
            raise SystemExit(f"--toolkits entry '{item}' must be LABEL=PATH")
        label, path = item.split("=", 1)
        toolkits[label.strip()] = Path(path).expanduser()
    return toolkits


def parse_matrix(spec: str, toolkits: dict[str, Path], cards: list[dict]):
    """'12:all,13:3060' -> list of (toolkit_label, card) pairs to run."""
    plan = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        if ":" not in item:
            raise SystemExit(f"--matrix entry '{item}' must be LABEL:CARDS")
        label, card_spec = item.split(":", 1)
        label = label.strip()
        if label not in toolkits:
            raise SystemExit(f"--matrix names toolkit '{label}' not in --toolkits")
        wanted = [c.strip().lower() for c in card_spec.split("+") if c.strip()]
        for card in cards:
            hay = card["name"].lower()
            if "all" in wanted or any(w in hay for w in wanted):
                plan.append((label, card))
    return plan


def nvcc_version(nvcc: Path) -> str:
    out = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True)
    m = re.search(r"release (\d+\.\d+), V([\d.]+)", out.stdout)
    return m.group(2) if m else "unknown"


def nvcc_supports(nvcc: Path, sm: str) -> tuple[bool, str]:
    """Probe whether this nvcc can emit code for ``sm`` (e.g. 'sm_52')."""
    with tempfile.TemporaryDirectory(prefix="yapocis-nvcc-") as tmp:
        src = Path(tmp) / "probe.cu"
        src.write_text(
            "__global__ void k(float *a) { a[threadIdx.x] = 1.0f; }\n", encoding="utf-8"
        )
        proc = subprocess.run(
            [str(nvcc), f"-arch={sm}", "--cubin", "-o", os.devnull, str(src)],
            capture_output=True,
            text=True,
        )
    if proc.returncode == 0:
        return True, ""
    msg = (proc.stderr or proc.stdout).strip().splitlines()
    return False, msg[-1] if msg else f"nvcc exit {proc.returncode}"


# ============================================================================
# Worker (child side): one card, one toolkit
# ============================================================================


class Timer:
    """Collect setup, first-call and steady-state timings for one workload."""

    def __init__(self, repeats: int):
        self.repeats = repeats
        self.setup_s = 0.0

    def setup(self, fn):
        """Time a constructor; yapocis compiles kernels at construction."""
        t0 = time.perf_counter()
        obj = fn()
        self.setup_s = time.perf_counter() - t0
        return obj

    def run(self, fn, sync=None) -> dict:
        t0 = time.perf_counter()
        result = fn()
        if sync:
            sync()
        first = time.perf_counter() - t0
        samples = []
        for _ in range(self.repeats):
            t0 = time.perf_counter()
            result = fn()
            if sync:
                sync()
            samples.append(time.perf_counter() - t0)
        return {
            "first_call_s": first,
            "median_s": statistics.median(samples),
            "min_s": min(samples),
            "max_s": max(samples),
            "samples": samples,
            "result": result,
        }


def _fingerprint(a) -> float:
    """Cheap, deterministic scalar for cross-card consistency checks."""
    import numpy as np

    a = np.asarray(a)
    if a.dtype.kind == "c":
        a = np.abs(a)
    return float(np.mean(a.astype(np.float64)))


def workload_memcpy(size: int, timer: Timer) -> dict:
    """Host<->device bandwidth through PyCUDA, pageable host memory."""
    import numpy as np
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray

    from yapocis.cu.device_manager import get_device_manager

    timer.setup(get_device_manager)  # same context the library workloads use
    nbytes = 256 << 20  # 256 MiB
    host = np.random.random_sample(nbytes // 4).astype(np.float32)
    dev = gpuarray.empty_like(gpuarray.to_gpu(host))
    out = np.empty_like(host)

    def h2d():
        dev.set(host)
        return None

    def d2h():
        dev.get(ary=out)
        return None

    r_h2d = timer.run(h2d, sync=cuda.Context.synchronize)
    r_d2h = timer.run(d2h, sync=cuda.Context.synchronize)
    gib = nbytes / (1 << 30)
    return {
        "metric": "GiB/s",
        "first_call_s": r_h2d["first_call_s"],
        "median_s": r_h2d["median_s"] + r_d2h["median_s"],
        "min_s": r_h2d["min_s"] + r_d2h["min_s"],
        "h2d_gibps": gib / r_h2d["median_s"],
        "d2h_gibps": gib / r_d2h["median_s"],
        "throughput": gib / r_h2d["median_s"],
        "throughput_note": "H2D",
        "fingerprint": _fingerprint(out),
        "work_pixels": nbytes // 4,
    }


def workload_gaussian(size: int, timer: Timer) -> dict:
    """Separable Gaussian (convolvesep) at the widest built-in scale."""
    import numpy as np

    from yapocis.lib.gaussian import Gaussian

    g = timer.setup(Gaussian)
    scale = g.get_scales() - 1
    img = (np.random.random_sample((size, size)) * 256.0).astype(np.float32)
    r = timer.run(lambda: g.gauss_image(img, scale))
    return {
        "metric": "Mpix/s",
        "throughput": size * size / r["median_s"] / 1e6,
        "fingerprint": _fingerprint(r["result"]),
        "work_pixels": size * size,
        "detail": f"scale {scale}, width {g.get_gaussian_kernel(scale).width}",
        **{k: r[k] for k in ("first_call_s", "median_s", "min_s", "max_s")},
    }


def workload_median3x3(size: int, timer: Timer) -> dict:
    """Ping-pong iterations of the resident 3x3 median, one transfer each way."""
    import numpy as np

    from yapocis.lib.median import Median

    iterations = 50
    m = timer.setup(Median)
    img = (np.random.random_sample((size, size)) * 256.0).astype(np.float32)
    r = timer.run(lambda: m.median3x3(img, iterations=iterations))
    return {
        "metric": "Mpix/s",
        "throughput": size * size * iterations / r["median_s"] / 1e6,
        "fingerprint": _fingerprint(r["result"]),
        "work_pixels": size * size * iterations,
        "detail": f"{iterations} iterations",
        **{k: r[k] for k in ("first_call_s", "median_s", "min_s", "max_s")},
    }


def workload_mandelbrot(size: int, timer: Timer) -> dict:
    """Mandelbrot escape iteration, compute-bound complex64 kernel."""
    import numpy as np

    from yapocis.lib.mandelbrot import Mandelbrot

    maxiter = 1024
    mb = timer.setup(Mandelbrot)
    re_ = np.linspace(-2.0, 1.0, size, dtype=np.float32)
    im_ = np.linspace(-1.5, 1.5, size, dtype=np.float32)
    xx, yy = np.meshgrid(re_, im_, indexing="xy")
    q = (xx + 1j * yy).astype(np.complex64)
    r = timer.run(lambda: mb.mandelbrot(q, maxiter))
    return {
        "metric": "Mpix/s",
        "throughput": size * size / r["median_s"] / 1e6,
        "fingerprint": _fingerprint(r["result"]),
        "work_pixels": size * size,
        "detail": f"maxiter {maxiter}",
        **{k: r[k] for k in ("first_call_s", "median_s", "min_s", "max_s")},
    }


def workload_anisodiff(size: int, timer: Timer) -> dict:
    """Iterated Perona-Malik anisotropic diffusion, one transfer each way."""
    import numpy as np

    from yapocis.lib.anisodiff import Anisodiff

    n = 50
    ad = timer.setup(Anisodiff)
    img = (np.random.random_sample((size, size)) * 256.0).astype(np.float32)
    r = timer.run(lambda: ad.anisodiff_n(img, n=n, scale=3.0, step_size=0.2))
    return {
        "metric": "Mpix/s",
        "throughput": size * size * n / r["median_s"] / 1e6,
        "fingerprint": _fingerprint(r["result"]),
        "work_pixels": size * size * n,
        "detail": f"{n} iterations",
        **{k: r[k] for k in ("first_call_s", "median_s", "min_s", "max_s")},
    }


def workload_hsi(size: int, timer: Timer) -> dict:
    """RGB to HSI colour-space conversion on a 3-channel image."""
    import numpy as np

    from yapocis.lib.hsi import HSI

    hsi = timer.setup(HSI)
    rgb = np.random.random_sample((size, size, 3)).astype(np.float32)
    r = timer.run(lambda: hsi.rgb2hsi3(rgb))
    return {
        "metric": "Mpix/s",
        "throughput": size * size / r["median_s"] / 1e6,
        "fingerprint": _fingerprint(r["result"]),
        "work_pixels": size * size,
        **{k: r[k] for k in ("first_call_s", "median_s", "min_s", "max_s")},
    }


def workload_graphics4(size: int, timer: Timer) -> dict:
    """KernelDraw4 display list of random lines and circles on an RGBA canvas."""
    import numpy as np

    from yapocis.lib.kerneldraw4 import KernelDraw4

    count = 2000
    rng = np.random.default_rng(1234)
    canvas = np.zeros((size, size, 4), dtype=np.float32)
    ad = timer.setup(lambda: KernelDraw4(canvas))
    xy = rng.integers(0, size, (count, 2))
    dxy = rng.integers(-50, 50, (count, 2))
    lines = np.hstack([xy, xy + dxy]).tolist()
    circles = np.hstack([xy, rng.integers(2, 50, (count, 1))]).tolist()
    colors = rng.random((count, 4)).astype(np.float32)

    def draw():
        canvas[...] = 0.0
        ad.graphics.write(canvas)
        ad.lines(lines, colors=colors, distance=1.5)
        ad.circles(circles, colors=colors)
        ad.render()
        return ad.read()

    r = timer.run(draw)
    return {
        "metric": "Mpix/s",
        "throughput": size * size / r["median_s"] / 1e6,
        "fingerprint": _fingerprint(r["result"]),
        "work_pixels": size * size,
        "detail": f"{count} lines + {count} circles",
        **{k: r[k] for k in ("first_call_s", "median_s", "min_s", "max_s")},
    }


WORKLOAD_FUNCS = {
    "memcpy": workload_memcpy,
    "gaussian": workload_gaussian,
    "median3x3": workload_median3x3,
    "mandelbrot": workload_mandelbrot,
    "anisodiff": workload_anisodiff,
    "hsi": workload_hsi,
    "graphics4": workload_graphics4,
}


def run_worker(args) -> int:
    """Child process body: run every requested workload on the visible card."""
    import numpy as np
    import pycuda
    import pycuda.driver as cuda

    os.environ["YAPOCIS_BACKEND"] = "cuda"
    np.random.seed(1234)

    cuda.init()
    if cuda.Device.count() != 1:
        print(
            f"worker expects exactly one visible device, saw {cuda.Device.count()}",
            file=sys.stderr,
        )
    dev = cuda.Device(0)
    nvcc = shutil.which("nvcc") or "nvcc"
    info = {
        "device": dev.name(),
        "compute_capability": "{}.{}".format(*dev.compute_capability()),
        "total_memory_mib": dev.total_memory() >> 20,
        "sm_count": dev.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT),
        "clock_mhz": dev.get_attribute(cuda.device_attribute.CLOCK_RATE) // 1000,
        "nvcc": nvcc,
        "nvcc_version": nvcc_version(Path(nvcc)),
        "cuda_driver_api": "{}.{}".format(*divmod(cuda.get_driver_version(), 1000)),
        "pycuda_version": pycuda.VERSION_TEXT,
        "python": platform.python_version(),
    }

    timer = Timer(args.repeats)
    results = {}
    for name in args.workloads.split(","):
        fn = WORKLOAD_FUNCS[name]
        print(
            f"[{info['device']} / nvcc {info['nvcc_version']}] {name} ...", flush=True
        )
        t0 = time.perf_counter()
        try:
            res = fn(args.size, timer)
            res["status"] = "ok"
            res["setup_s"] = timer.setup_s
            res["wall_s"] = time.perf_counter() - t0
            print(
                f"    setup {res['setup_s']:7.2f} s   first {res['first_call_s']*1e3:9.2f} ms   "
                f"median {res['median_s']*1e3:9.3f} ms   "
                f"{res['throughput']:9.2f} {res['metric']}",
                flush=True,
            )
        except Exception as exc:  # report, keep going with the other workloads
            import traceback

            traceback.print_exc()
            res = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        results[name] = res

    payload = {
        "info": info,
        "size": args.size,
        "repeats": args.repeats,
        "results": results,
    }
    Path(args.json_out).write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8"
    )
    return 0


# ============================================================================
# Parent: orchestration
# ============================================================================


def worker_env(toolkit: Path, card: dict, use_cache: bool) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = card["uuid"]
    env["PATH"] = f"{toolkit / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["CUDA_HOME"] = str(toolkit)
    env["CUDA_PATH"] = str(toolkit)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        p for p in (str(toolkit / "lib64"), env.get("LD_LIBRARY_PATH", "")) if p
    )
    env["YAPOCIS_BACKEND"] = "cuda"
    env.pop("YAPOCIS_DEVICE", None)
    if use_cache:
        env.pop("PYCUDA_DISABLE_CACHE", None)
    else:
        env["PYCUDA_DISABLE_CACHE"] = "1"
    return env


def run_plan(plan, toolkits, args, run_dir: Path) -> list[dict]:
    """Run one worker per (toolkit, card); return per-pair result records."""
    records = []
    for label, card in plan:
        toolkit = toolkits[label]
        nvcc = toolkit / "bin" / "nvcc"
        record = {
            "toolkit": label,
            "toolkit_path": str(toolkit),
            "card": card,
            "status": "ok",
        }
        tag = f"cuda{label}_{card['short'].replace(' ', '_')}"
        if not nvcc.exists():
            record.update(status="missing", reason=f"{nvcc} not found")
            records.append(record)
            print(f"== {tag}: toolkit missing ({nvcc})")
            continue
        record["nvcc_version"] = nvcc_version(nvcc)
        ok, why = nvcc_supports(nvcc, card["sm"])
        if not ok:
            record.update(status="unsupported", reason=why)
            records.append(record)
            print(
                f"== {tag}: {card['sm']} unsupported by nvcc {record['nvcc_version']}: {why}"
            )
            continue

        json_out = run_dir / f"{tag}.json"
        log_path = run_dir / f"{tag}.log"
        cmd = [
            sys.executable,
            "-m",
            "yapocis.scripts.cuda_cards",
            "--worker",
            "--size",
            str(args.size),
            "--repeats",
            str(args.repeats),
            "--workloads",
            args.workloads,
            "--json-out",
            str(json_out),
        ]
        print(
            f"== {tag}: {card['name']} ({card['sm']}) with nvcc {record['nvcc_version']}"
        )
        t0 = time.perf_counter()
        with open(log_path, "w", encoding="utf-8") as log:
            try:
                proc = subprocess.run(
                    cmd,
                    env=worker_env(toolkit, card, args.use_cache),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                )
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        record["wall_s"] = time.perf_counter() - t0
        record["log"] = str(log_path)
        if rc != 0 or not json_out.exists():
            record.update(status="failed", reason=f"worker exit {rc}, see {log_path}")
            print(f"   FAILED ({record['reason']})")
            tail = log_path.read_text(encoding="utf-8").splitlines()[-5:]
            for line in tail:
                print("   | " + line)
        else:
            payload = json.loads(json_out.read_text(encoding="utf-8"))
            record["info"] = payload["info"]
            record["results"] = payload["results"]
            for name, res in payload["results"].items():
                if res["status"] == "ok":
                    print(
                        f"   {name:11s} median {res['median_s']*1e3:9.3f} ms  "
                        f"{res['throughput']:9.2f} {res['metric']}"
                    )
                else:
                    print(f"   {name:11s} ERROR {res['error']}")
        records.append(record)
    return records


# ============================================================================
# Reporting
# ============================================================================


def _cell(v) -> str:
    """Markdown table cell: never contains a newline or a bare pipe."""
    return str(v).replace("\n", "\\n").replace("|", "\\|")


def md_table(headers: list[str], rows: list[list]) -> str:
    """Markdown table surrounded by blank lines, per the report conventions."""
    lines = [
        "| " + " | ".join(_cell(h) for h in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return "\n".join(lines) + "\n"


def _label(rec: dict) -> str:
    return f"{rec['card']['short']} / CUDA {rec['toolkit']}"


def build_report(
    records: list[dict], cards: list[dict], toolkits, args, started, apps
) -> str:
    workloads = args.workloads.split(",")
    ok = [r for r in records if r["status"] == "ok"]
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    host = platform.node()
    driver = cards[0]["driver"] if cards else "unknown"

    out = []
    out.append(f"# CUDA per-card performance: {host}\n")
    out.append(
        f"Generated {now}. Image size {args.size}x{args.size}, "
        f"{args.repeats} timed repeats per workload (median reported), "
        f"{'PyCUDA compile cache enabled' if args.use_cache else 'fresh nvcc compile (PyCUDA cache disabled)'}. "
        f"Driver {driver}. Wall time {time.perf_counter() - started:.0f} s.\n"
    )

    out.append("## Toolkits\n")
    rows = []
    for label, path in toolkits.items():
        nvcc = path / "bin" / "nvcc"
        ver = nvcc_version(nvcc) if nvcc.exists() else "missing"
        rows.append([f"CUDA {label}", str(path), ver])
    out.append(md_table(["Toolkit", "Path", "nvcc"], rows))

    out.append("## Cards\n")
    rows = [[c["index"], c["name"], c["cc"], c["memory_mib"], c["uuid"]] for c in cards]
    out.append(md_table(["Index", "Card", "CC", "VRAM (MiB)", "UUID"], rows))

    out.append("## Other processes on the cards at start\n")
    if apps:
        out.append(
            "These processes held the cards while the benchmark ran; the cards are "
            "shared, so their activity adds noise to the timings.\n"
        )
        by_uuid = {c["uuid"]: c["short"] for c in cards}
        rows = [
            [by_uuid.get(a["uuid"], a["uuid"]), a["pid"], a["mib"], a["name"]]
            for a in apps
        ]
        out.append(md_table(["Card", "PID", "Memory (MiB)", "Process"], rows))
    else:
        out.append("None.\n")

    out.append("## Run matrix\n")
    rows = []
    for r in records:
        info = r.get("info", {})
        rows.append(
            [
                _label(r),
                r["card"]["sm"],
                r.get("nvcc_version", "-"),
                r["status"],
                r.get("reason", f"{r.get('wall_s', 0):.0f} s"),
                info.get("sm_count", "-"),
                info.get("clock_mhz", "-"),
            ]
        )
    out.append(
        md_table(
            ["Pair", "Arch", "nvcc", "Status", "Note / wall", "SMs", "Clock (MHz)"],
            rows,
        )
    )

    # Per-workload tables
    out.append("## Results by workload\n")
    out.append(
        "Setup is the library constructor, which renders the Mako templates and "
        "runs nvcc for every kernel in the interface the first time that interface "
        "is used in the process (a later workload sharing an interface shows ~0); "
        "median and min are "
        "steady-state host-side wall times including host-device transfers, "
        "as the yapocis library performs them.\n"
    )
    for name in workloads:
        out.append(f"### {name}\n")
        detail = next(
            (
                r["results"][name].get("detail")
                for r in ok
                if r["results"].get(name, {}).get("detail")
            ),
            None,
        )
        if detail:
            out.append(f"{WORKLOAD_FUNCS[name].__doc__.strip()} ({detail}).\n")
        else:
            out.append(f"{WORKLOAD_FUNCS[name].__doc__.strip()}\n")
        rows = []
        for r in ok:
            res = r["results"].get(name)
            if res is None:
                continue
            if res["status"] != "ok":
                rows.append([_label(r), "error", "-", "-", "-", _cell(res["error"])])
                continue
            if name == "memcpy":
                thr = f"{res['h2d_gibps']:.2f} H2D / {res['d2h_gibps']:.2f} D2H {res['metric']}"
            else:
                thr = f"{res['throughput']:.1f} {res['metric']}"
            rows.append(
                [
                    _label(r),
                    f"{res['setup_s']:.2f}",
                    f"{res['median_s']*1e3:.3f}",
                    f"{res['min_s']*1e3:.3f}",
                    thr,
                    f"{res['fingerprint']:.6g}",
                ]
            )
        out.append(
            md_table(
                [
                    "Pair",
                    "Setup (s)",
                    "Median (ms)",
                    "Min (ms)",
                    "Throughput",
                    "Fingerprint",
                ],
                rows,
            )
        )

    # Summary: relative steady-state time vs fastest pair per workload
    out.append("## Summary: median time relative to the fastest pair\n")
    out.append(
        "Each cell is median time divided by the fastest median for that workload "
        "(1.00 = fastest). Compile-time-free; memcpy uses H2D+D2H time.\n"
    )
    headers = ["Pair"] + workloads + ["Geo. mean"]
    rows = []
    best = {}
    for name in workloads:
        vals = [
            r["results"][name]["median_s"]
            for r in ok
            if r["results"].get(name, {}).get("status") == "ok"
        ]
        best[name] = min(vals) if vals else None
    for r in ok:
        row = [_label(r)]
        ratios = []
        for name in workloads:
            res = r["results"].get(name, {})
            if res.get("status") == "ok" and best[name]:
                ratio = res["median_s"] / best[name]
                ratios.append(ratio)
                row.append(f"{ratio:.2f}")
            else:
                row.append("-")
        gm = statistics.geometric_mean(ratios) if ratios else float("nan")
        row.append(f"{gm:.2f}")
        rows.append((gm, row))
    rows.sort(key=lambda t: t[0])
    out.append(md_table(headers, [row for _, row in rows]))

    # Toolkit-vs-toolkit on the same card
    pairs_by_card: dict[str, list[dict]] = {}
    for r in ok:
        pairs_by_card.setdefault(r["card"]["uuid"], []).append(r)
    multi = {k: v for k, v in pairs_by_card.items() if len(v) > 1}
    if multi:
        out.append("## Toolkit comparison on the same card\n")
        out.append(
            "Ratio of median time under each later toolkit to the first toolkit "
            "listed for that card (below 1.00 means the later toolkit is faster). "
            "The setup column compares constructor (nvcc compile) times the same way.\n"
        )
        rows = []
        for recs in multi.values():
            base = recs[0]
            for other in recs[1:]:
                for name in workloads:
                    a = base["results"].get(name, {})
                    b = other["results"].get(name, {})
                    if a.get("status") == "ok" and b.get("status") == "ok":
                        rows.append(
                            [
                                base["card"]["short"],
                                name,
                                f"CUDA {other['toolkit']} vs {base['toolkit']}",
                                f"{b['median_s']/a['median_s']:.3f}",
                                f"{b['setup_s']/a['setup_s']:.3f}",
                                f"{a['median_s']*1e3:.3f}",
                                f"{b['median_s']*1e3:.3f}",
                            ]
                        )
        out.append(
            md_table(
                [
                    "Card",
                    "Workload",
                    "Comparison",
                    "Median ratio",
                    "Setup ratio",
                    "Base median (ms)",
                    "Other median (ms)",
                ],
                rows,
            )
        )

    # Consistency
    out.append("## Result consistency\n")
    out.append(
        "Mean of each workload's output, relative to the first successful pair. "
        "Mandelbrot and median are exact operations and should agree to the last digit; "
        "float workloads should agree to fast-math tolerance.\n"
    )
    rows = []
    for name in workloads:
        if name == "memcpy":
            continue
        ref = None
        for r in ok:
            res = r["results"].get(name, {})
            if res.get("status") != "ok":
                continue
            if ref is None:
                ref = res["fingerprint"]
                rows.append([name, _label(r), f"{ref:.9g}", "reference"])
            else:
                dev = abs(res["fingerprint"] - ref) / (abs(ref) or 1.0)
                rows.append(
                    [name, _label(r), f"{res['fingerprint']:.9g}", f"{dev:.2e}"]
                )
    out.append(md_table(["Workload", "Pair", "Mean", "Relative deviation"], rows))

    skipped = [r for r in records if r["status"] != "ok"]
    if skipped:
        out.append("## Skipped or failed pairs\n")
        rows = [[_label(r), r["status"], r.get("reason", "")] for r in skipped]
        out.append(md_table(["Pair", "Status", "Reason"], rows))

    return "\n".join(out) + "\n"


def write_pdf(md_path: Path) -> Path | None:
    if shutil.which("pandoc") is None:
        print("pandoc not found; skipping PDF")
        return None
    pdf_path = md_path.with_suffix(".pdf")
    cmd = [
        "pandoc",
        str(md_path),
        "-o",
        str(pdf_path),
        "-V",
        "geometry:margin=1.5cm",
        "-V",
        "geometry:landscape",
        "-V",
        "fontsize=9pt",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"pandoc failed: {proc.stderr.strip()}")
        return None
    return pdf_path


def record_db(records: list[dict], args) -> None:
    """Record each workload as a run in the yapocis performance DuckDB."""
    try:
        from yapocis.utils.performance import save_performance_data
    except Exception as exc:  # duckdb missing etc.
        print(f"performance DB unavailable ({exc}); skipping")
        return
    for r in records:
        if r["status"] != "ok":
            continue
        info = r["info"]
        for name, res in r["results"].items():
            if res["status"] != "ok":
                continue
            save_performance_data(
                test_name=f"cuda_cards.{name}[{args.size}]",
                file_path=os.path.abspath(__file__),
                backend="cuda",
                timing_data={
                    "setup": res["setup_s"],
                    "first_call": res["first_call_s"],
                    "median": res["median_s"],
                    "min": res["min_s"],
                },
                device_name=f"{r['card']['name']} [CUDA {r['toolkit']}]",
                driver_version=f"driver {r['card']['driver']}, nvcc {info['nvcc_version']}",
            )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--toolkits",
        default=DEFAULT_TOOLKITS,
        help="LABEL=PATH,... (default: %(default)s)",
    )
    p.add_argument(
        "--matrix",
        default=DEFAULT_MATRIX,
        help="LABEL:CARD+CARD,... where CARD is a name substring or 'all' (default: %(default)s)",
    )
    p.add_argument(
        "--size",
        type=int,
        default=1024,
        help="image edge in pixels (default: %(default)s)",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="timed repeats after the first call (default: %(default)s)",
    )
    p.add_argument(
        "--workloads",
        default=",".join(WORKLOADS),
        help="comma list from: " + ", ".join(WORKLOADS),
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="report directory (default: $YAPOCIS_PERFORMANCE_DIR or ./performance)",
    )
    p.add_argument(
        "--use-cache",
        action="store_true",
        help="keep the PyCUDA compile cache (setup times then exclude nvcc)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=WORKER_TIMEOUT_S,
        help="per-worker timeout in seconds",
    )
    p.add_argument(
        "--no-db",
        action="store_true",
        help="do not record runs in the performance DuckDB",
    )
    p.add_argument("--no-pdf", action="store_true", help="skip pandoc PDF conversion")
    p.add_argument(
        "--list",
        action="store_true",
        help="show cards, toolkits and the run plan, then exit",
    )
    # worker mode
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--json-out", default=None, help=argparse.SUPPRESS)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    unknown = [w for w in args.workloads.split(",") if w not in WORKLOAD_FUNCS]
    if unknown:
        raise SystemExit(f"unknown workloads: {unknown}; choose from {WORKLOADS}")
    if args.worker:
        return run_worker(args)

    if sys.platform != "linux":
        print(
            "warning: written for Linux hosts; nvidia-smi and toolkit paths may differ"
        )

    cards = query_cards()
    toolkits = parse_toolkits(args.toolkits)
    plan = parse_matrix(args.matrix, toolkits, cards)

    print(f"Cards ({len(cards)}):")
    for c in cards:
        print(
            f"  [{c['index']}] {c['name']:28s} {c['sm']}  {c['memory_mib']:6d} MiB  {c['uuid']}"
        )
    print("Toolkits:")
    for label, path in toolkits.items():
        nvcc = path / "bin" / "nvcc"
        print(
            f"  CUDA {label:4s} {path}  nvcc {nvcc_version(nvcc) if nvcc.exists() else 'MISSING'}"
        )
    print("Plan:")
    for label, c in plan:
        print(f"  CUDA {label:4s} on {c['name']}")
    if args.list:
        return 0
    if not plan:
        raise SystemExit("run plan is empty; check --matrix against the card names")

    started = time.perf_counter()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(
        args.output_dir or os.environ.get("YAPOCIS_PERFORMANCE_DIR") or "performance"
    )
    run_dir = out_dir / f"cuda_cards_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    apps = query_compute_apps()
    records = run_plan(plan, toolkits, args, run_dir)

    report = build_report(records, cards, toolkits, args, started, apps)
    md_path = out_dir / f"cuda_cards_{stamp}.md"
    md_path.write_text(report, encoding="utf-8")
    json_path = out_dir / f"cuda_cards_{stamp}.json"
    json_path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {md_path}")
    print(f"Data:   {json_path}")
    if not args.no_pdf:
        pdf = write_pdf(md_path)
        if pdf:
            print(f"PDF:    {pdf}")
    if not args.no_db:
        record_db(records, args)
    return 0 if any(r["status"] == "ok" for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())

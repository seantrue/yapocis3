"""Host-numerics fixtures for bit-exact ports (`yapocis-tools bundle --host-fixtures DIR`).

The Rust port of Virtual Impressionist (impressionist-rs, on yapocis-rs) reproduces
numpy's host-side numerics bit for bit: the PCG64 generator and its draws, pairwise
sums, quantiles, sorts, and numpy's own SIMD implementations of float32 sin/cos.
Those results depend on the host (numpy's SIMD dispatch, libm and BLAS differ by
CPU), so this module records what *this* host's numpy computes, under
``DIR/<host slug>/``: an ``index.json`` naming the host, the library versions and
every case, plus one ``.npz`` per case holding the inputs and outputs. A Rust test
replays the cases recorded on its own host and compares bit patterns, with no
tolerance.

Sections (``--host-fixtures-sections``): ``random`` (SeedSequence/PCG64/Generator),
``simd_math``, ``reduce``, ``hist``, and the numpy behaviours PlumCraft's array engine
needs: ``elementwise`` (minimum, maximum, clip, sign, mod, nan_to_num), ``extrema``
(min, max, nanmin, nanmax, integer cumsum), ``order`` (sort, searchsorted, lexsort,
quantile, median, nanpercentile, unique), ``fft`` (fftshift, ifftshift) and ``complex``
(complex64 multiply: numpy's ufunc loop over arrays and its scalar path, which round
differently on some hosts).
Each random case seeds one generator and makes a sequence of calls, so the
interleaving of 64-bit and buffered 32-bit draws is covered too.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def cpu_brand() -> str:
    """The CPU's marketing name, or the machine type when it can't be read."""
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.machine()


def host_slug() -> str:
    """``<system>-<machine>-<cpu brand>``, slugged: the per-host fixture directory."""
    return _slug(f"{platform.system()}-{platform.machine()}-{cpu_brand()}")


# ---------------------------------------------------------------------------
# random: SeedSequence / PCG64 / Generator
# ---------------------------------------------------------------------------

# Each case: a seed (int or list of ints, as default_rng takes it) and the calls made
# on that one generator, in order. A call is (method, kwargs, output name).
RANDOM_CASES: list[tuple[str, Any, list[tuple[str, dict[str, Any], str]]]] = [
    ("f64_seed0", 0, [("random", {"size": 1000}, "random_f64")]),
    ("f64_seed42", 42, [("random", {"size": 1000}, "random_f64")]),
    ("f32_seed0", 0, [("random", {"size": 1001, "dtype": "float32"}, "random_f32")]),
    (
        "f32_then_f64_list_seed",
        [0, 3735928559],
        [
            ("random", {"size": 7, "dtype": "float32"}, "a"),
            ("random", {"size": 5}, "b"),
            ("random", {"size": 7, "dtype": "float32"}, "c"),
        ],
    ),
    ("big_seed", 1 << 40, [("random", {"size": 100}, "random_f64")]),
    ("huge_seed", (1 << 70) + 12345, [("random", {"size": 100}, "random_f64")]),
    (
        "integers_small",
        7,
        [
            ("integers", {"low": 0, "high": 7, "size": 2000}, "i7"),
            ("integers", {"low": -5, "high": 6, "size": 100}, "i_neg"),
            ("integers", {"low": 0, "high": 1, "size": 10}, "i_one"),
        ],
    ),
    ("integers_u32_edge", 8, [("integers", {"low": 0, "high": 1 << 32, "size": 300}, "i32max")]),
    ("integers_64", 9, [("integers", {"low": 0, "high": 1 << 40, "size": 300}, "i40")]),
    ("integers_huge", 10, [("integers", {"low": 0, "high": 1000003, "size": 3000}, "i1m")]),
    (
        "integers_broadcast",
        11,
        [
            ("integers_broadcast", {"low": 0, "highs": [3, 10, 1 << 20, 2, 1 << 33, 97, 1, 5]}, "ib"),
            ("random", {"size": 3, "dtype": "float32"}, "f"),
            ("integers_broadcast", {"low": 0, "highs": [1000] * 50}, "ib2"),
        ],
    ),
    (
        "mixed_buffer",
        12,
        [
            ("random", {"size": 1, "dtype": "float32"}, "f1"),
            ("integers", {"low": 0, "high": 1000, "size": 3}, "i1"),
            ("random", {"size": 2}, "d1"),
            ("integers", {"low": 0, "high": 1 << 40, "size": 2}, "i2"),
            ("random", {"size": 3, "dtype": "float32"}, "f2"),
            ("standard_normal", {"size": 2}, "n1"),
            ("random", {"size": 1, "dtype": "float32"}, "f3"),
        ],
    ),
    ("standard_normal", 13, [("standard_normal", {"size": 20000}, "n")]),
    ("normal", 14, [("normal", {"loc": 2.5, "scale": 3.0, "size": 5000}, "n")]),
    ("uniform", 15, [("uniform", {"low": -1.0, "high": 4.5, "size": 5000}, "u")]),
    ("exponential", 16, [("exponential", {"scale": 1.5, "size": 20000}, "e")]),
    ("standard_exponential", 17, [("standard_exponential", {"size": 20000}, "e")]),
    (
        "geometric",
        18,
        [
            ("geometric", {"p": 0.5, "size": 3000}, "g_half"),
            ("geometric", {"p": 0.1, "size": 3000}, "g_tenth"),
            ("geometric", {"p": 1.0 / 3.0, "size": 300}, "g_third"),
            ("geometric", {"p": 0.9999, "size": 300}, "g_high"),
        ],
    ),
    (
        "choice_floyd",
        19,
        [
            ("choice", {"pop": 100, "size": 10}, "c1"),
            ("choice", {"pop": 20000, "size": 300}, "c2"),
            ("choice", {"pop": 50, "size": 50}, "c3"),
            ("choice", {"pop": 10000, "size": 1000}, "c4"),
        ],
    ),
    (
        "choice_tail_shuffle",
        20,
        [
            ("choice", {"pop": 20000, "size": 1000}, "c1"),
            ("choice", {"pop": 1_900_000, "size": 200_000}, "c2"),
            ("choice", {"pop": 10001, "size": 201}, "c3"),
        ],
    ),
    (
        "permutation_shuffle",
        21,
        [
            ("permutation", {"n": 1000}, "p1"),
            ("permutation", {"n": 5}, "p2"),
            ("permutation", {"n": 1 << 17}, "p3"),
            ("shuffle_arange", {"n": 300}, "s1"),
        ],
    ),
    ("seed_list_pool_overflow", [1, 2, 3, 4, 5, 6], [("random", {"size": 50}, "r")]),
    ("seed_list_zero_crc", [0, 0], [("random", {"size": 50, "dtype": "float32"}, "r")]),
]


def _run_random_call(rng: np.random.Generator, method: str, kw: dict[str, Any]) -> np.ndarray:
    if method == "random":
        dtype = np.float32 if kw.get("dtype") == "float32" else np.float64
        return np.asarray(rng.random(kw["size"], dtype=dtype))
    if method == "integers":
        return np.asarray(rng.integers(kw["low"], kw["high"], size=kw["size"]))
    if method == "integers_broadcast":
        return np.asarray(rng.integers(kw["low"], np.asarray(kw["highs"], dtype=np.int64)))
    if method == "choice":
        return np.asarray(rng.choice(kw["pop"], size=kw["size"], replace=False))
    if method == "permutation":
        return np.asarray(rng.permutation(kw["n"]))
    if method == "shuffle_arange":
        a = np.arange(kw["n"], dtype=np.int64)
        rng.shuffle(a)
        return a
    if method in ("standard_normal", "standard_exponential"):
        return np.asarray(getattr(rng, method)(kw["size"]))
    if method in ("normal", "uniform", "exponential", "geometric"):
        return np.asarray(getattr(rng, method)(**kw))
    raise ValueError(method)


def export_random(out_dir: str) -> list[dict[str, Any]]:
    cases = []
    for name, seed, calls in RANDOM_CASES:
        rng = np.random.default_rng(seed)
        arrays = {}
        recorded = []
        for method, kw, out_name in calls:
            arrays[out_name] = _run_random_call(rng, method, kw)
            recorded.append({"method": method, "args": kw, "out": out_name})
        file = f"random.{name}.npz"
        np.savez(os.path.join(out_dir, file), **arrays)
        cases.append(
            {
                "func": "random",
                "name": name,
                "file": file,
                # as decimal strings: JSON readers would round a 70-bit seed to a float
                "seed": [str(v) for v in (seed if isinstance(seed, list) else [seed])],
                "calls": recorded,
            }
        )
    return cases


def fma_contraction() -> dict[str, bool]:
    """Whether this host's numpy contracted `a + b * c` into a fused multiply-add
    in `random_normal` and `random_uniform` (clang does on arm64; the manylinux
    x86-64 builds do not). The Rust port must use the same arithmetic."""
    import math

    out = {}
    rng = np.random.default_rng(12345)
    z = rng.standard_normal(20000)
    got = np.random.default_rng(12345).normal(2.5, 3.0, 20000)
    plain = np.float64(2.5) + np.float64(3.0) * z
    fused = np.array([math.fma(3.0, float(v), 2.5) for v in z])
    out["normal_fma"] = bool((got != plain).any() and (got == fused).all())
    out["normal_plain"] = bool((got == plain).all())
    rng = np.random.default_rng(12345)
    u = rng.random(20000)
    got = np.random.default_rng(12345).uniform(-1.0, 4.5, 20000)
    plain = np.float64(-1.0) + np.float64(5.5) * u
    fused = np.array([math.fma(5.5, float(v), -1.0) for v in u])
    out["uniform_fma"] = bool((got != plain).any() and (got == fused).all())
    out["uniform_plain"] = bool((got == plain).all())
    return out


# ---------------------------------------------------------------------------
# simd_math: numpy's own SIMD transcendentals vs libm
# ---------------------------------------------------------------------------


def _simd_math_inputs_f32() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2026)
    u = rng.random(50_000, dtype=np.float32) * 2 - 1
    specials = np.array(
        [0.0, -0.0, 1e-45, -1e-45, 1e-38, 3.4e38, -3.4e38, np.inf, -np.inf, np.nan,
         np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi, -np.pi / 2, 117435.992, 117436.0,
         71476.0625, 71476.07, -117435.992, 1e6, 1e10, 0.1, 0.5, 1.0, 2.0, 8.0, 9.0],
        dtype=np.float32,
    )
    return {
        "unit": u,
        "pi": u * np.float32(np.pi),
        "hundred": u * np.float32(100),
        "furrow": rng.random(50_000, dtype=np.float32) * np.float32(2e5),
        "quadrant_edges": (np.arange(-2000, 2000, dtype=np.float32) * np.float32(np.pi / 4)),
        "specials": specials,
        "tanh_range": u * np.float32(12),
        "tanh_fine": rng.random(50_000, dtype=np.float32) * np.float32(0.3),
    }


def export_simd_math(out_dir: str) -> list[dict[str, Any]]:
    cases = []
    inputs = _simd_math_inputs_f32()
    for func in ("sin", "cos", "tanh"):
        for name, x in inputs.items():
            if func != "tanh" and name.startswith("tanh"):
                continue
            y = getattr(np, func)(x)
            file = f"simd_math.{func}_f32.{name}.npz"
            np.savez(os.path.join(out_dir, file), x=x, y=y)
            cases.append({"func": f"simd_math.{func}_f32", "name": name, "file": file})
    rng = np.random.default_rng(2027)
    for name, x in {
        "unit": rng.random(100_000) * 2 - 1,
        "range": (rng.random(100_000) * 2 - 1) * 25,
        "fine": rng.random(100_000) * 0.3,
        "specials": inputs["specials"].astype(np.float64),
    }.items():
        y = np.tanh(x)
        file = f"simd_math.tanh_f64.{name}.npz"
        np.savez(os.path.join(out_dir, file), x=x, y=y)
        cases.append({"func": "simd_math.tanh_f64", "name": name, "file": file})
    # Where numpy equals libm on this host (so the Rust port may call libm)
    # is recorded too, for the test to assert its assumptions
    return cases


def libm_agreement() -> dict[str, bool]:
    """Which float32/float64 ufuncs equal the C library on this host (the Rust port
    calls libm for those and ports numpy's SIMD code for the others)."""
    import ctypes
    import ctypes.util

    libm = ctypes.CDLL(ctypes.util.find_library("m") or ctypes.util.find_library("c"))
    rng = np.random.default_rng(3)
    x32 = (rng.random(20_000, dtype=np.float32) * 2 - 1) * np.float32(50)
    x64 = (rng.random(20_000) * 2 - 1) * 50
    out = {}
    for name, fn, cname, x, cty, dt in (
        ("sin_f32", np.sin, "sinf", x32, ctypes.c_float, np.float32),
        ("cos_f32", np.cos, "cosf", x32, ctypes.c_float, np.float32),
        ("exp_f32", np.exp, "expf", x32 / np.float32(10), ctypes.c_float, np.float32),
        ("log_f32", np.log, "logf", np.abs(x32) + np.float32(1e-3), ctypes.c_float, np.float32),
        ("tanh_f32", np.tanh, "tanhf", x32 / np.float32(5), ctypes.c_float, np.float32),
        ("sin_f64", np.sin, "sin", x64, ctypes.c_double, np.float64),
        ("cos_f64", np.cos, "cos", x64, ctypes.c_double, np.float64),
        ("exp_f64", np.exp, "exp", x64 / 10, ctypes.c_double, np.float64),
        ("log_f64", np.log, "log", np.abs(x64) + 1e-3, ctypes.c_double, np.float64),
        ("tanh_f64", np.tanh, "tanh", x64 / 5, ctypes.c_double, np.float64),
    ):
        f = getattr(libm, cname)
        f.restype = cty
        f.argtypes = [cty]
        ref = np.array([f(float(v)) for v in x], dtype=dt)
        out[name] = bool((fn(x) == ref).all())
    return out


# ---------------------------------------------------------------------------
# reduce: pairwise sums, means, variances
# ---------------------------------------------------------------------------


def export_reduce(out_dir: str) -> list[dict[str, Any]]:
    cases = []
    rng = np.random.default_rng(99)
    big = 2.0**60
    lengths = list(range(0, 300)) + [1000, 4096, 100_000, 262_147]
    for dt, tag in ((np.float32, "f32"), (np.float64, "f64")):
        arrays = {}
        for n in lengths:
            a = ((rng.random(n) - 0.5) * 1e3).astype(dt)
            arrays[f"a{n}"] = a
        # cancellation patterns that expose the summation order
        for n in (7, 8, 9, 100, 129, 1000, 5000):
            a = np.ones(n, dtype=dt)
            a[0] = dt(big if dt is np.float64 else 2.0**30)
            a[-1] = -a[0]
            arrays[f"cancel{n}"] = a
        arrays["negzero"] = np.array([-0.0, -0.0, -0.0], dtype=dt)
        arrays["nan"] = np.array([1.0, np.nan, 2.0], dtype=dt)
        results = {}
        for k, a in arrays.items():
            results[f"sum.{k}"] = np.asarray(np.sum(a))
            if a.size:
                results[f"mean.{k}"] = np.asarray(np.mean(a))
                results[f"var0.{k}"] = np.asarray(np.var(a))
                results[f"std0.{k}"] = np.asarray(np.std(a))
            if a.size > 1:
                results[f"var1.{k}"] = np.asarray(np.var(a, ddof=1))
                results[f"std1.{k}"] = np.asarray(np.std(a, ddof=1))
        file = f"reduce.{tag}.npz"
        np.savez(os.path.join(out_dir, file), **{f"in.{k}": v for k, v in arrays.items()}, **results)
        cases.append({"func": f"reduce.{tag}", "name": tag, "file": file, "inputs": list(arrays), "results": list(results)})
    return cases


# ---------------------------------------------------------------------------
# hist: histogram (uniform bins, density), linspace, interp, cumsum, histeq
# ---------------------------------------------------------------------------


def export_hist(out_dir: str) -> list[dict[str, Any]]:
    rng = np.random.default_rng(1234)
    arrays = {
        "unit": rng.random(50_000, dtype=np.float32),
        "skewed": (rng.random(50_000, dtype=np.float32) ** 3),
        "narrow": (np.float32(0.3) + rng.random(2000, dtype=np.float32) * np.float32(1e-3)),
        "small": rng.random(7, dtype=np.float32),
        "ties": np.repeat(rng.random(50, dtype=np.float32), 40),
        "constant": np.full(100, 0.25, dtype=np.float32),
        "image": (rng.random((200, 300), dtype=np.float32) ** 2).ravel(),
    }
    results = {}
    for k, a in arrays.items():
        for nbins in (256, 64):
            hist, bins = np.histogram(a, nbins, density=True)
            results[f"hist{nbins}.{k}"] = hist
            results[f"bins{nbins}.{k}"] = bins
            counts, _ = np.histogram(a, nbins)
            results[f"counts{nbins}.{k}"] = counts
        # VI's core.utils.histeq
        hist, bins = np.histogram(a.ravel(), 256, density=True)
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]
        results[f"cdf.{k}"] = cdf
        results[f"histeq.{k}"] = np.interp(a.ravel(), bins[:-1], cdf).astype(np.float32)
    # interp on its own: float32 xp/fp (converted to double inside), float64 x
    xp = np.sort(rng.random(300, dtype=np.float32))
    fp = rng.random(300, dtype=np.float32)
    x = rng.random(20_000, dtype=np.float32)
    arrays["interp_xp"], arrays["interp_fp"], arrays["interp_x"] = xp, fp, x
    results["interp.f32"] = np.interp(x, xp, fp)
    xp64 = np.sort(rng.random(50)); fp64 = rng.random(50); x64 = rng.random(100_000) * 1.2 - 0.1
    arrays["interp64_xp"], arrays["interp64_fp"], arrays["interp64_x"] = xp64, fp64, x64
    results["interp.f64"] = np.interp(x64, xp64, fp64)
    results["linspace.f32"] = np.linspace(np.float32(0.1), np.float32(0.9), 257, dtype=np.float32)
    results["linspace.f32b"] = np.linspace(np.float32(-3.5), np.float32(12.25), 65, dtype=np.float32)
    results["cumsum.f64"] = np.cumsum(rng.random(1000))
    # linspace with Python-float bounds and dtype=float32: computed in float64, then cast
    results["linspace.py_f32a"] = np.linspace(-2.0, 1.0, 1000, dtype=np.float32)
    results["linspace.py_f32b"] = np.linspace(0.1, 0.7, 333, dtype=np.float32)
    results["linspace.py_f32c"] = np.linspace(-1.5, -1.5, 9, dtype=np.float32)
    file = "hist.npz"
    np.savez(os.path.join(out_dir, file), **{f"in.{k}": v for k, v in arrays.items()}, **results)
    return [{"func": "hist", "name": "hist", "file": file, "inputs": list(arrays), "results": list(results)}]


# ---------------------------------------------------------------------------
# elementwise, extrema, order, fft: what PlumCraft's array engine reproduces
# ---------------------------------------------------------------------------

_F32_SPECIALS = [
    0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.5, -2.5, 7.0, -7.0, 3.0, -3.0, 360.0, -360.0,
    180.0, 1e-45, -1e-45, 1e-38, 3.4e38, -3.4e38, 1e10, -1e10, np.inf, -np.inf, np.nan,
]


def _specials_f32() -> np.ndarray:
    return np.array(_F32_SPECIALS, dtype=np.float32)


def _save(out_dir: str, func: str, arrays: dict[str, np.ndarray], results: dict[str, np.ndarray]) -> dict[str, Any]:
    file = f"{func}.npz"
    np.savez(os.path.join(out_dir, file), **{f"in.{k}": v for k, v in arrays.items()}, **results)
    return {"func": func, "name": func, "file": file, "inputs": list(arrays), "results": list(results)}


def export_elementwise(out_dir: str) -> list[dict[str, Any]]:
    """Binary ufuncs over every pair of specials (SIMD body and tail), and over long random
    arrays with NaN and signed zeros sprinkled in; unary ones over the same."""
    import warnings

    rng = np.random.default_rng(4242)
    sp = _specials_f32()
    pa, pb = np.meshgrid(sp, sp, indexing="ij")
    rand_a = ((rng.random(10_007, dtype=np.float32) - np.float32(0.5)) * np.float32(2000)).astype(np.float32)
    rand_b = ((rng.random(10_007, dtype=np.float32) - np.float32(0.5)) * np.float32(2000)).astype(np.float32)
    for arr in (rand_a, rand_b):
        idx = rng.integers(0, arr.size, 300)
        arr[idx[:100]] = np.nan
        arr[idx[100:200]] = 0.0
        arr[idx[200:]] = -0.0
    arrays = {
        "pairs_a": pa.ravel().copy(),
        "pairs_b": pb.ravel().copy(),
        "rand_a": rand_a,
        "rand_b": rand_b,
        "turns": np.array([360.0, -360.0, 6.2831855, 1.0, 0.0, np.inf], dtype=np.float32),
    }
    results: dict[str, np.ndarray] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tag in ("pairs", "rand"):
            a, b = arrays[f"{tag}_a"], arrays[f"{tag}_b"]
            results[f"minimum.{tag}"] = np.minimum(a, b)
            results[f"maximum.{tag}"] = np.maximum(a, b)
            results[f"mod.{tag}"] = np.mod(a, b)
            results[f"sign.{tag}"] = np.sign(a)
            results[f"nan_to_num.{tag}"] = np.nan_to_num(a)
            results[f"clip0.{tag}"] = np.clip(a, 0, None)
        # np.mod by a Python-float turn, as a scalar (the research software's Wrap Angle)
        for i, t in enumerate(arrays["turns"]):
            results[f"mod_scalar{i}.rand"] = np.mod(arrays["rand_a"], float(t))
            results[f"mod_scalar{i}.pairs"] = np.mod(arrays["pairs_a"], float(t))
    return [_save(out_dir, "elementwise", arrays, results)]


def export_extrema(out_dir: str) -> list[dict[str, Any]]:
    """np.min, np.max, np.nanmin, np.nanmax of float32 arrays of many lengths, with NaN
    at different places, signed zeros and all-NaN arrays; np.cumsum of int64."""
    import warnings

    rng = np.random.default_rng(777)
    arrays: dict[str, np.ndarray] = {}
    for n in list(range(1, 70)) + [127, 128, 129, 1000, 4099, 100_003]:
        arrays[f"r{n}"] = ((rng.random(n, dtype=np.float32) - np.float32(0.5)) * np.float32(100)).astype(np.float32)
        if n >= 3:
            a = arrays[f"r{n}"].copy()
            a[rng.integers(0, n)] = np.nan
            arrays[f"nan{n}"] = a
            z = np.where(rng.random(n) < 0.5, np.float32(0.0), np.float32(-0.0)).astype(np.float32)
            arrays[f"zeros{n}"] = z
            arrays[f"negzero_first{n}"] = np.concatenate([[np.float32(-0.0)], np.zeros(n - 1, np.float32)])
            arrays[f"poszero_first{n}"] = np.concatenate([[np.float32(0.0)], -np.zeros(n - 1, np.float32)])
            arrays[f"allnan{n}"] = np.full(n, np.nan, np.float32)
            inf = arrays[f"r{n}"].copy()
            inf[0], inf[-1] = np.inf, -np.inf
            arrays[f"inf{n}"] = inf
    results: dict[str, np.ndarray] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, a in arrays.items():
            results[f"min.{k}"] = np.asarray(np.min(a))
            results[f"max.{k}"] = np.asarray(np.max(a))
            results[f"nanmin.{k}"] = np.asarray(np.nanmin(a))
            results[f"nanmax.{k}"] = np.asarray(np.nanmax(a))
    counts = rng.integers(0, 1000, 4096).astype(np.int64)
    arrays["counts"] = counts
    results["cumsum_i64.counts"] = np.cumsum(counts)
    wrap = np.array([2**62, 2**62, 2**62, -5], dtype=np.int64)
    arrays["wrap"] = wrap
    with np.errstate(all="ignore"):
        results["cumsum_i64.wrap"] = np.cumsum(wrap)
    return [_save(out_dir, "extrema", arrays, results)]


def export_order(out_dir: str) -> list[dict[str, Any]]:
    """Sorts, searches, order statistics and unique, as numpy orders values (NaN last)."""
    import warnings

    rng = np.random.default_rng(31337)
    sp = _specials_f32()
    arrays: dict[str, np.ndarray] = {}
    results: dict[str, np.ndarray] = {}
    # sort (values, with signed zeros mixed in so their arrangement shows)
    for n in (1, 2, 15, 16, 17, 100, 1000, 65_537):
        a = ((rng.random(n, dtype=np.float32) - np.float32(0.5)) * np.float32(10)).astype(np.float32)
        a[rng.random(n) < 0.05] = np.nan
        a[rng.random(n) < 0.05] = 0.0
        a[rng.random(n) < 0.05] = -0.0
        a[rng.random(n) < 0.1] = np.float32(1.5)  # ties
        arrays[f"sort{n}"] = a
        results[f"sort.sort{n}"] = np.sort(a)
    arrays["sortspecials"] = np.tile(sp, 9)
    results["sort.sortspecials"] = np.sort(arrays["sortspecials"])
    # searchsorted on float64, NaN last
    hay = np.sort(np.concatenate([rng.random(200), [0.25, 0.25, 0.25, np.nan, np.nan]]))
    needles = np.concatenate([rng.random(500) * 1.2 - 0.1, [0.25, np.nan, -np.inf, np.inf, 0.0, 1.0]])
    arrays["hay"], arrays["needles"] = hay, needles
    results["searchsorted_left"] = np.searchsorted(hay, needles, side="left").astype(np.int64)
    results["searchsorted_right"] = np.searchsorted(hay, needles, side="right").astype(np.int64)
    # lexsort: keys with many ties; the last key is the primary one
    k = 3000
    keys = [rng.integers(0, 4, k).astype(np.float32) for _ in range(4)]
    keys[1][rng.random(k) < 0.02] = np.nan
    for i, key in enumerate(keys):
        arrays[f"lexkey{i}"] = key
    results["lexsort"] = np.lexsort(keys).astype(np.int64)
    # quantile (linear, Python-float q), median, nanpercentile (list q)
    qs = [0.0, 0.01, 0.1, 0.25, 0.3333333333333333, 0.5, 0.7, 0.9, 0.99, 1.0]
    arrays["qs"] = np.array(qs)
    for n in (1, 2, 3, 4, 5, 10, 101, 1000, 4096):
        a = ((rng.random(n, dtype=np.float32)) * np.float32(7) - np.float32(2)).astype(np.float32)
        arrays[f"stat{n}"] = a
        results[f"quantile.stat{n}"] = np.array([np.quantile(a, q) for q in qs], dtype=np.float32)
        results[f"median.stat{n}"] = np.asarray(np.median(a))
        b = a.copy()
        b[rng.random(n) < 0.2] = np.nan
        arrays[f"nanstat{n}"] = b
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j, (q1, q2) in enumerate([(2.0, 98.0), (0.0, 100.0), (1.0, 99.0), (0.5, 99.5), (25.0, 75.0)]):
                results[f"nanpercentile{j}.nanstat{n}"] = np.nanpercentile(b, [q1, q2])
    arrays["pct_pairs"] = np.array([(2.0, 98.0), (0.0, 100.0), (1.0, 99.0), (0.5, 99.5), (25.0, 75.0)])
    arrays["nanstat_allnan"] = np.full(5, np.nan, np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results["nanpercentile0.nanstat_allnan"] = np.nanpercentile(arrays["nanstat_allnan"], [2.0, 98.0])
    # unique with its inverse, int64
    for name, u in {
        "labels": rng.integers(0, 50_000, 200_000),
        "sparse": rng.integers(-(2**40), 2**40, 5000),
        "few": np.array([5, 5, 5, -1, 3, 3, 9]),
        "one": np.array([7]),
    }.items():
        u = u.astype(np.int64)
        arrays[f"uniq_{name}"] = u
        values, inverse = np.unique(u, return_inverse=True)
        results[f"unique.{name}"] = values.astype(np.int64)
        results[f"unique_inverse.{name}"] = inverse.astype(np.int64).ravel()
    return [_save(out_dir, "order", arrays, results)]


def export_fft(out_dir: str) -> list[dict[str, Any]]:
    """np.fft.fftshift and ifftshift of 2-D planes of odd and even sizes."""
    arrays: dict[str, np.ndarray] = {}
    results: dict[str, np.ndarray] = {}
    for h, w in ((4, 6), (5, 7), (1, 1), (1, 8), (9, 1), (6, 5), (33, 64)):
        a = np.arange(h * w, dtype=np.float32).reshape(h, w)
        arrays[f"p{h}x{w}"] = a
        results[f"fftshift.p{h}x{w}"] = np.fft.fftshift(a)
        results[f"ifftshift.p{h}x{w}"] = np.fft.ifftshift(a)
    return [_save(out_dir, "fft", arrays, results)]


def _cpu_features() -> dict[str, bool]:
    """The CPU features numpy dispatches its SIMD loops on (the ones that change results)."""
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as feat
    except ImportError:  # numpy 1.x
        from numpy.core._multiarray_umath import __cpu_features__ as feat
    return {k: bool(feat.get(k, False)) for k in ("ASIMD", "FMA3", "AVX2", "AVX512F")}


def _parts(name: str, z: np.ndarray) -> dict[str, np.ndarray]:
    z = np.asarray(z, dtype=np.complex64)
    return {f"{name}.re": np.ascontiguousarray(z.real), f"{name}.im": np.ascontiguousarray(z.imag)}


def export_complex(out_dir: str) -> list[dict[str, Any]]:
    """complex64 ``a * b`` (and ``z * z``) through numpy's ufunc loop: every pair of complex
    specials, long random arrays with NaN, infinities and signed zeros, every length up to
    33 (SIMD body and tail), a strided view, and an array times a scalar. Then the same
    products as numpy scalars, which go through numpy's compiled C rather than the loop
    and round differently on some hosts. Stored as separate real and imaginary float32
    parts; the case also names the CPU features numpy dispatched on."""
    import warnings

    rng = np.random.default_rng(6464)
    # 11 specials as real and imaginary parts: 121 values, 14,641 pairs
    sp = np.array([0.0, -0.0, 1.0, -1.0, 0.5, 1e-45, 3.4e38, -3.4e38, np.inf, -np.inf, np.nan], dtype=np.float32)
    re, im = np.meshgrid(sp, sp, indexing="ij")
    specials = (re.ravel() + 1j * im.ravel()).astype(np.complex64)
    pa, pb = np.meshgrid(specials, specials, indexing="ij")
    pairs_a, pairs_b = pa.ravel().copy(), pb.ravel().copy()

    def rand(n: int) -> np.ndarray:
        z = (rng.standard_normal(n) * 50 + 1j * rng.standard_normal(n) * 50).astype(np.complex64)
        idx = rng.integers(0, n, 120)
        z.real[idx[:20]] = np.nan
        z.imag[idx[20:40]] = np.nan
        z.real[idx[40:60]] = np.inf
        z.imag[idx[60:80]] = -np.inf
        z.real[idx[80:100]] = -0.0
        z.imag[idx[100:]] = 0.0
        return z

    rand_a, rand_b = rand(10_007), rand(10_007)
    arrays: dict[str, np.ndarray] = {}
    for name, z in (("pairs_a", pairs_a), ("pairs_b", pairs_b), ("rand_a", rand_a), ("rand_b", rand_b)):
        arrays.update(_parts(name, z))
    results: dict[str, np.ndarray] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results.update(_parts("mul.pairs", pairs_a * pairs_b))
        results.update(_parts("mul.rand", rand_a * rand_b))
        results.update(_parts("square.rand", rand_a * rand_a))
        for n in range(1, 34):
            results.update(_parts(f"mul.len{n}", rand_a[:n] * rand_b[:n]))
        results.update(_parts("mul.stride3", rand_a[::3] * rand_b[::3]))
        results.update(_parts("mul.by_scalar", rand_a * rand_b[0]))
        scal = [rand_a[i] * rand_b[i] for i in range(2_000)] + [pairs_a[i] * pairs_b[i] for i in range(0, pairs_a.size, 7)]
        results.update(_parts("mul.scalars", np.array(scal, dtype=np.complex64)))
    case = _save(out_dir, "complex", arrays, results)
    case["cpu_features"] = _cpu_features()
    return [case]


SECTIONS = {
    "random": export_random,
    "simd_math": export_simd_math,
    "reduce": export_reduce,
    "hist": export_hist,
    "elementwise": export_elementwise,
    "extrema": export_extrema,
    "order": export_order,
    "fft": export_fft,
    "complex": export_complex,
}


def export_host_fixtures(out_root: str, sections: list[str] | None = None, log=print) -> str:
    """Write ``out_root/<host slug>/index.json`` and the cases' ``.npz`` files."""
    slug = host_slug()
    out_dir = os.path.join(out_root, slug)
    os.makedirs(out_dir, exist_ok=True)
    index: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "host": slug,
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu": cpu_brand(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "fma": fma_contraction(),
        "libm": libm_agreement(),
        "cases": [],
    }
    for section in sections or list(SECTIONS):
        cases = SECTIONS[section](out_dir)
        log(f"host-fixtures[{section}]: {len(cases)} cases")
        index["cases"].extend(cases)
    path = os.path.join(out_dir, "index.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(index, f, indent=1)
    return path

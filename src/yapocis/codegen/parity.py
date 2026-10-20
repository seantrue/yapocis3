#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.
"""
Parity fixtures: kernel outputs from the Python runtime for other runtimes to match.

For every bundled kernel that takes only ``in`` arrays, ``out_like`` outputs,
derived dimensions and scalars, build deterministic inputs, run the kernel on
one backend, and save inputs and outputs as ``.npz`` next to an ``index.json``::

    <out>/parity/<backend>/<device>/index.json
    <out>/parity/<backend>/<device>/<interface id>.<kernel>.npz   (in_0, in_1, ..., out_0, ...)

``<device>`` is :func:`device_slug` of the device the backend opens: results
can differ between devices in the last bit (fast-math, fused multiply-add), so
a consumer compares only against fixtures recorded on its own device.

Kernels whose output is not reproducible run-to-run (out-of-bounds reads with
the synthetic shapes, uninitialised memory) are skipped, as are interfaces in
``SKIP``. A consumer replays each case through its own runtime and compares
against the saved outputs with ``allclose(rtol, atol)``.

Scalars take the values in ``SCALARS`` (by parameter name, else by dtype).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from typing import Any

import numpy as np

from yapocis.core import load_program
from yapocis.core.interface import get_interface

from .bundle import Manifest, load_bundle_config

# Interfaces not covered by the automatic generator: packed/structured inputs
# (graphics), models (cnn, vgg16), multi-pass host logic (truetrace), FFT (needs
# power-of-two sizes and explicit local sizes; the FFT library port has its own tests).
SKIP = {"cnn", "vgg16", "graphics", "graphics4", "truetrace", "fft"}

# Individual kernels the Python library layer never calls because their device
# layout is not portable: float3 is 16 bytes on OpenCL/Metal and 12 on CUDA, so
# the 3-plane anisodiff kernel reads past the buffer (lib/anisodiff.py pads to 4).
SKIP_KERNELS = {("anisodiff", "filterImage3")}


def skipped(interface: str, kernel: str) -> bool:
    if interface in SKIP or (interface, kernel) in SKIP_KERNELS:
        return True
    # convolve's `<name>_fast` kernels process `chunk` elements per thread and
    # expect a global size of size/chunk; with the default global size they
    # write far past the output, so their results are not comparable.
    return interface.split(".")[0] == "convolve" and kernel.endswith("_fast")


# Synthetic image shape for array inputs (yapocis convention: shape[0]=width, shape[1]=height)
SHAPE = (48, 40)
DEPTH_BY_INTERFACE = {"hsi": 3, "colormath": 3}
KERNEL_SHAPES = {
    # (interface, kernel) -> shape for array inputs
    ("hsi", "rgb2hsi3"): (48, 40, 3),
    ("hsi", "hsi2rgb3"): (48, 40, 3),
    ("colormath", "transform"): (48, 40, 3),
    ("colormath", "xyz2xyY"): (48, 40, 3),
    ("colormath", "xyY2xyz"): (48, 40, 3),
    ("zcs", "zcs2"): (48, 40, 2),
    ("zcs", "zcs3"): (48, 40, 3),
    ("zcs", "zcs4"): (48, 40, 4),
    ("anisodiff", "filterImage2"): (48, 40, 2),
    ("anisodiff", "filterImage3"): (48, 40, 3),
    ("anisodiff", "filterImage4"): (48, 40, 4),
    ("mandelbrot", "mandelbrot"): (64 * 64,),
    ("audio", None): (4096,),
}
# Explicit global sizes where the Python library layer passes one
GLOBAL_BY_DEPTH = {"hsi", "colormath", "zcs", "anisodiff"}

SCALARS: dict[str, Any] = {
    "maxiter": 40,
    "reach": 3,
    "stride": 3,
    "threshold": 0.01,
    "scale": 3.0,
    "size_step": 0.2,
    "step_size": 0.2,
    "transform_id": 28,  # srgb_d65.rgb2xyz
    "vertical": 1,
    "low": 0.2,
    "high": 0.8,
    "input_min": 0.0,
    "input_max": 1.0,
    "chunk": 64,
    "iterations": 2,
    "float32": 0.5,
    "int32": 3,
}

RTOL = 1e-5
ATOL = 1e-5


def _scalar_for(param) -> Any:
    if param.name in SCALARS:
        return SCALARS[param.name]
    return SCALARS.get(param.np_dtype, 1)


def _array_for(
    rng: np.random.Generator, dtype: str, shape: tuple[int, ...]
) -> np.ndarray:
    if dtype == "complex64":
        re = rng.uniform(-2.0, 1.0, size=shape).astype(np.float32)
        im = rng.uniform(-1.5, 1.5, size=shape).astype(np.float32)
        return (re + 1j * im).astype(np.complex64)
    if dtype.startswith("int") or dtype.startswith("uint"):
        return rng.integers(0, 100, size=shape).astype(dtype)
    return rng.uniform(0.0, 1.0, size=shape).astype(dtype)


def _shape_for(iface: str, kernel: str) -> tuple[int, ...]:
    if (iface, kernel) in KERNEL_SHAPES:
        return KERNEL_SHAPES[(iface, kernel)]
    if (iface, None) in KERNEL_SHAPES:
        return KERNEL_SHAPES[(iface, None)]
    if iface in DEPTH_BY_INTERFACE:
        return (*SHAPE, DEPTH_BY_INTERFACE[iface])
    return SHAPE


def eligible(kernel) -> bool:
    for p in kernel.parameters:
        if p.derived or not p.hint:
            continue
        if p.hint != "in":
            return False
    return any(p.hint == "out_like" for p in kernel.parameters)


def build_case(iface_name: str, kernel, seed: int) -> tuple[list[Any], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    shape = _shape_for(iface_name, kernel.name)
    args: list[Any] = []
    for p in kernel.parameters:
        if p.derived:
            continue
        if p.hint:
            args.append(_array_for(rng, p.np_dtype, shape))
        else:
            args.append(_scalar_for(p))
    kwargs: dict[str, Any] = {}
    if iface_name in GLOBAL_BY_DEPTH and len(shape) == 3:
        kwargs["global_size"] = int(np.prod(shape[:2]))
    return args, kwargs


def _as_tuple(result) -> tuple[np.ndarray, ...]:
    if result is None:
        return ()
    if isinstance(result, tuple):
        return result
    return (result,)


def _reproducible(a: tuple[np.ndarray, ...], b: tuple[np.ndarray, ...]) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if x.shape != y.shape or x.dtype != y.dtype:
            return False
        if not np.array_equal(x, y, equal_nan=True):
            return False
    return True


def device_name(backend: str) -> str:
    """Name of the device ``backend`` opens by default (honours ``YAPOCIS_DEVICE``)."""
    program = load_program("demo", backend=backend)
    device = getattr(getattr(program, "dmgr", None), "device", None)
    name = getattr(device, "name", device)
    # pycuda's Device.name is a method; pyopencl's and pymetallic's are attributes.
    return str(name() if callable(name) else name)


def device_slug(name: str) -> str:
    """Directory name for a device: ``"Apple M4 Max"`` -> ``"apple-m4-max"``."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def export_parity(
    out_dir: str,
    manifest: Manifest,
    backend: str,
    names: list[str] | None = None,
    seed: int = 20261018,
    log: Callable[..., None] = print,
) -> str:
    """Run every eligible kernel on ``backend`` and write the parity fixtures."""
    config = load_bundle_config()
    device = device_name(backend)
    parity_dir = os.path.join(out_dir, "parity", backend, device_slug(device))
    os.makedirs(parity_dir, exist_ok=True)
    for stale in os.listdir(parity_dir):
        if stale.endswith(".npz"):
            os.remove(os.path.join(parity_dir, stale))
    index: dict[str, Any] = {
        "backend": backend,
        "device": device,
        "seed": seed,
        "rtol": RTOL,
        "atol": ATOL,
        "cases": [],
    }
    for entry in manifest.interfaces:
        if entry.name in SKIP or (names and entry.name not in names):
            continue
        context = config.variants.get(entry.name, {}).get(entry.variant, {})
        try:
            program = load_program(entry.name, backend=backend, **context)
        except Exception as e:  # noqa: BLE001
            log(f"{entry.id}: SKIP load: {type(e).__name__}: {e}")
            continue
        spec = get_interface(entry.interface_text)
        for kernel in spec.kernels:
            if not eligible(kernel) or skipped(entry.name, kernel.name):
                continue
            args, kwargs = build_case(entry.name, kernel, seed)
            callable_ = program.callable[kernel.name]
            try:
                first = _as_tuple(
                    callable_(
                        *[a.copy() if isinstance(a, np.ndarray) else a for a in args],
                        **kwargs,
                    )
                )
                second = _as_tuple(
                    callable_(
                        *[a.copy() if isinstance(a, np.ndarray) else a for a in args],
                        **kwargs,
                    )
                )
            except Exception as e:  # noqa: BLE001
                log(f"{entry.id}.{kernel.name}: SKIP run: {type(e).__name__}: {e}")
                continue
            if not _reproducible(first, second):
                log(f"{entry.id}.{kernel.name}: SKIP not reproducible")
                continue
            if any(
                np.isnan(o).any() for o in first if np.issubdtype(o.dtype, np.floating)
            ):
                log(f"{entry.id}.{kernel.name}: SKIP NaN output")
                continue
            fname = f"{entry.id}.{kernel.name}.npz"
            arrays: dict[str, np.ndarray] = {}
            case_args = []
            for i, a in enumerate(args):
                if isinstance(a, np.ndarray):
                    arrays[f"in_{i}"] = a
                    case_args.append({"array": f"in_{i}"})
                else:
                    case_args.append({"scalar": a})
            for i, o in enumerate(first):
                arrays[f"out_{i}"] = o
            np.savez_compressed(os.path.join(parity_dir, fname), **arrays)  # type: ignore[arg-type]
            index["cases"].append(
                {
                    "interface": entry.id,
                    "kernel": kernel.name,
                    "file": fname,
                    "args": case_args,
                    "outputs": [f"out_{i}" for i in range(len(first))],
                    "global_size": kwargs.get("global_size"),
                }
            )
            log(f"{entry.id}.{kernel.name}: {len(first)} output(s)")
        try:
            program.purge()
        except Exception:  # noqa: BLE001
            pass
    path = os.path.join(parity_dir, "index.json")
    with open(path, "w") as f:
        json.dump(index, f, indent=1)
        f.write("\n")
    return path

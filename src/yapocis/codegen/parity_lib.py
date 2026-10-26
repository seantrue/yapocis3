#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.
"""
Library-level parity fixtures: outputs of the Python library wrappers for other
runtimes to match.

Kernel-level parity (:mod:`yapocis.codegen.parity`) covers kernels with plain
array inputs. The library wrappers add host logic (display-list packing, text
rasterisation, multi-pass pipelines, CNN orchestration) that a port must also
reproduce, so each library area registers a generator here that runs the
Python wrapper on a backend and saves inputs and outputs::

    <out>/parity/lib/<backend>/<device>/index.json
    <out>/parity/lib/<backend>/<device>/<generator>.<case>.npz

``<device>`` is :func:`yapocis.codegen.parity.device_slug` of the device the
backend opens, as for kernel-level fixtures.

A generator is ``fn(out_dir: str, backend: str, log) -> list[dict]`` returning
one record per case with at least ``name`` and ``file``; the record may carry any
extra JSON-serialisable metadata the consumer needs (shapes, scalar parameters,
tolerances). Arrays go into the ``.npz`` under keys the record names.

Register with ``@generator("name")``. Generator modules are imported at the
bottom of this file so that importing :mod:`yapocis.codegen.parity_lib` registers
all of them.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

Generator = Callable[[str, str, Callable[..., None]], list[dict[str, Any]]]

GENERATORS: dict[str, Generator] = {}

RTOL = 1e-4
ATOL = 1e-4


def generator(name: str) -> Callable[[Generator], Generator]:
    """Register a library parity generator under ``name``."""

    def register(fn: Generator) -> Generator:
        GENERATORS[name] = fn
        return fn

    return register


def parity_lib_dir(out_dir: str, backend: str, device: str) -> str:
    from yapocis.codegen.parity import device_slug

    path = os.path.join(out_dir, "parity", "lib", backend, device_slug(device))
    os.makedirs(path, exist_ok=True)
    return path


def export_parity_lib(
    out_dir: str,
    backend: str,
    names: list[str] | None = None,
    log: Callable[..., None] = print,
) -> str:
    """Run the registered generators on ``backend`` and write the index."""
    from yapocis.codegen.parity import device_name

    os.environ["YAPOCIS_BACKEND"] = backend
    device = device_name(backend)
    target = parity_lib_dir(out_dir, backend, device)
    index: dict[str, Any] = {
        "backend": backend,
        "device": device,
        "rtol": RTOL,
        "atol": ATOL,
        "generators": {},
    }
    # A run restricted to some generators keeps the others' entries, so areas
    # can be regenerated independently.
    path = os.path.join(target, "index.json")
    if names and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                previous = json.load(f)
            index["generators"] = {
                k: v
                for k, v in previous.get("generators", {}).items()
                if k not in names
            }
        except (OSError, ValueError):
            pass
    for name, fn in GENERATORS.items():
        if names and name not in names:
            continue
        for stale in os.listdir(target):
            if stale.startswith(f"{name}.") and stale.endswith(".npz"):
                os.remove(os.path.join(target, stale))
        try:
            cases = fn(target, backend, log)
        except Exception as e:  # noqa: BLE001
            log(f"parity-lib {name}: SKIP {type(e).__name__}: {e}")
            continue
        index["generators"][name] = cases
        log(f"parity-lib {name}: {len(cases)} case(s)")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(index, f, indent=1)
        f.write("\n")
    return path


# Generator modules register themselves on import. Add one line per area.
# fmt: off
# isort: off
from . import parity_convolve  # noqa: E402,F401  # Convolve1D / Gaussian runtime weights
from . import parity_cnn  # noqa: E402,F401  # cnn kernels, vgg16, resnet50
from . import parity_processing  # noqa: E402,F401  # truetrace, imageops, simplex, anisodiff, fft
from . import parity_graphics  # noqa: E402,F401  # graphics, graphics4, kerneldraw, typeset, paths
# isort: on
# fmt: on

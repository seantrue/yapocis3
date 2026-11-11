"""Kernel manager for the ``rustcpu`` backend (yapocis-rs's CPU backend).

An interface is rendered here exactly as ``yapocis-tools bundle`` renders it
(its manifest entry and OpenCL source, see ``codegen.bundle``), and loaded on
yapocis-rs's CPU device, which runs a Rust port of each OpenCL kernel. The Rust
runtime derives dimensions, ``out_like`` outputs and global sizes, as it does
for its own callers, so this module only converts arguments and packages
results.

As on numbax, a NumPy array is its own device buffer: ``write``/``read`` are
no-ops, ``resident`` arrays are used in place, and every call runs to
completion before it returns (``async_exec`` is accepted and returns None).
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any

import numpy as np

from yapocis.codegen.bundle import build_interface_entry

# Keyword arguments that configure the manager, not the interface's templates.
_MANAGER_KWARGS = ("device_name", "debug", "options", "async_write")

_device = None


def _key_default(o: object) -> object:
    """JSON for the program-cache key: arrays and numpy scalars exactly, with dtype and shape.

    ``repr`` keeps only numpy's print precision (8 decimal places), elides arrays over
    1000 elements and follows ``np.set_printoptions``, so different weights shared a key.
    """
    if isinstance(o, np.ndarray):
        return ["ndarray", str(o.dtype), list(o.shape), o.tolist()]
    if isinstance(o, np.generic):
        return [type(o).__name__, o.item()]
    return repr(o)


def _engine():
    """The `yapocis_rs` module, or an ImportError saying how to install it."""
    try:
        import yapocis_rs
    except ImportError as e:
        raise ImportError(
            "the rustcpu backend needs yapocis-rs's Python bindings (module yapocis_rs), "
            "which are not installed: build the wheel from yapocis-rs's bindings/python and "
            "install it with `uv pip install --no-deps` (see docs/quickstart/backends.md), "
            "or choose another backend"
        ) from e
    return yapocis_rs


def _get_device(device_name=None):
    """The CPU device (there is one); `device_name` may be None, 0, "cpu" or "rustcpu:cpu"."""
    global _device
    yapocis_rs = _engine()

    if device_name not in (None, 0, "0", "cpu", "rustcpu:cpu"):
        # Let yapocis-rs report what it accepts.
        return yapocis_rs.Device(str(device_name))
    if _device is None:
        _device = yapocis_rs.Device()
    return _device


def _sizes(size) -> list[int] | None:
    """A global or local size as a list of axis lengths (an int is one axis)."""
    if size is None:
        return None
    if isinstance(size, (int, np.integer)):
        return [int(size)]
    return [int(s) for s in size]


class Kernel:
    """One kernel of a loaded interface, called like the GPU backends' kernels.

    Positional arguments are the kernel's user arguments. Keyword arguments
    ``global_size`` and ``local_size`` are passed to the launch; ``async_exec``,
    ``resident`` and ``wait`` are accepted for compatibility and change nothing.
    Returns the ``inout``, ``out`` and ``out_like`` arrays in declaration order:
    one array, a tuple, or None.
    """

    def __init__(self, program, entry: dict[str, Any], kernel_manager: KernelManager):
        self.program = program
        self.name = entry["name"]
        self.kernel_manager = kernel_manager
        # (name, kind, role, dtype) of each user argument, in call order.
        self.specs = []
        for user_arg in entry["user_args"]:
            p = next(
                p
                for p in entry["params"]
                if p["name"] == user_arg and p["kind"] in ("buffer", "scalar")
            )
            self.specs.append((p["name"], p["kind"], p["role"], np.dtype(p["dtype"])))

    def write(self, a, **kwargs):
        return a

    def read(self, a, **kwargs):
        return a

    def ensure(self, a, **kwargs):
        return a

    def __call__(self, *args, **kwargs):
        km = self.kernel_manager
        km.calls += 1
        t = time.time()
        global_size = _sizes(kwargs.pop("global_size", None))
        local_size = _sizes(kwargs.pop("local_size", None))
        async_exec = kwargs.pop("async_exec", km._default_async)
        kwargs.pop("resident", None)
        kwargs.pop("wait", None)
        if kwargs:
            raise TypeError(
                f"{self.name}: unexpected keyword arguments {sorted(kwargs)}"
            )
        if len(args) != len(self.specs):
            names = ", ".join(s[0] for s in self.specs)
            raise TypeError(
                f"{self.name} takes {len(self.specs)} arguments ({names}), got {len(args)}"
            )

        prepared = []
        by_name = {}
        copy_back = []  # (caller's array, converted copy the kernel writes)
        for arg, (name, kind, role, dtype) in zip(args, self.specs, strict=True):
            if kind == "scalar":
                prepared.append(float(arg))
                continue
            if role == "in":
                a = np.ascontiguousarray(arg, dtype=dtype)
            elif (
                isinstance(arg, np.ndarray)
                and arg.dtype == dtype
                and arg.flags.c_contiguous
                and arg.flags.writeable
            ):
                a = arg
            else:
                # Like the GPU backends' `astype(dtype, copy=False)`: the kernel
                # writes a converted copy; copy it back when the caller's array
                # can take it.
                a = np.ascontiguousarray(arg, dtype=dtype).copy()
                if isinstance(arg, np.ndarray) and arg.flags.writeable:
                    copy_back.append((arg, a))
            prepared.append(a)
            by_name[name] = a

        outputs = self.program.call(
            self.name, prepared, global_size=global_size, local_size=local_size
        )
        for original, converted in copy_back:
            np.copyto(original, converted, casting="unsafe")
        km.runtime += time.time() - t
        if async_exec:
            return None
        rvals = [
            array if array is not None else by_name[param] for param, array in outputs
        ]
        if not rvals:
            return None
        return rvals[0] if len(rvals) == 1 else tuple(rvals)


class KernelManager:
    """One interface loaded on yapocis-rs's CPU device; kernels are attributes."""

    # Loaded programs, by (interface, rendered context): rendering is the slow part.
    _programs: dict[tuple[str, str], tuple[Any, dict[str, Any], str]] = {}

    def __init__(
        self, interface_spec: str, device_name=None, debug: bool = False, **context
    ):
        template_context = {
            k: v for k, v in context.items() if k not in _MANAGER_KWARGS
        }
        key = (
            interface_spec,
            json.dumps(template_context, sort_keys=True, default=_key_default),
        )
        if key not in self._programs:
            entry, sources = build_interface_entry(
                interface_spec, "default", template_context, ("ocl",)
            )
            entry_dict = asdict(entry)
            device = _get_device(device_name)
            program = device.load(json.dumps(entry_dict), sources["ocl"])
            self._programs[key] = (program, entry_dict, sources["ocl"])
        program, entry, source = self._programs[key]
        self.program = program
        self.entry = entry
        self.source = source
        self.interface = entry["interface_text"]
        self.calls = 0
        self.runtime = 0.0
        self.snapshots = []
        self.pending_events = []
        self._default_async = False
        self.callable = {k["name"]: Kernel(program, k, self) for k in entry["kernels"]}

    def __getattr__(self, attr: str):
        callable_ = self.__dict__.get("callable", {})
        if attr not in callable_:
            raise AttributeError(attr)
        return callable_[attr]

    # NumPy arrays are the device buffers.
    def read(self, a, **kwargs):
        return a

    def write(self, a, **kwargs):
        return a

    def ensure(self, a, **kwargs):
        return a

    def purge(self):
        return None

    def reset(self):
        self.calls = 0
        self.runtime = 0.0

    def snapshot(self, **kwargs):
        snap = {
            "interfacename": self.entry["name"],
            "calls": self.calls,
            "runtime": self.runtime,
            "device": "cpu",
            "backend": "rustcpu",
        }
        snap.update(kwargs)
        self.snapshots.append(snap)
        return snap

    def wait_all(self):
        """Every call has finished when it returns; nothing to wait for."""
        self.pending_events.clear()

    @contextmanager
    def async_mode(self):
        """Accepted for compatibility: calls still run to completion."""
        old = self._default_async
        self._default_async = True
        try:
            yield self
        finally:
            self.wait_all()
            self._default_async = old

    @classmethod
    def load_program(
        cls, interface_spec: str, device_name=None, debug: bool = False, **context
    ):
        return cls(interface_spec, device_name=device_name, debug=debug, **context)

    @classmethod
    def import_program(
        cls,
        interface_spec: str,
        device_name=None,
        namespace: bool = False,
        debug: bool = False,
        import_kernels=None,
        **context,
    ):
        program = cls.load_program(
            interface_spec, device_name=device_name, debug=debug, **context
        )
        import inspect

        target = inspect.getmodule(inspect.currentframe().f_back.f_code)
        for name, kernel in program.callable.items():
            if import_kernels == "*" or name in context:
                setattr(target, context.get(name, name), kernel)
        if namespace:
            return program, program.callable
        return program, list(program.callable.values())


import_program = KernelManager.import_program
load_program = KernelManager.load_program

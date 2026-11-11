from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from collections.abc import Callable
from contextlib import contextmanager

from mako.template import Template

from yapocis.core.interface import Interface, get_interface, load_interface
from yapocis.numbax.kernel import Kernel
from yapocis.utils.internals import get_yapocis_dir, get_yapocis_dirs


def _load_numba_kernel_module(interface_name: str):
    """
    Load a Numba kernel module, searching YAPOCIS_PROJECT directories first.

    Search order:
    1. YAPOCIS_PROJECT/kernels/numbax/{interface_name}.py
    2. Current working directory/kernels/numbax/{interface_name}.py
    3. yapocis.kernels.numbax.{interface_name} (standard import)

    Args:
        interface_name: Name of the interface (e.g., "mandelbrot")

    Returns:
        Loaded module or None if not found

    Raises:
        ModuleNotFoundError: If module not found in any location
    """
    # First, search in yapocis directories (project, cwd); yapocis's own kernels are
    # imported by their package name below.
    package_dir = os.path.realpath(get_yapocis_dir())
    for base_dir in get_yapocis_dirs():
        if os.path.realpath(base_dir) == package_dir:
            continue
        kernel_path = os.path.join(
            base_dir, "kernels", "numbax", f"{interface_name}.py"
        )
        if os.path.exists(kernel_path):
            # Load module from file path
            spec = importlib.util.spec_from_file_location(
                f"yapocis_ext.kernels.numbax.{interface_name}", kernel_path
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                return module

    # Fallback to standard import
    mod_name = f"yapocis.kernels.numbax.{interface_name}"
    module = importlib.import_module(mod_name)
    # Until October 2026 these were loaded from their file as yapocis_ext.kernels.numbax.*,
    # and numba's on-disk caches (cache=True) pickled that module name. Keep the alias so
    # those caches still load instead of raising ModuleNotFoundError.
    sys.modules.setdefault(f"yapocis_ext.kernels.numbax.{interface_name}", module)
    return module


_DYNAMIC_INTERFACES = frozenset({"convolve", "convolvesep"})


class KernelManager:
    """KernelManager for Numba backend.

    Binds interface kernels to Python/Numba functions in
    `yapocis.numbax.kernels.<interface>` module.
    """

    def __init__(
        self,
        interface: Interface,
        device_name: str | None = None,
        debug: bool = False,
        **context,
    ):
        self.interface = interface
        self.callable = {}
        self.calls = 0
        self.runtime = 0.0
        self.context = context
        self.snapshots = []
        # Async execution support (no-op for CPU backend)
        self.pending_events = []
        self._default_async = False
        # Load module with Python implementations (searches YAPOCIS_PROJECT first)
        try:
            mod = _load_numba_kernel_module(interface.name)
        except ModuleNotFoundError:
            # Defer implementation errors to call time; allow smoke-compile to pass
            self.module_missing = True  # type: ignore[attr-defined]
            self.missing_module_name = f"yapocis.kernels.numbax.{interface.name}"  # type: ignore[attr-defined]
            return
        # Wire up kernels. Kernels generated from the context (convolve, convolvesep) are
        # built for this manager only and never stored on the shared module (#64).
        built: dict[str, dict[str, Callable]] = {}
        mod_name = mod.__name__
        for k in interface.kernels:
            name = k.alias if k.alias else k.name
            func_name = name
            if interface.name in _DYNAMIC_INTERFACES:
                fn = self._make_dynamic_fn(interface.name, k, built)
            elif hasattr(mod, func_name):
                fn = getattr(mod, func_name)
            else:
                fn = None
            if fn is None:
                # Provide a lazy stub that raises when called, so compile-only paths work
                def _stub(*args, _f=func_name, **kwargs):
                    raise NotImplementedError(
                        f"Numba backend: function '{_f}' not implemented in {mod_name}"
                    )

                fn = _stub
            self.callable[k.name] = Kernel(name, fn, k.parameters, kernel_manager=self)

    def _make_dynamic_fn(self, iface_name: str, kernel, built: dict):
        """This manager's function for a kernel generated from the context.

        Functions are built once per base name into ``built`` (one manager's memo),
        never on the shared kernel module.
        """
        name = kernel.alias or kernel.name
        ctx = self.context
        if iface_name == "convolve":
            from yapocis.kernels.numbax.convolve import make_convolve_functions

            base = name.removesuffix("_fast")
            if base not in built:
                kmain, kfast = make_convolve_functions(base, ctx.get("conv"))
                built[base] = {base: kmain, base + "_fast": kfast}
            return built[base].get(name)
        if iface_name == "convolvesep":
            from yapocis.kernels.numbax.convolvesep import make_sep_functions

            base = name
            # Normalize suffixes to the base kernel name
            if base.endswith(("_h", "_v", "_res")):
                base = base.rsplit("_", 1)[0]
            if base not in built:
                convs = dict(ctx.get("convs", []))
                km, kh, kv = make_sep_functions(base, convs)
                built[base] = {base: km, base + "_h": kh, base + "_v": kv}
            return built[base].get(name)
        return None

    def __getattr__(self, attr: str) -> Callable:
        if attr not in self.callable:
            # Behave like normal attribute resolution so getattr(..., default) works
            raise AttributeError(attr)
        return self.callable[attr]

    # Optional lifecycle helpers to mirror other backends
    def purge(self):  # no-op for in-memory backend
        return None

    def reset(self):
        self.calls = 0
        self.runtime = 0.0
        return None

    def snapshot(self, **kwargs):
        """Record a lightweight snapshot for test parity.

        Mirrors the API of other backends so tests can call snapshot()
        without special-casing the numba backend.
        """
        snap = {
            "interfacename": self.interface.name,
            "calls": self.calls,
            "runtime": self.runtime,
            "device": "cpu",
            "backend": "numbax",
        }
        snap.update(kwargs)
        self.snapshots.append(snap)
        return snap

    @classmethod
    def load_program(
        cls,
        interface_spec: str,
        device_name: str | None = None,
        debug: bool = False,
        **context,
    ):
        # Load and merge YAML context with programmatic context
        from yapocis.core.interface_context import get_interface_context

        merged_context = get_interface_context(interface_spec, **context)

        # Load and render interface (expand Mako templates)
        source = load_interface(interface_spec)
        if merged_context:
            try:
                source = str(Template(source).render(**merged_context))
            except Exception:
                # If rendering fails, keep original source
                pass
        interface = get_interface(source)
        km = KernelManager(interface, device_name, debug=debug, **context)
        # Expose the rendered interface source for tests and tooling
        km.source = source
        return km

    def read(self, a, **kwargs):
        return a

    def write(self, a, **kwargs):
        return a

    def wait_all(self):
        """
        Wait for all pending async operations to complete.

        No-op for Numba backend since execution is synchronous on CPU.
        Provided for API compatibility with GPU backends.
        """
        # CPU backend executes synchronously - nothing to wait for
        self.pending_events.clear()

    @contextmanager
    def async_mode(self):
        """
        Context manager for async kernel execution.

        No-op for Numba backend since CPU execution is inherently synchronous.
        Provided for API compatibility with GPU backends.

        Usage:
            with prog.async_mode():
                prog.kernel1(...)  # Still executes synchronously
                prog.kernel2(...)  # Still executes synchronously

        Yields:
            self: The kernel manager instance
        """
        old_default = self._default_async
        self._default_async = True
        try:
            yield self
        finally:
            self.wait_all()
            self._default_async = old_default

    @classmethod
    def import_program(
        cls,
        interface_spec: str,
        device_name: str | None = None,
        namespace: bool = False,
        debug: bool = False,
        import_kernels=None,
        **context,
    ):
        program = cls.load_program(
            interface_spec, device_name=device_name, debug=debug, **context
        )
        import inspect

        caller = inspect.currentframe().f_back
        target = inspect.getmodule(caller.f_code)
        for name, callable in program.callable.items():
            if import_kernels == "*" or name in context:
                name_as = context.get(name, name)
                setattr(target, name_as, callable)
        if namespace:
            return program, program.callable
        else:
            return program, list(program.callable.values())


import_program = KernelManager.import_program
load_program = KernelManager.load_program

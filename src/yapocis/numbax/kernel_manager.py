from __future__ import annotations

import importlib
from contextlib import contextmanager
from typing import Callable, Optional

from mako.template import Template
from yapocis.numbax.kernel import Kernel

from yapocis.core.interface import Interface, get_interface, load_interface


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
        # Load module with Python implementations
        mod_name = f"yapocis.kernels.numbax.{interface.name}"
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            # Defer implementation errors to call time; allow smoke-compile to pass
            self.module_missing = True  # type: ignore[attr-defined]
            self.missing_module_name = mod_name  # type: ignore[attr-defined]
            return
        # Wire up kernels
        for k in interface.kernels:
            name = k.alias if k.alias else k.name
            func_name = name
            if not hasattr(mod, func_name):
                # Dynamic fallbacks for parameterized interfaces
                fn = self._make_dynamic_fn(interface.name, k, mod)
                if fn is None:
                    # Provide a lazy stub that raises when called, so compile-only paths work
                    def _stub(*args, **kwargs):
                        raise NotImplementedError(
                            f"Numba backend: function '{func_name}' not implemented in {mod_name}"
                        )

                    fn = _stub
            else:
                fn = getattr(mod, func_name)
            self.callable[k.name] = Kernel(name, fn, k.parameters, kernel_manager=self)

    def _make_dynamic_fn(self, iface_name: str, kernel, mod):
        """Create missing functions for parameterized interfaces on demand.

        Uses kernel.alias (when present) as the logical base name so that
        extensions like "_res", "_h", and "_v" resolve to the same base
        implementation.
        """
        name = kernel.name
        alias = getattr(kernel, "alias", None)
        ctx = self.context
        if iface_name == "convolve":
            from yapocis.kernels.numbax.convolve import make_convolve_functions

            base = (alias or name).replace("_fast", "")
            kmain, kfast = make_convolve_functions(base, ctx.get("conv"))
            setattr(mod, base, kmain)
            setattr(mod, base + "_fast", kfast)
            # Return function matching the requested callable name
            return getattr(mod, alias or name)
        if iface_name == "convolvesep":
            from yapocis.kernels.numbax.convolvesep import make_sep_functions

            convs_list = ctx.get("convs", [])
            convs = {n: v for (n, v) in convs_list}
            base = alias or name
            # Normalize suffixes to the base kernel name
            if base.endswith(("_h", "_v", "_res")):
                base = base.rsplit("_", 1)[0]
            km, kh, kv = make_sep_functions(base, convs)
            setattr(mod, base, km)
            setattr(mod, base + "_h", kh)
            setattr(mod, base + "_v", kv)
            # Return function matching the requested callable name
            return getattr(mod, alias or name)
        if iface_name == "operators":
            # operators provided in module statically
            return None
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
        setattr(km, "source", source)
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

from __future__ import annotations

"""
KernelManager for Metal (PyMetallic) backend.

Renders Mako templates to Metal shader source, builds a program via PyMetallic,
and exposes callable kernel functions mirroring the OpenCL KernelManager API.
"""

import os
from typing import Callable, Optional

try:
    import pymetallic as pm  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyMetallic is required for the metallic backend") from e

from mako.lookup import TemplateLookup
from mako.template import Template

from yapocis.metallic.device_manager import get_device_manager
from yapocis.core.interface import Interface, get_interface, load_interface
from yapocis.metallic.kernel import Kernel
from yapocis.utils.internals import get_yapocis_dir
from yapocis.utils.stereo_typing import Context


def _find_kernel_template(base_filename: str) -> str:
    # Prefer Metal-specific templates, fallback to generic kernels if present
    base_dir = os.path.join(get_yapocis_dir(), "metalkernels")
    cand = os.path.join(base_dir, f"{base_filename}.mako")
    if os.path.exists(cand):
        return cand
    # Fallback to OCL kernels for early development; may not compile on Metal
    ocl_dir = os.path.join(get_yapocis_dir(), "kernels")
    cand2 = os.path.join(ocl_dir, f"{base_filename}.mako")
    if os.path.exists(cand2):
        return cand2
    raise FileNotFoundError(
        f"No template found for {base_filename}; looked in {base_dir} and {ocl_dir}"
    )


def render_program(base_filename: str, **context: Context) -> str:
    tfilename = _find_kernel_template(base_filename)
    template_directories = [os.path.dirname(tfilename)]
    # Backend-specific cache directory to prevent clashes with OCL templates
    mako_modules = os.path.join(get_yapocis_dir(), "mako_modules", "metal")
    mylookup = TemplateLookup(
        directories=template_directories, module_directory=mako_modules
    )
    filename = os.path.split(tfilename)[-1]
    t = mylookup.get_template(filename)
    source = str(t.render(**context))
    return source


def render_interface(source: str, **context: Context) -> str:
    if not context:
        return source
    t = Template(source)
    return str(t.render(**context))


class KernelManager:
    def __init__(
        self,
        interface: Interface,
        device_name: str | int | None,
        debug: bool = False,
        async_write: bool = False,
        options=None,
        **context: Context,
    ):
        self.interface = interface
        self.dmgr = get_device_manager(device_name=device_name)
        try:
            source = render_program(interface.name, **context)
        except Exception:
            print(f"Source rendering issue for {interface.name}")
            print(context)
            raise
        self.source = source
        if debug:
            print("Metal Source")
            print(source)
        # pymetallic compiles kernels on demand; keep source handy
        self.callable: dict[str, Callable] = {}
        self.calls = 0
        self.runtime = 0.0
        self.async_write = async_write
        for kernel in interface.kernels:
            name = kernel.alias if getattr(kernel, "alias", None) else kernel.name
            pm_kernel = pm.Kernel(self.dmgr.device, source=source, func=name)
            self.callable[kernel.name] = Kernel(
                name,
                self,
                pm_kernel,
                kernel.parameters,
                async_write=async_write,
            )

    def read(self, a):
        return self.dmgr.read(a)

    def write(self, a, async_write=None):
        async_write = self.async_write if async_write is None else async_write
        return self.dmgr.write(a, wait=async_write)

    def purge(self):
        self.stats()
        return self.dmgr.purge()

    def reset(self):
        self.dmgr.reset()
        self.runtime = self.calls = 0

    def snapshot(self, **kwargs):
        d = dict(
            interfacename=self.interface.name,
            calls=self.calls,
            runtime=self.runtime,
            device=self.dmgr.device_name,
            backend="metal",
        )
        d.update(kwargs)
        self.dmgr.snapshot(**d)

    def __getattr__(self, attr: str) -> Callable:
        if attr not in self.callable:
            print(
                "Kernel %s not found, %s available" % (attr, list(self.callable.keys()))
            )
            raise KeyError(attr)
        return self.callable[attr]

    def stats(self):
        d = dict(
            interfacename=self.interface.name,
            calls=self.calls,
            runtime=self.runtime,
            device=self.dmgr.device_name,
        )
        d.update(self.dmgr.stats())
        return d

    def __str__(self):
        s = " ".join([f"{key}: {value}" for key, value in self.stats().items()])
        return s

    @classmethod
    def load_program(
        cls,
        interface_spec: str,
        device_name: Optional[str | int] = None,
        debug: bool = False,
        options=None,
        **context: Context,
    ):
        source = load_interface(interface_spec)
        source = render_interface(source, **context)
        interface = get_interface(source)
        if debug:
            print("Interface", interface)
        return KernelManager(
            interface, device_name, debug=debug, options=options, **context
        )

    @classmethod
    def import_program(
        cls,
        interface_spec: str,
        device_name: Optional[str | int] = None,
        namespace: bool = False,
        debug: bool = False,
        import_kernels=None,
        options=None,
        **context: Context,
    ):
        program = cls.load_program(
            interface_spec,
            device_name=device_name,
            debug=debug,
            options=options,
            **context,
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

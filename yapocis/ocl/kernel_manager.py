#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

import os
from typing import Any, Callable

import pyopencl
from mako.lookup import TemplateLookup
from mako.template import Template

from yapocis.kernels import find_kernel
from yapocis.ocl.device_manager import get_device_manager
from yapocis.core.interface import Interface, get_interface, load_interface
from yapocis.ocl.kernel import Kernel
from yapocis.utils.internals import find_yapocis_file, get_yapocis_dir
from yapocis.utils.stereo_typing import Context


def render_program(base_filename: str, **context: Context) -> str:
    """
    Takes the basename of a mako template file, and uses context to generate an OpenCL program
    """
    tfilename = find_kernel(base_filename)
    template_directories = [os.path.dirname(tfilename)]
    mako_modules = os.path.join(get_yapocis_dir(), "mako_modules", "ocl")
    mylookup = TemplateLookup(directories=template_directories, module_directory=mako_modules)
    filename = os.path.split(tfilename)[-1]
    t = mylookup.get_template(filename)
    source = str(t.render(**context))
    return source


def render_interface(source: str, **context: Context) -> str:
    """
    Takes the source for a mako definition of a yapocis.rpc1 interface and generates
    a parsable definition. Useful when the template can generate multiple parameterized
    routines.
    """
    if not context:
        return source
    t = Template(source)
    return str(t.render(**context))


class KernelManager:
    """
    Takes a parsed yapocis interface and returns a class/module like structure
    which supports calling into kernel methods of the interface.
    Manages buffers on the OpenCL device to prevent over-allocation.
    """

    def __init__(self, interface: Interface, device_name: str, debug: bool = False, async_write: bool = False,
                 options = None,
                 **context: Context):
        self.interface = interface
        self.dmgr = get_device_manager(device_name=device_name)
        try:
            source = render_program(interface.name, **context)
        except:
            print(f"Source rendering issue for {interface.name}")
            print(context)
            raise
        self.source = source
        if debug:
            print("Source")
            print(source)
        self.mf = pyopencl.mem_flags
        self.program = self.dmgr.build(source)
        self.callable = {}
        self.calls = 0
        self.runtime = 0.0
        self.async_write = async_write
        for kernel in interface.kernels:
            name = kernel.alias if kernel.alias else kernel.name
            self.callable[kernel.name] = Kernel(name, self,
                                                getattr(self.program, name),
                                                kernel.parameters,
                                                async_write=async_write)

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
        d = dict(interfacename=self.interface.name,
                 calls=self.calls,
                 runtime=self.runtime,
                 device=self.dmgr.device_name)
        d.update(kwargs)
        self.dmgr.snapshot(**d)

    def __getattr__(self, attr: str) -> Callable:
        "Any unbound attribute is checked to see if it is a callable"
        if not attr in self.callable:
            print("Kernel %s not found, %s available" % (attr, list(self.callable.keys())))
            raise KeyError(attr)
        return self.callable[attr]

    def stats(self):
        d = dict(interfacename=self.interface.name,
                 calls=self.calls,
                 runtime=self.runtime,
                 device=self.dmgr.device_name)
        d.update(self.dmgr.stats())
        return d

    def __str__(self):
        s = " ".join([f"{key}: {value}" for key, value in self.stats().items()])
        return s

    @classmethod
    def load_program(cls, interface_spec: str, device_name: str | None = None, debug: bool = False,
                     options=None,
                     **context: Context):
        """
        Factory returning runnable interface based on a interface specification.
        """
        assert device_name is None or isinstance(device_name,(str,int,bytes)), device_name
        source = load_interface(interface_spec)
        source = render_interface(source, **context)
        interface = get_interface(source)
        if debug:
            print("Interface", interface)
        return KernelManager(interface, device_name, debug=debug, options=options, **context)

    @classmethod
    def import_program(cls, interface_spec: str, device_name: str | None = None,
                       namespace: bool = False, debug: bool = False, import_kernels=None,
                       options=None, **context: Context):
        program = cls.load_program(interface_spec, device_name=device_name, debug=debug, options=options, **context)
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

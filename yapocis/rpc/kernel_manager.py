#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import os
from typing import Optional, Callable

import pyopencl
from mako.lookup import TemplateLookup
from mako.template import Template

from yapocis.rpc.device_manager import get_device_manager
from yapocis.rpc.kernel import Kernel
from yapocis.rpc.interface import Interface, get_interface, load_interface
from yapocis.utils.internals import find_yapocis_file
from yapocis.utils.typing import Context


def render_program(base_filename: str, **context: Context) -> str:
    """
    Takes the basename of a mako template file, and uses context to generate an OpenCL program
    """
    tfilename = None
    for ext in [".cl", ".mako"]:
        filename = base_filename + ext
        tfilename = find_yapocis_file(["kernels", "."], filename)
        if tfilename:
            break
    if not tfilename:
        raise Exception("Can't find template '%s'" % filename)
    mylookup = TemplateLookup(directories=[os.path.dirname(tfilename)])
    t = Template(filename=tfilename, lookup=mylookup)
    source = str(t.render(**context))
    return source


def render_interface(source: str, **context: Context) -> str:
    """
    Takes the source for a mako definition of a yapocis.rpc1 interface and generates
    a parsable definition. Useful when the template can generate multiple parameterized
    routines.
    """
    t = Template(source)
    return str(t.render(**context))


class KernelManager:
    """
    Takes a parsed yapocis interface and returns a class/module like structure
    which supports calling into kernel methods of the interface.
    Manages buffers on the OpenCL device to prevent over-allocation.
    """

    def __init__(self, interface: Interface, device_name: str, debug: bool = False, async_write: bool = False,
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
    def load_program(cls, interface_spec: str, device_name: Optional[str] = None, debug: bool = False,
                     **context: Context):
        """
        Factory returning runnable interface based on a interface specification.
        """
        source = load_interface(interface_spec)
        source = render_interface(source, **context)
        interface = get_interface(source)
        if debug:
            print("Interface", interface)
        return KernelManager(interface, device_name, debug=debug, **context)

    @classmethod
    def import_program(cls, interface_spec: str, device_name: Optional[str] = None,
                       namespace: bool = False, debug: bool = False, import_kernels=None, **context: Context):
        program = cls.load_program(interface_spec, device_name=device_name, debug=debug, **context)
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

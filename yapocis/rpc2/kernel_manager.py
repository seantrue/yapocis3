#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import os
from typing import Optional, Callable

import pyopencl
from mako.lookup import TemplateLookup
from mako.template import Template

from yapocis.rpc.device_manager import DeviceManager
from yapocis.rpc.kernel import Kernel
from yapocis.rpc.interface import InterfaceCL, get_interface, load_interface
from yapocis.utils.basic import find_yapocis_file
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
    Takes the source for a mako definition of a yapocis.rpc interface and generates
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

    def __init__(self, interface: InterfaceCL, device_name: str, debug: bool = False, **context: Context):
        self.dmgr = DeviceManager(device_name=device_name)
        try:
            source = render_program(interface.interface_name, **context)
        except:
            print(f"Source rendering issue for {interface.interface_name}")
            print(context)
            raise
        self.source = source
        if debug:
            print("Source")
            print(source)
        self.mf = pyopencl.mem_flags
        interface.kernel_manager = self.dmgr.build(source)
        self.callable = {}
        self.interface = interface
        program = interface.kernel_manager
        self.hits = 0
        self.misses = 0
        self.calls = 0
        self.runtime = 0.0
        for kernel in interface.kernels():
            alias = interface.kernel_alias(kernel)
            self.callable[kernel] = Kernel(self, getattr(program, alias), interface.kernel_params(kernel))
            self.callable[kernel].name = kernel

    def __getattr__(self, attr: str) -> Callable:
        "Any unbound attribute is checked to see if it is a callable"
        if not attr in self.callable:
            print("Kernel %s not found, %s available" % (attr, list(self.callable.keys())))
            raise KeyError(attr)
        return self.callable[attr]

    @property
    def stats(self):
        self.purge()
        return dict(interfacename=self.interface.interface_name,
                    calls=self.calls,
                    hits=self.hits,
                    misses=self.misses,
                    buffers=len(self.buffers),
                    runtime=self.runtime,
                    device=self.device_name)

    def __str__(self):
        self.purge()
        s = " ".join([f"{key}: {value}" for key,value in self.stats.items()])
        return s

    @classmethod
    def load_program(cls, interface_spec: str, device_name: Optional[str] = None, debug: bool = False, **context: Context):
        """
        Factory returning runnable interface based on a interface specification.
        """
        source = load_interface(interface_spec)
        source = render_interface(source, **context)
        interface = get_interface(source)
        interface.source = source
        if debug:
            print("Interface", interface)
        return KernelManager(interface, device_name, debug=debug, **context)

    @classmethod
    def import_program(cls, interface_spec: str, device_name: Optional[str] = None,
                       namespace: bool = False, debug: bool = False, import_kernels=None, **context: Context):
        program = load_program(interface_spec, device_name=device_name, debug=debug, **context)
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
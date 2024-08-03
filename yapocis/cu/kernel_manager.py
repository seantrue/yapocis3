import os
from typing import Callable, Optional, Dict, List
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from mako.lookup import TemplateLookup
from mako.template import Template

# Assuming these utility functions exist in a similar structure
from yapocis.kernels import find_kernel
from yapocis.utils.internals import find_yapocis_file, get_yapocis_dir
from yapocis.utils.stereo_typing import Context

class CudaKernel:
    def __init__(self, name: str, function, parameters: List[str]):
        self.name = name
        self.function = function
        self.parameters = parameters

    def __call__(self, *args, **kwargs):
        # Here you'd implement the logic to call the CUDA kernel
        # This would involve setting up grid and block dimensions,
        # preparing arguments, and launching the kernel
        pass

class CudaKernelManager:
    def __init__(self, interface: Dict, debug: bool = False, **context: Context):
        self.interface = interface
        self.context = context
        self.debug = debug
        self.source = self.render_program(interface['name'], **context)
        if debug:
            print("Source:")
            print(self.source)
        self.module = SourceModule(self.source)
        self.callable: Dict[str, CudaKernel] = {}
        self.calls = 0
        self.runtime = 0.0

        for kernel in interface['kernels']:
            name = kernel.get('alias', kernel['name'])
            self.callable[kernel['name']] = CudaKernel(
                name,
                self.module.get_function(name),
                kernel['parameters']
            )

    @staticmethod
    def render_program(base_filename: str, **context: Context) -> str:
        tfilename = find_kernel(base_filename)
        template_directories = [os.path.dirname(tfilename)]
        mako_modules = os.path.join(get_yapocis_dir(), "mako_modules")
        mylookup = TemplateLookup(directories=template_directories, module_directory=mako_modules)
        filename = os.path.split(tfilename)[-1]
        t = mylookup.get_template(filename)
        source = str(t.render(**context))
        return source

    def __getattr__(self, attr: str) -> Callable:
        if attr not in self.callable:
            print(f"Kernel {attr} not found, available kernels: {list(self.callable.keys())}")
            raise KeyError(attr)
        return self.callable[attr]

    def stats(self):
        return {
            "interface_name": self.interface['name'],
            "calls": self.calls,
            "runtime": self.runtime
        }

    def __str__(self):
        return " ".join([f"{key}: {value}" for key, value in self.stats().items()])

    @classmethod
    def load_program(cls, interface_spec: str, debug: bool = False, **context: Context):
        # Here you'd implement the logic to load and parse the interface specification
        # For this example, we'll assume it returns a dict with the interface structure
        interface = cls.load_interface(interface_spec)
        return cls(interface, debug=debug, **context)

    @staticmethod
    def load_interface(interface_spec: str) -> Dict:
        # Implement the logic to load and parse the interface specification
        # This is a placeholder implementation
        pass

    @classmethod
    def import_program(cls, interface_spec: str, namespace: bool = False,
                       debug: bool = False, import_kernels=None, **context: Context):
        program = cls.load_program(interface_spec, debug=debug, **context)
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

import_program = CudaKernelManager.import_program
load_program = CudaKernelManager.load_program
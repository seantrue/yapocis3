#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import os
import time
import numpy as np
from typing import List, Optional, Callable, Tuple, Any, Union

import pyopencl
from mako.lookup import TemplateLookup
from mako.template import Template

from yapocis.rpc.engines import get_engine, Engine
from yapocis.rpc.parser import InterfaceCL, get_interface
from yapocis.utils.basic import find_yapocis_file
from yapocis.utils.typing import Context, Array


def render_program(base_filename: str, **context: Context) -> str:
    """
    Takes the basename of a mako template file, and uses context to generate an OpenCL program
    """
    tfilename = None
    for ext in [".cl",".mako"]:
        filename = base_filename + ext
        tfilename = find_yapocis_file(["kernels", "."], filename)
        if tfilename:
            break
    if not tfilename:
        raise Exception("Can't find template '%s'" % filename)
    mylookup = TemplateLookup(directories=[os.path.dirname(tfilename)])
    t = Template(filename=tfilename, lookup=mylookup)
    source =  str(t.render(**context))
    return source


def render_interface(source: str, **context: Context) -> str:
    """
    Takes the source for a mako definition of a yapocis.rpc interface and generates
    a parsable definition. Useful when the template can generate multiple parameterized
    routines.
    """
    t = Template(source)
    return str(t.render(**context))


class Program:
    """
    Takes a parsed yapocis interface and returns a class/module like structure
    which supports calling into kernel methods of the interface.
    Manages buffers on the OpenCL device to prevent over-allocation.
    """

    def __init__(self, interface: InterfaceCL, engine_name: str, debug: bool = False, **context: Context):
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
        self.engine = get_engine(engine_name)
        interface.program = self.engine.build(source)
        self.callable = {}
        self.context_meta = self.engine.context_meta
        self.ctx = self.engine.ctx
        self.queue = self.engine.queue
        self.interface = interface
        program = interface.program
        self.hits = 0
        self.misses = 0
        self.calls = 0
        self.runtime = 0.0
        for kernel in interface.kernels():
            alias = interface.kernel_alias(kernel)
            self.callable[kernel] = Kernel(self, getattr(program, alias), interface.kernel_params(kernel), self.engine)

    def __getattr__(self, attr: str) -> Callable:
        "Any unbound attribute is checked to see if it is a callable"
        if not attr in self.callable:
            print("Kernel %s not found, %s available" % (attr, list(self.callable.keys())))
            raise KeyError(attr)
        return self.callable[attr]

    def purge(self):
        self.engine.purge()

    def read(self, a: Array):
        return self.engine.read(a)

    def write(self, a: Array):
        return self.engine.write(a)

    def ensure(self, a: Array):
        return self.engine.ensure(a)

    def stats(self):
        return dict(interfacename=self.interface.interface_name,
                    calls=self.calls,
                    hits=self.engine.bmgr.hits,
                    misses=self.engine.bmgr.misses,
                    buffers=len(self.engine.bmgr.buffers),
                    runtime=self.runtime,
                    engine=self.engine.device_name)

    def __str__(self):
        self.purge()
        return "%(interfacename)s engine:%(engine)s calls:%(calls)s hits:%(hits)s misses:%(misses)s buffers:%(buffers)s runtime:%(runtime)s" % self.stats()


def load_interface(interface_spec:str) -> str:
    """
    Find an interface, or return source if it is an interface spec.
    Args:
        interface_spec: either an interface name or a spec

    Returns:
        an interface specification
    """
    if interface_spec.strip().startswith("interface "):
        return interface_spec
    filename= f"{interface_spec}.interface"
    path = find_yapocis_file(["interfaces", "."], filename)
    if path is None:
        raise RuntimeError(f"Can't find {filename}")
    with open(path) as inf:
        return inf.read()


def load_program(interface_spec: str, engine_name: Optional[str] = None, debug: bool = False, **context: Context):
    """
    Primary interface for yapocis.rpc
    Factory returning runnable interface based on a interface specification.
    """
    source = load_interface(interface_spec)
    source = render_interface(source, **context)
    interface = get_interface(source)
    interface.source = source
    if debug: print("Interface", interface)
    return Program(interface, engine_name, debug=debug, **context)


def import_program(interface_spec: str, engine_name:Optional[str]=None,
                   namespace:bool=False, debug: bool = False, import_kernels=None, **context: Context):
    program = load_program(interface_spec, engine_name=engine_name,debug=debug,**context)
    import inspect
    caller = inspect.currentframe().f_back
    target = inspect.getmodule(caller.f_code)
    for name, callable in program.callable.items():
        if import_kernels == "*" or name in context:
            name_as = context.get(name,name)
            setattr(target, name_as, callable)
    if namespace:
        return program, program.callable
    else:
        return program, list(program.callable.values())


class Kernel:
    """
    Provide a callable interface to an OpenCL kernel function.
    """

    def __init__(self, program:pyopencl.Program, kernel:Callable, params:List[Tuple], engine:Engine):
        self.program = program
        self.kernel = kernel
        self.params = params
        self.ctx = engine.ctx
        self.queue = engine.queue
        self.autoflatten = engine.autoflatten
        self.map_arguments()

    def read(self, a:Array):
        return self.program.read(a)

    def write(self, a:Array):
        return self.program.write(a)

    def ensure(self, a:Array):
        return self.program.ensure(a)

    def map_arguments(self):
        argspecs = [None] * len(self.params)
        posbyname = {}
        iarg = 0
        for iparam, (bufferHint, dtype, isbuffer, name) in enumerate(self.params):
            if bufferHint in ("outlike", "sizeof", "widthof", "heightof", "depthof"):
                argspecs[iparam] = (bufferHint, name)
            else:
                posbyname[name] = iarg
                argspecs[iparam] = (bufferHint, iarg)
                iarg += 1
        self.argspecs = argspecs
        self.posbyname = posbyname

    def prepare_arguments(self, args:List[Any]):
        "Takes the actual arguments and maps to the arguments needed by the kernel"
        assert len(self.posbyname) == len(args), f" Got {len(args)}, expected {len(self.posbyname)} arguments, [{self.posbyname}]"
        self.returns = []
        realargs = [None] * len(self.params)
        for iparam, (bufferHint, value) in enumerate(self.argspecs):
            (_, dtype, isbuffer, name) = self.params[iparam]
            if dtype in ("char", "uchar"):
                dtype = "uint8"
            if dtype:
                dtype = getattr(np, dtype)
            else:
                dtype = None
            if not isbuffer:
                assert dtype, "Must have type for scalar argument %s" % iparam
                arg = args[value]
                realargs[iparam] = dtype(arg)
                continue
            methodname = "handle_%s" % bufferHint
            realargs[iparam] = getattr(self, methodname)(args, value, dtype)
        if self.global_size is None:
            for arg in args:
                if hasattr(arg, "size"):
                    if self.autoflatten:
                        self.global_size = arg.size
                    else:
                        self.global_size = arg.shape
                    break
        return realargs

    def handle_sizeof(self, args, name, dtype):
        arg = args[self.posbyname[name]]
        if not dtype:
            dtype = np.int32
        return dtype(arg.size)

    # Assume row major storage.
    # addr(x,y) = (x*height)+y
    #  x = addr/height
    #  y = (addr - (addr/height)*height)
    def handle_widthof(self, args, name, dtype):
        arg = args[self.posbyname[name]]
        if not dtype:
            dtype = np.int32
        return dtype(arg.shape[0])

    def handle_heightof(self, args, name, dtype):
        arg = args[self.posbyname[name]]
        if not dtype:
            dtype = np.int32
        return dtype(arg.shape[1])

    def handle_depthof(self, args, name, dtype):
        arg = args[self.posbyname[name]]
        if not dtype:
            dtype = np.int32
        return dtype(arg.shape[2])

    def handle_outlike(self, args, name, dtype):
        arg = args[self.posbyname[name]]
        if not dtype:
            dtype = arg.dtype
        a = np.zeros_like(arg, dtype=dtype)
        buf = self.ensure(a)
        self.returns.append((a, buf))
        return buf

    def handle_in(self, args, pos, dtype):
        arg = args[pos]
        assert dtype
        if arg.dtype != dtype:
            arg = arg.astype(dtype)
        buf = self.write(arg)
        return buf

    def handle_inout(self, args, pos, dtype):
        arg = args[pos]
        assert dtype
        if arg.dtype != dtype:
            arg = arg.astype(dtype)
        buf = self.write(arg)
        self.returns.append((arg, buf))
        return buf

    def handle_out(self, args, pos, dtype):
        arg = args[pos]
        assert dtype
        if arg.dtype != dtype:
            arg = arg.astype(dtype)
        buf = self.ensure(arg)
        self.returns.append((arg, buf))
        return buf

    handle_resident = handle_out

    def __call__(self, *args:List[Any], **kwargs):
        "Call a kernel, prepare arguments, and track performance"
        self.program.calls += 1
        t = time.time()
        self.global_size = kwargs.pop("global_size", None)
        self.local_size = kwargs.pop("local_size", None)
        args = self.prepare_arguments(args)
        if not isinstance(self.global_size, (tuple, list)):
            self.global_size = (self.global_size,)
        self.evt = self.kernel(self.queue, self.global_size, self.local_size, *args)
        poll = kwargs.pop("poll", None)
        if poll:
            while (True):
                status = self.evt.get_info(pyopencl.event_info.COMMAND_EXECUTION_STATUS)
                if status == pyopencl.command_execution_status.COMPLETE:
                    break
                time.sleep(poll)
        else:
            self.evt.wait()
        # pyopencl.enqueue_barrier(self.queue)
        rval = self.prepare_return()
        self.program.runtime += (time.time() - t)
        return rval

    def prepare_return(self) ->Union[Any, Tuple[Any], None]:
        "Takes the return value buffers (out and outlike) and prepares proper Numpy arrays"
        rvals = []
        for arg, buf in self.returns:
            self.read(arg)
            rvals.append(arg)
        self.returns = []
        if len(rvals) == 0:
            return None
        elif len(rvals) == 1:
            return rvals[0]
        return tuple(rvals)
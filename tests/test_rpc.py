#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import numpy as np

from yapocis.rpc.buffers import BufferManager, CPU_ENGINE
from yapocis.rpc.parser import get_interface
from yapocis.rpc.engines import get_engine, get_available
from yapocis.rpc.programs import load_program, import_program


def test_buffers():
    bmgr = BufferManager(CPU_ENGINE)
    a = np.random.sample((100,)).astype(np.float32)
    b = a.copy()
    bmgr.write_buffer(a)
    bmgr.read_buffer(a)
    assert a.sum() == b.sum()
    b[:] = 0.0
    assert a.sum() != b.sum()
    assert len(list(bmgr.arrays.keys())) == 1
    del a
    assert len(list(bmgr.arrays.keys())) == 0
    a = np.random.sample((100, 100)).astype(np.float32)
    b = a.copy()
    bmgr.write_buffer(a)
    bmgr.read_buffer(a)
    assert a.sum() == b.sum()
    b[:, :] = 0.0
    assert a.sum() != b.sum()
    bmgr.read_buffer(a)
    try:
        bmgr.read_buffer(b)
        assert False, "Should have generated a Value error"
    except:
        pass
    bmgr.write_buffer(b)
    bmgr.read_buffer(b)
    assert b.sum() == 0.0
    print(id(a), id(b))
    assert len(list(bmgr.arrays.keys())) == 2
    a = None
    assert len(list(bmgr.arrays.keys())) == 1
    b = None
    print(list(bmgr.arrays.keys()), id(a), id(b))
    # Todo: why does the b buffer stay?
    # assert len(bmgr.arrays.keys())==0


def test_getinterfacecl():
    interface = get_interface(
        """
        interface boundedmedian {
             kernel boundedmedian(sizeof int input, in float32 *input, in int32 *zcs, outlike int16 input, out short *trace);
             alias bm as boundedmedian(in int32 offset, resident float32 *input, in int32 *zcs, resident float *input, out int16 *trace);
          };
        """
    )
    print("Interface:", interface.interface_name)
    for kernel in interface.kernels():
        print("Kernel: %s alias for %s" % (kernel, interface.kernel_alias(kernel)))
        symbols = {}
        iparam = 0
        for param in interface.kernel_params(kernel):
            assert len(param) == 4
            print("Param:", param, end=' ')
            direction, dtype, isbuffer, name = param
            assert direction in ("in", "out", "inout", "outlike", "resident", "sizeof", "widthof", "heightof")
            if direction == "outlike":
                assert name in symbols
                iparam, olparam = symbols[name]
                dtype, isbuffer = olparam[1], olparam[2]
                print("->", olparam, end=' ')
                assert isbuffer
            else:
                symbols[name] = iparam, param
                iparam += 1
            assert isbuffer in ("*", "")
            assert dtype in ("int16", "int32", "float32", "uint16", "uint32", "complex64", "int", "float", "short")
            togpu = direction in ("in", "inout")
            fromgpu = direction in ("out", "inout", "outlike")
            if isbuffer:
                if direction == "resident":
                    print("Buffer is resident on GPU.", end=' ')
                elif direction == "outlike":
                    print("Allocate a buffer like %(name)s (position=%(iparam)s). (%(olparam)s)" % locals(), end=' ')
                elif direction in ("sizeof", "heightof", "widthof"):
                    print("Pass in %s of %s" % (direction, name), end=' ')
                else:
                    print("Coerce %(name)s to numpy.%(dtype)s. " % locals(), end=' ')
                    print("Allocate a buffer for %(name)s. " % locals(), end=' ')
                if togpu:
                    print("Copy to GPU. ", end=' ')
                if fromgpu:
                    print("Copy back from GPU.", end=' ')
            else:
                assert not fromgpu
                print("Use as parameter.", end=' ')
            print()
        print()


def test_compiling():
    interfaces = [("convolve", dict(name="convolve", conv=[1, 2, 3, 4, 3, 2, 1])),
                  ("median3x3", dict(steps=[9], width=9)),
                  ("gradient", {}),
                  ("convolves", dict(convs=[("boxone", [1, 1, 1]), ("triangle", [.5, 1, .5])])),
                  ("mandelbrot", {}),
                  ("demo", {}),
                  ("hsi", {}),
                  ]

    for engine_name in get_available().keys():
        for itest, (interface, context) in enumerate(interfaces):
            program = load_program(interface, engine_name=engine_name, debug=True, **context)
            print("Interface", program.interface.interface_name)
            for kernel in program.interface.kernels():
                print("Kernel", kernel)
                print("Params", program.interface.kernel_params(kernel))
                print("OpenCL entry", getattr(program.interface.program, program.interface.kernel_alias(kernel)))
                print("Callable", getattr(program, kernel))
            print()


def test_context():
    engine = get_engine()
    a = np.random.sample((100,)).astype(np.float32)
    b = a.copy()
    # b[0]=b[1]
    engine.write(a)
    engine.read(a)
    diff = np.abs(a - b).sum()
    assert diff == 0.0
    del a
    try:
        engine.read(b)
        print("Did not fail on unmapped array")
    except ValueError:
        pass


def test_demo():
    _,entrypoints = import_program("demo",sum="sum")
    assert sum == entrypoints[0]
    a = np.array([0, 1, 2, 3]).astype(np.float32)
    b = np.array([1, 1, 1, 1]).astype(np.float32)
    assert sum(a, b).sum() == 10


def test_import():
    import_program("demo", sum="frabjous_day")
    assert callable(frabjous_day)

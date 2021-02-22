#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import numpy as np

from yapocis.rpc.interface import get_interface
from yapocis.rpc.device import get_device, get_available_device_names, get_available_devices, get_platforms
from yapocis.rpc.device_manager import DeviceManager
from yapocis.rpc.kernel_manager import load_program, import_program


def test_buffers():
    bmgr = DeviceManager()
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


def test_getinterface():
    interface = get_interface(
        """
        interface boundedmedian {
             kernel boundedmedian(sizeof input, in float input, widthof zcs, in int zcs, outlike input, out short trace);
             alias bm as boundedmedian(in int offset, resident float input, widthof zcs, in int zcs, resident float output, out short trace);
          };
        """
    )


def test_compiling():
    interfaces = [
        ("convolve", dict(name="convolve", conv=[1, 2, 3, 4, 3, 2, 1])),
        ("median3x3", dict(steps=[9], width=9)),
        ("gradient", {}),
        ("convolves", dict(convs=[("boxone", [1, 1, 1]), ("triangle", [.5, 1, .5])])),
        ("mandelbrot", {}),
        ("demo", {}),
        ("hsi", {}),
        ("operators",dict(operators=[("add", "+"), ("sub", "-"), ("mul", "*"), ("div", "/")]))
    ]

    for device_name in get_available_device_names():
        for itest, (interface, context) in enumerate(interfaces):
            print(device_name, interface)
            program = load_program(interface, device_name=device_name, debug=False, **context)
            print("Interface", program.interface.name)
            print(f"Kernels: {len(program.interface.kernels)}")


def test_context():
    engine = DeviceManager()
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
    _, entrypoints = import_program("demo", sum="sum")
    assert sum == entrypoints[0]
    a = np.array([0, 1, 2, 3]).astype(np.float32)
    b = np.array([1, 1, 1, 1]).astype(np.float32)
    assert sum(a, b).sum() == 10


def test_import():
    import_program("demo", sum="frabjous_day")
    assert callable(frabjous_day)

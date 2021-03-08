#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import pytest
import time
import numpy as np

from yapocis.rpc.interface import get_interface
from yapocis.rpc.device import get_device, get_available_device_names, get_available_devices, get_platforms
from yapocis.rpc.device_manager import DeviceManager
from yapocis.rpc.buffer_manager import BufferManager
from yapocis.rpc.kernel_manager import load_program, import_program
from . import show_snapshots



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
    bmgr.snapshot(name="buffers")

def test_buffers_async_write():
    bmgr = DeviceManager(timed=True, device_name=2)
    a = np.random.sample((2000, 1000)).astype(np.float32)
    t = a.sum()
    for i in range(1000):
        bmgr.write(a, wait=False)
    a[:,:] = 0.0
    bmgr.read(a)
    assert a.sum() == t
    print(bmgr.snapshot(name="async_write"))
    del bmgr
    bmgr = DeviceManager(timed=True, device_name=2)
    arrays = []
    a = np.random.sample((2000, 1000)).astype(np.float32)
    t = a.sum()
    for i in range(1000):
        bmgr.write(a, wait=True)
    a[:,:] = 0.0
    bmgr.read(a)
    assert a.sum() == t
    print(bmgr.snapshot(name="sync_write"))
    del bmgr



def test_getinterface():
    interface = get_interface(
        """\
        interface boundedmedian {
             kernel boundedmedian(sizeof input, in float input, widthof zcs, in int zcs, outlike input, out short trace);
             alias bm as boundedmedian(int offset, resident float input, widthof zcs, in int zcs, resident float output, out short trace);
             extension res of boundedmedian(int offset, resident float input, widthof zcs, resident int zcs, resident float output, resident short trace);
          };
        """
    )
    assert len(interface.kernels) == 3
    assert len(interface.aliases) == 2


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
    with pytest.raises(ValueError):
        engine.read(b)


def test_demo_kernel():
    _, entrypoints = import_program("demo", import_kernels="*")
    print(sum,sum_res)
    assert sum == entrypoints[0]
    assert sum_res == entrypoints[1]
    a = np.array([0, 1, 2, 3]).astype(np.float32)
    b = np.ones_like(a)
    c = np.zeros_like(a)
    ids = (id(a),id(b),id(c))
    # Send two arrays in, allocate and retrieve results
    d = sum(a, b)
    assert id(d) not in ids
    assert d.sum() == 10
    d1 =  sum_res(a,b,c,resident=(a,b))
    assert id(d1) == id(c)
    assert id(d) != id(d1)
    assert d.sum() == 10
    c1 = sum_res(a,b, c)
    assert id(c) == id(c1)
    assert c1.sum() == 10
    c[:] = 0
    a[:] = 0
    sum_res(c,c, a, resident=(c,a))
    assert a.sum() == 0
    sum_res.read(a)
    assert a.sum() == 20

def test_import():
    import_program("demo", sum="frabjous_day")
    assert callable(frabjous_day)

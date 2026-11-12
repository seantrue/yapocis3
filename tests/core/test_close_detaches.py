"""
close() leaves no strong reference to arrays, and fast free survives close (#100).

Each case builds its own DeviceManager, bypassing the cached get_device_manager, so
closing it can't drop buffers that other tests registered. On CUDA, constructing a
DeviceManager makes its new context current, so the CUDA case runs only in its own
pytest process: it is skipped unless YAPOCIS_TEST_CLOSE_CUDA=1.
"""

from __future__ import annotations

import gc
import importlib
import os
import weakref

import numpy as np
import pytest

from yapocis.core import get_test_backends

MODULES = {
    "metal": "yapocis.metallic.device_manager",
    "ocl": "yapocis.ocl.device_manager",
    "cuda": "yapocis.cu.device_manager",
}


def _fresh_manager(backend):
    if backend not in get_test_backends():
        pytest.skip(f"{backend} not in the test backends")
    try:
        module = importlib.import_module(MODULES[backend])
        return module.DeviceManager(None)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{backend} not available: {e}")


@pytest.fixture
def no_gc():
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("metal", marks=pytest.mark.metal),
        pytest.param("ocl", marks=pytest.mark.ocl),
    ],
)
def test_close_detaches(backend, no_gc):
    dm = _fresh_manager(backend)
    live = np.zeros(8, dtype=np.float32)
    y = np.arange(100, dtype=np.float32)
    q = np.arange(50, dtype=np.float32)
    dm.write(y)
    dm.write(q)
    ky = id(y)
    del q  # its ref is queued before close
    dm.close()  # must not raise on the queued ref
    del y
    dm.write(live)  # one manager call after close
    assert ky not in dm.buffers, "fast free stopped after close()"
    assert ky not in dm._store
    z = np.arange(10, dtype=np.float32)
    dm.write(z)
    z[:] = 0
    dm.read(z)
    np.testing.assert_array_equal(z, np.arange(10, dtype=np.float32))
    dm.close()


@pytest.mark.cuda
@pytest.mark.skipif(
    os.environ.get("YAPOCIS_TEST_CLOSE_CUDA") != "1",
    reason="CUDA close runs in its own process: set YAPOCIS_TEST_CLOSE_CUDA=1",
)
def test_close_detaches_cuda(no_gc):
    dm = _fresh_manager("cuda")
    y = np.arange(100, dtype=np.float32)
    dm.write(y)
    dm.close()
    assert len(dm.buffers) == 0
    assert not getattr(dm, "_store", None)
    assert not getattr(dm, "_dead", None)
    del y
    z = np.arange(10, dtype=np.float32)
    dm.write(z)
    zr = weakref.ref(z)
    del z
    assert zr() is None, "close() left the manager holding arrays strongly"
    w = np.arange(10, dtype=np.float32)
    dm.write(w)
    w[:] = 0
    dm.read(w)
    np.testing.assert_array_equal(w, np.arange(10, dtype=np.float32))

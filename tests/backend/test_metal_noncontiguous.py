"""
Metal: non-C-contiguous arrays (strided views, Fortran order, transposes) as kernel arguments (#89).

The Metal device manager used to make a contiguous copy and register the buffer under the
copy's id. The copy's weak entry died when the call returned, so later lookups for the
caller's array missed: strided inputs leaked their buffers, strided outputs and writes could
not be read back, and an F-ordered input with an ``out_like`` result failed at read-back.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

pytestmark = pytest.mark.metal


@pytest.fixture(scope="module")
def metal():
    try:
        import pymetallic  # noqa: F401

        from yapocis.core import get_manager

        mgr = get_manager("metal")
        demo = mgr.load_program("demo")
        graphics = mgr.load_program("graphics")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Metal not available: {e}")
    return demo, graphics


@pytest.fixture
def demo(metal):
    prog = metal[0]
    # Start each test with no orphaned buffers. Collect first: arrays that earlier
    # tests left in reference cycles die only in a full collection, and a purge
    # before it would leave their buffers counted as leaks. (Kernels no longer
    # keep their last call's arguments, #100.)
    gc.collect()
    prog.dmgr.purge()
    return prog


def test_strided_in_leaves_no_orphan_buffer(demo):
    dmgr = demo.dmgr
    a = np.arange(64, dtype=np.float32)[::2]
    b = np.arange(64, dtype=np.float32)[1::2]
    c = np.zeros(32, dtype=np.float32)
    entries = len(dmgr._store)
    released = dmgr.released
    for _ in range(5):
        demo.sum_res(a, b, c, global_size=a.size)
    gc.collect()
    np.testing.assert_array_equal(c, a + b)
    assert dmgr.count_leaked_buffers() == 0
    # count_leaked_buffers() drains first (#100), so it can't see a buffer that was
    # registered under a temporary copy: the copy dies and fast free drops its entry.
    # Check the store instead. Each argument is registered under the caller's array,
    # and no buffer was registered under an array that then died.
    for arr in (a, b, c):
        assert (
            dmgr._lookup(arr) is not None
        ), "a strided argument has no buffer of its own"
    assert dmgr.released == released, "buffers were registered under temporary copies"
    assert len(dmgr._store) == entries + 3


def test_strided_out_is_written(demo):
    a = np.arange(16, dtype=np.float32)
    b = np.full(16, 7, dtype=np.float32)
    c = np.zeros(32, dtype=np.float32)[::2]
    r = demo.sum_res(a, b, c, global_size=a.size)
    assert r is c
    np.testing.assert_array_equal(c, a + b)


def test_strided_write_then_read_round_trips(demo):
    nc = np.arange(32, dtype=np.float32)[::2]
    demo.dmgr.write(nc)
    assert demo.dmgr.find_buffer(nc, demo.dmgr.READ) is not None


def test_fortran_order_input_works(demo):
    f = np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4))
    out = demo.sum(f, np.zeros_like(f), global_size=f.size)
    np.testing.assert_array_equal(out, f)


def test_transposed_view_input_works(demo):
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = base.T  # an F-ordered view
    out = demo.sum(t, np.ones_like(t), global_size=t.size)
    np.testing.assert_array_equal(out, t + 1)


def test_strided_inout_writes_back_into_its_base(metal, demo):
    graphics = metal[1]
    base = np.ones((8, 6), dtype=np.float32)
    view = base[::2]  # every other row: not C-contiguous
    # One rectangle that covers the whole image, value 5, opaque, no antialiasing.
    coords = np.array([[-100.0, -100.0, 100.0, 100.0]], dtype=np.float32)
    aa_widths = np.zeros(1, dtype=np.float32)
    colors = np.array([[5.0, 1.0]], dtype=np.float32)
    graphics.rects(view, coords, aa_widths, colors, global_size=view.size)
    np.testing.assert_array_equal(base[::2], 5.0)
    np.testing.assert_array_equal(base[1::2], 1.0)

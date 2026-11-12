"""
Buffer lifetime and identity in the device managers (#100).

- Kernels don't keep their last call's arguments, and async calls hold the device
  buffers they used (not the arrays) until ``wait_all()``.
- A buffer is found for an array only when the store entry under ``id(a)`` holds a
  weak reference whose referent is ``a``; ``id()`` is reused after an array dies.
- A device buffer is released at the next device-manager call after its array dies
  ("fast free"), without a purge.

Tests that read ``dmgr._store`` test the invariant, not the API. The fixture disables
the cycle collector, so every release here comes from reference counting alone.
"""

from __future__ import annotations

import gc
import threading
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from yapocis.core import get_manager, get_test_backends
from yapocis.core.device_mgr_base import DeviceManagerBase

BACKENDS = [
    pytest.param("metal", marks=pytest.mark.metal),
    pytest.param("ocl", marks=pytest.mark.ocl),
    pytest.param("cuda", marks=pytest.mark.cuda),
]

_programs: dict[str, SimpleNamespace] = {}


def _load(backend: str) -> SimpleNamespace:
    if backend not in get_test_backends():
        pytest.skip(f"{backend} not in the test backends")
    if backend not in _programs:
        try:
            mgr = get_manager(backend)
            imageops = mgr.load_program("imageops")
            demo = mgr.load_program("demo")
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"{backend} not available: {e}")
        _programs[backend] = SimpleNamespace(
            backend=backend, imageops=imageops, demo=demo, dmgr=imageops.dmgr
        )
    return _programs[backend]


@pytest.fixture(params=BACKENDS)
def env(request):
    e = _load(request.param)
    gc.collect()
    e.imageops.wait_all()
    e.demo.wait_all()
    e.dmgr.purge()
    gc.disable()
    try:
        yield e
    finally:
        gc.enable()


def _rand(n, dtype=np.float32, seed=100):
    return np.random.default_rng(seed).random(n).astype(dtype)


def _is_buffer(x) -> bool:
    """A device buffer in a prepared-argument tuple (not a scalar or a dimension array)."""
    return not isinstance(x, (np.ndarray, np.generic, int, float))


# --- Item 1: kernels don't keep arguments; async calls hold buffers -------------------


def test_kernel_drops_current_args(env):
    prog = env.imageops
    x = np.linspace(0, 1, 64, dtype=np.float32)
    xr = weakref.ref(x)
    out = prog.stretch(x, 0.0, 1.0, 0.0, 2.0)
    np.testing.assert_allclose(out, x * 2.0, atol=1e-5)
    del x
    assert xr() is None, "a kernel kept its last call's argument"
    for name, kernel in prog.callable.items():
        assert kernel._current_args is None, name


def test_current_args_cleared_on_error(env):
    kernel = env.imageops.callable["stretch"]
    x = np.linspace(0, 1, 64, dtype=np.float32)
    with pytest.raises(Exception):  # noqa: B017 - the wrong number of arguments
        env.imageops.stretch(x, 0.0)
    assert kernel._current_args is None


def test_async_returns_cleared_and_inflight_held(env):
    prog, dmgr = env.imageops, env.dmgr
    kernel = prog.callable["stretch"]
    # float64 into a float32 kernel makes an astype temporary; out_like makes a result
    # the caller never sees. Both die when the call returns.
    x64 = np.linspace(0, 1, 4096, dtype=np.float64)
    assert prog.stretch(x64, 0.0, 1.0, 0.0, 2.0, async_exec=True) is None
    assert kernel.returns == []
    dmgr.count_leaked_buffers()  # one manager call: drains the dead arrays' entries
    held = [b for b in prog._inflight[-1] if _is_buffer(b)]
    registered = list(dmgr.buffers.values())
    orphans = [b for b in held if not any(b is r for r in registered)]
    assert orphans, "expected a buffer dropped by the manager but held in flight"
    if env.backend == "metal":
        wr = weakref.ref(orphans[0])
        del held, registered, orphans
        assert wr() is not None, "the in-flight hold should keep the buffer alive"
        prog.wait_all()
        assert wr() is None, "wait_all should release the in-flight buffer"
    else:
        del held, registered, orphans
        prog.wait_all()
    assert prog._inflight == []


def test_async_inflight_soak(env):
    """Async calls with dtype-converting temporaries, draining between calls."""
    prog, dmgr = env.demo, env.dmgr
    rng = np.random.default_rng(7)
    sizes = [7, 64, 1000, 4096, 65536, 1 << 20]
    weights = [0.25, 0.25, 0.2, 0.15, 0.1, 0.05]
    dtypes = [np.float32, np.float64, np.int32, np.float16]
    live = np.zeros(8, dtype=np.float32)
    dmgr.write(live)
    calls = 0
    while calls < 500:
        batch = []
        for _ in range(25):
            n = int(rng.choice(sizes, p=weights))
            da = dtypes[int(rng.integers(len(dtypes)))]
            db = dtypes[int(rng.integers(len(dtypes)))]
            a = (rng.random(n) * 4).astype(da)
            b = (rng.random(n) * 4).astype(db)
            c = np.zeros(n, dtype=np.float32)
            prog.sum_res(a, b, c, global_size=n, async_exec=True)
            batch.append((c, a.astype(np.float32) + b.astype(np.float32)))
            del a, b
            dmgr.find_buffer(live, dmgr.WRITE)  # a manager call: drains dead entries
            calls += 1
        prog.wait_all()
        for c, expected in batch:
            dmgr.read(c)
            np.testing.assert_allclose(c, expected, rtol=1e-3, atol=1e-2)
        del batch, c, expected


def test_async_resident_pingpong(env):
    """impressionist core/shims.py:64-73: resident ping-pong under async_mode()."""
    prog, dmgr = env.demo, env.dmgr
    n = 4096
    before = len(dmgr.buffers)
    a = _rand(n, seed=1)
    b = _rand(n, seed=2)
    c = np.zeros(n, dtype=np.float32)
    for arr in (a, b, c):
        dmgr.write(arr)
    expected = a.copy()
    with prog.async_mode():
        for _ in range(50):
            prog.sum_res(a, b, c, global_size=a.size, resident=(a, b, c))
            a, c = c, a
            expected = expected + b
    dmgr.read(a)
    np.testing.assert_allclose(a, expected, rtol=1e-5)
    del a, b, c, arr
    dmgr.count_leaked_buffers()  # one manager call
    assert len(dmgr.buffers) == before


def test_purge_releases_inflight(env):
    """D9: KernelManager.purge() waits for pending async work first."""
    prog, dmgr = env.imageops, env.dmgr
    x64 = np.linspace(0, 1, 4096, dtype=np.float64)
    for _ in range(5):
        prog.stretch(x64, 0.0, 1.0, 0.0, 2.0, async_exec=True)
    prog.purge()
    assert prog.pending_events == []
    assert prog._inflight == []
    assert dmgr.count_leaked_buffers() == 0


# --- Item 2: the guarded lookup ---------------------------------------------------------


def test_lookup_rejects_foreign_entry(env):
    dmgr = env.dmgr
    b = np.zeros(16, dtype=np.float32)
    other = np.ones(16, dtype=np.float32)
    sentinel = object()
    k = id(b)
    try:
        dmgr._store[k] = (weakref.KeyedRef(other, None, k), sentinel)
        assert dmgr._lookup(b) is None
        assert dmgr.find_buffer(b, dmgr.WRITE) is None
        with pytest.raises(ValueError):
            dmgr.find_buffer(b, dmgr.READ)
        tmp = np.zeros(3, dtype=np.float32)
        dead = weakref.KeyedRef(tmp, None, k)
        del tmp
        assert dead() is None
        dmgr._store[k] = (dead, sentinel)
        assert dmgr._lookup(b) is None
        assert dmgr.find_buffer(b, dmgr.WRITE) is None
    finally:
        dmgr._store.pop(k, None)


def test_find_buffer_is_centralized(env):
    """D4: one concrete find_buffer in the base class."""
    assert type(env.dmgr).find_buffer is DeviceManagerBase.find_buffer


def _force_collision(dmgr, old_n, old_dt, new_n, new_dt, tries=10_000):
    """Register X, drop it, and return a new array that has id(X). Failed attempts are dropped."""
    x = _rand(old_n, old_dt, seed=3)
    dmgr.write(x)
    k = id(x)
    del x
    for _ in range(tries):
        y = np.empty(new_n, dtype=new_dt)
        if id(y) == k:
            return y
        del y
    pytest.skip("no id collision in 10,000 tries")


@pytest.mark.parametrize(
    "old_n,old_dt,new_n,new_dt",
    [
        (16, np.float32, 1 << 20, np.float32),  # bigger
        (1 << 20, np.float32, 16, np.float32),  # smaller
        (4096, np.float64, 4096, np.float32),
        (4096, np.int32, 4096, np.float32),
        (1000, np.float32, 1000, np.float32),  # same
    ],
    ids=["bigger", "smaller", "f64-f32", "i32-f32", "same"],
)
def test_forced_id_collision(env, old_n, old_dt, new_n, new_dt):
    prog, dmgr = env.imageops, env.dmgr
    rng = np.random.default_rng(5)
    # lookup and read
    y = _force_collision(dmgr, old_n, old_dt, new_n, new_dt)
    assert dmgr.find_buffer(y, dmgr.WRITE) is None
    with pytest.raises(ValueError):
        dmgr.read(y)
    del y
    # kernel `in`
    y = _force_collision(dmgr, old_n, old_dt, new_n, new_dt)
    y[:] = rng.random(new_n).astype(new_dt)
    out = prog.stretch(y, 0.0, 1.0, 0.0, 2.0)
    np.testing.assert_allclose(out, np.clip(y, 0.0, 1.0) * np.float32(2.0), atol=1e-4)
    del y, out
    # resident input and output (histogram_res)
    y = _force_collision(dmgr, old_n, old_dt, new_n, new_dt)
    y[:] = rng.random(new_n).astype(new_dt)
    yf = y.astype(np.float32, copy=False)
    hst = np.zeros(256, dtype=np.int32)
    prog.histogram_res(yf, 0.0, 1.0, hst, resident=[yf, hst])
    prog.read(hst)
    expected = prog.histogram(yf, 0.0, 1.0, np.zeros(256, dtype=np.int32))
    assert int(hst.sum()) == new_n
    np.testing.assert_array_equal(hst, expected)


@pytest.mark.parametrize(
    "old_n,old_dt,new_n,new_dt",
    [
        (16, np.float32, 1 << 20, np.float32),  # bigger
        (1 << 20, np.float32, 16, np.float32),  # smaller
        (4096, np.float64, 4096, np.float32),
        (1000, np.float32, 1000, np.float32),  # same
    ],
    ids=["bigger", "smaller", "f64-f32", "same"],
)
def test_forced_id_collision_undrained(env, old_n, old_dt, new_n, new_dt):
    """The ``ref() is a`` guard on its own: the stale entry is still in the store.

    ``_lookup`` drains first, so in ``test_forced_id_collision`` the drain removes
    the dead array's entry before the guard runs. Here the drain is held off, as
    for a lookup that runs before another thread's drain, so the new array with
    the old id meets the old entry, and only the guard rejects it. Once the drain
    runs, it must leave the new array's entry alone (token match).
    """
    prog, dmgr = env.imageops, env.dmgr
    x = _rand(old_n, old_dt, seed=3)
    dmgr.write(x)
    k = id(x)
    stale = dmgr._store[k]
    dmgr._drain = lambda: None  # hold off the drain on this manager
    try:
        del x
        assert dmgr._dead, "the dead array's ref should be queued"
        y = None
        for _ in range(10_000):
            y = np.empty(new_n, dtype=new_dt)
            if id(y) == k:
                break
            del y
            y = None
        if y is None:
            pytest.skip("no id collision in 10,000 tries")
        assert (
            dmgr._store.get(k) is stale
        ), "the stale entry should still be in the store"
        assert dmgr._lookup(y) is None
        assert dmgr.find_buffer(y, dmgr.WRITE) is None
        with pytest.raises(ValueError):
            dmgr.read(y)
        # A kernel call registers y under the same key, replacing the stale entry.
        y[:] = np.random.default_rng(5).random(new_n).astype(new_dt)
        out = prog.stretch(y, 0.0, 1.0, 0.0, 2.0)
        np.testing.assert_allclose(
            out, np.clip(y, 0.0, 1.0) * np.float32(2.0), atol=1e-4
        )
        del out
    finally:
        del dmgr._drain
    stale = None
    dmgr.count_leaked_buffers()  # drains the old ref, whose entry y has replaced
    assert dmgr._lookup(y) is not None, "the drain removed the live array's entry"
    assert dmgr._store[k][0]() is y


# --- Item 4: fast buffer free -----------------------------------------------------------


def test_fast_free_without_purge(env):
    dmgr = env.dmgr
    live = np.zeros(8, dtype=np.float32)
    dmgr.write(live)
    x = _rand(1000)
    buf = dmgr.write(x)
    k = id(x)
    wr = weakref.ref(buf) if env.backend == "metal" else None
    del buf
    released = getattr(dmgr, "released", 0)
    purged = dmgr.purged
    del x
    dmgr.find_buffer(live, dmgr.WRITE)  # one manager call
    assert k not in dmgr.buffers, "the dead array's buffer is still registered"
    assert k not in dmgr._store
    assert dmgr.count_leaked_buffers() == 0
    assert dmgr.released == released + 1
    assert dmgr.purged == purged
    if wr is not None:
        assert wr() is None


def test_fast_free_one_ref_per_array(env):
    dmgr = env.dmgr
    n0 = len(dmgr._store)
    x = _rand(64)
    first = dmgr.write(x)
    ref0 = dmgr._store[id(x)][0]
    wr = weakref.ref(first) if env.backend == "metal" else None
    del first
    for i in range(10):
        dmgr.write(x)
        assert dmgr._store[id(x)][0] is ref0
        if i == 0 and wr is not None:
            assert wr() is None, "Metal's second write should drop the first buffer"
    dmgr.count_leaked_buffers()  # a drain leaves the live entry alone
    assert id(x) in dmgr._store
    assert len(dmgr._store) == n0 + 1


def test_resource_usage_drains_first(env):
    dmgr = env.dmgr
    count0 = dmgr.check_resource_usage(warn=False)["buffer_count"]
    stats0 = dmgr.stats()
    x = _rand(100)
    dmgr.write(x)
    del x
    assert dmgr.check_resource_usage(warn=False)["buffer_count"] == count0
    stats = dmgr.stats()
    assert stats["buffers"] == stats0["buffers"]
    assert stats["released"] == stats0.get("released", 0) + 1


def test_drain_skips_missing_entry(env):
    dmgr = env.dmgr
    x = _rand(100)
    dmgr.write(x)
    k = id(x)
    del x
    assert dmgr._dead, "the dead array's ref should be queued"
    dmgr._store.pop(k)  # as _clear_store or the purge sweep would
    released = dmgr.released
    dmgr.count_leaked_buffers()  # one manager call; must not raise
    assert dmgr.released == released


def test_compat_views(env):
    """N1: arrays and buffers stay as read-only views of the store."""
    dmgr = env.dmgr
    x = _rand(100)
    buf = dmgr.write(x)
    k = id(x)
    assert dmgr.arrays[k] is x
    assert dmgr.buffers[k] is buf
    n = len(dmgr.arrays)
    del x, buf
    assert k not in dmgr.arrays  # no manager call: as with a WeakValueDictionary
    assert len(dmgr.arrays) == n - 1
    with pytest.raises(TypeError):
        dmgr.buffers[1] = 0
    with pytest.raises(AttributeError):
        dmgr.buffers.clear()
    with pytest.raises(AttributeError):
        dmgr.arrays.clear()
    with pytest.raises(AttributeError):
        dmgr.arrays = {}


_RLOCK = type(threading.RLock())


@pytest.mark.parametrize("backend", BACKENDS)
def test_reentrant_release(backend):
    """D2: a release that calls back into the manager, on the thread holding the lock.

    Every manager call runs in a worker thread, so a lock that isn't re-entrant
    fails at the join's timeout instead of hanging the run (this test doesn't use
    the ``env`` fixture, whose purge would deadlock on the main thread). The
    release hook looks at the store when it is called, so "the store is updated
    before anything is released" is checked whatever id the next array gets.
    """
    e = _load(backend)
    dmgr = e.dmgr
    gc.collect()
    state: dict = {}
    calls: list = []
    errors: list = []

    def free_hook(buf):
        calls.append(buf)
        if len(calls) == 1:
            entry = dmgr._store.get(state["k"])
            if entry is not None and entry[0] is state["ref"]:
                errors.append(
                    "a buffer was released before its store entry was removed"
                )
            dmgr.write(state["y"])  # re-enters the manager while purge holds the lock

    def run():
        try:
            e.imageops.wait_all()
            e.demo.wait_all()
            dmgr.purge()
            x = _rand(100, seed=11)
            dmgr.write(x)
            state["k"] = id(x)
            state["ref"] = dmgr._store[id(x)][0]
            del x  # its ref is queued
            state["y"] = _rand(100, seed=12)
            dmgr._free_buffer = free_hook
            try:
                dmgr.purge_buffers()
            finally:
                del dmgr._free_buffer
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    gc.disable()
    try:
        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout=30)
        deadlocked = t.is_alive()
    finally:
        gc.enable()
    assert isinstance(
        dmgr._lock, _RLOCK
    ), f"the manager's lock is a {type(dmgr._lock).__name__}, not an RLock (deadlocked: {deadlocked})"
    assert not deadlocked, "purge_buffers deadlocked on a re-entrant release"
    assert not errors, errors
    assert calls, "the dead array's buffer was not released"
    y = state["y"]
    assert dmgr.arrays.get(id(y)) is y
    assert dmgr._lookup(y) is not None
    assert all(ref() is not None for ref, _ in dmgr._store.values())


class _FakeDeviceManager(DeviceManagerBase):
    """Bookkeeping only: no device."""

    def __init__(self):
        super().__init__(None, None, "fake", None)

    def make_buffer(self, a):
        raise NotImplementedError

    def read_buffer(self, a):
        raise NotImplementedError

    def write_buffer(self, a, wait=True):
        raise NotImplementedError

    def build(self, source, options=None):
        raise NotImplementedError


class _Sentinel:
    __slots__ = ("owner",)

    def __init__(self, owner):
        self.owner = owner


def test_store_threads_smoke():
    """Register, look up and drop from 8 threads; a smoke test for free-threaded builds."""
    dmgr = _FakeDeviceManager()
    errors = []

    def worker(t):
        try:
            for i in range(5000):
                a = np.empty(1 + (i + t) % 7, dtype=np.float32)
                s = _Sentinel(t)
                dmgr._register(a, s)
                got = dmgr._lookup(a)
                if got is not s:
                    errors.append((t, i, got))
                del a, s, got
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:5]
    dmgr._drain()
    assert len(dmgr._store) == 0

#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.

import subprocess
import sys
import textwrap
import threading
import uuid

import numpy as np
import pytest
from matplotlib import pyplot as plt

from yapocis.core import get_backends, get_test_backends
from yapocis.lib.bound_kernels import BoundKernels
from yapocis.lib.convolve import Convolve1D
from yapocis.utils.stereo_typing import *

from . import StopWatcher


@pytest.mark.smoke
@pytest.mark.parametrize("backend", get_test_backends())
def test_convolve(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    a = np.random.uniform(-1, 1, size=1000)
    cnv = Convolve1D(name="convolve", conv=[0.01] * 100)
    with StopWatcher("convolve", kernel=cnv) as sw:
        with sw.checkpoint("convolve_fast"):
            b = cnv.convolve_fast(a, 100)
    plt.plot(b)
    plt.show()


# #64: kernels generated from the context (Convolve1D, convolvesep) must use their own
# weights, even when an earlier kernel had the same name. Each test makes its kernel names
# unique, so test order inside one pytest process can't hide or cause a failure.

K1 = [0.25, 0.5, 0.25]
K2 = [1.0, 0.0, -1.0, 0.0, 2.0]

needs_numbax = pytest.mark.skipif(
    "numbax" not in get_backends(), reason="numbax backend not available"
)


def _tag() -> str:
    return uuid.uuid4().hex[:8]


def _run(owner, name, a, *extra):
    return np.asarray(getattr(owner, name)(a.copy(), *extra))


@pytest.mark.parametrize("backend", get_test_backends())
def test_second_kernel_of_the_same_name_uses_its_own_weights(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    a = np.arange(4096, dtype=np.float32) ** 0.5
    shared, fresh = "k" + _tag(), "f" + _tag()
    c1 = Convolve1D(name=shared, conv=K1)
    r1 = _run(c1, shared, a)
    c2 = Convolve1D(name=shared, conv=K2)
    r2 = _run(c2, shared, a)
    rf = _run(Convolve1D(name=fresh, conv=K2), fresh, a)
    assert np.array_equal(r2, rf), "second same-name kernel did not use its own weights"
    assert not np.array_equal(r1, r2)
    assert np.array_equal(_run(c1, shared, a), r1), "first kernel changed weights"


# numba only: the GPU _fast kernels give partial output, a separate bug (#80)
@needs_numbax
def test_numba_fast_variant_of_the_same_name_uses_its_own_weights(monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", "numbax")
    a = np.arange(4096, dtype=np.float32) ** 0.5
    shared, fresh = "k" + _tag(), "f" + _tag()
    Convolve1D(name=shared, conv=K1)
    c2 = Convolve1D(name=shared, conv=K2)
    cf = Convolve1D(name=fresh, conv=K2)
    assert np.array_equal(
        _run(c2, shared + "_fast", a, 64), _run(cf, fresh + "_fast", a, 64)
    )


@pytest.mark.parametrize("backend", get_test_backends())
def test_second_convolvesep_set_with_shared_names_uses_its_own_taps(
    backend, monkeypatch
):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    class Sep(BoundKernels):
        INTERFACE = "convolvesep"

    shared, fresh = f"g{_tag()}5", f"h{_tag()}5"
    w1 = np.array([1, 4, 6, 4, 1], np.float32) / 16
    w2 = np.array([1, 0, -2, 0, 1], np.float32)
    img = np.random.default_rng(0).random((64, 48), dtype=np.float32)

    def run_h(name, w):
        s = Sep(convs=[(name, w)])
        i, o = img.copy(), np.zeros_like(img)
        s.write(i)
        s.write(o)
        getattr(s, name + "_h")(i, o)
        s.read(o)
        return o

    run_h(shared, w1)
    assert np.array_equal(run_h(shared, w2), run_h(fresh, w2))


# A kernel named like a module global must not bind or overwrite it. Run in a child
# process: before the fix it broke convolve.py for the rest of the process.
@needs_numbax
def test_numba_kernel_name_cannot_bind_or_overwrite_a_module_global():
    code = textwrap.dedent("""
        import os, sys
        os.environ["YAPOCIS_BACKEND"] = "numbax"
        import numpy as np
        from yapocis.lib.convolve import Convolve1D
        a = np.arange(64, dtype=np.float32)
        k = [1.0, 0.0, -1.0, 0.0, 2.0]
        ref = np.asarray(Convolve1D(name="before", conv=k).before(a.copy()))
        try:
            out = np.asarray(Convolve1D(name="np", conv=k).np(a.copy()))
            assert np.array_equal(out, ref), "name='np' did not convolve"
            after = np.asarray(Convolve1D(name="after", conv=k).after(a.copy()))
            assert np.array_equal(after, ref), "module broken after name='np'"
        except Exception as e:
            print("FAIL", type(e).__name__, e)
            sys.exit(1)
        print("OK")
        """)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr[-2000:]


@needs_numbax
def test_numba_concurrent_same_name_kernels_use_their_own_weights(monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", "numbax")
    a = np.arange(4096, dtype=np.float32) ** 0.5
    shared = "t" + _tag()
    weights = [[1.0, float(i + 1), 1.0] for i in range(8)]
    refs = [
        _run(Convolve1D(name=f"r{i}{shared}", conv=w), f"r{i}{shared}", a)
        for i, w in enumerate(weights)
    ]
    barrier = threading.Barrier(len(weights))
    results = [None] * len(weights)

    def work(i):
        barrier.wait()
        results[i] = _run(Convolve1D(name=shared, conv=weights[i]), shared, a)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(weights))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wrong = [i for i in range(len(weights)) if not np.array_equal(results[i], refs[i])]
    assert (
        not wrong
    ), f"{len(wrong)} of {len(weights)} threads got other weights: {wrong}"


# rustcpu caches loaded programs by interface and context for the whole process. The key
# used repr() of the weights, which keeps numpy's print precision (8 decimal places),
# elides arrays over 1000 elements and follows np.set_printoptions, so different weights
# shared one program. Each case's weights are float64 ndarrays with equal reprs (a Python
# list never collided: json writes floats exactly).


def _key_case_ninth_digit():
    w1 = np.array([0.1234567812, 0.25, 0.5, 0.25, 0.1234567812])
    w2 = w1.copy()
    w2[0] = w2[4] = 0.1234567849
    return "sep", w1, w2, None


def _key_case_small_taps():
    w1 = np.array([0.00123456781, 0.5, 0.00123456781])
    w2 = w1.copy()
    w2[0] = w2[2] = 0.00123456849
    return "conv", w1, w2, None


def _key_case_long():
    w1 = np.zeros(1001)
    w1[0] = w1[-1] = w1[500] = 1.0
    w2 = w1.copy()
    w2[500] = 2.0
    return "conv", w1, w2, None


def _key_case_printoptions():
    return "conv", np.array([0.5012, 0.25]), np.array([0.5014, 0.25]), 3


@pytest.mark.skipif(
    "rustcpu" not in get_backends(), reason="rustcpu backend not available"
)
@pytest.mark.parametrize(
    "case",
    [
        _key_case_ninth_digit,
        _key_case_small_taps,
        _key_case_long,
        _key_case_printoptions,
    ],
    ids=["ninth_digit", "small_taps_seventh_digit", "1001_taps", "printoptions"],
)
def test_rustcpu_same_name_kernel_gets_its_own_program(case):
    import contextlib

    from yapocis.rustcpu.kernel_manager import KernelManager

    kind, w1, w2, precision = case()
    shared, fresh = f"rk{_tag()}", f"rf{_tag()}"
    rng = np.random.default_rng(64)

    if kind == "sep":

        class Sep(BoundKernels):
            INTERFACE = "convolvesep"

        img = rng.random((64, 48), dtype=np.float32)

        def run(name, w):
            s = Sep(convs=[(name, w)], backend="rustcpu")
            i, o = img.copy(), np.zeros_like(img)
            s.write(i)
            s.write(o)
            getattr(s, name + "_h")(i, o)
            s.read(o)
            return o

    else:
        a = rng.random(4096, dtype=np.float32)

        def run(name, w):
            return _run(Convolve1D(name=name, conv=w, backend="rustcpu"), name, a)

    printing = (
        np.printoptions(precision=precision) if precision else contextlib.nullcontext()
    )
    with printing:
        assert repr(w1) == repr(w2), "the case no longer collides under repr()"
        r1 = run(shared, w1)
        before = len(KernelManager._programs)
        r2 = run(shared, w2)
        grew = len(KernelManager._programs) - before
        rf = run(fresh, w2)
    same = np.array_equal(r2, rf)
    assert same, "the second same-name kernel did not use its own weights"
    assert grew == 1, "the second same-name kernel reused the first one's program"
    if kind == "conv" and precision is None:
        assert not np.array_equal(r1, r2)

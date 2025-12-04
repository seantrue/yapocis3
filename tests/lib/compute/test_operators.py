#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.

"""Tests for operators interface - binary operations (add, sub, mul, divop)"""

import numpy as np
import pytest

from yapocis.core import get_test_backends
from yapocis.lib.operators import Operators

from . import StopWatcher


@pytest.mark.parametrize("backend", get_test_backends())
@pytest.mark.smoke
def test_add(backend, monkeypatch):
    """Test addition operator"""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    ops = Operators()
    a = np.random.uniform(0, 10, 1000).astype(np.float32)
    b = np.random.uniform(0, 10, 1000).astype(np.float32)

    with StopWatcher("add", kernel=ops) as sw:
        with sw.checkpoint("add"):
            result = ops.add(a, b)

    expected = a + b
    np.testing.assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize("backend", get_test_backends())
def test_sub(backend, monkeypatch):
    """Test subtraction operator"""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    ops = Operators()
    a = np.random.uniform(0, 10, 1000).astype(np.float32)
    b = np.random.uniform(0, 10, 1000).astype(np.float32)

    with StopWatcher("sub", kernel=ops) as sw:
        with sw.checkpoint("sub"):
            result = ops.sub(a, b)

    expected = a - b
    np.testing.assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize("backend", get_test_backends())
def test_mul(backend, monkeypatch):
    """Test multiplication operator"""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    ops = Operators()
    a = np.random.uniform(0, 10, 1000).astype(np.float32)
    b = np.random.uniform(0, 10, 1000).astype(np.float32)

    with StopWatcher("mul", kernel=ops) as sw:
        with sw.checkpoint("mul"):
            result = ops.mul(a, b)

    expected = a * b
    np.testing.assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize("backend", get_test_backends())
def test_divop(backend, monkeypatch):
    """Test division operator"""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    ops = Operators()
    a = np.random.uniform(1, 10, 1000).astype(np.float32)
    b = np.random.uniform(1, 10, 1000).astype(np.float32)

    with StopWatcher("divop", kernel=ops) as sw:
        with sw.checkpoint("divop"):
            result = ops.divop(a, b)

    expected = a / b
    np.testing.assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize("backend", get_test_backends())
def test_div_alias(backend, monkeypatch):
    """Test div alias (backward compatibility for divop)"""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    ops = Operators()
    a = np.random.uniform(1, 10, 1000).astype(np.float32)
    b = np.random.uniform(1, 10, 1000).astype(np.float32)

    with StopWatcher("div_alias", kernel=ops) as sw:
        with sw.checkpoint("div"):
            result = ops.div(a, b)

    expected = a / b
    np.testing.assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize("backend", get_test_backends())
def test_add_resident(backend, monkeypatch):
    """Test addition with resident buffers"""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    ops = Operators()
    a = np.random.uniform(0, 10, 1000).astype(np.float32)
    b = np.random.uniform(0, 10, 1000).astype(np.float32)
    output = np.zeros_like(a)

    with StopWatcher("add_resident", kernel=ops) as sw:
        with sw.checkpoint("write"):
            ops.write(a)
            ops.write(b)
            ops.write(output)

        with sw.checkpoint("add_res"):
            ops.add_res(a, b, output)

        with sw.checkpoint("read"):
            result = ops.read(output)

    expected = a + b
    np.testing.assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize("backend", get_test_backends())
def test_div_resident(backend, monkeypatch):
    """Test division with resident buffers"""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)

    ops = Operators()
    a = np.random.uniform(1, 10, 1000).astype(np.float32)
    b = np.random.uniform(1, 10, 1000).astype(np.float32)
    output = np.zeros_like(a)

    with StopWatcher("div_resident", kernel=ops) as sw:
        with sw.checkpoint("write"):
            ops.write(a)
            ops.write(b)
            ops.write(output)

        with sw.checkpoint("div_res"):
            ops.div_res(a, b, output)

        with sw.checkpoint("read"):
            result = ops.read(output)

    expected = a / b
    np.testing.assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize("backend", get_test_backends())
def test_all_operators(backend, monkeypatch):
    """Test all operators in sequence"""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    ops = Operators()

    a = np.random.uniform(1, 10, (100, 100)).astype(np.float32)
    b = np.random.uniform(1, 10, (100, 100)).astype(np.float32)

    with StopWatcher("all_operators", kernel=ops) as sw:
        with sw.checkpoint("add"):
            r1 = ops.add(a, b)
        np.testing.assert_allclose(r1, a + b, rtol=1e-6)

        with sw.checkpoint("sub"):
            r2 = ops.sub(a, b)
        np.testing.assert_allclose(r2, a - b, rtol=1e-6)

        with sw.checkpoint("mul"):
            r3 = ops.mul(a, b)
        np.testing.assert_allclose(r3, a * b, rtol=1e-6)

        with sw.checkpoint("div"):
            r4 = ops.divop(a, b)
        np.testing.assert_allclose(r4, a / b, rtol=1e-6)


@pytest.mark.parametrize("backend", get_test_backends())
def test_operators_2d_arrays(backend, monkeypatch):
    """Test operators with 2D arrays"""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    ops = Operators()

    a = np.random.uniform(0, 10, (512, 512)).astype(np.float32)
    b = np.random.uniform(0, 10, (512, 512)).astype(np.float32)

    with StopWatcher("operators_2d", kernel=ops) as sw:
        with sw.checkpoint("add_2d"):
            result = ops.add(a, b)

        expected = a + b
        np.testing.assert_allclose(result, expected, rtol=1e-6)
        assert result.shape == a.shape


@pytest.mark.parametrize("backend", get_test_backends())
def test_operators_edge_cases(backend, monkeypatch):
    """Test operators with edge cases"""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    ops = Operators()

    # Test with zeros
    a = np.zeros(100, dtype=np.float32)
    b = np.ones(100, dtype=np.float32)

    result = ops.add(a, b)
    np.testing.assert_allclose(result, b, rtol=1e-6)

    # Test with negative numbers
    a = np.random.uniform(-10, 0, 100).astype(np.float32)
    b = np.random.uniform(-10, 0, 100).astype(np.float32)

    result = ops.mul(a, b)
    expected = a * b
    np.testing.assert_allclose(result, expected, rtol=1e-6)

    # Test division by small numbers
    a = np.ones(100, dtype=np.float32)
    b = np.full(100, 0.001, dtype=np.float32)

    result = ops.divop(a, b)
    expected = a / b
    np.testing.assert_allclose(result, expected, rtol=1e-5)

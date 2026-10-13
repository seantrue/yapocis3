"""The gradient kernel's values, on every backend and the numbax pure-numpy fallback.

Convention (all backends): central differences over 2 * reach,
    d_row = I[row + r, col] - I[row - r, col],  d_col = I[row, col + r] - I[row, col - r],
grad = hypot(d_row, d_col), theta = atan2(d_col, d_row) / pi in (-1, 1], measured from +row
toward +col, and both are 0 within `reach` of the border.
"""

import numpy as np
import pytest

from yapocis.core import get_test_backends
from yapocis.lib.gradient import Gradient
from yapocis.numbax.kernel_manager import _load_numba_kernel_module

H, W = 61, 97  # non-square and odd, so a transposed axis shows up


def reference(img, r):
    h, w = img.shape
    grad = np.zeros_like(img)
    theta = np.zeros_like(img)
    d_col = img[r:h - r, 2 * r:] - img[r:h - r, :w - 2 * r]
    d_row = img[2 * r:, r:w - r] - img[:h - 2 * r, r:w - r]
    grad[r:h - r, r:w - r] = np.hypot(d_row, d_col)
    theta[r:h - r, r:w - r] = np.arctan2(d_col, d_row) / np.pi
    return grad, theta


def images():
    rows, cols = np.mgrid[0:H, 0:W].astype(np.float32)
    return {
        "ramp_cols": cols / W,
        "ramp_rows": rows / H,
        "ramp_diagonal": (rows + cols) / (H + W),
        "random": np.random.default_rng(0).random((H, W), dtype=np.float32),
    }


def angle_error(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 2.0 - d)  # -1 and +1 are the same angle


def check(grad, theta, img, r):
    ref_grad, ref_theta = reference(img, r)
    np.testing.assert_allclose(grad, ref_grad, atol=1e-5)
    inner = (slice(r, H - r), slice(r, W - r))
    # Angles are only defined where the gradient is not ~0
    defined = ref_grad[inner] > 1e-4
    assert angle_error(theta[inner], ref_theta[inner])[defined].max() < 1e-4
    border = np.ones_like(img, dtype=bool)
    border[inner] = False
    assert not grad[border].any() and not theta[border].any()


@pytest.mark.parametrize("backend", get_test_backends())
@pytest.mark.parametrize("reach", [1, 3])
@pytest.mark.parametrize("name", list(images()))
def test_gradient_values(backend, reach, name, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    img = images()[name]
    grad, theta = Gradient().gradient(img, reach)
    check(np.asarray(grad), np.asarray(theta), img, reach)


@pytest.mark.parametrize(("name", "expected"), [("ramp_cols", 0.5), ("ramp_rows", 0.0),
                                                 ("ramp_diagonal", 0.25)])
@pytest.mark.parametrize("backend", get_test_backends())
def test_gradient_directions(backend, name, expected, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    _, theta = Gradient().gradient(images()[name], 3)
    assert float(theta[H // 2, W // 2]) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("numba", [True, False], ids=["numba", "numpy_fallback"])
@pytest.mark.parametrize("reach", [1, 3])
@pytest.mark.parametrize("name", list(images()))
def test_numbax_gradient_paths(numba, reach, name, monkeypatch):
    module = _load_numba_kernel_module("gradient")
    monkeypatch.setattr(module, "HAVE_NUMBA", numba)
    img = images()[name]
    grad, theta = np.zeros_like(img), np.zeros_like(img)
    module.gradient(W, H, img, reach, grad, theta)
    check(grad, theta, img, reach)

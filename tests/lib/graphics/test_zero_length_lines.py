"""A zero-length line draws a dot at its start on every backend, as numbax always has. Metal,
OpenCL and CUDA computed the projection as 0/0 and skipped the line, so a stroke made of
single-point segments vanished on the GPU but not on numbax."""

import numpy as np
import pytest

from yapocis.core import get_test_backends
from yapocis.lib.graphics import Graphics1
from yapocis.lib.graphics4 import Graphics4

LINE = [2, 5, 12, 5]
DOT = [7, 9, 7, 9]


def draw1(coords):
    img = np.zeros((24, 24), np.float32)
    g = Graphics1()
    g.lines(img, coords, distance=1.0, color=1.0, alpha=1.0, aa_width=0.0)
    return g.read(img)


def draw4(coords):
    img = np.zeros((24, 24, 4), np.float32)
    g = Graphics4()
    g.lines(img, coords, distance=1.0, colors=[[1.0, 1.0, 1.0, 1.0]] * len(coords), aa_width=0.0)
    return g.read(img)


@pytest.mark.parametrize("backend", get_test_backends())
@pytest.mark.parametrize("draw", [draw1, draw4], ids=["graphics", "graphics4"])
@pytest.mark.parametrize("coords", [[DOT], [LINE, DOT], [DOT, LINE]], ids=["dot", "line_dot", "dot_line"])
def test_zero_length_line_is_a_dot(backend, draw, coords, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    got = draw(coords)
    monkeypatch.setenv("YAPOCIS_BACKEND", "numbax")
    want = draw(coords)
    ink = got if got.ndim == 2 else got[..., 0]
    assert ink[DOT[1], DOT[0]] > 0.5, f"{backend}: no dot at the zero-length line"
    np.testing.assert_array_equal(got, want)

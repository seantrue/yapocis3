"""KernelDraw4 primitives draw the same pixels on every backend, and mesh triangles
draw each shared-edge pixel once.

Numba truncated float coordinates to integers, and its line kernel used a narrower
coverage rule than OCL/Metal/CUDA. Triangles count pixels on any edge as inside, so a
translucent mesh drawn with them was blended twice along each shared edge; mesh
triangles use a top-left fill rule instead.
"""

import numpy as np
import pytest

from yapocis.core import get_test_backends
from yapocis.lib.kerneldraw4 import KernelDraw4

TRANSLUCENT = [0.0, 0.0, 0.0, 0.6]


def draw(backend, monkeypatch, primitive, *args, **kwargs):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    kd = KernelDraw4(np.ones((50, 50, 4), np.float32))
    for call in primitive:
        getattr(kd, call[0])(*call[1:], color=TRANSLUCENT, **kwargs)
    kd.render()
    return kd.read()[..., 0].copy()


TRIANGLE = np.array([10.3, 10.7, 10.3, 40.2, 40.6, 10.7], np.float32)

# (row, col) coordinates at fractional positions, so coordinate truncation shows
PRIMITIVES = {
    "triangle": [("triangle", TRIANGLE)],
    "mesh_triangle": [("mesh_triangle", TRIANGLE)],
    "line": [("line", np.array([5.2, 5.7, 44.1, 40.3], np.float32))],
    "vertical_line": [("line", np.array([5.0, 20.5, 45.0, 20.5], np.float32))],
    "rect": [("rect", np.array([10.4, 12.6, 30.5, 35.2], np.float32))],
    "circle": [("circle", np.array([25.3, 24.6, 9.5], np.float32))],
}

# A square as two triangles sharing the diagonal, at integer coordinates so pixels
# lie exactly on the shared edge
SQUARE = (
    np.array([10, 10, 10, 40, 40, 40], np.float32),
    np.array([10, 10, 40, 40, 40, 10], np.float32),
)


@pytest.mark.parametrize("name", list(PRIMITIVES))
@pytest.mark.parametrize("aa_width", [0.0, 0.5])
def test_backends_draw_same_pixels(name, aa_width, monkeypatch):
    backends = get_test_backends()
    if len(backends) < 2:
        pytest.skip("needs two backends")
    kwargs = {"aa_width": aa_width}
    if PRIMITIVES[name][0][0] == "line":
        kwargs["width"] = 1.0
    ref = draw(backends[0], monkeypatch, PRIMITIVES[name], **kwargs)
    for backend in backends[1:]:
        got = draw(backend, monkeypatch, PRIMITIVES[name], **kwargs)
        np.testing.assert_allclose(
            got, ref, atol=1e-5, err_msg=f"{backends[0]} vs {backend}"
        )


@pytest.mark.parametrize("backend", get_test_backends())
def test_mesh_shared_edge_blended_once(backend, monkeypatch):
    # Every pixel of the square is covered exactly once: uniform 1 - 0.6
    out = draw(
        backend, monkeypatch, [("mesh_triangle", t) for t in SQUARE], aa_width=0.0
    )
    np.testing.assert_allclose(out[11:40, 11:40], 0.4, atol=1e-5)


@pytest.mark.parametrize("backend", get_test_backends())
def test_triangles_keep_inclusive_edges(backend, monkeypatch):
    # Plain triangles still draw pixels on every edge: the shared diagonal is covered
    # by both, and the vertex and edge pixels of a lone triangle are drawn
    out = draw(backend, monkeypatch, [("triangle", t) for t in SQUARE], aa_width=0.0)
    assert np.isclose(out[25, 25], 0.4 * 0.4)  # on the shared diagonal, blended twice
    lone = draw(
        backend,
        monkeypatch,
        [("triangle", np.array([10, 10, 10, 22, 22, 16], np.float32))],
        aa_width=0.0,
    )
    assert lone[10, 10] < 0.5  # first vertex drawn
    assert ((lone < 0.5).sum()) == 85


# Zero-area triangles: collinear corners, along a diagonal and along a row, and two
# equal corners. The GPU kernels' sign test counted every pixel on the line through
# such a triangle's corners as inside, drawing it across the whole image (numbax drew
# nothing); triangles now skip zero area, as mesh_triangles already did.
DEGENERATE = [
    [10, 10, 20, 20, 30, 30],
    [10, 32, 20, 32, 30, 32],
    [10, 10, 10, 10, 20, 15],
]


def _rustcpu_draws_degenerate() -> bool:
    """True only while the installed yapocis_rs still ports the old `triangles`, which
    draws a zero-area triangle's line. Temporary: remove with the xfail below once
    CI's YAPOCIS_RS_REF includes yapocis-rs's matching CPU-port fix."""
    from yapocis.lib.graphics import Graphics1

    image = np.ones((16, 16), np.float32)
    Graphics1(backend="rustcpu").triangles(
        image, [[2, 2, 4, 4, 6, 6]], colors=[0.0], alphas=[1.0]
    )
    return bool((image != 1).any())


@pytest.mark.parametrize("backend", get_test_backends())
@pytest.mark.parametrize("channels", [1, 4])
@pytest.mark.parametrize("kernel", ["triangles", "mesh_triangles"])
@pytest.mark.parametrize("aa_width", [0.0, 1.0])
def test_degenerate_triangles_draw_nothing(backend, channels, kernel, aa_width):
    from yapocis.lib.graphics import Graphics1
    from yapocis.lib.graphics4 import Graphics4

    if backend == "rustcpu" and kernel == "triangles" and _rustcpu_draws_degenerate():
        pytest.xfail("installed yapocis_rs predates yapocis-rs's zero-area triangles fix")
    if channels == 1:
        g, shape = Graphics1(backend=backend), (64, 64)
        colors = {"colors": [0.0], "alphas": [1.0]}
    else:
        g, shape = Graphics4(backend=backend), (64, 64, 4)
        colors = {"colors": [[0.0, 0.0, 0.0, 1.0]]}
    for tri in DEGENERATE:
        image = np.ones(shape, np.float32)
        getattr(g, kernel)(image, [tri], aa_widths=[aa_width], **colors)
        assert (image == 1).all(), f"{backend} {kernel} drew {tri}"
    # A proper triangle still draws, and only inside its box
    image = np.ones(shape, np.float32)
    getattr(g, kernel)(image, [[10, 10, 30, 12, 15, 25]], aa_widths=[aa_width], **colors)
    drawn = (image != 1) if channels == 1 else (image != 1).any(axis=2)
    rows, cols = np.nonzero(drawn)
    assert len(rows) > 100
    assert cols.min() >= 10 and cols.max() <= 30 and rows.min() >= 10 and rows.max() <= 25

"""truetrace.fill_regions / trace_mask: traced shapes rasterised back through the
mesh_triangles kernel (the potrace replacement used by Virtual Impressionist)."""

import numpy as np
import pytest

from yapocis.lib import truetrace


def _ring_mask(h=64, w=80):
    mask = np.zeros((h, w), np.uint8)
    mask[10:50, 12:68] = 1  # a block
    mask[22:38, 30:50] = 0  # with a hole
    mask[55:58, 5:8] = 1  # a 9 px speckle
    return mask


def test_fill_regions_square_with_hole_matches_mask():
    mask = _ring_mask()
    rings = truetrace.trace_bitmap(mask, simplify_tol_px=0.0, min_area_px2=0.0)
    regions = truetrace.build_regions_with_holes(rings)
    assert len(regions) == 2 and len(regions[0]["holes"]) + len(regions[1]["holes"]) == 1
    img = truetrace.fill_regions(regions, mask.shape, curve="none")
    assert img.dtype == np.float32 and img.shape == mask.shape
    ink = img == 0.0
    # marching squares at 0.5 on the mask puts the contour half a pixel outside
    # the ink pixels, so every ink pixel is inside and no paper pixel is
    assert ink[mask == 1].all()
    assert not ink[mask == 0].any()


@pytest.mark.parametrize("engine", ["cpu", "gpu"])
def test_trace_mask_smooths_and_drops_speckles(engine):
    mask = _ring_mask()
    img = truetrace.trace_mask(mask, iso=0.0, min_area_px2=4.0, engine=engine)
    ink = img == 0.0
    assert ink[55:58, 5:8].all()  # the 9 px speckle (area 8.5 > 4) stays
    assert ink[12:48, 14:28].all()  # the block's body is ink
    assert not ink[24:36, 32:48].any()  # the hole is paper
    assert np.unique(img).tolist() in ([0.0, 1.0], [1.0])


def test_cpu_and_gpu_engines_agree_on_the_block():
    mask = _ring_mask()
    a = truetrace.trace_mask(mask, engine="cpu", simplify_tol_px=0.0)
    b = truetrace.trace_mask(mask, engine="gpu", simplify_tol_px=0.0)
    # the jump flood and the exact EDT agree on these distances
    assert (a == b).all()


def _thin_strokes(h=96, w=96):
    """1 px strokes, as the pen styles draw them: a diagonal, a ring and a comb."""
    mask = np.zeros((h, w), np.uint8)
    for i in range(10, 80):
        mask[i, i] = mask[i, i + 1] = 1
    yy, xx = np.mgrid[:h, :w]
    mask[np.abs(np.hypot(yy - 48, xx - 60) - 25) < 0.6] = 1
    mask[60:90, 5:40:3] = 1
    mask[60, 5:40] = 1
    return mask


@pytest.mark.parametrize("engine", ["cpu", "gpu"])
@pytest.mark.parametrize("iso", [0.5, 0.75, 0.9, 1.0, 1.5])
def test_trace_mask_inward_iso_draws_no_ink_off_the_strokes(engine, iso):
    # Until 2026-10-21 chaining snapped endpoints to 0.25 px: above iso 0.5 the
    # two outlines of a 1 px stroke fused, rings broke open, and the even-odd
    # fill drew wedges of ink across the paper. At iso 1.0 the contour also ran
    # through the strokes' pixel centres.
    from scipy import ndimage

    mask = _thin_strokes()
    near = ndimage.binary_dilation(mask, iterations=2)
    img = truetrace.trace_mask(mask, iso=iso, curve="none", engine=engine)
    assert not ((img == 0.0) & ~near).any()


def test_chain_segments_exact_keeps_close_contours_apart():
    # two vertical segments 0.25 px apart, which a 0.25 px snap would merge
    segs = np.array([[1.125, 0, 1.125, 1], [1.125, 1, 1.125, 2],
                     [0.875, 0, 0.875, 1], [0.875, 1, 0.875, 2]], np.float32)
    exact = truetrace.chain_segments_to_polylines(truetrace.sort_segments(segs))
    assert sorted(len(p) for p in exact) == [3, 3]
    assert {float(p[0, 0]) for p in exact} == {0.875, 1.125}

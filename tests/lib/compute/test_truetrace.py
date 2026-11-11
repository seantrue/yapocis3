from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from yapocis.core import get_backends, get_test_backends
from yapocis.lib.truetrace import (
    TrueTrace,
    build_circle_mask,
    build_regions_with_holes,
    chain_segments_to_polylines,
    close_and_filter_polylines,
    douglas_peucker,
    marching_squares_cpu,
    off_field_iso,
    sdf_from_mask_cpu,
    sort_segments,
    trace_bitmap,
)

from ...test_truetrace_fill import _ring_mask, _thin_strokes
from . import (
    ASIZE,
    StopWatcher,
)


def rasterize_regions(regions, size):
    """Rasterize regions-with-holes to a binary mask using PIL.
    Coordinates are in pixel space (origin at top-left).
    """
    w, h = size
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    for reg in regions:
        outer = [(float(x), float(y)) for (x, y) in reg["outer"]]
        draw.polygon(outer, fill=1)
        for hole in reg["holes"]:
            hole_pts = [(float(x), float(y)) for (x, y) in hole]
            draw.polygon(hole_pts, fill=0)
    return (np.array(img) > 0).astype(np.uint8)


def ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    """Simple global SSIM for 2D arrays (no windowing)."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a /= 1.0 if a.max() <= 1 else 255.0
    b /= 1.0 if b.max() <= 1 else 255.0
    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a2 = a.var()
    sigma_b2 = b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sigma_a2 + sigma_b2 + c2)
    return float(num / den) if den != 0 else 1.0


def render_segments_to_array(
    segments: np.ndarray, size, line_width: int = 1, aa: int = 4
) -> np.ndarray:
    """Render line segments into a grayscale image array using PIL.

    segments: array-like of shape (N, 4) with (x0, y0, x1, y1) in pixel coordinates.
    size: (width, height) of target image.
    line_width: line thickness in pixels.
    aa: supersampling factor for antialiasing.
    """
    w, h = size
    scale = max(1, int(aa))
    img = Image.new("L", (w * scale, h * scale), 0)
    draw = ImageDraw.Draw(img)
    for s in segments:
        x0, y0, x1, y1 = map(float, s[:4])
        draw.line(
            [
                (x0 + 0.5) * scale,
                (y0 + 0.5) * scale,
                (x1 + 0.5) * scale,
                (y1 + 0.5) * scale,
            ],
            fill=255,
            width=max(1, int(line_width * scale)),
        )
    if scale != 1:
        img = img.resize((w, h), resample=Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


@pytest.mark.smoke
@pytest.mark.parametrize("backend", get_test_backends())
def test_truetrace_matches_original_fill_iou_ssim(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    # Create a simple binary mask (circle)
    W = H = ASIZE
    mask = build_circle_mask(W, H, r=min(W, H) // 3)

    # Original CPU pipeline → regions
    rings_cpu = trace_bitmap(
        mask, simplify_tol_px=0.75, min_area_px2=4.0, chain_tol_px=0.25
    )
    regions_cpu = build_regions_with_holes(rings_cpu)

    # New GPU pipeline → segments → rings → regions
    tracer = TrueTrace()
    with StopWatcher("truetrace", kernel=tracer) as sw:
        with sw.checkpoint("trace_segments"):
            segs, (w2, h2) = tracer.trace_segments(mask, iso=0.0)
        assert (w2, h2) == (W, H)

        # Render segments to an array and save for visual inspection
        seg_img = render_segments_to_array(segs, (W, H), line_width=1, aa=4)
        seg_img = seg_img / seg_img.max()
        sw.show(seg_img)

    polys = chain_segments_to_polylines(segs, tol=0.25)
    simp = [douglas_peucker(p, 0.75) for p in polys if p.shape[0] >= 3]
    rings_gpu = close_and_filter_polylines(simp, close_tol_px=1.0, min_area_px2=4.0)
    regions_gpu = build_regions_with_holes(rings_gpu)

    # Rasterize both region sets to binary masks
    m_cpu = rasterize_regions(regions_cpu, (W, H))
    m_gpu = rasterize_regions(regions_gpu, (W, H))

    inter = np.logical_and(m_cpu, m_gpu).sum()
    union = np.logical_or(m_cpu, m_gpu).sum()
    iou = inter / union if union else 1.0
    ssim = ssim_simple(m_cpu, m_gpu)

    # Expect very high agreement; allow small tolerance for contour discretization
    assert iou > 0.97, f"IoU too low: {iou}"
    assert ssim > 0.977, f"SSIM too low: {ssim}"


# #67: truetrace's numbax jump flood. The numbax kernels were stubs that left the field
# at +-1e30, so iso had no effect there; they are now a port of the OpenCL jump flood.


def _bars_mask():
    m = np.zeros((90, 130), np.uint8)
    m[10:30, 10:100] = 1
    m[40:85, 20:35] = 1
    m[50:60, 60:120] = 1
    return m


def _disc_mask(n=256):
    yy, xx = np.mgrid[:n, :n]
    m = (np.hypot(yy - n / 2 + 0.3, xx - n / 2 - 0.7) < n / 3).astype(np.uint8)
    m[np.hypot(yy - n / 2, xx - n / 2) < n / 9] = 0
    return m


def _stored_mask(name):
    """Masks too irregular to draw here, stored bit-packed: "blobs", 300x420, 25
    random ellipses xor'd together (numpy default_rng seed 11); "noise", 256x256,
    5% random pixels (seed 9)."""
    with np.load(Path(__file__).with_name("truetrace_masks.npz")) as z:
        shape = tuple(int(n) for n in z[name + "_shape"])
        return np.unpackbits(z[name], count=shape[0] * shape[1]).reshape(shape)


def _field(tracer, mask):
    """The signed field trace_segments hands to marching_squares, as (H, W)."""
    captured = {}
    kmgr = tracer.kmgr
    orig = kmgr.marching_squares

    def spy(dist, iso, *a, **k):
        captured["dist"] = np.array(dist, copy=True)
        return orig(dist, iso, *a, **k)

    kmgr.marching_squares = spy
    try:
        tracer.trace_segments(mask, iso=0.0)
    finally:
        kmgr.marching_squares = orig
    return captured["dist"].T


@pytest.mark.parametrize("iso", [0.75, 1.5, -1.0])
@pytest.mark.parametrize("backend", get_test_backends())
def test_trace_segments_follow_iso_on_every_backend(backend, iso, monkeypatch):
    """Within a few pixels of the edge the jump flood is exact, so the traced segments
    equal marching squares on the exact field."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    mask = _ring_mask()
    segs, _ = TrueTrace().trace_segments(mask, iso=iso)
    sdf = sdf_from_mask_cpu(mask)
    want = sort_segments(marching_squares_cpu(sdf, off_field_iso(sdf, iso)))
    assert segs.shape == want.shape
    assert np.allclose(segs, want, atol=1e-4)


@pytest.mark.parametrize("backend", get_test_backends())
def test_signed_field_is_exact_near_the_boundary(backend, monkeypatch):
    """Where |sdf| <= 2 the field equals the exact one. Further in, the jump flood is
    approximate (up to about 1.5 px), so there is no bound there."""
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    for mask in (_ring_mask(), _bars_mask(), _thin_strokes()):
        sdf = sdf_from_mask_cpu(mask)
        field = _field(TrueTrace(), mask)
        band = np.abs(sdf) <= 2
        assert np.abs(field - sdf)[band].max() <= 1e-4


@pytest.mark.skipif(
    "numbax" not in get_backends(), reason="numbax backend not available"
)
def test_numbax_jump_flood_matches_the_device_kernels():
    """numbax's field is OpenCL's and rustcpu's to the bit; Metal and CUDA differ from
    them by sqrt rounding (~1e-6), which now and then flips a tie (up to ~0.33 px on
    under 0.01% of pixels). Run once, comparing numbax with each other backend."""
    others = [b for b in get_test_backends() if b != "numbax"]
    if not others:
        pytest.skip("no other backend to compare numbax with")
    masks = {
        "ring": _ring_mask(),
        "bars": _bars_mask(),
        "disc256": _disc_mask(),
        "blobs": _stored_mask("blobs"),
        "noise5": _stored_mask("noise"),
        "strokes": _thin_strokes(),
    }
    numbax = TrueTrace(backend="numbax")
    tracers = {b: TrueTrace(backend=b) for b in others}
    for name, mask in masks.items():
        mine = _field(numbax, mask)
        assert np.isfinite(mine).all(), name
        for b, tracer in tracers.items():
            theirs = _field(tracer, mask)
            if b in ("ocl", "rustcpu"):
                assert np.array_equal(mine, theirs), (name, b)
            else:
                off = np.abs(mine - theirs) > 1e-4
                assert off.mean() <= 1e-3, (name, b, int(off.sum()))
    if "ocl" in tracers:
        blobs = masks["blobs"]
        mine, _ = numbax.trace_segments(blobs, iso=20.0)
        theirs, _ = tracers["ocl"].trace_segments(blobs, iso=20.0)
        assert mine.shape == theirs.shape
        assert np.allclose(mine, theirs, atol=1e-4)

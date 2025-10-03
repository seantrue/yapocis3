import numpy as np
from PIL import Image, ImageDraw

from yapocis.lib.sdfpath import (
    build_circle_mask,
    trace_bitmap,
    chain_segments_to_polylines,
    douglas_peucker,
    close_and_filter_polylines,
    build_regions_with_holes,
)
from yapocis.lib.sdfpath import SDFPath


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


def test_truetrace_matches_original_fill_iou_ssim():
    # Create a simple binary mask (circle)
    W = H = 128
    mask = build_circle_mask(W, H, r=min(W, H) // 3)

    # Original CPU pipeline → regions
    rings_cpu = trace_bitmap(
        mask, simplify_tol_px=0.75, min_area_px2=4.0, chain_tol_px=0.25
    )
    regions_cpu = build_regions_with_holes(rings_cpu)

    # New GPU pipeline → segments → rings → regions
    tracer = SDFPath()
    segs, (w2, h2) = tracer.trace_segments(mask, iso=0.0)
    assert (w2, h2) == (W, H)
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

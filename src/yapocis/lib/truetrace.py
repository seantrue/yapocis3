"""
TrueTrace: GPU kernels + CPU helpers for bitmap contour tracing

Overview
- GPU kernels (via yapocis) implement a Jump Flood Algorithm (JFA) Euclidean
  distance transform and Marching Squares to extract iso-contour segments.
- CPU helpers provide image thresholding (Otsu/manual), a high-quality EDT,
  CPU Marching Squares, polyline chaining, Douglas–Peucker simplification,
  region nesting (outer/holes), and SVG writers.

Primary entrypoints
- Class `TrueTrace(BoundKernels)` — GPU tracer. Key method:
  - `trace_segments(mask_u8, iso=0.0) -> (segs, (W,H))` — returns Nx4 float32
    segments `[x0, y0, x1, y1]` at a given iso on the SDF (0.0=zero-crossing).

Core helpers (CPU)
- `load_mask_from_image(path, invert=False, threshold=None)` — grayscale→binary mask
- `trace_bitmap(mask, simplify_tol_px=0.75, min_area_px2=4.0, chain_tol_px=0.0)` —
  convenience to extract closed rings from a binary mask fully on CPU.
- `write_svg_regions(path, regions, w, h, stroke_wxidth=1.0, fill=None, curve='none', tension=0.0)` —
  write nested regions (outer+holes) as a single SVG with even-odd fill.
- `trace_regions(mask, iso=0.0, ...)` / `regions_svg(regions, w, h, ...)` — the outlines
  `trace_mask` fills, and those outlines as an SVG document (text).

Notes
- Shapes for GPU interface use (W,H) order internally; the wrapper transposes
  the input HxW mask and uses contiguous arrays to avoid extra allocations.
- Backends: honors `YAPOCIS_BACKEND` and `YAPOCIS_DEVICE`; you may also pass
  `backend=` and `device_name=` when constructing the wrapper.
"""

from __future__ import annotations

import math
from html import escape

import numpy as np

from .bound_kernels import BoundKernels


class TrueTrace(BoundKernels):
    INTERFACE = "truetrace"

    def trace_segments(
        self, mask_u8: np.ndarray, iso: float = 0.0
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Run JFA + Marching Squares on a binary mask.

        Args:
            mask_u8: HxW uint8/char array, values 0/1 (or 0/255; nonzero treated as 1).
            iso: iso value on the signed distance field.
                 - 0.0 traces the zero-crossing (contour between foreground/background)
                 - Positive moves contour outward, negative inward (in pixels)

        Returns:
            segs: Nx4 float32 array of segments [x0, y0, x1, y1].
            size: (W, H)
        """
        assert mask_u8.ndim == 2, mask_u8.shape
        H, W = mask_u8.shape

        # Coerce mask to int32 to match interface type, map nonzero to 1
        m = np.ascontiguousarray((mask_u8 != 0).astype(np.int32).T)  # shape (W,H)
        # One thread per pixel on every launch: left to the backend, OpenCL and
        # CUDA sized jfa_step by its first array-like argument, the scalar jump,
        # and ran a single thread, so the field was never propagated there.
        global_size = (W * H,)

        def nearest(mask_int: np.ndarray) -> np.ndarray:
            """Jump-flood distance (pixels) from every pixel to the nearest
            nonzero pixel of mask_int (0 on those pixels)."""
            seed_a = np.full((W, H, 4), -1.0, dtype=np.float32)
            seed_b = np.full((W, H, 4), -1.0, dtype=np.float32)
            dist = np.full((W, H), 1e30, dtype=np.float32)
            self.kmgr.init_seed_from_mask(mask_int, seed_a, dist, global_size=global_size)
            max_dim = max(W, H)
            jump = 1
            while (jump << 1) < max_dim:
                jump <<= 1
            cur_in, cur_out = seed_a, seed_b
            while jump >= 1:
                self.kmgr.jfa_step(np.int32(jump), cur_in, dist, cur_out, global_size=global_size)
                cur_in, cur_out = cur_out, cur_in
                jump >>= 1
            return dist

        # A signed distance field, positive inside (sdf_from_mask_cpu's
        # convention): distance to the background minus distance to the
        # foreground, so iso 0 is the boundary between the two pixel classes.
        # Until 2026-10-21 this was the unsigned distance to the foreground, whose
        # zero contour ran through the boundary foreground pixels' centres.
        dist = nearest(1 - m) - nearest(m)
        iso = off_field_iso(dist, iso)

        # Marching squares: estimate upper bound of segments and allocate
        max_segments = max(0, (W - 1) * (H - 1) * 2)
        seg_counter = np.zeros((1,), dtype=np.uint32)
        segments_out = np.full((max_segments * 4,), np.nan, dtype=np.float32)

        self.kmgr.marching_squares(
            dist, np.float32(iso), seg_counter, segments_out, global_size=global_size
        )

        # Read counter and slice segments
        count = int(seg_counter[0])
        count = max(0, min(count, max_segments))
        segs = (
            segments_out[: count * 4].reshape((-1, 4))
            if count > 0
            else np.empty((0, 4), dtype=np.float32)
        )
        # The kernel's atomic counter emits segments in an arbitrary order, which
        # would change where chaining starts each ring and so what Douglas-Peucker
        # keeps: sort them (by x0, y0, x1, y1) so the trace is repeatable and the
        # same in every runtime.
        return sort_segments(segs), (W, H)


# ----------------------------- Helpers (CPU + I/O) ----------------------------


def sort_segments(segs: np.ndarray) -> np.ndarray:
    """Segments `[x0, y0, x1, y1]` in a canonical order (by x0, then y0, x1, y1),
    so chaining and simplification do not depend on the order a kernel's atomic
    counter or a scan emitted them in."""
    if segs.shape[0] == 0:
        return segs
    return segs[np.lexsort((segs[:, 3], segs[:, 2], segs[:, 1], segs[:, 0]))]

INF32 = np.float32(1e30)


def load_mask_from_image(
    path: str, invert: bool = False, threshold: float | None = None
) -> np.ndarray:
    """Load an image and produce a binary mask.

    Args:
        path: Path to image. Loaded in grayscale.
        invert: If True, flips foreground/background after thresholding.
        threshold: If None, uses Otsu's method on [0..255] grayscale.
                   If in [0,1], treated as fraction of 255; otherwise [0..255].

    Returns:
        HxW uint8 array with values {0,1}.
    """
    try:
        from PIL import Image  # pillow
    except Exception as e:
        raise SystemExit(
            "Pillow is required to load images. Install with: pip install pillow"
            + str(e)
        )

    img = Image.open(path).convert("L")  # grayscale 0..255
    g = np.asarray(img, dtype=np.uint8)

    if threshold is None:
        thr = otsu_threshold_u8(g)
    else:
        thr = float(threshold)
        if 0.0 <= thr <= 1.0:
            thr = thr * 255.0
    mask = (g >= thr).astype(np.uint8)
    if invert:
        mask = 1 - mask
    return mask


def otsu_threshold_u8(gray: np.ndarray) -> float:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-20)
    idx = np.nanargmax(sigma_b2)
    return float(idx)


def edt_1d(f: np.ndarray) -> np.ndarray:
    n = f.shape[0]
    v = np.empty(n, dtype=np.int32)
    z = np.empty(n + 1, dtype=np.float64)
    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf
    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * (q - v[k]))
        while k > 0 and s <= z[k]:
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * (q - v[k]))
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf
    k = 0
    g = np.empty(n, dtype=np.float64)
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        dq = q - v[k]
        g[q] = dq * dq + f[v[k]]
    return g


def edt_2d(mask_bool: np.ndarray) -> np.ndarray:
    inf = 1e12
    f = np.where(mask_bool, 0.0, inf).astype(np.float64)
    tmp = np.empty_like(f)
    for y in range(f.shape[0]):
        tmp[y, :] = edt_1d(f[y, :])
    out = np.empty_like(tmp)
    for x in range(tmp.shape[1]):
        out[:, x] = edt_1d(tmp[:, x])
    out = np.sqrt(out, dtype=np.float64)
    return out


def sdf_from_mask_cpu(mask_u8: np.ndarray) -> np.ndarray:
    mask_bool = mask_u8.astype(bool)
    d_true = edt_2d(mask_bool)  # distance to foreground (True)
    d_false = edt_2d(~mask_bool)  # distance to background (False)
    # Positive inside convention: +dist to background inside, -dist outside
    sdf = d_false - d_true
    return sdf.astype(np.float32)


def marching_squares_cpu(dist: np.ndarray, iso: float = 0.0) -> np.ndarray:
    h, w = dist.shape
    segs: list[tuple[float, float, float, float]] = []
    for y in range(h - 1):
        drow0 = dist[y]
        drow1 = dist[y + 1]
        for x in range(w - 1):
            v00 = drow0[x] - iso
            v10 = drow0[x + 1] - iso
            v11 = drow1[x + 1] - iso
            v01 = drow1[x] - iso
            c = 0
            if v00 > 0:
                c |= 1
            if v10 > 0:
                c |= 2
            if v11 > 0:
                c |= 4
            if v01 > 0:
                c |= 8
            if c == 0 or c == 15:
                continue

            def lerp(t, a, b):
                return a + t * (b - a)

            eps = 1e-20
            t0 = v00 / (v00 - v10 + eps)
            p0 = (lerp(t0, float(x), float(x + 1)), float(y))
            t1 = v10 / (v10 - v11 + eps)
            p1 = (float(x + 1), lerp(t1, float(y), float(y + 1)))
            t2 = v01 / (v01 - v11 + eps)
            p2 = (lerp(t2, float(x), float(x + 1)), float(y + 1))
            t3 = v00 / (v00 - v01 + eps)
            p3 = (float(x), lerp(t3, float(y), float(y + 1)))
            P = (p0, p1, p2, p3)

            def emit(a, b):
                segs.append((P[a][0], P[a][1], P[b][0], P[b][1]))

            if c in (1, 14):
                emit(3, 0)
            elif c in (2, 13):
                emit(0, 1)
            elif c in (3, 12):
                emit(3, 1)
            elif c in (4, 11):
                emit(1, 2)
            elif c in (6, 9):
                emit(0, 2)
            elif c in (7, 8):
                emit(3, 2)
            elif c in (5, 10):
                d = v00 * v11 - v10 * v01
                if c == 5:
                    if d > 0:
                        emit(3, 0)
                        emit(1, 2)
                    else:
                        emit(0, 1)
                        emit(2, 3)
                else:  # c == 10
                    if d > 0:
                        emit(0, 1)
                        emit(2, 3)
                    else:
                        emit(3, 0)
                        emit(1, 2)
    if not segs:
        return np.empty((0, 4), dtype=np.float32)
    return np.asarray(segs, dtype=np.float32)


def write_svg_segments(path: str, segs: np.ndarray, w: int, h: int):
    from xml.etree.ElementTree import Element, ElementTree, SubElement

    svg = Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        width=str(w),
        height=str(h),
        viewBox=f"0 0 {w} {h}",
    )
    g = SubElement(
        svg, "g", attrib={"fill": "none", "stroke": "black", "stroke-width": "0.75"}
    )
    for x0, y0, x1, y1 in segs:
        SubElement(
            g, "line", x1=f"{x1}", y1=f"{h - 1 - y1}", x2=f"{x0}", y2=f"{h - 1 - y0}"
        )
    ElementTree(svg).write(path)


def off_field_iso(dist: np.ndarray, iso: float) -> np.float32:
    """``iso`` raised in 1e-3 px steps until no value of the field equals it.

    The field's values are pixel-centre distances (1, sqrt 2, 2, ...): at an iso
    equal to one of them, marching squares puts contour points on those pixels'
    centres, where the outlines on either side of a stroke meet, and chaining
    can't tell them apart. Values at the iso already count as outside, so raising
    it changes no classification, only moves the points off the centres."""
    iso = np.float32(iso)
    while np.any(dist == iso):
        iso = np.float32(iso + np.float32(1e-3))
    return iso


def chain_segments_to_polylines(
    segs: np.ndarray, tol: float = 0.0
) -> list[np.ndarray]:
    """Segments joined end to end into polylines (closed rings end where they
    start). Endpoints match exactly by default: marching squares computes a
    crossing shared by two cells with the same arithmetic in both, so the copies
    are identical floats. ``tol > 0`` snaps endpoints to a ``tol`` grid first,
    which merges distinct contours closer than ``tol``: until 2026-10-21
    ``trace_mask`` chained with 0.25, and at iso above 0.5 the two outlines of a
    1 px stroke (2 * (1 - iso) px apart) fused, broke rings open and closed them
    with long chords that the even-odd fill turned into large black wedges."""
    if segs.size == 0:
        return []

    def q2(p):
        if tol <= 0:
            return (float(p[0]), float(p[1]))
        return (round(float(p[0]) / tol) * tol, round(float(p[1]) / tol) * tol)

    pts: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for x0, y0, x1, y1 in segs:
        a = q2((x0, y0))
        b = q2((x1, y1))
        pts.setdefault(a, []).append(b)
        pts.setdefault(b, []).append(a)
    visited_edges: dict[tuple[tuple[float, float], tuple[float, float]], bool] = {}
    polylines: list[np.ndarray] = []

    def pop_edge(u):
        lst = pts.get(u)
        while lst:
            v = lst.pop()
            e = (u, v) if u < v else (v, u)
            if not visited_edges.get(e):
                visited_edges[e] = True
                return v
        return None

    starts = [p for p, nbrs in pts.items() if len(nbrs) != 2]
    if not starts:
        starts = [next(iter(pts.keys()))]

    seen_nodes = set()
    for s in starts:
        if s in seen_nodes and not pts.get(s):
            continue
        path = [s]
        u = s
        while True:
            v = pop_edge(u)
            if v is None:
                break
            path.append(v)
            seen_nodes.add(u)
            u = v
        if len(path) >= 2:
            polylines.append(np.array(path, dtype=np.float32))

    for u in list(pts.keys()):
        while True:
            v = pop_edge(u)
            if v is None:
                break
            path = [u, v]
            w = v
            while True:
                nxt = pop_edge(w)
                if nxt is None:
                    break
                path.append(nxt)
                w = nxt
            if len(path) >= 2:
                polylines.append(np.array(path, dtype=np.float32))

    return polylines


def douglas_peucker(poly: np.ndarray, tol: float) -> np.ndarray:
    if poly.shape[0] <= 2 or tol <= 0:
        return poly
    closed = np.linalg.norm(poly[0] - poly[-1]) < 1e-9
    if closed:
        poly = poly[:-1]
        if poly.shape[0] <= 2:
            return np.vstack([poly, poly[0]]) if closed else poly
    keep = np.zeros(poly.shape[0], dtype=bool)
    keep[0] = True
    keep[-1] = True
    stack: list[tuple[int, int]] = [(0, poly.shape[0] - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        a, b = poly[i0], poly[i1]
        seg = b - a
        seg2 = float(np.dot(seg, seg))
        pts = poly[i0 + 1 : i1]
        if pts.size == 0:
            continue
        if seg2 <= 0:
            dists = np.linalg.norm(pts - a, axis=1)
        else:
            ap = pts - a
            t = np.clip((ap @ seg) / seg2, 0.0, 1.0)
            proj = a + t[:, None] * seg
            dists = np.linalg.norm(pts - proj, axis=1)
        j_rel = int(np.argmax(dists))
        dmax = float(dists[j_rel])
        j = i0 + 1 + j_rel
        if dmax > tol:
            keep[j] = True
            stack.append((i0, j))
            stack.append((j, i1))
    out = poly[keep]
    if closed:
        out = np.vstack([out, out[0]])
    return out


douglas_peuker = douglas_peucker


def close_and_filter_polylines(
    polys: list[np.ndarray], close_tol_px: float = 1.0, min_area_px2: float = 0.0
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for poly in polys:
        if poly.shape[0] < 2:
            continue
        if np.linalg.norm(poly[0] - poly[-1]) > close_tol_px:
            poly = np.vstack([poly, poly[0]])
        if poly.shape[0] < 4:
            continue
        A = polygon_area(poly)
        if abs(A) < float(min_area_px2):
            continue
        out.append(poly)
    return out


def polygon_area(poly: np.ndarray) -> float:
    """Signed shoelace area over consecutive points, in float64 with an exactly
    rounded sum (math.fsum), so the value doesn't depend on the host's BLAS
    (np.dot of float32 rows runs the platform's sdot, whose summation order
    differs between Accelerate and OpenBLAS) and the Rust port can match it."""
    p = np.asarray(poly, dtype=np.float64)
    x = p[:, 0]
    y = p[:, 1]
    return 0.5 * math.fsum((x[:-1] * y[1:] - y[:-1] * x[1:]).tolist())


def point_in_polygon(pt: tuple[float, float], poly: np.ndarray) -> bool:
    """Even-odd test in float64 (float32 vertices promoted once; numpy's scalar
    rules would otherwise compare and interpolate in float32)."""
    x, y = float(pt[0]), float(pt[1])
    cnt = 0
    p = np.asarray(poly, dtype=np.float64).tolist()
    for i in range(len(p) - 1):
        x0, y0 = p[i]
        x1, y1 = p[i + 1]
        if (y0 > y) != (y1 > y):
            t = (y - y0) / (y1 - y0 + 1e-20)
            xr = x0 + t * (x1 - x0)
            if xr > x:
                cnt ^= 1
    return bool(cnt)


def build_regions_with_holes(rings: list[np.ndarray]) -> list[dict]:
    """Group closed rings into nested regions, assigning holes.

    - Normalizes ring orientation to CCW for outers and CW for holes.
    - Uses centroid-in-polygon tests to find the smallest containing outer.

    Args:
        rings: list of closed rings (Nx2 float arrays, last point == first).

    Returns:
        list of dicts: { 'outer': ring, 'holes': [ring, ...] }, sorted by area.
    """
    if not rings:
        return []
    recs = []
    for r in rings:
        A = polygon_area(r)
        rr = r
        if A < 0:
            rr = r[::-1].copy()
            A = -A
        # float64, exactly rounded: np.mean of a float32 column sums in float32
        n = rr.shape[0] - 1
        cx = math.fsum(rr[:-1, 0].astype(np.float64).tolist()) / n
        cy = math.fsum(rr[:-1, 1].astype(np.float64).tolist()) / n
        recs.append({"ring": rr, "area": A, "centroid": (cx, cy)})
    recs.sort(key=lambda t: t["area"])
    parents = [-1] * len(recs)
    for i in range(len(recs)):
        c = recs[i]["centroid"]
        best = -1
        best_area = 1e30
        for j in range(i + 1, len(recs)):
            if recs[j]["area"] < best_area and point_in_polygon(c, recs[j]["ring"]):
                best = j
                best_area = recs[j]["area"]
        parents[i] = best
    # Nesting depth decides: an unparented ring, or one inside a hole (even
    # depth), is an outer; a ring directly inside an outer (odd depth) is that
    # outer's hole, reversed to CW. (Until 2026-10-21 a top-level ring was also
    # appended as a hole of a nonexistent parent and real holes were lost.)
    regions: dict[int, dict] = {}
    for i, p in enumerate(parents):
        depth = 0
        q = p
        while q != -1:
            depth += 1
            q = parents[q]
        if depth % 2 == 0:
            regions.setdefault(i, {"outer": recs[i]["ring"], "holes": []})
        else:
            hole = recs[i]["ring"][::-1].copy()
            regions.setdefault(p, {"outer": recs[p]["ring"], "holes": []})[
                "holes"
            ].append(hole)
    ordered_keys = sorted(
        regions.keys(), key=lambda k: polygon_area(regions[k]["outer"])
    )
    return [regions[k] for k in ordered_keys]


def _ring_to_cubic_beziers(
    ring: np.ndarray, tension: float = 0.0
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    pts = ring[:-1]
    n = pts.shape[0]
    if n < 2:
        return []
    s = (1.0 - float(tension)) / 6.0
    beziers = []
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        c1 = p1 + s * (p2 - p0)
        c2 = p2 - s * (p3 - p1)
        beziers.append((p1, c1, c2, p2))
    return beziers


def write_svg_regions(
    path: str,
    regions: list[dict],
    w: int,
    h: int,
    stroke_width: float = 1.0,
    fill: str | None = None,
    curve: str = "none",
    tension: float = 0.0,
):
    """Write nested regions (outer+holes) to an SVG file.

    Args:
        path: Output SVG filename.
        regions: As produced by build_regions_with_holes.
        w, h: Canvas size in pixels.
        stroke_width: Stroke width of the outline in pixels.
        fill: Fill color; None for no fill (default).
        curve: 'none' for polylines, 'cubic' for Catmull–Rom→Bezier smoothed.
        tension: Smoothing factor in [0,1]; 0=smoothest, 1=straight.

    Returns:
        The output path (convenience).
    """
    from xml.etree.ElementTree import Element, ElementTree, SubElement

    def _fmt_xy(xy: np.ndarray) -> tuple[float, float]:
        return float(xy[0]), float(h - 1 - xy[1])

    def path_from_ring_poly(r: np.ndarray) -> str:
        cmds = [f"M {_fmt_xy(r[0])[0]:.3f} {_fmt_xy(r[0])[1]:.3f}"]
        for i in range(1, r.shape[0]):
            x, y = _fmt_xy(r[i])
            cmds.append(f"L {x:.3f} {y:.3f}")
        cmds.append("Z")
        return " ".join(cmds)

    def path_from_ring_cubic(r: np.ndarray) -> str:
        bez = _ring_to_cubic_beziers(r, tension=tension)
        if not bez:
            return path_from_ring_poly(r)
        p1, c1, c2, p2 = bez[0]
        x0, y0 = _fmt_xy(p1)
        cmds = [f"M {x0:.3f} {y0:.3f}"]
        for p1, c1, c2, p2 in bez:
            x1, y1 = _fmt_xy(c1)
            x2, y2 = _fmt_xy(c2)
            x3, y3 = _fmt_xy(p2)
            cmds.append(f"C {x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f} {x3:.3f} {y3:.3f}")
        cmds.append("Z")
        return " ".join(cmds)

    svg = Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        width=str(w),
        height=str(h),
        viewBox=f"0 0 {w} {h}",
    )
    attrib = {
        "fill": "none" if fill is None else fill,
        "stroke": "black",
        "stroke-width": f"{stroke_width:.3f}",
        "fill-rule": "evenodd",
    }
    g = SubElement(svg, "g", attrib=attrib)
    for reg in regions:
        if curve == "cubic":
            d = path_from_ring_cubic(reg["outer"])
        else:
            d = path_from_ring_poly(reg["outer"])
        for hole in reg["holes"]:
            if curve == "cubic":
                d += " " + path_from_ring_cubic(hole)
            else:
                d += " " + path_from_ring_poly(hole)
        SubElement(g, "path", d=d)
    ElementTree(svg).write(path)
    return path


def trace_bitmap(
    mask_u8: np.ndarray,
    simplify_tol_px: float = 0.75,
    min_area_px2: float = 4.0,
    chain_tol_px: float = 0.0,
) -> list[np.ndarray]:
    """Extract closed rings from a binary mask fully on CPU.

    Pipeline: Marching Squares (iso=0.5 on mask) → chain segments → simplify
    with Douglas–Peucker → close and filter by area.

    Args:
        mask_u8: HxW uint8 mask {0,1}.
        simplify_tol_px: Douglas–Peucker tolerance in pixels.
        min_area_px2: Discard rings with |area| below this value.
        chain_tol_px: Endpoint snapping tolerance when chaining segments (0:
            exact endpoints, see ``chain_segments_to_polylines``).

    Returns:
        list of closed rings, each an (N,2) float32 array (last == first).
    """
    segs = marching_squares_cpu(mask_u8.astype(np.float32), iso=0.5)
    polys = chain_segments_to_polylines(segs, tol=chain_tol_px)
    simp = []
    for p in polys:
        if p.shape[0] < 3:
            continue
        simp.append(douglas_peucker(p, simplify_tol_px))
    rings = close_and_filter_polylines(
        simp, close_tol_px=1.0, min_area_px2=min_area_px2
    )
    return rings


def build_circle_mask(w: int, h: int, r: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w // 2, h // 2
    return (((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r).astype(np.uint8)


# ----------------------------- Filling traced regions ------------------------


def flatten_cubic(p0, p1, p2, p3, segments: int = 8) -> list[tuple[float, float]]:
    """The cubic Bézier (p0, p1, p2, p3) as ``segments`` + 1 points, by uniform
    parameter steps (the same arithmetic as yapocis-rs ``paths::flatten_cubic``)."""
    n = max(int(segments), 1)
    pts = []
    for i in range(n + 1):
        t = i / n
        mt = 1.0 - t
        b0, b1, b2, b3 = mt * mt * mt, 3.0 * mt * mt * t, 3.0 * mt * t * t, t * t * t
        pts.append(
            (
                b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0],
                b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1],
            )
        )
    return pts


def ring_to_polygon(
    ring: np.ndarray, curve: str = "cubic", tension: float = 0.0, segments: int = 8
) -> list[tuple[float, float]]:
    """A closed ring (last == first) as a polygon: its vertices as they are
    (``curve='none'``), or the Catmull-Rom cubics of ``_ring_to_cubic_beziers``
    flattened with ``segments`` steps each (``curve='cubic'``)."""
    if curve == "none":
        return [(float(x), float(y)) for x, y in ring[:-1]]
    if curve != "cubic":
        raise ValueError(f"curve must be 'none' or 'cubic', not {curve!r}")
    poly: list[tuple[float, float]] = []
    for p1, c1, c2, p2 in _ring_to_cubic_beziers(ring, tension=tension):
        pts = flatten_cubic(
            (float(p1[0]), float(p1[1])),
            (float(c1[0]), float(c1[1])),
            (float(c2[0]), float(c2[1])),
            (float(p2[0]), float(p2[1])),
            segments,
        )
        poly.extend(pts[:-1])  # the last point is the next bezier's first
    return poly


def fill_polygons(
    polys: list[list[tuple[float, float]]],
    shape: tuple[int, int],
    ink: float = 0.0,
    paper: float = 1.0,
) -> np.ndarray:
    """Even-odd scanline fill of closed polygons (float64 vertices, implicitly
    closed), as an (H, W) float32 raster of ``paper`` with ``ink`` inside.

    Pixel (row, col) is sampled at (x=col, y=row), the grid marching squares
    emits. An edge counts on scanline y when min(y0, y1) <= y < max(y0, y1), its
    crossing is x0 + (y - y0) * (x1 - x0) / (y1 - y0), and each pair of sorted
    crossings [xa, xb) covers the pixels with xa <= col < xb. Pure IEEE float64
    with no triangulation and no GPU, so every host and the Rust port give the
    same pixels (earcut triangulations differed between the C++ and Rust
    libraries, and the kernel's per-triangle coverage with them).
    """
    h, w = shape
    out = np.full((h, w), paper, dtype=np.float32)
    edges = []
    for poly in polys:
        n = len(poly)
        if n < 3:
            continue
        for i in range(n):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % n]
            if y0 != y1:
                edges.append((x0, y0, x1, y1))
    if not edges:
        return out
    e = np.array(edges, dtype=np.float64)
    x0, y0, x1, y1 = e[:, 0], e[:, 1], e[:, 2], e[:, 3]
    ylo = np.minimum(y0, y1)
    yhi = np.maximum(y0, y1)
    y_first = max(int(math.ceil(float(ylo.min()))), 0)
    y_last = min(int(math.ceil(float(yhi.max()))), h)
    for y in range(y_first, y_last):
        hit = (ylo <= y) & (y < yhi)
        if not hit.any():
            continue
        xc = x0[hit] + (y - y0[hit]) * (x1[hit] - x0[hit]) / (y1[hit] - y0[hit])
        xc.sort()
        for k in range(0, len(xc) - 1, 2):
            a = max(int(math.ceil(xc[k])), 0)
            b = min(int(math.ceil(xc[k + 1])), w)
            if a < b:
                out[y, a:b] = ink
    return out


def fill_regions(
    regions: list[dict],
    shape: tuple[int, int],
    curve: str = "cubic",
    tension: float = 0.0,
    segments: int = 8,
    ink: float = 0.0,
    paper: float = 1.0,
) -> np.ndarray:
    """Rasterise nested regions (``build_regions_with_holes`` output) as ``ink``
    on a ``paper`` canvas of ``shape`` (H, W): every ring, outer or hole,
    flattened by ``ring_to_polygon`` and filled together with the even-odd rule
    (``fill_polygons``), so the nesting takes care of the holes.

    Until 2026-10-21 each region was triangulated with earcut and drawn through
    the graphics4 ``mesh_triangles`` kernel; the C++ and Rust earcut libraries
    triangulate differently and the kernel's coverage depended on that.

    Returns an (H, W) float32 array of ``paper`` and ``ink`` values.
    """
    polys = []
    for region in regions:
        rings = [region["outer"]] + list(region.get("holes", []))
        for r in rings:
            poly = ring_to_polygon(r, curve=curve, tension=tension, segments=segments)
            if len(poly) >= 3:
                polys.append(poly)
    return fill_polygons(polys, shape, ink=ink, paper=paper)


def regions_svg(
    regions: list[dict],
    w: int,
    h: int,
    curve: str = "none",
    tension: float = 0.0,
    ink: str = "#000000",
    paper: str | None = "#ffffff",
) -> str:
    """Nested regions as an SVG document: the same outlines ``fill_regions`` fills,
    filled even-odd in ``ink`` on a ``paper`` rectangle (None: transparent).

    ``curve='cubic'`` writes the real cubics of ``_ring_to_cubic_beziers`` where
    ``fill_regions`` flattens them. Coordinates keep the image's y-down, and the group
    is moved by half a pixel because ``fill_polygons`` samples pixel (row, col) at
    (x=col, y=row) while SVG's pixel centres are at (col + 0.5, row + 0.5).

    ``write_svg_regions`` is a different thing: outlines stroked, y flipped, for
    looking at the tracer's output.
    """
    if curve not in ("none", "cubic"):
        raise ValueError(f"curve must be 'none' or 'cubic', not {curve!r}")

    def num(v) -> str:
        return f"{float(v):.2f}".rstrip("0").rstrip(".")

    def ring_d(ring: np.ndarray) -> str:
        bez = _ring_to_cubic_beziers(ring, tension=tension) if curve == "cubic" else []
        if bez:
            cmds = [f"M{num(bez[0][0][0])} {num(bez[0][0][1])}"]
            for _, c1, c2, p2 in bez:
                cmds.append(
                    f"C{num(c1[0])} {num(c1[1])} {num(c2[0])} {num(c2[1])} "
                    f"{num(p2[0])} {num(p2[1])}"
                )
        else:
            pts = ring[:-1]
            cmds = [f"M{num(pts[0][0])} {num(pts[0][1])}"]
            cmds.extend(f"L{num(x)} {num(y)}" for x, y in pts[1:])
        return "".join(cmds) + "Z"

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">'
    ]
    if paper is not None:
        out.append(f'<rect width="{w}" height="{h}" fill="{escape(paper)}"/>')
    out.append(
        f'<g fill="{escape(ink)}" fill-rule="evenodd" transform="translate(0.5 0.5)">'
    )
    for region in regions:
        rings = [region["outer"]] + list(region.get("holes", []))
        out.append(f'<path d="{"".join(ring_d(r) for r in rings)}"/>')
    out.append("</g></svg>\n")
    return "\n".join(out)


def trace_regions(
    mask: np.ndarray,
    iso: float = 0.0,
    simplify_tol_px: float = 0.75,
    min_area_px2: float = 4.0,
    chain_tol_px: float = 0.0,
    engine: str = "gpu",
    device_name: str | None = None,
) -> list[dict]:
    """Trace a binary mask (nonzero = ink) into nested regions, the outlines
    ``trace_mask`` fills (``build_regions_with_holes`` output): for ``regions_svg``,
    or anything else that wants the geometry rather than the raster.

    Pipeline: the signed distance field of the mask (GPU jump flood, or the CPU
    EDT with ``engine='cpu'``) traced by marching squares at ``iso`` (0 is the
    mask's boundary; negative moves it into the ink, so ``-0.5`` keeps only pixels
    fully inside the shapes, as thresholding potrace's antialiased output does),
    chained into polylines, simplified with Douglas-Peucker (``simplify_tol_px``),
    closed and filtered by area (``min_area_px2``, potrace's turdsize), and nested
    into regions with holes.
    """
    mask_u8 = (np.asarray(mask) != 0).astype(np.uint8)
    if engine == "gpu":
        tracer = TrueTrace(device_name=device_name)
        segs, _ = tracer.trace_segments(mask_u8, iso=iso)
    elif engine == "cpu":
        sdf = sdf_from_mask_cpu(mask_u8)
        segs = sort_segments(marching_squares_cpu(sdf, iso=off_field_iso(sdf, iso)))
    else:
        raise ValueError(f"engine must be 'gpu' or 'cpu', not {engine!r}")
    polys = chain_segments_to_polylines(segs, tol=chain_tol_px)
    simp = [douglas_peucker(p, simplify_tol_px) for p in polys if p.shape[0] >= 3]
    rings = close_and_filter_polylines(simp, close_tol_px=1.0, min_area_px2=min_area_px2)
    return build_regions_with_holes(rings)


def trace_mask(
    mask: np.ndarray,
    iso: float = 0.0,
    simplify_tol_px: float = 0.75,
    min_area_px2: float = 4.0,
    chain_tol_px: float = 0.0,
    curve: str = "cubic",
    tension: float = 0.0,
    segments: int = 8,
    engine: str = "gpu",
    device_name: str | None = None,
) -> np.ndarray:
    """Trace a binary mask (nonzero = ink) into smooth filled shapes: a raster of
    1.0 (paper) and 0.0 (ink) the size of the mask, the way potrace's PGM backend
    is used by Virtual Impressionist: ``trace_regions``, filled by the even-odd
    scanline of ``fill_regions``.
    """
    regions = trace_regions(
        mask, iso=iso, simplify_tol_px=simplify_tol_px, min_area_px2=min_area_px2,
        chain_tol_px=chain_tol_px, engine=engine, device_name=device_name,
    )
    h, w = np.asarray(mask).shape
    return fill_regions(regions, (h, w), curve=curve, tension=tension, segments=segments)

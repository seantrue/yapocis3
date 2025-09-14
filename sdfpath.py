"""
Hero Demo: Potrace-like Tracing (Closed SVG Curves)
- Focus: robust contour extraction from B/W images with optional simplification
- Drop-in replacement for prior V-carve/G-code path: this emits **closed SVG paths** only
- Still uses safe buffer init patterns in GPU parts if you keep the OpenCL bits around

USAGE EXAMPLES
--------------
# 1) Trace a black/white image to closed SVG polylines (auto-threshold)
python sdfpath.py --image ./mask.png --out traced.svg

# 2) Control simplification (in pixels) and minimum area filter
python sdfpath.py --image ./mask.png --simplify-tol-px 0.75 --min-area-px2 25 --out traced.svg

# 3) Invert foreground/background or set manual threshold
python sdfpath.py --image ./mask.png --invert --threshold 180 --out traced.svg
"""
from __future__ import annotations
import math
import struct
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np

# ----------------------------- Optional PyOpenCL -------------------------------
try:
    import pyopencl as cl
    HAVE_OPENCL = True
except Exception:
    cl = None  # type: ignore
    HAVE_OPENCL = False


# ----------------------------- Utilities & Guards -----------------------------

INF32 = np.float32(1e30)


def bytes_pattern_of(val: np.generic) -> bytes:
    return np.array(val).tobytes()


def try_create_ctx_queue() -> Tuple[Optional["cl.Context"], Optional["cl.CommandQueue"]]:
    if not HAVE_OPENCL:
        return None, None
    plats = cl.get_platforms()
    if not plats:
        return None, None
    devs = plats[0].get_devices()
    if not devs:
        return None, None
    ctx = cl.Context(devices=devs)
    queue = cl.CommandQueue(ctx)
    return ctx, queue


def ensure_fill(queue: "cl.CommandQueue", buf: "cl.Buffer", pattern: bytes):
    cl.enqueue_fill_buffer(queue, buf, pattern, 0, buf.size)
    queue.finish()


def sanity_no_nan_f32(arr: np.ndarray):
    if not np.all(np.isfinite(arr)):
        bad = arr[~np.isfinite(arr)]
        raise RuntimeError(f"Non-finite values detected (CPU path): {bad[:8]}")


# ----------------------------- Image Loading ----------------------------------

def load_mask_from_image(path: str, invert: bool = False, threshold: Optional[float] = None) -> np.ndarray:
    try:
        from PIL import Image  # pillow
    except Exception as e:
        raise SystemExit("Pillow is required to load images. Install with: pip install pillow" + str(e))

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


# ----------------------------- CPU SDF (EDT) ----------------------------------

def edt_1d(f: np.ndarray) -> np.ndarray:
    n = f.shape[0]
    v = np.empty(n, dtype=np.int32)
    z = np.empty(n + 1, dtype=np.float64)
    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf
    for q in range(1, n):
        s = ((f[q] + q*q) - (f[v[k]] + v[k]*v[k])) / (2.0*(q - v[k]))
        while k > 0 and s <= z[k]:
            k -= 1
            s = ((f[q] + q*q) - (f[v[k]] + v[k]*v[k])) / (2.0*(q - v[k]))
        k += 1
        v[k] = q
        z[k] = s
        z[k+1] = np.inf
    k = 0
    g = np.empty(n, dtype=np.float64)
    for q in range(n):
        while z[k+1] < q:
            k += 1
        dq = q - v[k]
        g[q] = dq*dq + f[v[k]]
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
    d_true = edt_2d(mask_bool)      # distance to foreground (True)
    d_false = edt_2d(~mask_bool)    # distance to background (False)
    # Positive inside convention: +dist to background inside, -dist outside
    sdf = d_false - d_true
    return sdf.astype(np.float32)


# ----------------------------- CPU Marching Squares ----------------------------

def marching_squares_cpu(dist: np.ndarray, iso: float = 0.0) -> np.ndarray:
    h, w = dist.shape
    segs: List[Tuple[float, float, float, float]] = []
    for y in range(h - 1):
        drow0 = dist[y]
        drow1 = dist[y + 1]
        for x in range(w - 1):
            v00 = drow0[x] - iso
            v10 = drow0[x + 1] - iso
            v11 = drow1[x + 1] - iso
            v01 = drow1[x] - iso
            c = 0
            if v00 > 0: c |= 1
            if v10 > 0: c |= 2
            if v11 > 0: c |= 4
            if v01 > 0: c |= 8
            if c == 0 or c == 15:
                continue
            def lerp(t, a, b):
                return a + t*(b-a)
            eps = 1e-20
            t0 = v00/(v00 - v10 + eps); p0 = (lerp(t0, float(x), float(x+1)), float(y))
            t1 = v10/(v10 - v11 + eps); p1 = (float(x+1), lerp(t1, float(y), float(y+1)))
            t2 = v01/(v01 - v11 + eps); p2 = (lerp(t2, float(x), float(x+1)), float(y+1))
            t3 = v00/(v00 - v01 + eps); p3 = (float(x), lerp(t3, float(y), float(y+1)))
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
                # Asymptotic decider for ambiguous cases
                # Use bilinear invariant: d = v00*v11 - v10*v01
                d = v00*v11 - v10*v01
                if c == 5:
                    if d > 0:
                        emit(3, 0); emit(1, 2)
                    else:
                        emit(0, 1); emit(2, 3)
                else:  # c == 10
                    if d > 0:
                        emit(0, 1); emit(2, 3)
                    else:
                        emit(3, 0); emit(1, 2)
    if not segs:
        return np.empty((0, 4), dtype=np.float32)
    return np.asarray(segs, dtype=np.float32)


# ----------------------------- OpenCL Kernels ---------------------------------

KERNEL_SRC = r"""
inline int2 make_int2(int x, int y) { int2 v; v.x = x; v.y = y; return v; }

__kernel void init_seed_from_mask(
    __global const uchar *mask, // 0 or 1 per pixel
    const int width,
    const int height,
    __global int2 *seed,        // nearest seed coordinate per pixel; (-1,-1) if none
    __global float *dist        // distance to nearest seed; INF if none
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;
    const int idx = y * width + x;
    if (mask[idx] != 0) {
        seed[idx] = make_int2(x, y);
        dist[idx] = 0.0f;
    } else {
        seed[idx] = make_int2(-1, -1);
        dist[idx] = (float)1e30f;
    }
}

__kernel void jfa_step(
    const int width,
    const int height,
    const int jump,
    __global const int2 *seed_in,
    __global float *dist_inout,
    __global int2 *seed_out
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;
    const int idx = y * width + x;

    int2 best_seed = seed_in[idx];
    float best_dist = dist_inout[idx];

    const int dirs[8][2] = {{-1,-1},{0,-1},{1,-1},{-1,0},{1,0},{-1,1},{0,1},{1,1}};
    for (int k=0;k<8;++k) {
        int nx = x + dirs[k][0]*jump;
        int ny = y + dirs[k][1]*jump;
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        const int nidx = ny * width + nx;
        int2 nseed = seed_in[nidx];
        if (nseed.x < 0) continue;
        float dx = (float)(nseed.x - x);
        float dy = (float)(nseed.y - y);
        float d2 = dx*dx + dy*dy;
        if (d2 < best_dist*best_dist) {
            best_dist = sqrt(d2);
            best_seed = nseed;
        }
    }
    seed_out[idx] = best_seed;
    dist_inout[idx] = best_dist;
}

__kernel void marching_squares(
    const int width,
    const int height,
    __global const float *dist,
    const float iso,
    __global uint *seg_counter,
    __global float *segments_out
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width-1 || y >= height-1) return;

    const int idx00 = y*width + x;
    const int idx10 = y*width + (x+1);
    const int idx11 = (y+1)*width + (x+1);
    const int idx01 = (y+1)*width + x;

    float v00 = dist[idx00] - iso;
    float v10 = dist[idx10] - iso;
    float v11 = dist[idx11] - iso;
    float v01 = dist[idx01] - iso;

    int c = 0;
    if (v00 > 0) c |= 1;
    if (v10 > 0) c |= 2;
    if (v11 > 0) c |= 4;
    if (v01 > 0) c |= 8;

    if (c == 0 || c == 15) return;

    #define LERP(t, a, b) ((a) + (t)*((b)-(a)))
    float2 p[4];
    float t0 = v00/(v00 - v10 + 1e-20f); p[0].x = LERP(t0, (float)x, (float)(x+1)); p[0].y = (float)y;
    float t1 = v10/(v10 - v11 + 1e-20f); p[1].x = (float)(x+1); p[1].y = LERP(t1, (float)y, (float)(y+1));
    float t2 = v01/(v01 - v11 + 1e-20f); p[2].x = LERP(t2, (float)x, (float)(x+1)); p[2].y = (float)(y+1);
    float t3 = v00/(v00 - v01 + 1e-20f); p[3].x = (float)x; p[3].y = LERP(t3, (float)y, (float)(y+1));

    int pairs[2][2]; int nseg = 0;
    switch(c) {
        case 1: case 14: pairs[0][0]=3; pairs[0][1]=0; nseg=1; break;
        case 2: case 13: pairs[0][0]=0; pairs[0][1]=1; nseg=1; break;
        case 3: case 12: pairs[0][0]=3; pairs[0][1]=1; nseg=1; break;
        case 4: case 11: pairs[0][0]=1; pairs[0][1]=2; nseg=1; break;
        case 6: case 9:  pairs[0][0]=0; pairs[0][1]=2; nseg=1; break;
        case 7: case 8:  pairs[0][0]=3; pairs[0][1]=2; nseg=1; break;
        case 5: case 10: {
            float d = v00*v11 - v10*v01; // asymptotic decider
            if (c == 5) {
                if (d > 0) { pairs[0][0]=3; pairs[0][1]=0; pairs[1][0]=1; pairs[1][1]=2; }
                else       { pairs[0][0]=0; pairs[0][1]=1; pairs[1][0]=2; pairs[1][1]=3; }
            } else {
                if (d > 0) { pairs[0][0]=0; pairs[0][1]=1; pairs[1][0]=2; pairs[1][1]=3; }
                else       { pairs[0][0]=3; pairs[0][1]=0; pairs[1][0]=1; pairs[1][1]=2; }
            }
            nseg=2; break; }
        default: return;
    }

    for (int i=0;i<nseg;++i) {
        uint slot = atomic_inc(seg_counter);
        const uint base = slot * 4u;
        float2 a = p[pairs[i][0]];
        float2 b = p[pairs[i][1]];
        segments_out[base+0] = a.x; segments_out[base+1] = a.y;
        segments_out[base+2] = b.x; segments_out[base+3] = b.y;
    }
}

    for (int i=0;i<nseg;++i) {
        uint slot = atomic_inc(seg_counter);
        const uint base = slot * 4u;
        float2 a = p[pairs[i][0]];
        float2 b = p[pairs[i][1]];
        segments_out[base+0] = a.x; segments_out[base+1] = a.y;
        segments_out[base+2] = b.x; segments_out[base+3] = b.y;
    }
}
"""


# ----------------------------- SVG Writer -------------------------------------

def write_svg_segments(path: str, segs: np.ndarray, w: int, h: int):
    from xml.etree.ElementTree import Element, SubElement, ElementTree
    svg = Element('svg', xmlns="http://www.w3.org/2000/svg", width=str(w), height=str(h), viewBox=f"0 0 {w} {h}")
    g = SubElement(svg, 'g', attrib={'fill':'none','stroke':'black','stroke-width':'0.75'})
    for x0, y0, x1, y1 in segs:
        SubElement(g, 'line', x1=f"{x1}", y1=f"{h-1-y1}", x2=f"{x0}", y2=f"{h-1-y0}")
    ElementTree(svg).write(path)


# ----------------------------- Potrace-like tracing ----------------------------

# Segment chaining: build polylines from unordered [x0,y0,x1,y1] segments
from typing import List, Tuple, Dict

def chain_segments_to_polylines(segs: np.ndarray, tol: float = 1e-3) -> List[np.ndarray]:
    """Greedy chaining of segment soup into polylines. segs: [N,4] in pixel coords.
    Returns list of (M_i, 2) arrays. Works by quantizing endpoints and walking adjacency.
    """
    if segs.size == 0:
        return []
    # Quantize endpoints for stable hashing
    def q2(p):
        return (round(float(p[0])/tol)*tol, round(float(p[1])/tol)*tol)
    # Build adjacency
    pts: Dict[Tuple[float,float], List[Tuple[float,float]]] = {}
    for x0,y0,x1,y1 in segs:
        a = q2((x0, y0)); b = q2((x1, y1))
        pts.setdefault(a, []).append(b)
        pts.setdefault(b, []).append(a)
    visited_edges: Dict[Tuple[Tuple[float,float], Tuple[float,float]], bool] = {}
    polylines: List[np.ndarray] = []

    def pop_edge(u):
        lst = pts.get(u)
        while lst:
            v = lst.pop()
            e = (u, v) if u < v else (v, u)
            if not visited_edges.get(e):
                visited_edges[e] = True
                return v
        return None

    # Start with nodes of odd degree or 1 (path ends), then cycles
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

    # Drain remaining cycles
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

# Douglas–Peucker simplification (polyline)

def douglas_peucker(poly: np.ndarray, tol: float) -> np.ndarray:
    """Simplify an open polyline with Douglas–Peucker. tol is in same units as poly.
    If closed (first≈last), we simplify the open form and leave closure to the caller.
    """
    if poly.shape[0] <= 2 or tol <= 0:
        return poly
    closed = np.linalg.norm(poly[0] - poly[-1]) < 1e-9
    if closed:
        poly = poly[:-1]
        if poly.shape[0] <= 2:
            return np.vstack([poly, poly[0]]) if closed else poly
    keep = np.zeros(poly.shape[0], dtype=bool)
    keep[0] = True; keep[-1] = True
    stack: List[Tuple[int,int]] = [(0, poly.shape[0]-1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        a, b = poly[i0], poly[i1]
        seg = b - a
        seg2 = float(np.dot(seg, seg))
        pts = poly[i0+1:i1]
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

# Back-compat alias (common misspelling without the 'c')
douglas_peuker = douglas_peucker

def close_and_filter_polylines(polys: List[np.ndarray], close_tol_px: float = 1.0, min_area_px2: float = 0.0) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for poly in polys:
        if poly.shape[0] < 2:
            continue
        # Close if endpoints near
        if np.linalg.norm(poly[0] - poly[-1]) > close_tol_px:
            poly = np.vstack([poly, poly[0]])
        # Need at least a triangle
        if poly.shape[0] < 4:  # including repeated first/last
            continue
        # Area filter
        A = polygon_area(poly)
        if abs(A) < float(min_area_px2):
            continue
        out.append(poly)
    return out


def polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))


def point_in_polygon(pt: Tuple[float, float], poly: np.ndarray) -> bool:
    # Ray casting (even-odd rule). Assumes poly is closed (first==last).
    x, y = pt
    cnt = 0
    for i in range(poly.shape[0] - 1):
        x0, y0 = poly[i]
        x1, y1 = poly[i+1]
        if ((y0 > y) != (y1 > y)):
            t = (y - y0) / (y1 - y0 + 1e-20)
            xr = x0 + t * (x1 - x0)
            if xr > x:
                cnt ^= 1
    return bool(cnt)


def build_regions_with_holes(rings: List[np.ndarray]) -> List[dict]:
    """Group rings into regions (outer with optional holes) by nesting.
    Returns list of { 'outer': ndarray, 'holes': [ndarray, ...] }
    Orientation: outer CCW (area>0), holes CW (area<0).
    """
    if not rings:
        return []
    # Normalize orientation and compute bbox/centroid
    recs = []
    for r in rings:
        A = polygon_area(r)
        rr = r
        if A < 0:
            rr = r[::-1].copy()  # make CCW
            A = -A
        # centroid for simple polys
        cx = float(np.mean(rr[:-1, 0]))
        cy = float(np.mean(rr[:-1, 1]))
        recs.append({'ring': rr, 'area': A, 'centroid': (cx, cy)})
    # Sort by area ascending so smaller rings find nearest parent
    recs.sort(key=lambda t: t['area'])
    parents = [-1] * len(recs)
    for i in range(len(recs)):
        c = recs[i]['centroid']
        best = -1
        best_area = 1e30
        for j in range(i+1, len(recs)):
            if recs[j]['area'] < best_area and point_in_polygon(c, recs[j]['ring']):
                best = j
                best_area = recs[j]['area']
        parents[i] = best
    # Build outer→holes mapping
    regions: Dict[int, dict] = {}
    for i, p in enumerate(parents):
        if p == -1:
            regions[i] = {'outer': recs[i]['ring'], 'holes': []}
        else:
            # ensure hole orientation CW
            hole = recs[i]['ring'][::-1].copy()
            while p != -1 and parents[p] != -1:
                # even-odd correction for deep nesting: flip on each level
                hole = hole[::-1].copy()
                p = parents[p]
            regions.setdefault(p, {'outer': recs[p]['ring'], 'holes': []})['holes'].append(hole)
    # Emit in increasing outer area
    ordered_keys = sorted(regions.keys(), key=lambda k: polygon_area(regions[k]['outer']))
    return [regions[k] for k in ordered_keys]


def _ring_to_cubic_beziers(ring: np.ndarray, tension: float = 0.0) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Convert a closed ring (N,2, last==first) to a list of cubic Bezier segments.
    Uniform Catmull–Rom spline converted to Cubic Beziers with adjustable tension.
    tension in [0,1]: 0 = classic Catmull–Rom (smoothest), 1 = straight segments.
    Control scale s = (1 - tension) / 6.
    """
    pts = ring[:-1]  # drop duplicate last
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


def write_svg_regions(path: str, regions: List[dict], w: int, h: int, stroke_width: float = 1.0, fill: Optional[str] = None, curve: str = 'none', tension: float = 0.0):
    """Write regions as SVG. If curve=='cubic', output Catmull–Rom→Bezier smoothed paths.
    tension∈[0,1]: 0 smooth, 1 straight. """
    from xml.etree.ElementTree import Element, SubElement, ElementTree

    def _fmt_xy(xy: np.ndarray) -> Tuple[float, float]:
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
        # Move to first point
        p1, c1, c2, p2 = bez[0]
        x0, y0 = _fmt_xy(p1)
        cmds = [f"M {x0:.3f} {y0:.3f}"]
        for (p1, c1, c2, p2) in bez:
            x1,y1 = _fmt_xy(c1); x2,y2 = _fmt_xy(c2); x3,y3 = _fmt_xy(p2)
            cmds.append(f"C {x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f} {x3:.3f} {y3:.3f}")
        cmds.append("Z")
        return " ".join(cmds)

    svg = Element('svg', xmlns="http://www.w3.org/2000/svg", width=str(w), height=str(h), viewBox=f"0 0 {w} {h}")
    attrib={'fill':'none' if fill is None else fill, 'stroke':'black', 'stroke-width':f'{stroke_width:.3f}', 'fill-rule':'evenodd'}
    g = SubElement(svg, 'g', attrib=attrib)
    for reg in regions:
        if curve == 'cubic':
            d = path_from_ring_cubic(reg['outer'])
        else:
            d = path_from_ring_poly(reg['outer'])
        for hole in reg['holes']:
            if curve == 'cubic':
                d += ' ' + path_from_ring_cubic(hole)
            else:
                d += ' ' + path_from_ring_poly(hole)
        SubElement(g, 'path', d=d)
    ElementTree(svg).write(path)
    return path
    """Write each region as a <path> that includes its outer and hole subpaths.
    Uses fill-rule="evenodd"; by default we draw outlines only (fill none).
    """
    from xml.etree.ElementTree import Element, SubElement, ElementTree
    def path_from_ring(r: np.ndarray) -> str:
        cmds = [f"M {r[0,0]:.3f} {h-1-r[0,1]:.3f}"]
        for i in range(1, r.shape[0]):
            cmds.append(f"L {r[i,0]:.3f} {h-1-r[i,1]:.3f}")
        cmds.append("Z")
        return " ".join(cmds)

    svg = Element('svg', xmlns="http://www.w3.org/2000/svg", width=str(w), height=str(h), viewBox=f"0 0 {w} {h}")
    g = SubElement(svg, 'g', attrib={'fill':'none','stroke':'black','stroke-width':f'{stroke_width:.3f}','fill-rule':'evenodd'})
    for reg in regions:
        d = path_from_ring(reg['outer'])
        for hole in reg['holes']:
            d += ' ' + path_from_ring(hole)
        SubElement(g, 'path', d=d)
    ElementTree(svg).write(path)


def trace_bitmap(mask_u8: np.ndarray, simplify_tol_px: float = 0.75, min_area_px2: float = 4.0, chain_tol_px: float = 0.25) -> List[np.ndarray]:
    """Extract closed rings from a binary mask using marching squares at iso=0.5.
    Returns list of closed rings (numpy arrays Nx2, pixel coords, last==first).
    """
    segs = marching_squares_cpu(mask_u8.astype(np.float32), iso=0.5)
    polys = chain_segments_to_polylines(segs, tol=chain_tol_px)
    # Simplify each before closing/area filters to keep topology cleaner
    simp = []
    for p in polys:
        if p.shape[0] < 3:
            continue
        simp.append(douglas_peucker(p, simplify_tol_px))
    rings = close_and_filter_polylines(simp, close_tol_px=1.0, min_area_px2=min_area_px2)
    return rings

# ----------------------------- GPU Path ---------------------------------------



def run_gpu(mask_u8: np.ndarray, params: "DemoParams"):
    assert HAVE_OPENCL, "PyOpenCL not available"
    ctx, queue = try_create_ctx_queue()
    if ctx is None or queue is None:
        raise RuntimeError("No OpenCL context/devices found")

    W, H = mask_u8.shape[1], mask_u8.shape[0]
    N = W * H

    prg = cl.Program(ctx, KERNEL_SRC).build()
    mf = cl.mem_flags

    mask_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mask_u8)
    seed_a = cl.Buffer(ctx, mf.READ_WRITE, size=N * np.dtype(np.int32).itemsize * 2)
    seed_b = cl.Buffer(ctx, mf.READ_WRITE, size=N * np.dtype(np.int32).itemsize * 2)
    dist_d = cl.Buffer(ctx, mf.READ_WRITE, size=N * np.dtype(np.float32).itemsize)

    ensure_fill(queue, seed_a, bytes_pattern_of(np.int32(-1)))
    ensure_fill(queue, seed_b, bytes_pattern_of(np.int32(-1)))
    ensure_fill(queue, dist_d, bytes_pattern_of(np.float32(1e30)))

    gsize = (W, H)
    prg.init_seed_from_mask(queue, gsize, None, mask_d, np.int32(W), np.int32(H), seed_a, dist_d)

    max_dim = max(W, H)
    jump = 1
    while (jump << 1) < max_dim:
        jump <<= 1
    cur_in, cur_out = seed_a, seed_b
    while jump >= 1:
        prg.jfa_step(queue, gsize, None, np.int32(W), np.int32(H), np.int32(jump), cur_in, dist_d, cur_out)
        queue.finish()
        cur_in, cur_out = cur_out, cur_in
        jump >>= 1

    max_segments = (W-1) * (H-1) * 2
    seg_counter = cl.Buffer(ctx, mf.READ_WRITE, size=4)
    ensure_fill(queue, seg_counter, bytes_pattern_of(np.uint32(0)))
    seg_out = cl.Buffer(ctx, mf.READ_WRITE, size=max_segments * 4 * np.dtype(np.float32).itemsize)
    ensure_fill(queue, seg_out, bytes_pattern_of(np.float32(np.nan)))

    prg.marching_squares(queue, gsize, None, np.int32(W), np.int32(H), dist_d, np.float32(0.0), seg_counter, seg_out)
    queue.finish()

    seg_count_h = np.empty(1, dtype=np.uint32)
    cl.enqueue_copy(queue, seg_count_h, seg_counter, is_blocking=True)
    count = int(seg_count_h[0])
    count = min(count, max_segments)

    segs_h = np.empty(count*4, dtype=np.float32)
    cl.enqueue_copy(queue, segs_h, seg_out, is_blocking=True)
    segs_h = segs_h.reshape(-1, 4)
    return segs_h, (W, H)


# ----------------------------- CPU Path ---------------------------------------

def run_cpu(mask_u8: np.ndarray, params: "DemoParams"):
    sdf = sdf_from_mask_cpu(mask_u8)
    sanity_no_nan_f32(sdf)
    segs = marching_squares_cpu(sdf, iso=params.iso)
    H, W = sdf.shape
    return segs, (W, H), sdf


# ----------------------------- Driver -----------------------------------------

@dataclass
class DemoParams:
    width: int = 512
    height: int = 512
    source: str = "circle"               # "circle" | "image"
    image_path: Optional[str] = None
    threshold: Optional[float] = None
    invert: bool = False
    svg_out: str = "traced.svg"
    force_cpu: bool = True               # tracing pipeline uses CPU by default
    # Tracing controls
    simplify_tol_px: float = 0.75
    min_area_px2: float = 4.0
    chain_tol_px: float = 0.25
    stroke_width: float = 1.0
    # Curve output
    curve: str = 'none'                  # 'none' or 'cubic'
    tension: float = 0.0                 # 0=smooth Catmull–Rom, 1=straight


def build_circle_mask(w: int, h: int, r: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w//2, h//2
    return (((xx - cx)**2 + (yy - cy)**2) <= r*r).astype(np.uint8)


def run_demo(params: DemoParams = DemoParams()):
    # Source mask
    if params.source == "image":
        if not params.image_path:
            raise SystemExit("--image requires --image PATH")
        mask = load_mask_from_image(params.image_path, invert=params.invert, threshold=params.threshold)
        H, W = mask.shape
    else:
        W, H = params.width, params.height
        mask = build_circle_mask(W, H, r=min(W, H)//3)

    # Trace to closed rings
    rings = trace_bitmap(mask, simplify_tol_px=params.simplify_tol_px, min_area_px2=params.min_area_px2, chain_tol_px=params.chain_tol_px)
    # Group into regions with holes
    regions = build_regions_with_holes(rings)
    write_svg_regions(params.svg_out, regions, W, H, stroke_width=params.stroke_width, curve=params.curve, tension=params.tension)
    n_rings = sum(1 + len(r['holes']) for r in regions)
    print(f"Traced: {len(regions)} regions, {n_rings} rings → wrote {params.svg_out} ({W}x{H})")

# ----------------------------- CLI --------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Potrace-like tracing: bitmap → closed SVG curves")
    p.add_argument('--image', type=str, default=None, help='Path to a black/white image to use as mask')
    p.add_argument('--invert', action='store_true', help='Invert mask after thresholding')
    p.add_argument('--threshold', type=float, default=None, help='Manual threshold (0..255 or 0..1). Default: Otsu')
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--height', type=int, default=512)
    p.add_argument('--out', type=str, default='traced.svg')
    p.add_argument('--simplify-tol-px', type=float, default=0.75, help='Douglas–Peucker tolerance in pixels')
    p.add_argument('--min-area-px2', type=float, default=4.0, help='Discard rings with |area| below this (pixel^2)')
    p.add_argument('--chain-tol-px', type=float, default=0.25, help='Endpoint snapping tolerance when chaining segments (pixels)')
    p.add_argument('--stroke-width', type=float, default=1.0, help='SVG stroke width (pixels)')
    p.add_argument('--curve', type=str, default='none', choices=['none','cubic'], help='Output curves as cubic Beziers')
    p.add_argument('--tension', type=float, default=0.0, help='Cubic curve tension: 0=smooth CR, 1=straight lines')
    args = p.parse_args()

    src = 'image' if args.image is not None else 'circle'
    params = DemoParams(
        width=args.width, height=args.height, source=src, image_path=args.image,
        threshold=args.threshold, invert=args.invert, svg_out=args.out,
        simplify_tol_px=args.simplify_tol_px, min_area_px2=args.min_area_px2, stroke_width=args.stroke_width,
    )
    run_demo(params)

from __future__ import annotations

import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - fallback when numba unavailable
    HAVE_NUMBA = False


    def njit(*args, **kwargs):  # type: ignore
        def _wrap(f):
            return f

        return _wrap


@njit(cache=True, fastmath=True)
def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Smooth Hermite interpolation between 0 and 1."""
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True, fastmath=True)
def _blend_scalar(dst: np.ndarray, x: int, y: int, c: float, alpha: float):
    if 0 <= y < dst.shape[0] and 0 <= x < dst.shape[1]:
        dst[y, x] = (1.0 - alpha) * dst[y, x] + alpha * c


@njit(cache=True, fastmath=True)
def _draw_line_scalar(
        img: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        width: float,
        aa_width: float,
        color: float,
        alpha: float,
):
    xmin = int(max(0, min(x0, x1) - width - aa_width - 1))
    xmax = int(min(img.shape[1] - 1, max(x0, x1) + width + aa_width + 1))
    ymin = int(max(0, min(y0, y1) - width - aa_width - 1))
    ymax = int(min(img.shape[0] - 1, max(y0, y1) + width + aa_width + 1))
    vx = x1 - x0
    vy = y1 - y0
    seg2 = vx * vx + vy * vy + 1e-20
    threshold = width * 0.5
    thr2 = threshold * threshold
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            wx = x - x0
            wy = y - y0
            t = (wx * vx + wy * vy) / seg2
            if t < 0:
                px, py = x0, y0
            elif t > 1:
                px, py = x1, y1
            else:
                px = x0 + t * vx
                py = y0 + t * vy
            dx = x - px
            dy = y - py
            dist2 = dx * dx + dy * dy

            if aa_width > 0.0:
                # Anti-aliased edge
                dist = np.sqrt(dist2)
                edge_outer = threshold + aa_width
                edge_inner = threshold - aa_width
                blend_t = _smoothstep(edge_outer, edge_inner, dist)
                if blend_t > 0.0:
                    _blend_scalar(img, x, y, color, alpha * blend_t)
            else:
                # Hard edge (original behavior)
                if dist2 <= thr2:
                    _blend_scalar(img, x, y, color, alpha)


@njit(cache=True, fastmath=True)
def _draw_rect_scalar(
        img: np.ndarray, x0: int, y0: int, x1: int, y1: int, aa_width: float, color: float, alpha: float
):
    if x0 <= x1:
        xmin, xmax = int(x0), int(x1)
    else:
        xmin, xmax = int(x1), int(x0)
    if y0 <= y1:
        ymin, ymax = int(y0), int(y1)
    else:
        ymin, ymax = int(y1), int(y0)

    # Expand bounds for anti-aliasing
    if aa_width > 0.0:
        xmin_draw = max(0, int(xmin - aa_width - 1))
        xmax_draw = min(img.shape[1] - 1, int(xmax + aa_width + 1))
        ymin_draw = max(0, int(ymin - aa_width - 1))
        ymax_draw = min(img.shape[0] - 1, int(ymax + aa_width + 1))
    else:
        xmin_draw = max(0, xmin)
        xmax_draw = min(img.shape[1] - 1, xmax)
        ymin_draw = max(0, ymin)
        ymax_draw = min(img.shape[0] - 1, ymax)

    if xmin_draw > xmax_draw or ymin_draw > ymax_draw:
        return

    for y in range(ymin_draw, ymax_draw + 1):
        for x in range(xmin_draw, xmax_draw + 1):
            inside = (xmin <= x <= xmax and ymin <= y <= ymax)

            if aa_width > 0.0 and inside:
                # Anti-aliased edges using distance to nearest edge
                dist_l = float(x) - float(xmin)
                dist_r = float(xmax) - float(x)
                dist_b = float(y) - float(ymin)
                dist_t = float(ymax) - float(y)
                edge_dist = min(min(dist_l, dist_r), min(dist_b, dist_t))

                blend_t = _smoothstep(0.0, aa_width, edge_dist)
                _blend_scalar(img, x, y, color, alpha * blend_t)
            elif inside:
                # Hard edge (original behavior)
                _blend_scalar(img, x, y, color, alpha)


@njit(cache=True, fastmath=True)
def _draw_circle_scalar(
        img: np.ndarray, cx: int, cy: int, r: float, aa_width: float, color: float, alpha: float
):
    xmin = int(max(0, cx - r - aa_width - 1))
    xmax = int(min(img.shape[1] - 1, cx + r + aa_width + 1))
    ymin = int(max(0, cy - r - aa_width - 1))
    ymax = int(min(img.shape[0] - 1, cy + r + aa_width + 1))
    for y in range(ymin, ymax + 1):
        dy = y - cy
        for x in range(xmin, xmax + 1):
            dx = x - cx
            dist = np.sqrt(dx * dx + dy * dy)

            if aa_width > 0.0:
                # Anti-aliased edge
                edge_outer = r + aa_width
                edge_inner = r - aa_width
                blend_t = _smoothstep(edge_outer, edge_inner, dist)
                if blend_t > 0.0:
                    _blend_scalar(img, x, y, color, alpha * blend_t)
            else:
                # Hard edge (original behavior)
                if dist <= r:
                    _blend_scalar(img, x, y, color, alpha)


@njit(cache=True, fastmath=True)
def _edge(ax, ay, bx, by, px, py):
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax)


@njit(cache=True, fastmath=True)
def _dist_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Compute distance from point (px, py) to line segment (a, b)."""
    pax = px - ax
    pay = py - ay
    bax = bx - ax
    bay = by - ay
    dot_ba = bax * bax + bay * bay
    if dot_ba == 0.0:
        return np.sqrt(pax * pax + pay * pay)
    h = max(0.0, min(1.0, (pax * bax + pay * bay) / dot_ba))
    dx = pax - h * bax
    dy = pay - h * bay
    return np.sqrt(dx * dx + dy * dy)


@njit(cache=True, fastmath=True)
def _draw_triangle_scalar(
        img: np.ndarray, x0, y0, x1, y1, x2, y2, aa_width: float, color: float, alpha: float
):
    # Expand bounds for anti-aliasing
    if aa_width > 0.0:
        xmin = int(max(0, min(x0, x1, x2) - aa_width - 1))
        xmax = int(min(img.shape[1] - 1, max(x0, x1, x2) + aa_width + 1))
        ymin = int(max(0, min(y0, y1, y2) - aa_width - 1))
        ymax = int(min(img.shape[0] - 1, max(y0, y1, y2) + aa_width + 1))
    else:
        xmin = int(max(0, min(x0, x1, x2)))
        xmax = int(min(img.shape[1] - 1, max(x0, x1, x2)))
        ymin = int(max(0, min(y0, y1, y2)))
        ymax = int(min(img.shape[0] - 1, max(y0, y1, y2)))

    area = _edge(x0, y0, x1, y1, x2, y2)
    if area == 0:
        return
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            w0 = _edge(x1, y1, x2, y2, x, y)
            w1 = _edge(x2, y2, x0, y0, x, y)
            w2 = _edge(x0, y0, x1, y1, x, y)
            inside = (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0)

            if aa_width > 0.0 and inside:
                # Anti-aliased edges using distance to nearest edge
                dist1 = _dist_to_segment(float(x), float(y), float(x0), float(y0), float(x1), float(y1))
                dist2 = _dist_to_segment(float(x), float(y), float(x1), float(y1), float(x2), float(y2))
                dist3 = _dist_to_segment(float(x), float(y), float(x2), float(y2), float(x0), float(y0))
                edge_dist = min(min(dist1, dist2), dist3)

                blend_t = _smoothstep(0.0, aa_width, edge_dist)
                if blend_t > 0.0:
                    _blend_scalar(img, x, y, color, alpha * blend_t)
            elif inside:
                # Hard edge (original behavior)
                _blend_scalar(img, x, y, color, alpha)


@njit(cache=True, fastmath=True)
def _draw_ellipse_scalar(
        img: np.ndarray, cx: int, cy: int, rx: float, ry: float, aa_width: float, color: float, alpha: float
):
    # Expand bounds for anti-aliasing
    if aa_width > 0.0:
        xmin = int(max(0, cx - rx - aa_width - 1))
        xmax = int(min(img.shape[1] - 1, cx + rx + aa_width + 1))
        ymin = int(max(0, cy - ry - aa_width - 1))
        ymax = int(min(img.shape[0] - 1, cy + ry + aa_width + 1))
    else:
        xmin = int(max(0, cx - rx - 1))
        xmax = int(min(img.shape[1] - 1, cx + rx + 1))
        ymin = int(max(0, cy - ry - 1))
        ymax = int(min(img.shape[0] - 1, cy + ry + 1))

    for y in range(ymin, ymax + 1):
        dy = float(y - cy)
        for x in range(xmin, xmax + 1):
            dx = float(x - cx)
            # Compute normalized distance
            delta_x = dx / rx
            delta_y = dy / ry
            d = np.sqrt(delta_x * delta_x + delta_y * delta_y)

            if aa_width > 0.0:
                # Anti-aliased edge using smoothstep
                # d < 1: inside, d = 1: on boundary, d > 1: outside
                avg_radius = (rx + ry) * 0.5
                dist_to_edge = (d - 1.0) * avg_radius
                edge_outer = aa_width
                edge_inner = -aa_width
                blend_t = _smoothstep(edge_outer, edge_inner, dist_to_edge)
                if blend_t > 0.0:
                    _blend_scalar(img, x, y, color, alpha * blend_t)
            else:
                # Hard edge (original behavior)
                if d <= 1.0:
                    _blend_scalar(img, x, y, color, alpha)


def lines(
        width: int,
        height: int,
        a: np.ndarray,
        nelements: int,
        coords: np.ndarray,
        widths: np.ndarray,
        aa_widths: np.ndarray,
        colors: np.ndarray,
):
    for i in range(nelements):
        # Swap coords to match OCL/Metal column-major convention
        # coords are passed as (row1, col1, row2, col2)
        x0 = int(coords[i, 1])  # col1
        y0 = int(coords[i, 0])  # row1
        x1 = int(coords[i, 3])  # col2
        y1 = int(coords[i, 2])  # row2
        w = float(widths[i])
        aa_w = float(aa_widths[i])
        if colors.ndim == 1:
            c = float(colors[i])
            alpha = 1.0
        else:
            c = float(colors[i, 0])
            alpha = float(colors[i, 1])
        _draw_line_scalar(a, x0, y0, x1, y1, w, aa_w, c, alpha)
    return a


def rects(
        width: int,
        height: int,
        a: np.ndarray,
        nelements: int,
        coords: np.ndarray,
        aa_widths: np.ndarray,
        colors: np.ndarray,
):
    for i in range(nelements):
        # Swap coords to match OCL/Metal column-major convention
        # coords are passed as (row1, col1, row2, col2)
        # but OCL treats them in column-major order
        x0 = int(coords[i, 1])  # col1
        y0 = int(coords[i, 0])  # row1
        x1 = int(coords[i, 3])  # col2
        y1 = int(coords[i, 2])  # row2
        aa_w = float(aa_widths[i])
        if colors.ndim == 1:
            c = float(colors[i])
            alpha = 1.0
        else:
            c = float(colors[i, 0])
            alpha = float(colors[i, 1])
        _draw_rect_scalar(a, x0, y0, x1, y1, aa_w, c, alpha)
    return a


def circles(
        width: int,
        height: int,
        a: np.ndarray,
        nelements: int,
        coords: np.ndarray,
        aa_widths: np.ndarray,
        colors: np.ndarray,
):
    for i in range(nelements):
        # Swap coords to match OCL/Metal column-major convention
        # coords are passed as (row, col, radius)
        cx = int(coords[i, 1])  # col
        cy = int(coords[i, 0])  # row
        r = float(coords[i, 2])
        aa_w = float(aa_widths[i])
        if colors.ndim == 1:
            c = float(colors[i])
            alpha = 1.0
        else:
            c = float(colors[i, 0])
            alpha = float(colors[i, 1])
        _draw_circle_scalar(a, cx, cy, r, aa_w, c, alpha)
    return a


def triangles(
        width: int,
        height: int,
        a: np.ndarray,
        nelements: int,
        coords: np.ndarray,
        aa_widths: np.ndarray,
        colors: np.ndarray,
):
    for i in range(nelements):
        # Swap coords to match OCL/Metal column-major convention
        # coords are passed as (row1, col1, row2, col2, row3, col3)
        x0 = int(coords[i, 1])  # col1
        y0 = int(coords[i, 0])  # row1
        x1 = int(coords[i, 3])  # col2
        y1 = int(coords[i, 2])  # row2
        x2 = int(coords[i, 5])  # col3
        y2 = int(coords[i, 4])  # row3
        aa_w = float(aa_widths[i])
        if colors.ndim == 1:
            c = float(colors[i])
            alpha = 1.0
        else:
            c = float(colors[i, 0])
            alpha = float(colors[i, 1])
        _draw_triangle_scalar(a, x0, y0, x1, y1, x2, y2, aa_w, c, alpha)
    return a


def ellipses(
        width: int,
        height: int,
        a: np.ndarray,
        nelements: int,
        coords: np.ndarray,
        aa_widths: np.ndarray,
        colors: np.ndarray,
):
    for i in range(nelements):
        # Swap coords to match OCL/Metal column-major convention
        # coords are passed as (row, col, width, height)
        cx = int(coords[i, 1])  # col
        cy = int(coords[i, 0])  # row
        # OCL/Metal also swap width/height for ellipses
        rx = float(coords[i, 3])  # height (swapped to match OCL)
        ry = float(coords[i, 2])  # width (swapped to match OCL)
        aa_w = float(aa_widths[i])
        if colors.ndim == 1:
            c = float(colors[i])
            alpha = 1.0
        else:
            c = float(colors[i, 0])
            alpha = float(colors[i, 1])
        _draw_ellipse_scalar(a, cx, cy, rx, ry, aa_w, c, alpha)
    return a


def composite(
    width: int,
    height: int,
    a: np.ndarray,
    nelements: int,
    coords: np.ndarray,
    pixels: np.ndarray,
    ipixels: np.ndarray,
    npixels: int,
):
    # coords: N x 4 (col, row, w, h), pixels: concatenated [val, alpha] pairs per pixel, ipixels: start index per block
    # In yapocis column-major convention:
    #   width = shape[0] = number of rows
    #   height = shape[1] = number of cols
    for i in range(nelements):
        # coords are passed as (col, row, block_width, block_height)
        x = int(coords[i, 0])  # col position in canvas
        y = int(coords[i, 1])  # row position in canvas
        w = int(coords[i, 2])  # block width (cols)
        h = int(coords[i, 3])  # block height (rows)
        start = int(ipixels[i])
        block = pixels[start : start + w * h].reshape(h, w, 2)
        # Clamp row bounds to width (number of rows), col bounds to height (number of cols)
        row0 = max(0, y)
        col0 = max(0, x)
        row1 = min(width, y + h)
        col1 = min(height, x + w)
        for row in range(row0, row1):
            for col in range(col0, col1):
                srcv = block[row - y, col - x, 0]
                al = block[row - y, col - x, 1]
                a[row, col] = (1.0 - al) * a[row, col] + al * srcv
    return a

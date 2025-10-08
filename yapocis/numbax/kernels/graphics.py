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
    color: float,
    alpha: float,
):
    xmin = int(max(0, min(x0, x1) - width - 1))
    xmax = int(min(img.shape[1] - 1, max(x0, x1) + width + 1))
    ymin = int(max(0, min(y0, y1) - width - 1))
    ymax = int(min(img.shape[0] - 1, max(y0, y1) + width + 1))
    vx = x1 - x0
    vy = y1 - y0
    seg2 = vx * vx + vy * vy + 1e-20
    thr2 = (width * 0.5) * (width * 0.5)
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
            if dx * dx + dy * dy <= thr2:
                _blend_scalar(img, x, y, color, alpha)


@njit(cache=True, fastmath=True)
def _draw_rect_scalar(
    img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: float, alpha: float
):
    if x0 <= x1:
        xmin, xmax = int(x0), int(x1)
    else:
        xmin, xmax = int(x1), int(x0)
    if y0 <= y1:
        ymin, ymax = int(y0), int(y1)
    else:
        ymin, ymax = int(y1), int(y0)
    xmin = max(0, xmin)
    xmax = min(img.shape[1] - 1, xmax)
    ymin = max(0, ymin)
    ymax = min(img.shape[0] - 1, ymax)
    if xmin > xmax or ymin > ymax:
        return
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            _blend_scalar(img, x, y, color, alpha)


@njit(cache=True, fastmath=True)
def _draw_circle_scalar(
    img: np.ndarray, cx: int, cy: int, r: float, color: float, alpha: float
):
    r2 = r * r
    xmin = int(max(0, cx - r - 1))
    xmax = int(min(img.shape[1] - 1, cx + r + 1))
    ymin = int(max(0, cy - r - 1))
    ymax = int(min(img.shape[0] - 1, cy + r + 1))
    for y in range(ymin, ymax + 1):
        dy = y - cy
        for x in range(xmin, xmax + 1):
            dx = x - cx
            if dx * dx + dy * dy <= r2:
                _blend_scalar(img, x, y, color, alpha)


@njit(cache=True, fastmath=True)
def _edge(ax, ay, bx, by, px, py):
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax)


@njit(cache=True, fastmath=True)
def _draw_triangle_scalar(
    img: np.ndarray, x0, y0, x1, y1, x2, y2, color: float, alpha: float
):
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
            if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                _blend_scalar(img, x, y, color, alpha)


@njit(cache=True, fastmath=True)
def _draw_ellipse_scalar(
    img: np.ndarray, cx: int, cy: int, rx: float, ry: float, color: float, alpha: float
):
    xmin = int(max(0, cx - rx - 1))
    xmax = int(min(img.shape[1] - 1, cx + rx + 1))
    ymin = int(max(0, cy - ry - 1))
    ymax = int(min(img.shape[0] - 1, cy + ry + 1))
    rx2 = rx * rx
    ry2 = ry * ry
    for y in range(ymin, ymax + 1):
        dy2 = float((y - cy) * (y - cy))
        for x in range(xmin, xmax + 1):
            dx2 = float((x - cx) * (x - cx))
            if dx2 / rx2 + dy2 / ry2 <= 1.0:
                _blend_scalar(img, x, y, color, alpha)


def lines(
    width: int,
    height: int,
    a: np.ndarray,
    nelements: int,
    coords: np.ndarray,
    widths: np.ndarray,
    colors: np.ndarray,
):
    for i in range(nelements):
        x0 = int(coords[i, 0])
        y0 = int(coords[i, 1])
        x1 = int(coords[i, 2])
        y1 = int(coords[i, 3])
        w = float(widths[i])
        if colors.ndim == 1:
            c = float(colors[i])
            alpha = 1.0
        else:
            c = float(colors[i, 0])
            alpha = float(colors[i, 1])
        _draw_line_scalar(a, x0, y0, x1, y1, w, c, alpha)
    return a


def rects(
    width: int,
    height: int,
    a: np.ndarray,
    nelements: int,
    coords: np.ndarray,
    colors: np.ndarray,
):
    for i in range(nelements):
        x0 = int(coords[i, 0])
        y0 = int(coords[i, 1])
        x1 = int(coords[i, 2])
        y1 = int(coords[i, 3])
        if colors.ndim == 1:
            c = float(colors[i])
            alpha = 1.0
        else:
            c = float(colors[i, 0])
            alpha = float(colors[i, 1])
        _draw_rect_scalar(a, x0, y0, x1, y1, c, alpha)
    return a


def circles(
    width: int,
    height: int,
    a: np.ndarray,
    nelements: int,
    coords: np.ndarray,
    colors: np.ndarray,
):
    for i in range(nelements):
        cx = int(coords[i, 0])
        cy = int(coords[i, 1])
        r = float(coords[i, 2])
        if colors.ndim == 1:
            c = float(colors[i])
            alpha = 1.0
        else:
            c = float(colors[i, 0])
            alpha = float(colors[i, 1])
        _draw_circle_scalar(a, cx, cy, r, c, alpha)
    return a


def triangles(
    width: int,
    height: int,
    a: np.ndarray,
    nelements: int,
    coords: np.ndarray,
    colors: np.ndarray,
):
    for i in range(nelements):
        x0 = int(coords[i, 0])
        y0 = int(coords[i, 1])
        x1 = int(coords[i, 2])
        y1 = int(coords[i, 3])
        x2 = int(coords[i, 4])
        y2 = int(coords[i, 5])
        if colors.ndim == 1:
            c = float(colors[i])
            alpha = 1.0
        else:
            c = float(colors[i, 0])
            alpha = float(colors[i, 1])
        _draw_triangle_scalar(a, x0, y0, x1, y1, x2, y2, c, alpha)
    return a


def ellipses(
    width: int,
    height: int,
    a: np.ndarray,
    nelements: int,
    coords: np.ndarray,
    colors: np.ndarray,
):
    for i in range(nelements):
        cx = int(coords[i, 0])
        cy = int(coords[i, 1])
        rx = float(coords[i, 2])
        ry = float(coords[i, 3])
        if colors.ndim == 1:
            c = float(colors[i])
            alpha = 1.0
        else:
            c = float(colors[i, 0])
            alpha = float(colors[i, 1])
        _draw_ellipse_scalar(a, cx, cy, rx, ry, c, alpha)
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
    # coords: N x 4 (x,y,w,h), pixels: concatenated [val, alpha] pairs per pixel, ipixels: start index per block
    for i in range(nelements):
        x = int(coords[i, 0])
        y = int(coords[i, 1])
        w = int(coords[i, 2])
        h = int(coords[i, 3])
        start = int(ipixels[i])
        block = pixels[start : start + w * h].reshape(h, w, 2)
        xs0 = max(0, y)
        ys0 = max(0, x)
        xs1 = min(height, y + h)
        ys1 = min(width, x + w)
        for yy in range(xs0, xs1):
            for xx in range(ys0, ys1):
                srcv = block[yy - y, xx - x, 0]
                al = block[yy - y, xx - x, 1]
                a[yy, xx] = (1.0 - al) * a[yy, xx] + al * srcv
    return a

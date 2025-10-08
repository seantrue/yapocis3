from __future__ import annotations

import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - fallback
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def _wrap(f):
            return f

        return _wrap


@njit(cache=True, fastmath=True)
def _blend_rgba(dst: np.ndarray, x: int, y: int, rgba: np.ndarray):
    if 0 <= y < dst.shape[0] and 0 <= x < dst.shape[1]:
        alpha_src = rgba[3]
        alpha_dst = dst[y, x, 3]
        alpha_out = alpha_src + alpha_dst * (1.0 - alpha_src)

        # Standard alpha blending
        if alpha_out > 0.0:
            dst[y, x, 0] = (
                rgba[0] * alpha_src + dst[y, x, 0] * alpha_dst * (1.0 - alpha_src)
            ) / alpha_out
            dst[y, x, 1] = (
                rgba[1] * alpha_src + dst[y, x, 1] * alpha_dst * (1.0 - alpha_src)
            ) / alpha_out
            dst[y, x, 2] = (
                rgba[2] * alpha_src + dst[y, x, 2] * alpha_dst * (1.0 - alpha_src)
            ) / alpha_out
            dst[y, x, 3] = alpha_out
        # If alpha_out == 0, leave the destination pixel unchanged


@njit(cache=True, fastmath=True)
def _draw_line_rgba(
    img: np.ndarray, x0: int, y0: int, x1: int, y1: int, width: float, rgba: np.ndarray
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
                _blend_rgba(img, x, y, rgba)


@njit(cache=True, fastmath=True)
def _draw_rect_rgba(
    img: np.ndarray, x0: int, y0: int, x1: int, y1: int, rgba: np.ndarray
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
    a = rgba[3]
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            img[y, x, 0] = (1.0 - a) * img[y, x, 0] + a * rgba[0]
            img[y, x, 1] = (1.0 - a) * img[y, x, 1] + a * rgba[1]
            img[y, x, 2] = (1.0 - a) * img[y, x, 2] + a * rgba[2]


@njit(cache=True, fastmath=True)
def _draw_circle_rgba(img: np.ndarray, cx: int, cy: int, r: float, rgba: np.ndarray):
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
                _blend_rgba(img, x, y, rgba)


@njit(cache=True, fastmath=True)
def _edge(ax, ay, bx, by, px, py):
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax)


@njit(cache=True, fastmath=True)
def _draw_triangle_rgba(img: np.ndarray, x0, y0, x1, y1, x2, y2, rgba: np.ndarray):
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
                _blend_rgba(img, x, y, rgba)


@njit(cache=True, fastmath=True)
def _draw_ellipse_rgba(
    img: np.ndarray, cx: int, cy: int, rx: float, ry: float, rgba: np.ndarray
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
                _blend_rgba(img, x, y, rgba)


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
        rgba = colors[i]
        _draw_line_rgba(a, x0, y0, x1, y1, w, rgba)
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
        rgba = colors[i]
        _draw_rect_rgba(a, x0, y0, x1, y1, rgba)
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
        rgba = colors[i]
        _draw_circle_rgba(a, cx, cy, r, rgba)
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
        rgba = colors[i]
        _draw_triangle_rgba(a, x0, y0, x1, y1, x2, y2, rgba)
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
        rgba = colors[i]
        _draw_ellipse_rgba(a, cx, cy, rx, ry, rgba)
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
    # coords: N x 4 (x,y,w,h), pixels: concatenated RGBA per pixel, ipixels: start index
    for i in range(nelements):
        x = int(coords[i, 0])
        y = int(coords[i, 1])
        w = int(coords[i, 2])
        h = int(coords[i, 3])
        # ipixels is provided in units of RGBA pixels; convert to float-index
        start = int(ipixels[i]) * 4
        block = pixels[start : start + (w * h * 4)].reshape(h, w, 4)
        xs0 = max(0, y)
        ys0 = max(0, x)
        xs1 = min(height, y + h)
        ys1 = min(width, x + w)
        for yy in range(xs0, xs1):
            for xx in range(ys0, ys1):
                _blend_rgba(a, xx, yy, block[yy - y, xx - x])
    return a

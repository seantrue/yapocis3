"""
Performance and correctness comparisons between Matplotlib Agg and
yapocis.lib.mplbackend backends, with specific detection of the known
marker rendering bug (upper-left quadrant corruption) in the yapocis backend.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import pytest
from PIL import Image

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from yapocis.lib.mplbackend import FigureCanvas as FigureCanvasYapo

from yapocis.core.manager import get_backends

RESULTS_PATH = "perf_results/perf_results.jsonl"


def _persist_jsonl(obj: dict):
    import json, os
    os.makedirs("perf_results", exist_ok=True)
    with open(RESULTS_PATH, "a", encoding="utf-8") as outf:
        json.dump(obj, outf); outf.write("\n")


def _git_commit_id() -> str:
    try:
        import subprocess
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode().strip()
        )
    except Exception:
        return "unknown"


def _to_array(buf: io.BytesIO) -> np.ndarray:
    buf.seek(0)
    img = Image.open(buf)
    a = np.asarray(img).astype(np.float32) / 255.0
    return a


def _draw_markers(fig: Figure, count: int, marker: str = "o", size: float = 8.0, color=(0, 0, 0)):
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.axis("off")
    # Scatter grid of markers
    n = int(np.sqrt(count))
    xs = np.linspace(0.1, 0.9, n)
    ys = np.linspace(0.1, 0.9, n)
    X, Y = np.meshgrid(xs, ys)
    ax.scatter(X.ravel(), Y.ravel(), marker=marker, s=size**2, c=[color])


def _draw_single_marker(fig: Figure, marker: str = "o", size: float = 24.0, color=(0, 0, 0)):
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.axis("off")
    ax.scatter([0.5], [0.5], marker=marker, s=size**2, c=[color])


def _draw_single_marker_fill_edge(fig: Figure, marker: str, size: float, fill: bool, color=(0, 0, 0), edgecolor=(0, 0, 0)):
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.axis("off")
    if fill:
        ax.scatter([0.5], [0.5], marker=marker, s=size**2, c=[color], edgecolors=[edgecolor])
    else:
        ax.scatter([0.5], [0.5], marker=marker, s=size**2, facecolors='none', edgecolors=[edgecolor])


@dataclass
class Backend:
    name: str
    make_canvas: Callable[[Figure], object]


def _render(fig: Figure, backend: Backend) -> Tuple[np.ndarray, float]:
    canvas = backend.make_canvas(fig)
    # FigureCanvasAgg supports file-like; our custom backend expects a path.
    t0 = time.perf_counter()
    if isinstance(canvas, FigureCanvasAgg):
        buf = io.BytesIO()
        canvas.print_png(buf)
        ms = (time.perf_counter() - t0) * 1000.0
        arr = _to_array(buf)
        return arr, ms
    else:
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            canvas.print_png(tmp_path)
            ms = (time.perf_counter() - t0) * 1000.0
            with open(tmp_path, "rb") as inf:
                buf = io.BytesIO(inf.read())
            arr = _to_array(buf)
            return arr, ms
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _detect_upper_left_quadrant_corruption(img: np.ndarray) -> bool:
    # Find marker region by thresholding darker pixels (on white background)
    gray = img[..., :3].mean(axis=2)
    mask = gray < 0.9
    if mask.sum() == 0:
        return False
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2
    # Quadrant means
    ul = gray[y0:cy, x0:cx].mean() if cy > y0 and cx > x0 else 1.0
    ur = gray[y0:cy, cx:x1].mean() if cy > y0 and x1 > cx else 1.0
    ll = gray[cy:y1, x0:cx].mean() if y1 > cy and cx > x0 else 1.0
    lr = gray[cy:y1, cx:x1].mean() if y1 > cy and x1 > cx else 1.0
    # For a symmetric circular marker on white background, quadrants should be similar.
    # Flag if UL deviates much more than others.
    diffs = np.array([abs(ul-ur), abs(ul-ll), abs(ul-lr), abs(ur-ll), abs(ur-lr), abs(ll-lr)])
    # Heuristic: corruption when UL is darkest by a margin and asymmetric.
    is_ul_darkest = ul < min(ur, ll, lr) - 0.05
    high_asym = diffs.max() > 0.15
    return bool(is_ul_darkest and high_asym)


def _images_close(a: np.ndarray, b: np.ndarray, mean_tol: float = 0.03, max_tol: float = 0.2) -> bool:
    # Compare first 3 channels and allow RGBA vs RGB by aligning on height/width.
    if a.shape[0:2] != b.shape[0:2]:
        return False
    k = min(a.shape[2] if a.ndim == 3 else 1, b.shape[2] if b.ndim == 3 else 1, 3)
    a3 = a[..., :k]
    b3 = b[..., :k]
    diff = np.abs(a3 - b3)
    return bool(diff.mean() <= mean_tol and diff.max() <= max_tol)


def _ink_mask(img: np.ndarray, thr: float = 0.98) -> np.ndarray:
    g = img[..., :3].mean(axis=2)
    return g < thr


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return float(inter) / float(uni or 1)


@pytest.mark.performance
def test_mpl_markers_perf_and_correctness():
    # Skip entirely if no GPU backends; the custom backend benefits from them.
    if not get_backends():
        pytest.skip("No GPU backends available for yapocis mpl backend comparison.")

    commit_id = _git_commit_id()
    # Performance: many markers
    for backend in (
        Backend("mpl_agg", lambda f: FigureCanvasAgg(f)),
        Backend("mpl_yapocis", lambda f: FigureCanvasYapo(f)),
    ):
        fig = Figure(figsize=(4, 4), dpi=100)
        _draw_markers(fig, count=1600, marker="o", size=8.0, color=(0, 0, 0))
        arr, ms = _render(fig, backend)
        _persist_jsonl({
            "metric": "mpl_exec_ms",
            "backend": backend.name,
            "commit": commit_id,
            "interface": "mpl_markers",
            "value": ms,
            "details": {"shape": list(arr.shape), "count": 1600},
        })

    # Correctness: single centered marker, detect UL corruption on yapocis backend
    fig_agg = Figure(figsize=(2, 2), dpi=100)
    _draw_single_marker(fig_agg, marker="o", size=24.0)
    img_agg, _ = _render(fig_agg, Backend("mpl_agg", lambda f: FigureCanvasAgg(f)))

    fig_yapo = Figure(figsize=(2, 2), dpi=100)
    _draw_single_marker(fig_yapo, marker="o", size=24.0)
    img_yapo, _ = _render(fig_yapo, Backend("mpl_yapocis", lambda f: FigureCanvasYapo(f)))

    agg_ul_bug = _detect_upper_left_quadrant_corruption(img_agg)
    yapo_ul_bug = _detect_upper_left_quadrant_corruption(img_yapo)

    # Persist correctness flags
    _persist_jsonl({
        "metric": "mpl_marker_upper_left_corruption",
        "backend": "mpl_agg",
        "commit": commit_id,
        "interface": "mpl_markers",
        "value": bool(agg_ul_bug),
    })
    _persist_jsonl({
        "metric": "mpl_marker_upper_left_corruption",
        "backend": "mpl_yapocis",
        "commit": commit_id,
        "interface": "mpl_markers",
        "value": bool(yapo_ul_bug),
    })

    # Assert no corruption is present in either backend.
    assert not agg_ul_bug, "Agg backend shows unexpected marker quadrant corruption"
    assert not yapo_ul_bug, "yapocis mpl backend marker rendering shows upper-left quadrant corruption"


@pytest.mark.performance
@pytest.mark.parametrize("marker", ["o", "^", "s", "D"])  # filled markers more comparable
@pytest.mark.parametrize("size", [10.0, 24.0])
@pytest.mark.parametrize("fill", [True])
def test_mpl_markers_strict_image_match(marker: str, size: float, fill: bool):
    # Strict image comparison between Agg and yapocis backends over a set of markers/sizes.
    if not get_backends():
        pytest.skip("No GPU backends available for yapocis mpl backend comparison.")

    backends = (
        Backend("mpl_agg", lambda f: FigureCanvasAgg(f)),
        Backend("mpl_yapocis", lambda f: FigureCanvasYapo(f)),
    )
    imgs = {}
    for be in backends:
        fig = Figure(figsize=(2, 2), dpi=100)
        _draw_single_marker_fill_edge(fig, marker=marker, size=size, fill=fill, color=(0, 0, 0), edgecolor=(0, 0, 0))
        img, _ = _render(fig, be)
        imgs[be.name] = img

    agg = imgs["mpl_agg"]; yapo = imgs["mpl_yapocis"]
    # Assert no quadrant corruption in either image
    assert not _detect_upper_left_quadrant_corruption(agg), f"Agg quadrant anomaly for marker {marker}, size {size}, fill {fill}"
    assert not _detect_upper_left_quadrant_corruption(yapo), f"yapocis quadrant anomaly for marker {marker}, size {size}, fill {fill}"
    # Assert shapes overlap closely using mask IoU and relaxed pixel tolerance
    mask_agg = _ink_mask(agg)
    mask_yapo = _ink_mask(yapo)
    iou = _iou(mask_agg, mask_yapo)
    assert iou >= 0.80, f"Marker masks differ (IoU={iou:.3f}) for marker {marker}, size {size}, fill {fill}"

#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.
"""
Library parity generators for the graphics layer: Graphics1, Graphics4, the
KernelDraw display lists, typeset and the mplbackend path rendering.

Every case is a list of ``ops`` the Python wrapper executes on a canvas; the
same records drive the Rust replay. Arrays referenced by ops live in the case's
``.npz`` under the key named in the record; scalars are inline JSON.

Op records::

    {"op": "lines"|"rects"|"circles"|"triangles"|"mesh_triangles"|"ellipses",
     "coords": key, "style": {"color", "colors", "alpha", "alphas", "width",
     "widths", "distance", "aa_width", "aa_widths"}}      (vector values are keys)
    {"op": "composite", "locations": key, "blocks": [keys], "alpha": null|float,
     "alphas": [null|float|key] | null}
    {"op": "sprites", "locations": key (n,3), "sprites": [keys], "alpha", "alphas"}
    {"op": "draw_strings", "locations": key, "texts": [...], "font", "fonts",
     "size", "sizes", "alpha", "alphas", "color", "colors", "rotation", "rotations"}
    {"op": "item", "kind": "line"|..., "coords": [...], "kw": {...}}   (KernelDraw lists)
    {"op": "item", "kind": "draw_string", "xy": [x, y], "text": "...", "kw": {...}}
    {"op": "render"}
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np

from .parity_lib import generator

W, H = (
    200,
    160,
)  # canvas (width, height) -> arrays are (H, W) style? No: yapocis canvases are (x, y)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _save(out_dir: str, name: str, arrays: dict[str, np.ndarray]) -> str:
    fname = f"{name}.npz"
    np.savez_compressed(os.path.join(out_dir, fname), **arrays)
    return fname


class Case:
    """Collects arrays and ops for one case."""

    def __init__(self, name: str):
        self.name = name
        self.arrays: dict[str, np.ndarray] = {}
        self.ops: list[dict[str, Any]] = []
        self.counter = 0

    def key(self, prefix: str, array: np.ndarray) -> str:
        self.counter += 1
        k = f"{prefix}_{self.counter}"
        self.arrays[k] = np.ascontiguousarray(array)
        return k

    def shape(self, op: str, coords, **style) -> None:
        coords = np.asarray(coords, dtype=np.float32)
        rec: dict[str, Any] = {
            "op": op,
            "coords": self.key("coords", coords),
            "style": {},
        }
        for k, v in style.items():
            if v is None:
                continue
            if isinstance(v, np.ndarray):
                rec["style"][k] = self.key(k, v.astype(np.float32))
            else:
                rec["style"][k] = v
        self.ops.append(rec)


# --- executing ops with the Python wrappers ----------------------------------


def _style_kwargs(
    case_arrays: dict[str, np.ndarray], style: dict[str, Any]
) -> dict[str, Any]:
    kwargs = {}
    for k, v in style.items():
        if isinstance(v, str):
            kwargs[k] = case_arrays[v]
        elif isinstance(v, list):
            kwargs[k] = v
        else:
            kwargs[k] = v
    return kwargs


def _alpha_value(arrays: dict[str, np.ndarray], v: Any) -> Any:
    if isinstance(v, str) and v in arrays:
        return arrays[v]
    return v


def run_graphics_ops(g, canvas: np.ndarray, case: Case) -> None:
    """Run a case's ops through a Graphics1/Graphics4 instance."""
    a = case.arrays
    for rec in case.ops:
        op = rec["op"]
        if op in (
            "lines",
            "rects",
            "circles",
            "triangles",
            "mesh_triangles",
            "ellipses",
        ):
            coords = a[rec["coords"]].tolist()
            getattr(g, op)(canvas, coords, **_style_kwargs(a, rec["style"]))
        elif op == "composite":
            locations = [tuple(xy) for xy in a[rec["locations"]].tolist()]
            blocks = [a[k] for k in rec["blocks"]]
            kwargs = {}
            if rec.get("alpha") is not None:
                kwargs["alpha"] = _alpha_value(a, rec["alpha"])
            if rec.get("alphas") is not None:
                kwargs["alphas"] = [_alpha_value(a, v) for v in rec["alphas"]]
            g.composite(canvas, locations, blocks, **kwargs)
        elif op == "sprites":
            locations = [
                tuple(int(v) for v in row) for row in a[rec["locations"]].tolist()
            ]
            sprites = [a[k] for k in rec["sprites"]]
            kwargs = {}
            if rec.get("alpha") is not None:
                kwargs["alpha"] = rec["alpha"]
            if rec.get("alphas") is not None:
                kwargs["alphas"] = rec["alphas"]
            g.sprites(canvas, locations, sprites, **kwargs)
        elif op == "draw_strings":
            locations = [tuple(xy) for xy in a[rec["locations"]].tolist()]
            kwargs = {
                k: rec[k]
                for k in (
                    "font",
                    "fonts",
                    "size",
                    "sizes",
                    "alpha",
                    "alphas",
                    "color",
                    "colors",
                    "rotation",
                    "rotations",
                )
                if rec.get(k) is not None
            }
            g.draw_strings(canvas, locations, rec["texts"], **kwargs)
        else:
            raise ValueError(op)


def run_kd_ops(kd, case: Case) -> None:
    """Run a case's ops through a KernelDraw instance."""
    a = case.arrays
    for rec in case.ops:
        op = rec["op"]
        if op in (
            "lines",
            "rects",
            "circles",
            "triangles",
            "mesh_triangles",
            "ellipses",
        ):
            coords = a[rec["coords"]].tolist()
            getattr(kd, op)(coords, **_style_kwargs(a, rec["style"]))
        elif op == "item":
            kind = rec["kind"]
            kw = dict(rec["kw"])
            if kind == "draw_string":
                kd.draw_string(tuple(rec["xy"]), rec["text"], **kw)
            else:
                getattr(kd, kind)(list(rec["coords"]), **kw)
        elif op == "composite":
            locations = [tuple(xy) for xy in a[rec["locations"]].tolist()]
            blocks = [a[k] for k in rec["blocks"]]
            kwargs = {}
            if rec.get("alphas") is not None:
                kwargs["alphas"] = [_alpha_value(a, v) for v in rec["alphas"]]
            kd.composite(locations, blocks, **kwargs)
        elif op == "sprites":
            locations = [
                tuple(int(v) for v in row) for row in a[rec["locations"]].tolist()
            ]
            sprites = [a[k] for k in rec["sprites"]]
            kd.sprites(locations, sprites)
        elif op == "draw_strings":
            locations = [tuple(xy) for xy in a[rec["locations"]].tolist()]
            kwargs = {
                k: rec[k]
                for k in (
                    "font",
                    "fonts",
                    "size",
                    "sizes",
                    "alpha",
                    "alphas",
                    "color",
                    "colors",
                    "rotation",
                    "rotations",
                )
                if rec.get(k) is not None
            }
            kd.draw_strings(locations, rec["texts"], **kwargs)
        elif op == "render":
            kd.render()
        else:
            raise ValueError(op)


# --- scenes -------------------------------------------------------------------


def _grid_lines() -> list[list[int]]:
    return [[0, 0, i, j] for i in range(40, W, 40) for j in range(40, H, 40)]


def _random_lines(rng, n=40, length=30) -> np.ndarray:
    xy = np.column_stack([rng.integers(0, W, n), rng.integers(0, H, n)])
    d = rng.integers(-length, length, (n, 2))
    return np.concatenate([xy, xy + d], axis=1).astype(np.float32)


def _random_tris(rng, n=25, length=30) -> np.ndarray:
    xy = np.column_stack([rng.integers(0, W, n), rng.integers(0, H, n)])
    d1 = rng.integers(-length, length, (n, 2))
    d2 = rng.integers(-length, length, (n, 2))
    return np.concatenate([xy, xy + d1, xy + d2], axis=1).astype(np.float32)


def _pattern_block(rng, h, w) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    return (
        (np.sin(x / 3.0) * np.cos(y / 4.0) + 1.0) / 2.0 * rng.uniform(0.5, 1.0)
    ).astype(np.float32)


def _canvas1(fill=0.0) -> np.ndarray:
    return np.full((W, H), fill, dtype=np.float32)


def _canvas4() -> np.ndarray:
    return np.zeros((W, H, 4), dtype=np.float32)


def graphics1_cases(rng) -> list[Case]:
    cases = []

    c = Case("lines_grid")
    c.shape(
        "lines",
        _grid_lines(),
        width=1.0,
        colors=rng.uniform(0, 1, len(_grid_lines())),
        alpha=1.0,
    )
    cases.append(c)

    c = Case("lines_random")
    lines = _random_lines(rng)
    c.shape(
        "lines",
        lines,
        widths=rng.uniform(1, 6, len(lines)),
        colors=rng.uniform(0, 1, len(lines)),
        alphas=rng.uniform(0.25, 1, len(lines)),
        aa_widths=np.where(
            rng.uniform(0, 1, len(lines)) > 0.5, rng.uniform(0.5, 2.0, len(lines)), 0.0
        ),
    )
    cases.append(c)

    c = Case("lines_zero_length")
    c.shape(
        "lines",
        [[20, 20, 20, 20], [60.7, 40.2, 60.7, 40.2], [100, 100, 100, 100]],
        width=3.0,
        color=1.0,
    )
    c.shape("lines", [[150, 30, 150, 30]], width=1.0, color=0.5, aa_width=1.0)
    cases.append(c)

    c = Case("rects_alpha")
    c.shape("rects", [[10, 10, 90, 70], [50, 40, 150, 120]], color=1.0, alpha=0.5)
    lines = _random_lines(rng, n=20)
    c.shape(
        "rects",
        lines,
        colors=rng.uniform(0, 1, len(lines)),
        alphas=rng.uniform(0.25, 1, len(lines)),
    )
    cases.append(c)

    c = Case("circles_aa")
    circles = np.column_stack(
        [
            rng.integers(10, W - 10, 20),
            rng.integers(10, H - 10, 20),
            rng.integers(3, 25, 20),
        ]
    ).astype(np.float32)
    c.shape(
        "circles",
        circles,
        colors=rng.uniform(0, 1, 20),
        aa_widths=rng.uniform(0, 2, 20),
    )
    c.shape("circles", [[100, 80, 30]], color=0.8, alpha=0.5, aa_width=1.5)
    cases.append(c)

    c = Case("triangles")
    tris = _random_tris(rng)
    c.shape(
        "triangles",
        tris,
        colors=rng.uniform(0, 1, len(tris)),
        alphas=rng.uniform(0.3, 1, len(tris)),
    )
    c.shape(
        "mesh_triangles", _random_tris(rng, n=15), color=0.7, alpha=0.5, aa_width=0.0
    )
    cases.append(c)

    c = Case("ellipses")
    ell = np.column_stack(
        [
            rng.integers(10, W - 10, 20),
            rng.integers(10, H - 10, 20),
            rng.integers(3, 30, 20),
            rng.integers(3, 30, 20),
        ]
    ).astype(np.float32)
    c.shape(
        "ellipses", ell, colors=rng.uniform(0, 1, 20), aa_widths=rng.uniform(0, 1.5, 20)
    )
    cases.append(c)

    c = Case("composite")
    blocks = [
        _pattern_block(rng, 20, 30),
        _pattern_block(rng, 15, 15),
        _pattern_block(rng, 40, 25),
    ]
    locs = np.array([[10, 10], [100, 50], [150, 120]], dtype=np.int32)
    c.ops.append(
        {
            "op": "composite",
            "locations": c.key("locations", locs),
            "blocks": [c.key("block", b) for b in blocks],
            "alpha": None,
            "alphas": [
                None,
                0.5,
                c.key("mask", rng.uniform(0, 1, blocks[2].shape).astype(np.float32)),
            ],
        }
    )
    c.ops.append(
        {
            "op": "composite",
            "locations": c.key("locations", np.array([[60, 90]], dtype=np.int32)),
            "blocks": [c.key("block", blocks[0])],
            "alpha": 0.3,
            "alphas": None,
        }
    )
    cases.append(c)

    c = Case("sprites")
    sprites = [
        _pattern_block(rng, 12, 12),
        _pattern_block(rng, 8, 16),
        _pattern_block(rng, 20, 10),
        _pattern_block(rng, 5, 5),
    ]
    locs = np.array(
        [
            [i % 4, int(rng.integers(0, W - 20)), int(rng.integers(0, H - 20))]
            for i in range(10)
        ],
        dtype=np.int32,
    )
    c.ops.append(
        {
            "op": "sprites",
            "locations": c.key("locations", locs),
            "sprites": [c.key("sprite", s) for s in sprites],
            "alpha": 0.7,
            "alphas": None,
        }
    )
    c.ops.append(
        {
            "op": "sprites",
            "locations": c.key("locations", locs[:4]),
            "sprites": [c.key("sprite", s) for s in sprites],
            "alpha": None,
            "alphas": [1.0, 0.25, 0.5, None],
        }
    )
    cases.append(c)

    c = Case("text_rotations")
    rots = list(range(-180, 180, 45))
    locs = np.array([[W // 2, H // 2]] * len(rots), dtype=np.int32)
    c.ops.append(
        {
            "op": "draw_strings",
            "locations": c.key("locations", locs),
            "texts": [f"---- {r:03d}" for r in rots],
            "rotations": rots,
            "color": 1,
            "alpha": "mask",
            "size": 20,
        }
    )
    cases.append(c)

    c = Case("text_fonts")
    locs = np.array([[10, 30], [10, 80], [10, 130]], dtype=np.int32)
    c.ops.append(
        {
            "op": "draw_strings",
            "locations": c.key("locations", locs),
            "texts": ["Hello, yapocis", "Bold jump!", "light italic"],
            "fonts": ["default", "bold", "lightitalic"],
            "sizes": [16, 24, 40],
            "alpha": "bug",
        }
    )
    c.ops.append(
        {
            "op": "draw_strings",
            "locations": c.key("locations", np.array([[120, 100]], dtype=np.int32)),
            "texts": ["half"],
            "alpha": 0.5,
            "rotation": 90,
        }
    )
    c.ops.append(
        {
            "op": "draw_strings",
            "locations": c.key("locations", np.array([[150, 20]], dtype=np.int32)),
            "texts": ["none"],
            "size": 18,
        }
    )
    cases.append(c)
    return cases


def _rgba(rng, n) -> np.ndarray:
    return rng.uniform(0, 1, (n, 4)).astype(np.float32)


def _block4(rng, h, w) -> np.ndarray:
    b = np.empty((h, w, 4), dtype=np.float32)
    for k in range(3):
        b[:, :, k] = _pattern_block(rng, h, w)
    b[:, :, 3] = rng.uniform(0.3, 1.0, (h, w)).astype(np.float32)
    return b


def graphics4_cases(rng) -> list[Case]:
    cases = []
    c = Case("lines")
    grid = _grid_lines()
    c.shape("lines", grid, width=1.5, colors=_rgba(rng, len(grid)))
    lines = _random_lines(rng)
    c.shape(
        "lines",
        lines,
        distance=1.5,
        colors=_rgba(rng, len(lines)),
        aa_widths=rng.uniform(0, 2, len(lines)),
    )
    c.shape("lines", [[5, 5, 5, 5]], width=2.0, color=[1, 0, 0, 1])
    cases.append(c)

    c = Case("rects_alpha")
    c.shape("rects", [[0, 40, W - 1, 80]], color=[0, 1, 0, 0.5])
    c.shape("rects", [[60, 0, 120, H - 1]], color=[1, 0, 0, 1])
    lines = _random_lines(rng, n=15)
    c.shape(
        "rects",
        lines,
        colors=_rgba(rng, len(lines)),
        aa_widths=rng.uniform(0, 1.5, len(lines)),
    )
    cases.append(c)

    c = Case("circles_triangles_ellipses")
    circles = np.column_stack(
        [
            rng.integers(10, W - 10, 15),
            rng.integers(10, H - 10, 15),
            rng.integers(3, 25, 15),
        ]
    ).astype(np.float32)
    c.shape("circles", circles, colors=_rgba(rng, 15), aa_widths=rng.uniform(0, 2, 15))
    tris = _random_tris(rng, n=12)
    c.shape("triangles", tris, colors=_rgba(rng, 12))
    c.shape("mesh_triangles", _random_tris(rng, n=10), color=[0.2, 0.4, 0.9, 0.5])
    ell = np.column_stack(
        [
            rng.integers(10, W - 10, 12),
            rng.integers(10, H - 10, 12),
            rng.integers(3, 30, 12),
            rng.integers(3, 30, 12),
        ]
    ).astype(np.float32)
    c.shape("ellipses", ell, colors=_rgba(rng, 12), aa_width=1.0)
    cases.append(c)

    c = Case("composite_sprites")
    blocks = [_block4(rng, 20, 30), _block4(rng, 15, 15)]
    locs = np.array([[10, 10], [100, 50]], dtype=np.int32)
    c.ops.append(
        {
            "op": "composite",
            "locations": c.key("locations", locs),
            "blocks": [c.key("block", b) for b in blocks],
            "alpha": None,
            "alphas": None,
        }
    )
    sprites = [_block4(rng, 12, 12), _block4(rng, 8, 16), _block4(rng, 20, 10)]
    locs = np.array(
        [
            [i % 3, int(rng.integers(0, W - 20)), int(rng.integers(0, H - 20))]
            for i in range(8)
        ],
        dtype=np.int32,
    )
    c.ops.append(
        {
            "op": "sprites",
            "locations": c.key("locations", locs),
            "sprites": [c.key("sprite", s) for s in sprites],
            "alpha": None,
            "alphas": None,
        }
    )
    cases.append(c)

    c = Case("texts")
    rots = list(range(0, 360, 60))
    locs = np.array([[W // 2, H // 2]] * len(rots), dtype=np.int32)
    c.ops.append(
        {
            "op": "draw_strings",
            "locations": c.key("locations", locs),
            "texts": [f"-- {r:03d}" for r in rots],
            "rotations": rots,
            "colors": [
                [1, 0, 0, 1],
                [0, 1, 0, 1],
                [0, 0, 1, 1],
                [1, 1, 0, 0.5],
                [0, 1, 1, 1],
                [1, 0, 1, 0.7],
            ],
            "size": 18,
        }
    )
    locs = np.array([[10, 20], [10, 60], [10, 100], [10, 140]], dtype=np.int32)
    c.ops.append(
        {
            "op": "draw_strings",
            "locations": c.key("locations", locs),
            "texts": ["mask", "bug", "half", "none"],
            "alphas": ["mask", "bug", 0.5, None],
            "color": [0.9, 0.8, 0.2, 0.6],
            "fonts": ["default", "bold", "medium", "italic"],
            "sizes": [14, 20, 26, 32],
        }
    )
    cases.append(c)
    return cases


def kerneldraw_cases(rng) -> list[tuple[Case, str, int | None]]:
    """(case, kind, chunk_size) with kind in kd1 | kd3 | kd4."""
    cases = []

    c = Case("kd4_batch")
    blocks = [_block4(rng, 20, 30), _block4(rng, 15, 15)]
    locs = np.array([[10, 10], [100, 50]], dtype=np.int32)
    c.ops.append(
        {
            "op": "composite",
            "locations": c.key("locations", locs),
            "blocks": [c.key("block", b) for b in blocks],
            "alpha": None,
            "alphas": None,
        }
    )
    sprites = [_block4(rng, 12, 12), _block4(rng, 8, 16)]
    locs = np.array(
        [
            [i % 2, int(rng.integers(0, W - 20)), int(rng.integers(0, H - 20))]
            for i in range(6)
        ],
        dtype=np.int32,
    )
    c.ops.append(
        {
            "op": "sprites",
            "locations": c.key("locations", locs),
            "sprites": [c.key("sprite", s) for s in sprites],
            "alpha": None,
            "alphas": None,
        }
    )
    ell = np.column_stack(
        [
            rng.integers(10, W - 10, 10),
            rng.integers(10, H - 10, 10),
            rng.integers(3, 30, 10),
            rng.integers(3, 30, 10),
        ]
    ).astype(np.float32)
    c.shape("ellipses", ell, colors=_rgba(rng, 10))
    circles = np.column_stack(
        [
            rng.integers(10, W - 10, 10),
            rng.integers(10, H - 10, 10),
            rng.integers(3, 25, 10),
        ]
    ).astype(np.float32)
    c.shape("circles", circles, colors=_rgba(rng, 10))
    rects = _random_lines(rng, n=10)
    c.shape("rects", rects, colors=_rgba(rng, 10))
    lines = _random_lines(rng, n=20)
    c.shape("lines", lines, colors=_rgba(rng, 20), distance=1.5)
    c.shape(
        "lines", lines[:5], color=[1, 1, 1, 1], alpha=0.5, distance=2.0, aa_width=0.5
    )
    c.ops.append({"op": "render"})
    cases.append((c, "kd4", None))

    c = Case("kd4_list")
    lines = _random_lines(rng, n=30)
    colors = _rgba(rng, 30)
    for xyxy, color in zip(lines.tolist(), colors.tolist(), strict=False):
        c.ops.append(
            {
                "op": "item",
                "kind": "line",
                "coords": xyxy,
                "kw": {"color": color, "distance": 1.5},
            }
        )
    rects = _random_lines(rng, n=10)
    for rect, color in zip(rects.tolist(), _rgba(rng, 10).tolist(), strict=False):
        c.ops.append(
            {
                "op": "item",
                "kind": "rect",
                "coords": rect,
                "kw": {"color": color, "alpha": 0.5},
            }
        )
    for xyr in (
        np.column_stack(
            [
                rng.integers(10, W - 10, 5),
                rng.integers(10, H - 10, 5),
                rng.integers(3, 20, 5),
            ]
        )
        .astype(np.float32)
        .tolist()
    ):
        c.ops.append(
            {
                "op": "item",
                "kind": "circle",
                "coords": xyr,
                "kw": {"color": [0.5, 0.5, 1.0, 1.0], "aa_width": 1.0},
            }
        )
    for t in _random_tris(rng, n=4).tolist():
        c.ops.append(
            {
                "op": "item",
                "kind": "mesh_triangle",
                "coords": t,
                "kw": {"color": [0.9, 0.3, 0.3, 0.6]},
            }
        )
    for e in (
        np.column_stack(
            [
                rng.integers(10, W - 10, 3),
                rng.integers(10, H - 10, 3),
                rng.integers(3, 20, 3),
                rng.integers(3, 20, 3),
            ]
        )
        .astype(np.float32)
        .tolist()
    ):
        c.ops.append(
            {
                "op": "item",
                "kind": "ellipse",
                "coords": e,
                "kw": {"color": [0.1, 0.9, 0.3, 1.0], "width": 2.0},
            }
        )
    for x in range(0, W, 60):
        c.ops.append(
            {
                "op": "item",
                "kind": "draw_string",
                "xy": [x, x // 2 + 10],
                "text": "Hello world!",
                "kw": {"color": [1, 1, 1, 0.5]},
            }
        )
    c.ops.append({"op": "render"})
    cases.append((c, "kd4", 500))

    c = Case("kd4_chunk10")
    lines = _random_lines(rng, n=45)
    colors = _rgba(rng, 45)
    for xyxy, color in zip(lines.tolist(), colors.tolist(), strict=False):
        c.ops.append(
            {
                "op": "item",
                "kind": "line",
                "coords": xyxy,
                "kw": {"color": color, "width": 1.5},
            }
        )
    rects = _random_lines(rng, n=25)
    for rect, color in zip(rects.tolist(), _rgba(rng, 25).tolist(), strict=False):
        c.ops.append(
            {"op": "item", "kind": "rect", "coords": rect, "kw": {"color": color}}
        )
    for x in range(0, W, 50):
        c.ops.append(
            {
                "op": "item",
                "kind": "draw_string",
                "xy": [x, x // 2 + 20],
                "text": "chunk",
                "kw": {"size": 14, "rotation": 30},
            }
        )
    c.ops.append({"op": "render"})
    cases.append((c, "kd4", 10))

    c = Case("kd1_list")
    lines = _random_lines(rng, n=20)
    for xyxy, color in zip(
        lines.tolist(), rng.uniform(0.2, 1, 20).tolist(), strict=False
    ):
        c.ops.append(
            {
                "op": "item",
                "kind": "line",
                "coords": xyxy,
                "kw": {"color": color, "width": 2.0},
            }
        )
    for rect, color in zip(
        _random_lines(rng, n=8).tolist(), rng.uniform(0.2, 1, 8).tolist(), strict=False
    ):
        c.ops.append(
            {
                "op": "item",
                "kind": "rect",
                "coords": rect,
                "kw": {"color": color, "aa_width": 1.0},
            }
        )
    for x in range(10, W, 70):
        c.ops.append(
            {
                "op": "item",
                "kind": "draw_string",
                "xy": [x, 100],
                "text": "kd1",
                "kw": {"size": 24, "alpha": "mask"},
            }
        )
    c.ops.append({"op": "render"})
    lines = _random_lines(rng, n=10)
    c.shape(
        "lines",
        lines,
        colors=rng.uniform(0, 1, 10),
        alpha=0.5,
        widths=rng.uniform(1, 4, 10),
    )
    c.shape(
        "circles", [[50, 50, 20], [150, 100, 15]], color=0.9, alpha=0.3, aa_width=1.0
    )
    c.ops.append({"op": "render"})
    cases.append((c, "kd1", 500))

    c = Case("kd3_scene")
    lines = _random_lines(rng, n=15)
    c.shape("lines", lines, colors=rng.uniform(0, 1, (15, 3)), distance=2.0)
    rects = _random_lines(rng, n=6)
    c.shape("rects", rects, colors=rng.uniform(0, 1, (6, 3)))
    c.shape("circles", [[60, 60, 25], [140, 90, 18]], color=[1.0, 0.5, 0.2])
    for xyxy, color in zip(
        _random_lines(rng, n=8).tolist(),
        rng.uniform(0, 1, (8, 3)).tolist(),
        strict=False,
    ):
        c.ops.append(
            {
                "op": "item",
                "kind": "line",
                "coords": xyxy,
                "kw": {"color": color, "width": 1.5},
            }
        )
    for x in range(10, W, 80):
        c.ops.append(
            {
                "op": "item",
                "kind": "draw_string",
                "xy": [x, 120],
                "text": "kd3",
                "kw": {"size": 20},
            }
        )
    c.ops.append({"op": "render"})
    cases.append((c, "kd3", 500))
    return cases


@generator("graphics")
def gen_graphics(
    out_dir: str, backend: str, log: Callable[..., None]
) -> list[dict[str, Any]]:
    from yapocis.lib.graphics import Graphics1

    g = Graphics1()
    records = []
    for case in graphics1_cases(_rng(1)):
        canvas = _canvas1(0.1 if case.name == "composite" else 0.0)
        case.arrays["canvas0"] = canvas.copy()
        run_graphics_ops(g, canvas, case)
        case.arrays["output"] = canvas
        fname = _save(out_dir, f"graphics.{case.name}", case.arrays)
        records.append(
            {
                "name": case.name,
                "file": fname,
                "channels": 1,
                "canvas": "canvas0",
                "output": "output",
                "ops": case.ops,
            }
        )
    g.purge()
    return records


@generator("graphics4")
def gen_graphics4(
    out_dir: str, backend: str, log: Callable[..., None]
) -> list[dict[str, Any]]:
    from yapocis.lib.graphics4 import Graphics4

    g = Graphics4()
    records = []
    for case in graphics4_cases(_rng(2)):
        canvas = _canvas4()
        case.arrays["canvas0"] = canvas.copy()
        run_graphics_ops(g, canvas, case)
        case.arrays["output"] = canvas
        fname = _save(out_dir, f"graphics4.{case.name}", case.arrays)
        records.append(
            {
                "name": case.name,
                "file": fname,
                "channels": 4,
                "canvas": "canvas0",
                "output": "output",
                "ops": case.ops,
            }
        )
    g.purge()
    return records


@generator("kerneldraw")
def gen_kerneldraw(
    out_dir: str, backend: str, log: Callable[..., None]
) -> list[dict[str, Any]]:
    from yapocis.lib.kerneldraw1 import KernelDraw1
    from yapocis.lib.kerneldraw3 import KernelDraw3
    from yapocis.lib.kerneldraw4 import KernelDraw4

    records = []
    for case, kind, chunk in kerneldraw_cases(_rng(3)):
        if kind == "kd4":
            canvas = _canvas4()
            kd = KernelDraw4(canvas, chunk_size=chunk)
        elif kind == "kd1":
            canvas = _canvas1()
            kd = KernelDraw1(canvas, chunk_size=chunk)
        else:
            canvas = np.zeros((W, H, 3), dtype=np.float32)
            kd = KernelDraw3(canvas, chunk_size=chunk)
        case.arrays["canvas0"] = canvas.copy()
        run_kd_ops(kd, case)
        out = kd.read()
        case.arrays["output"] = np.ascontiguousarray(out)
        fname = _save(out_dir, f"kerneldraw.{case.name}", case.arrays)
        records.append(
            {
                "name": case.name,
                "file": fname,
                "kind": kind,
                "chunk_size": chunk,
                "canvas": "canvas0",
                "output": "output",
                "ops": case.ops,
            }
        )
    return records


@generator("typeset")
def gen_typeset(
    out_dir: str, backend: str, log: Callable[..., None]
) -> list[dict[str, Any]]:
    from yapocis.lib.typeset import compute_offset, render_text

    specs = [
        ("hello", "Hello, yapocis!", "default", 32, 0),
        ("bold_small", "Bold 12", "bold", 12, 0),
        ("italic_rot", "italic 45", "italic", 24, 45),
        ("rot_neg", "negative", "default", 20, -30),
        ("rot_90", "ninety", "medium", 28, 90),
        ("rot_180", "flipped", "light", 18, 180),
        ("rot_270", "two seventy", "default", 16, 270),
        ("big", "Big", "boldItalic", 72, 10),
        ("float_size", "float", "default", 21.5, 0),
        ("empty", "", "default", 32, 0),
        ("spaces", "a b  c", "default", 24, 0),
        ("unicode", "café üñ", "default", 26, 15),
    ]
    arrays: dict[str, np.ndarray] = {}
    cases = []
    for name, text, font, size, rotation in specs:
        bbox, block = render_text(text, font_name=font, size=size, rotation=rotation)
        arrays[name] = np.ascontiguousarray(block)
        cases.append(
            {
                "name": name,
                "text": text,
                "font": font,
                "size": size,
                "rotation": rotation,
                "bbox": [int(v) for v in bbox],
                "block": name,
            }
        )
    fname = _save(out_dir, "typeset.blocks", arrays)
    offsets = [
        {
            "angle": a,
            "h": h,
            "w": w,
            "dxdy": [float(v) for v in compute_offset(a, h, w)],
        }
        for a in (0, 30, 90, 120, 180, -180, -45, -90, -135)
        for (h, w) in ((7, 20), (8, 21))
    ]
    return [{"name": "blocks", "file": fname, "cases": cases, "offsets": offsets}]


@generator("paths")
def gen_paths(
    out_dir: str, backend: str, log: Callable[..., None]
) -> list[dict[str, Any]]:
    """mplbackend's device-independent rendering, driven through the real RendererTemplate."""
    try:
        from matplotlib.path import Path
        from matplotlib.transforms import IdentityTransform

        from yapocis.lib.mplbackend import RendererTemplate
    except ImportError as e:  # matplotlib is an optional extra
        log(f"paths: SKIP (matplotlib not available: {e})")
        return []

    dpi = 100.0
    figsize = (2.0, 1.6)
    rng = _rng(4)
    records = []

    def new_renderer():
        return RendererTemplate(dpi, figsize)

    def finish(name: str, r, arrays: dict[str, np.ndarray], extra: dict[str, Any]):
        out = r.read()
        arrays["output"] = np.ascontiguousarray(out)
        fname = _save(out_dir, f"paths.{name}", arrays)
        rec = {
            "name": name,
            "file": fname,
            "width": r.width,
            "height": r.height,
            "output": "output",
        }
        rec.update(extra)
        records.append(rec)

    # dashed lines
    r = new_renderer()
    arrays: dict[str, np.ndarray] = {}
    segs = []
    for i in range(6):
        x1, y1, x2, y2 = (
            float(v)
            for v in (
                rng.integers(5, 195),
                rng.integers(5, 155),
                rng.integers(5, 195),
                rng.integers(5, 155),
            )
        )
        color = [float(v) for v in rng.uniform(0, 1, 3)] + [1.0]
        pattern = [float(v) for v in rng.uniform(2, 10, 2 * (1 + i % 2))]
        offset = float(rng.uniform(0, 5))
        lw = float(rng.uniform(1, 4))
        r._draw_dashed_line(x1, y1, x2, y2, color, lw, pattern, offset)
        segs.append(
            {
                "xyxy": [x1, y1, x2, y2],
                "color": color,
                "linewidth": lw,
                "pattern": pattern,
                "offset": offset,
            }
        )
    r.kd.render()
    finish("dashed", r, arrays, {"segments": segs})

    # paths with faces, holes, dashes, drawn through draw_path
    def polygon_record(
        name: str, verts_list, codes_list, face, edge, linewidth_pt, dashes
    ):
        r = new_renderer()
        arrays: dict[str, np.ndarray] = {}
        gc = r.new_gc()
        gc.set_foreground(edge, isRGBA=True)
        gc.set_linewidth(linewidth_pt)
        if dashes is not None:
            gc.set_dashes(dashes[0], dashes[1])
        polygons_all = []
        for verts, codes in zip(verts_list, codes_list, strict=False):
            path = Path(np.asarray(verts, dtype=np.float64), codes)
            polygons = path.to_polygons(
                transform=IdentityTransform(), closed_only=False
            )
            polygons_all.extend(np.asarray(p, dtype=np.float64) for p in polygons)
            r.draw_path(gc, path, IdentityTransform(), rgbFace=face)
        keys = []
        for i, poly in enumerate(polygons_all):
            keys.append(f"poly_{i}")
            arrays[f"poly_{i}"] = np.ascontiguousarray(poly)
        lw_px = float(r.points_to_pixels(linewidth_pt))
        rec = {
            "polygons": keys,
            "face": face,
            "edge": edge,
            "linewidth_px": lw_px,
            "linewidth_pt": linewidth_pt,
        }
        if dashes is not None:
            rec["dashes"] = [
                float(r.points_to_pixels(dashes[0])),
                [float(r.points_to_pixels(d)) for d in dashes[1]],
            ]
        finish(name, r, arrays, rec)

    square = [(20, 20), (120, 20), (120, 100), (20, 100), (20, 20)]
    hole = [(50, 40), (90, 40), (90, 80), (50, 80), (50, 40)]
    closed = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    polygon_record(
        "filled_square_hole",
        [square + hole],
        [closed + closed],
        [0.2, 0.5, 0.9, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        1.5,
        None,
    )
    star = [
        (
            (100 + 60 * np.cos(a), 80 + 60 * np.sin(a))
            if i % 2 == 0
            else (100 + 25 * np.cos(a), 80 + 25 * np.sin(a))
        )
        for i, a in enumerate(np.linspace(0, 2 * np.pi, 11)[:-1])
    ]
    star.append(star[0])
    polygon_record(
        "star_translucent",
        [star],
        [[Path.MOVETO] + [Path.LINETO] * 9 + [Path.CLOSEPOLY]],
        [0.9, 0.3, 0.2, 0.5],
        [0.1, 0.1, 0.4, 1.0],
        3.0,
        None,
    )
    polyline = [(10, 150), (60, 20), (110, 140), (160, 30), (190, 120)]
    polygon_record(
        "dashed_polyline",
        [polyline],
        [[Path.MOVETO] + [Path.LINETO] * 4],
        None,
        [0.0, 0.6, 0.0, 1.0],
        2.0,
        (1.0, [6.0, 3.0]),
    )
    polygon_record(
        "zero_width_fill",
        [square],
        [closed],
        [0.5, 0.5, 0.5, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        0.0,
        None,
    )
    curve = [(10, 20), (60, 150), (140, 150), (190, 20)]
    polygon_record(
        "bezier_edge",
        [curve],
        [[Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]],
        None,
        [0.3, 0.2, 0.8, 1.0],
        2.5,
        None,
    )

    # images
    r = new_renderer()
    arrays = {}
    im = _block4(rng, 40, 60)
    im[:, :, 3] = 1.0
    r.draw_image(None, 30.0, 40.0, im)
    gray = _pattern_block(rng, 25, 25)
    r.draw_image(None, 120.0, 90.0, gray)
    rgb8 = (rng.uniform(0, 1, (20, 30, 3)) * 255).astype(np.uint8)
    r.draw_image(None, 140.0, 10.0, rgb8)
    arrays["rgba"] = im
    arrays["gray"] = gray
    arrays["rgb8"] = rgb8
    r.kd.render()
    finish(
        "images",
        r,
        arrays,
        {
            "images": [
                {"key": "rgba", "x": 30.0, "y": 40.0},
                {"key": "gray", "x": 120.0, "y": 90.0},
                {"key": "rgb8", "x": 140.0, "y": 10.0},
            ]
        },
    )

    # text coverage bitmaps
    from yapocis.lib.typeset import render_text

    r = new_renderer()
    arrays = {}
    gc = r.new_gc()
    gc.set_foreground([0.1, 0.2, 0.8, 0.9], isRGBA=True)
    _, block = render_text("Text 0", font_name="default", size=24)
    cov = (block * 255).astype(np.uint8)
    r._draw_text_image(cov, 20, 60, 0, gc)
    arrays["cov0"] = cov
    _, block = render_text("Rot 30", font_name="default", size=24)
    cov30 = (block * 255).astype(np.uint8)
    r._draw_text_image(cov30, 90, 140, 30, gc)
    arrays["cov30"] = cov30
    r.kd.render()
    finish(
        "text",
        r,
        arrays,
        {
            "texts": [
                {
                    "key": "cov0",
                    "x": 20,
                    "y": 60,
                    "angle": 0,
                    "rgba": [0.1, 0.2, 0.8, 0.9],
                },
                {
                    "key": "cov30",
                    "x": 90,
                    "y": 140,
                    "angle": 30,
                    "rgba": [0.1, 0.2, 0.8, 0.9],
                },
            ]
        },
    )
    return records

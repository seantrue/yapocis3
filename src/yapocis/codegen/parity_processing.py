#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.
"""
Library parity generators for the processing wrappers: truetrace, imageops,
simplex, anisodiff and fft. See :mod:`yapocis.codegen.parity_lib`.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np

from .parity_lib import generator

SEED = 20261019


def _save(out_dir: str, name: str, **arrays: Any) -> str:
    fname = f"{name}.npz"
    np.savez_compressed(os.path.join(out_dir, fname), **arrays)
    return fname


def _rings_to_arrays(rings: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Concatenate (N,2) rings with an offsets array (ragged lists in npz)."""
    if not rings:
        return {
            "rings": np.zeros((0, 2), np.float32),
            "ring_offsets": np.zeros(1, np.int64),
        }
    offsets = np.cumsum([0] + [r.shape[0] for r in rings]).astype(np.int64)
    return {"rings": np.concatenate(rings).astype(np.float32), "ring_offsets": offsets}


# ----------------------------------------------------------------------------
# truetrace
# ----------------------------------------------------------------------------


@generator("truetrace")
def truetrace_cases(
    out_dir: str, backend: str, log: Callable[..., None]
) -> list[dict[str, Any]]:
    from yapocis.lib import truetrace as tt_mod
    from yapocis.lib.truetrace import TrueTrace

    tt = TrueTrace()
    cases: list[dict[str, Any]] = []

    # Circle with a hole: exercises every CPU helper and the GPU tracer.
    mask = tt_mod.build_circle_mask(48, 40, 15)
    hole = tt_mod.build_circle_mask(48, 40, 6)
    mask[hole == 1] = 0

    for iso in (0.0, 1.5):
        # The library launches one thread per pixel itself since 2026-10-21 (and
        # traces a signed field), so it is the reference; the exporter used to
        # carry its own per-pixel jump flood as a workaround.
        segs, (w, h) = tt.trace_segments(mask, iso)
        lib_segs = segs
        fname = _save(
            out_dir, f"truetrace.gpu_iso{iso}", mask=mask, segs=segs, lib_segs=lib_segs
        )
        cases.append(
            {
                "name": f"gpu_iso{iso}",
                "file": fname,
                "iso": iso,
                "size": [w, h],
                "lib_matches_per_pixel": bool(
                    segs.shape == lib_segs.shape
                    and np.allclose(np.sort(segs, axis=0), np.sort(lib_segs, axis=0))
                ),
            }
        )

    sdf = tt_mod.sdf_from_mask_cpu(mask)
    ms = tt_mod.marching_squares_cpu(sdf, 0.0)
    ms_mask = tt_mod.marching_squares_cpu(mask.astype(np.float32), 0.5)
    polys = tt_mod.chain_segments_to_polylines(ms_mask)
    simp = [tt_mod.douglas_peucker(p, 0.75) for p in polys if p.shape[0] >= 3]
    rings = tt_mod.trace_bitmap(mask)
    regions = tt_mod.build_regions_with_holes(rings)
    svg_cubic = tt_mod.write_svg_regions(
        os.path.join(out_dir, "truetrace.regions.svg"),
        regions,
        48,
        40,
        curve="cubic",
        tension=0.0,
    )
    seg_svg = os.path.join(out_dir, "truetrace.segments.svg")
    tt_mod.write_svg_segments(seg_svg, ms[:20], 48, 40)
    areas = np.array([tt_mod.polygon_area(r) for r in rings], dtype=np.float64)
    fname = _save(
        out_dir,
        "truetrace.cpu",
        mask=mask,
        sdf=sdf,
        ms_segs=ms,
        ms_mask_segs=ms_mask,
        **{f"poly_{i}": p for i, p in enumerate(polys)},
        **{f"simp_{i}": p for i, p in enumerate(simp)},
        **_rings_to_arrays(rings),
        areas=areas,
        **{f"region_{i}_outer": r["outer"] for i, r in enumerate(regions)},
        **{
            f"region_{i}_hole_{j}": h
            for i, r in enumerate(regions)
            for j, h in enumerate(r["holes"])
        },
        otsu=np.array(tt_mod.otsu_threshold_u8((mask * 200 + 30).astype(np.uint8))),
    )
    cases.append(
        {
            "name": "cpu",
            "file": fname,
            "n_polys": len(polys),
            "n_simp": len(simp),
            "n_rings": len(rings),
            "regions": [
                {"outer": len(r["outer"]), "holes": [len(h) for h in r["holes"]]}
                for r in regions
            ],
            "svg_regions": os.path.basename(svg_cubic),
            "svg_segments": os.path.basename(seg_svg),
        }
    )

    # A photograph: PIL grayscale (so decoder differences do not matter) + Otsu mask.
    from PIL import Image

    from yapocis.images import find_image

    path = find_image("lemon")
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    for invert in (False, True):
        m = tt_mod.load_mask_from_image(path, invert=invert)
        fname = _save(out_dir, f"truetrace.lemon_{int(invert)}", gray=gray, mask=m)
        cases.append(
            {
                "name": f"lemon_{int(invert)}",
                "file": fname,
                "invert": invert,
                "otsu": tt_mod.otsu_threshold_u8(gray),
                "image": "lemon.jpg",
            }
        )
    m = tt_mod.load_mask_from_image(path, threshold=0.4)
    fname = _save(out_dir, "truetrace.lemon_thr", gray=gray, mask=m)
    cases.append(
        {"name": "lemon_thr", "file": fname, "threshold": 0.4, "image": "lemon.jpg"}
    )
    # Filling traced regions (the potrace replacement): a block with a hole and a
    # speckle, through trace_mask on both engines, and fill_regions of the
    # unsmoothed rings
    block = np.zeros((64, 80), np.uint8)
    block[10:50, 12:68] = 1
    block[22:38, 30:50] = 0
    block[55:58, 5:8] = 1
    rings = tt_mod.trace_bitmap(block, simplify_tol_px=0.0, min_area_px2=0.0)
    regions = tt_mod.build_regions_with_holes(rings)
    filled_none = tt_mod.fill_regions(regions, block.shape, curve="none")
    filled_cubic = tt_mod.fill_regions(regions, block.shape, curve="cubic", segments=8)
    traced = {}
    for engine in ("gpu", "cpu"):
        for iso in (0.0, -1.0):
            traced[f"trace_{engine}_iso{iso}"] = tt_mod.trace_mask(
                block, iso=iso, engine=engine
            )
    # 1 px strokes traced inside the ink: until 2026-10-21 chaining snapped
    # endpoints to 0.25 px, which fused their two outlines above iso 0.5, and
    # iso 1.0 lies on the strokes' own field values (off_field_iso)
    thin = np.zeros((48, 64), np.uint8)
    for i in range(6, 40):
        thin[i, i] = thin[i, i + 1] = 1
    thin[30:44, 44:62:3] = 1
    thin[30, 44:62] = 1
    for engine in ("gpu", "cpu"):
        for iso in (0.75, 1.0):
            traced[f"thin_{engine}_iso{iso}"] = tt_mod.trace_mask(
                thin, iso=iso, curve="none", engine=engine
            )
    # The outlines those fills come from, as regions_svg writes them: text, byte for byte
    svgs = {}
    for engine in ("gpu", "cpu"):
        thin_regions = tt_mod.trace_regions(thin, iso=0.75, engine=engine)
        for curve in ("none", "cubic"):
            name = f"truetrace.fill.{engine}_{curve}.svg"
            with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
                f.write(tt_mod.regions_svg(thin_regions, 64, 48, curve=curve))
            svgs[f"{engine}_{curve}"] = name
    name = "truetrace.fill.styled.svg"
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        f.write(tt_mod.regions_svg(regions, 80, 64, curve="cubic", tension=0.5,
                                   ink="#1a2b3c", paper=None))
    svgs["styled"] = name
    fname = _save(
        out_dir,
        "truetrace.fill",
        mask=block,
        thin_mask=thin,
        filled_none=filled_none,
        filled_cubic=filled_cubic,
        **traced,
    )
    cases.append(
        {
            "name": "fill",
            "file": fname,
            "regions": len(regions),
            "traced": sorted(traced),
            "regions_svg": svgs,
        }
    )
    tt.purge()
    return cases


# ----------------------------------------------------------------------------
# imageops
# ----------------------------------------------------------------------------


@generator("imageops")
def imageops_cases(
    out_dir: str, backend: str, log: Callable[..., None]
) -> list[dict[str, Any]]:
    from yapocis.lib.imageops import Imageops

    rng = np.random.default_rng(SEED)
    ops = Imageops()
    im = rng.uniform(-0.2, 1.3, size=(64, 48)).astype(np.float32)
    big = rng.beta(2.0, 5.0, size=(200, 180)).astype(
        np.float32
    )  # > 32768 pixels: int32 wrap in histeq
    target = rng.uniform(0.0, 1.0, size=(64, 48)).astype(np.float32)
    cases = []
    out = {
        "clamp": ops.clamp_pixels(im, 0.0, 1.0),
        "clamp_wide": ops.clamp_pixels(im, 0.1, 0.9),
        "stretch": ops.stretch(im, -0.2, 1.3, 0.0, 1.0),
        "binarize": ops.binarize(im, 0.5),
        "histogram": ops.histogram(np.clip(im, 0, 1), n_bins=256),
        "histeq_small": ops.histeq(np.clip(im, 0, 1), nbr_bins=1024),
        "histeq_default": ops.histeq(np.clip(im, 0, 1)),
        "histeq_big": ops.histeq(big, nbr_bins=1024),
    }
    target_hist = ops.histogram(target, n_bins=256).astype(np.float64)
    out["histmatch"] = ops.histmatch(np.clip(im, 0, 1), target_hist)
    fname = _save(
        out_dir,
        "imageops.all",
        im=im,
        big=big,
        target=target,
        target_hist=target_hist,
        **out,
    )
    cases.append({"name": "all", "file": fname, "outputs": sorted(out)})
    ops.purge()
    return cases


# ----------------------------------------------------------------------------
# simplex
# ----------------------------------------------------------------------------


@generator("simplex")
def simplex_cases(
    out_dir: str, backend: str, log: Callable[..., None]
) -> list[dict[str, Any]]:
    from yapocis.lib.simplex import Simplex

    s = Simplex()
    out = {
        "noise2d": s.simplex2d(64, 48, seed=42, scale=0.05, octaves=4),
        "noise2d_tile": s.simplex2d(
            64,
            48,
            seed=7,
            scale=0.1,
            offset=(1.5, -2.0),
            octaves=3,
            persistence=0.6,
            lacunarity=2.5,
            tileable=True,
        ),
        "noise3d": s.simplex3d(
            16, 12, 8, seed=99, scale=0.2, offset=(0.1, 0.2, 0.3), octaves=2
        ),
        "turbulence": s.turbulence(48, 32, seed=5, scale=0.08, octaves=5),
        "ridged": s.ridged(48, 32, seed=5, scale=0.08, octaves=5),
    }
    fname = _save(out_dir, "simplex.all", **out)
    s.purge()
    return [{"name": "all", "file": fname, "outputs": sorted(out)}]


# ----------------------------------------------------------------------------
# anisodiff (the RGB and iterator variants; anisodiff_n is covered by lib tests)
# ----------------------------------------------------------------------------


@generator("anisodiff")
def anisodiff_cases(
    out_dir: str, backend: str, log: Callable[..., None]
) -> list[dict[str, Any]]:
    from yapocis.lib.anisodiff import Anisodiff

    rng = np.random.default_rng(SEED)
    an = Anisodiff()
    x = np.linspace(0, 1, 32, dtype=np.float32)[:, None]
    y = np.linspace(0, 1, 24, dtype=np.float32)[None, :]
    base = (0.3 + 0.4 * ((x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.06)).astype(np.float32)
    rgb = np.stack([base, base * 0.8 + 0.1, 1.0 - base], axis=2).astype(np.float32)
    rgb = np.clip(rgb + rng.normal(0, 0.02, rgb.shape).astype(np.float32), 0.0, 1.0)
    gray = np.ascontiguousarray(rgb[:, :, 0])
    frames = []
    it = an.anisodiff(gray, scale=3.0, step_size=0.2)
    for _ in range(3):
        frames.append(next(it))
    rgb_frames = []
    it2 = an.anisodiff_rgbI(rgb, scale=3.0, step_size=0.2)
    for _ in range(2):
        rgb_frames.append(next(it2))
    out = {
        "rgb_i_n": an.anisodiff_rgbI_n(rgb, 3, scale=3.0, step_size=0.2),
        "rgb_is_n": an.anisodiff_rgbIS_n(rgb, 3, scale=3.0, step_size=0.2),
        "gray_n": an.anisodiff_n(gray, 3, scale=3.0, step_size=0.2),
    }
    for i, f in enumerate(frames):
        out[f"frame_{i}"] = f
    for i, f in enumerate(rgb_frames):
        out[f"rgb_frame_{i}"] = f
    fname = _save(out_dir, "anisodiff.rgb", rgb=rgb, gray=gray, **out)
    an.purge()
    return [
        {
            "name": "rgb",
            "file": fname,
            "n": 3,
            "scale": 3.0,
            "step_size": 0.2,
            "frames": 3,
            "rgb_frames": 2,
        }
    ]


# ----------------------------------------------------------------------------
# fft (spectrogram and the analysis helpers)
# ----------------------------------------------------------------------------


@generator("fft")
def fft_cases(
    out_dir: str, backend: str, log: Callable[..., None]
) -> list[dict[str, Any]]:
    from yapocis.lib.fft import FFT

    rng = np.random.default_rng(SEED)
    f = FFT()
    sr = 8000.0
    t = np.arange(4096) / sr
    signal = (np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 1200 * t)).astype(
        np.float32
    )
    noise = rng.normal(0, 0.1, signal.shape).astype(np.float32)
    noisy = (signal + noise).astype(np.float32)
    other = np.roll(noisy, 7).astype(np.float32)
    times_b, freqs_b, spec_b = f.spectrogram(
        noisy, sample_rate=sr, window_size=256, hop_size=64
    )
    times_s, freqs_s, spec_s = f.spectrogram(
        noisy[:1500], sample_rate=sr, window_size=512, hop_size=256, return_db=False
    )
    coh_f, coh = f.coherence(noisy[:256], other[:256])
    out = {
        "spec_batch_times": times_b,
        "spec_batch_freqs": freqs_b,
        "spec_batch": spec_b,
        "spec_seq_times": times_s,
        "spec_seq_freqs": freqs_s,
        "spec_seq": spec_s,
        "bandpass": f.bandpass_filter(noisy[:256], 300.0, 600.0, sample_rate=sr),
        "coherence_freqs": coh_f,
        "coherence": coh,
        "estimate_frequency": np.array(
            f.estimate_frequency(noisy[:256], sample_rate=sr)
        ),
        "energy": np.array(f.energy(noisy)),
        "rms": np.array(f.rms(noisy)),
        "snr": np.array(f.snr(signal, noise)),
    }
    fname = _save(
        out_dir,
        "fft.analysis",
        signal=signal,
        noise=noise,
        noisy=noisy,
        other=other,
        **out,
    )
    f.purge()
    return [
        {"name": "analysis", "file": fname, "sample_rate": sr, "outputs": sorted(out)}
    ]

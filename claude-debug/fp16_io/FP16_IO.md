# FP16 image read/write without a full-size FP32 copy

Date: 2026-10-07, thresher (M4 Max). Test image: `test.jpg` tiled 5x5, giving 4060x4055x3 uint8 (49M values). Scripts are in this directory.

## Read: `imread(..., dtype=np.float16)`

Today `imread` returns `img.astype(np.float32) / max` (`utils/basic.py:221`). Getting FP16 means calling `.astype(np.float16)` afterwards, which costs one full-size FP32 array (189 MB here) and a second pass.

| Variant | ms | Peak MB | Bit-identical to FP32 route |
|---|---|---|---|
| FP32 today (baseline) | 18.0 | 377 | n/a (FP32) |
| FP16 via full FP32 | 20.0 | 377 | yes |
| 256-entry LUT, `lut[img]` (no FP32 at all) | 37.8 | 94 | yes |
| LUT via `np.take` | 78.0 | 471 | yes |
| `astype(f16)` then FP16 divide | 170.1 | 188 | yes |
| **FP32 slab of 64 rows, written into an FP16 output** | **18.1** | **97** | **yes** |

- **Recommended: the slab version.** It is as fast as today's FP32 read, uses a quarter of the memory, and gives identical bits. FP32 exists only in a slab of about 3 MB, never as a full-size image.
- If "no FP32 at all" is the requirement, the LUT is exact and uses the same memory, but it is 2x slower.
- Numpy does FP16 arithmetic in software, so any variant that does maths in FP16 is slow.
- Exactness: the 256-entry uint8 LUT, correctly rounded from float64, matches the FP32 double-rounding route on all 256 values. For uint16 sources it differs on 2 of 65536 values; the LUT is the more accurate of the two.
- The `longest=` resize path goes through `skimage.transform.resize`, which returns float64. Converting that directly to FP16 skips FP32 but not the float64 array. Avoiding float64 would mean resizing on uint8 (e.g. PIL), which changes the resampling result, so it is not proposed.

## Write: is there a common path?

There are three float→uint8 paths in `utils/basic.py`:

| Path | Used by | FP16 today |
|---|---|---|
| `byte_scale` (`:48`), via `to_image` | `to_image` → `sign`, graphics position tests, `utils/advanced.py` | Correct, but **771.8 ms** (FP16 min/max and arithmetic in software) |
| `array_to_image` (`:294`) | `sign_array` / viewer-queue saves (`showcheck`, etc.) | **Fails:** `TypeError: Cannot handle this data type: (1, 1, 3), <f2`. Its float branch only lists float32/float64 |
| `imwrite` (`:226`) | No callers | **Broken for every dtype:** `AttributeError: 'numpy.ndarray' object has no attribute 'to_image'`. Its `*= 255` is also undone by `to_image`'s auto-ranging |

`byte_scale` is the natural common funnel, and `to_image` already uses it. The same slab treatment applies: compute min/max over FP32 slabs, then scale each slab into a preallocated uint8 output.

| Write variant (byte_scale of the FP16 image) | ms | Peak MB | px different from float64-exact |
|---|---|---|---|
| Today: `byte_scale(f16)` | 771.8 | 188 | 0 |
| Today on an FP32 copy | 34.7 | 565 | 0 |
| 65536-entry LUT (bit pattern → byte) + bincount range | 123.6 | 424 | 0 |
| **Two-pass FP32 slabs (64 rows)** | **36.4** | **56** | **0** |

Caveat on unifying: `array_to_image` truncates, while `byte_scale` rounds (`+0.4999`). Routing `array_to_image` through `byte_scale` would shift some viewer-queue PNGs by one grey level. The minimal fix is to add float16 to `array_to_image`'s float branch and leave its truncation unchanged.

## Proposed change

1. A private helper in `utils/basic.py` that converts an integer image to `dtype` in [0, 1] through row slabs. `imread` gains `dtype=np.float32` and uses it; the default output is unchanged.
2. In `byte_scale`, take the FP16 input through the two-pass slab path. All other dtypes are unchanged.
3. Accept float16 in `array_to_image`, keeping truncation.
4. Fix `imwrite`: drop the `*= 255` and call `to_image(img, cmin=0, cmax=1)`, so it saves [0, 1] data without auto-ranging. Whether it should auto-range is a decision for the maintainer.

## Status (2026-10-07): implemented, uncommitted

- `utils/basic.py`: `imread(..., dtype=np.float32)`; `_to_unit_float` (slab conversion); `byte_scale` routes float16 to `_byte_scale_float16` (two-pass slabs); `array_to_image` widens float16 to float32; `imwrite` saves [0, 1] data with no rescaling: NaN→0, ±inf and out-of-range values are clipped, and `to_image` is called with `cmin=0, cmax=1`.
- Tests: `tests/integration/test_image_io.py` (19 tests). The default float32 `imread` output is bit-identical to the old code, both with and without `longest`. float16 matches `imread(...).astype(np.float16)` bit for bit.
- Regression run: graphics position validation, anisodiff and `tests/integration` (126 passed); `-m smoke` (121 passed, 1 skipped).
- `make test-smoke` is stale: it points at `tests/test_backend_smoke.py`, which does not exist. This was already the case before this change.

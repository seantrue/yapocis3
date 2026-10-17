# yapocis dependency licence scan (2026-10-17)

## Result

**No poison licences in anything yapocis ships or pulls in for users.** The core install and every user-facing extra (`opencl`, `metal`, `cuda`, `numba`, `mpl`) are permissive throughout. The only copyleft or proprietary licences are in the developer tooling (`dev` and `docs` extras), which is never redistributed.

## Method

- **Whole locked tree:** all 171 packages in `uv.lock`, covering every extra and platform, including Linux-only wheels. The licence of each exact locked version was read from PyPI (`license_expression`, `license` and the `License ::` classifiers) by `claude-debug/release_eval_license_scan.py`.
- **Which install pulls what:** `uv export --no-dev [--extra X]` for core and for each extra.
- **Bundled native code:** the licence files and `.dylib`s inside the installed runtime wheels (macOS, Python 3.13, clean venv from the yapocis wheel). Metadata alone misses vendored C libraries.

## Flagged packages and where they come from

| Package | Licence | Pulled in by | Verdict |
|---|---|---|---|
| siphash24 | Apache-2.0 OR LGPL-2.1-or-later | **core** (pyopencl → pytools) | OK: dual-licensed; choose Apache-2.0. |
| nvidia-*-cu12 (14 packages) | NVIDIA Proprietary; 3 are BSD/Apache but tagged "Other/Proprietary" | `dev` only (torch, Linux) | Dev tooling, not redistributed. `make install-dev` on Linux installs NVIDIA's EULA-licensed CUDA libraries through torch. |
| pygame | LGPL | `dev` only | Fine to use unmodified; not shipped. |
| astroid | LGPL-2.1-or-later | `docs`/`dev` (sphinx-autoapi) | Docs build only. |
| certifi, pathspec | MPL-2.0 | `docs`/`dev` | Weak copyleft, used unmodified; not shipped. |
| docutils | BSD / PSF / public domain; GPL-3 classifier | `docs`/`dev` | The GPL covers only `tools/editors/emacs/rst.el`, which is not in the wheel. |
| mypy-extensions | no PyPI metadata; installed METADATA says MIT | `dev` | OK. |
| pyrefly | no licence metadata at all; upstream LICENSE is MIT (Meta) | `dev` | OK. |

## Bundled native libraries in runtime wheels

| Wheel | Bundled code | Licence finding |
|---|---|---|
| numpy, scipy | libgfortran, libgcc, libquadmath | GPL-3.0-or-later **with the GCC Runtime Library Exception**, which permits shipping with non-GPL software. OK. |
| Pillow | freetype, harfbuzz, libjpeg, libtiff, libwebp, lcms2, openjpeg, liblzma, libxcb, ... | Its only GPL mention is in the XZ Utils notice; liblzma itself is public domain. OK. |
| imagecodecs | ~45 codec libraries (aom, jxl, openexr, zstd, ...) | It ships 67 licence files and none is GPL/LGPL. A "not for commercial use" hit refers to a journal article cited in the libjpeg README, not to code. OK. |
| numba, typing_extensions | (Python licence history text) | The GPL mention is the PSF history note. OK. |
| llvmlite | libllvmlite (LLVM) | Apache-2.0 WITH LLVM-exception, BSD-2. OK. |
| freetype-py | libfreetype | FreeType is FTL **or** GPLv2; FTL (BSD-style, with credit) applies. Not redistributed by yapocis. |
| pymetallic | libpymetallic | MIT (yours). |

## Not covered

- CUDA (`pycuda`) was checked from metadata only (MIT); it was not installed here, since it needs the CUDA toolkit.
- The native-library scan used the macOS wheels. Linux and Windows wheels of the same packages may bundle a different set (e.g. OpenBLAS builds), and were not inspected.

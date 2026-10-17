# yapocis PyPI release evaluation (2026-10-17)

Evaluated at HEAD `872e8e8`. Nothing was released, tagged, or changed in the repo.

## Verdict

**Not ready.** Licensing is clean. The blockers are versioning, the changelog, and a README that doesn't work. Each of these is a small fix.

## Blockers

| Area | Finding |
|---|---|
| Version | pyproject says `0.3.8`, but tag `v0.3.8` already exists and HEAD is 28 commits past it. A `0.3.9` bump was committed and then reverted. PyPI's latest is `0.2.4`, so no 0.3.x has ever been published. |
| Changelog | There is no `[0.3.8]` entry. `[Unreleased]` holds the docs rewrite that shipped in 0.3.8. PRs #32–#46 are not listed: colorspace fixes, the (x,y) drawing convention, CUDA fixes, FP16 I/O, yapocis-cuda-cards, and the anisodiff/gradient/split_channels fixes. |
| README code | `import core` / `from core import get_manager` fail with ModuleNotFoundError; the module is `yapocis.core`. `prog.mandelbrot(grid, maxiter=256)` fails because bound kernels reject keyword args. |
| README render | A broken code fence in the graphics example (L176–210) turns the comments into H1s and pulls "## Development" into a code block. PyPI renders the README as the project page. |
| `YAPOCIS_BACKEND=numba` | This is documented in the README (L156), CLAUDE.md and docs/quickstart. A clean install rejects it with `RuntimeError: backend='numba' is not one of available backends=['ocl', 'numbax']`. `get_backend()` accepts only `numbax`; `get_manager()` accepts both. Fix the code (accept the alias) or the docs. |
| Release tooling | `twine check` with the locked twine 6.2.0 fails: `'2.5' is not a valid metadata version`. Current hatchling emits Metadata-Version 2.5. twine 7.0.0 passes both artifacts. `make publish` uses the venv's twine, so bump it in `uv.lock`, and upload to TestPyPI first. |

## Licensing / contamination

| Item | Status |
|---|---|
| Project license | MIT (LICENSE.txt, metadata, classifier all agree). Source headers say "Sean D. True & Swapwizard LLC", but LICENSE.txt names only Sean D. True. Make them consistent. |
| Runtime deps (clean venv from the wheel) | All permissive: MIT, BSD, Apache, PSF, MIT-CMU. No GPL, LGPL or AGPL. `siphash24` is "Apache-2.0 OR LGPL-2.1+"; use the Apache option. FreeType inside freetype-py is FTL/GPLv2 dual-licensed; FTL is fine. |
| Full dependency-tree scan | **No poison licences** in core or any user-facing extra; copyleft/proprietary only in dev/docs tooling (torch's NVIDIA CUDA libs, pygame, astroid, MPL certifi/pathspec). See `release_eval_license_scan_2026-10-17.md`. |
| Optional deps | pycuda MIT, pymetallic MIT (your package), matplotlib PSF-style, mapbox-earcut ISC. |
| Dev-only deps | pygame is LGPL. It is never bundled, so it does not affect the wheel. |
| Ubuntu fonts (8 .ttf, ~2.4 MB) | Ubuntu Font Licence 1.0. `UFL.txt` ships next to them. OK. |
| `lib/typeset.py` | **Fixed.** The empty terms block now holds freetype-py's full BSD notice: the `wordle.py` header and the licence text from freetype-py's LICENSE.txt. |
| `kernels/{ocl,cuda}/hsi.mako` | The RGB-HSV code is from a StackOverflow answer (CC BY-SA, a share-alike license). The original is Sam Hocevar's lolengine post (WTFPL). Cite that source instead of SO to avoid the CC BY-SA question. Low risk. |
| `src/yapocis/images/*.jpg` | **Replaced.** The 13 classifier demo photos (probably ILSVRC; not from BSR) are now CC0 1.0 photos found through Openverse and listed in `src/yapocis/images/CREDITS.md`, with their sources: Wikimedia, rawpixel, Flickr, StockSnap. ResNet50 classifies 12 of the 13 as their own class; cassette_player (chosen by eye) comes out as "cassette", the tape beside the recorder. `test.jpg` is the package author's own photograph, distributed under the package's MIT licence (recorded in CREDITS.md). |
| Model weights | `.npz` is excluded from the wheel and sdist. The models README says to run `python scripts/download_vgg16.py`, which isn't in the wheel. Torchvision weights are BSD, but they were trained on ImageNet; mention that if you document downloads. |

**Dates (fixed):** copyright end years are 2026 in 73 source headers and `LICENSE.txt` (holder names unchanged). The ChatGPT-written Perona–Malik comment in the OCL/CUDA anisodiff kernels credited "Carlo Perona and João Malik"; it now says Pietro Perona and Jitendra Malik, IEEE PAMI 12(7), 1990.

**Known issue (shipping with it):** OpenCL FFT gives wrong results on the Tesla P40 (8 tests in `tests/lib/fft`). It predates 1.0.0 (it reproduces at `8b049af`); CUDA on the P40 and OpenCL on the 3080 pass. It is recorded in CHANGELOG under 1.0.0.

## Packaging

| Item | Status |
|---|---|
| Build | `uv build` works: 3.8 MB wheel, 4.5 MB sdist. |
| Clean install (py3.13, macOS) | Imports work. The default backend `ocl` and `numbax` both run Gaussian. `yapocis-tools backends` is correct. All 8 entry points answer `--help`. |
| `__version__` | Not defined in `yapocis/__init__.py`. |
| sdist contents | Includes oldtests/, `vad_yapocis.py`, `VAD_README.md`, `cq`, `activate.sh`, `.settings`, AI.md/CLAUDE.md/AGENTS.md. This is harmless but untidy; add sdist `only-include`/`exclude`. |
| `src/yapocis/lib/#gaussian.py#` | An Emacs autosave file tracked in git. It is **not** in the wheel or sdist, so just `git rm` it. |
| Bound-kernel `__getattr__` | Raises `KeyError` instead of `AttributeError`, so `hasattr(Gaussian(), "x")` raises. Minor, but it breaks introspection tools. |

## Docs

**Fixed (2026-10-17):** README examples and the `numba` alias; `.readthedocs.yml` (Python 3.13, ubuntu-24.04, installs the package); `conf.py` (version from package metadata, copyright, repo name, mocks for pycuda/pymetallic); the `docs` extra now uses sphinx-rtd-theme; `get_backends` imports (5 pages imported it from `yapocis.core.manager`, which doesn't define it); `yapocis.numbax` naming; the made-up `yapocis.assets.add_asset_path` (`YAPOCIS_PROJECT` is a single directory); the `yapocis-cuda-cards` CLI docs; the `yapocis-doctest` options. A fresh-venv Sphinx build passes, with 9 docstring-format warnings and no import failures, and the API pages are populated. The doc examples on the edited pages pass.

README links now use absolute `github.com/seantrue/yapocis3/blob/main/...` URLs (all 10 targets exist on main). The license badge is a static MIT badge, and the CI badge, which pointed to a workflow that doesn't exist, is removed. `twine check` passes. The two failing examples in `docs/library/channels.md` (`histeq` is an `Imageops` method; `rgb2hsi3` returns one array) are fixed: `yapocis-doctest docs/` gives 210 passed, 0 failed.

Docstring warnings are fixed. `sphinx-build -W` now passes with zero warnings: RST formatting in 5 docstrings, an unresolved `FreeTypeFont` forward reference, and PIL/pyparsing names leaking in through imports (now `ImageFont.*` and pyparsing's `DelimitedList`). The API page also listed `parse_interface_file`, which doesn't exist; it's `parse_interface_text`.

**Still open:** the GitHub repo is **private**, so every github.com link in the README and the pyproject URLs 404 for PyPI visitors until it is made public.

## Suggested order

1. Fix the README examples and code fence, fix `numba`/`numbax`, and fill in the typeset.py BSD notice.
2. Resolve where the images came from.
3. Write the changelog for 0.3.9 and bump to 0.3.9.
4. Upgrade twine, publish to TestPyPI, `pip install` from it, run the smoke tests.
5. Tag, update AI.md, publish.

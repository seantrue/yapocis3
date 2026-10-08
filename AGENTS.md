# Repository Guidelines

## Project Structure & Module Organization
- `yapocis/` – core package:
  - `ocl/` OpenCL devices, buffers, kernel build (`DeviceManager`, `KernelManager`).
  - `core/` shared utilities (backend selection, autotune, tunecache, interface parsing).
  - `cu/` and `cukernels/` CUDA support via CuPy and Mako templates.
  - `interfaces/`, `kernels/` interface DSL and Mako kernel templates.
- `tests/` – pytest suite (`test_*.py`).
- `examples/` – runnable demos and performance scripts.
- `docs/` – documentation (Read the Docs).
- `scripts/` – helper utilities (e.g., `scripts/imagewatcher.py`).

## Build, Test, and Development Commands
- Create env + install:
  - `make install-dev` (recommended)
  - Or manually: `uv venv .venv && source .venv/bin/activate && uv pip install -e ".[dev]"`
- Run tests:
  - `make test` (all available backends)
  - `make test-ocl` (OpenCL only)
  - `make test-metal` (Metal only)
  - `pytest -q` (direct pytest, example: `pytest tests/backend/test_smoke.py -q`)
- Run examples: `python examples/matplotlib_median_performance.py`.

## Coding Style & Naming Conventions
- Python, 4‑space indentation, PEP 8. Use type hints where practical.
- Names: modules/files `snake_case.py`; functions/vars `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE`.
- Keep functions small and documented; module/class/function docstrings use triple quotes with a one‑line summary.
- Prefer explicit imports within the `yapocis` namespace.

## Testing Guidelines
- Framework: pytest. Place tests in `tests/` named `test_*.py` with clear arrange/act/assert sections.
- Include GPU/CPU expectations and deterministic assertions. Add a focused test for each new kernel or interface.
- Quick loop: `pytest -k <keyword> -q`.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject (≤72 chars). Example: `Add CUDA tunecache path handling`.
- Reference issues: `Fixes #123` or `Refs #123` when applicable.
- PRs must include: summary, rationale, affected modules, how to test (commands), and backend/hardware tested (e.g., `OpenCL CPU`, `CUDA RTX 3060`). Add screenshots/SVGs if output changes; update docs/examples when behavior or APIs change.

## Security & Configuration Tips
- Do not commit secrets or large artifacts. Prefer sample data in `examples/`.
- Useful env vars: `YAPOCIS_BACKEND` (`ocl`|`cuda`), `YAPOCIS_DEVICE` (device name/index), `YAPOCIS_CACHE` (kernel cache dir), `YAPOCIS_TUNECACHE` (autotune JSON path), `YAPOCIS_PROJECT` (extra asset root), `YAPOCIS_VIEWER` (image viewer).
- See `OVERVIEW.md` for architecture details and asset lookup behavior.


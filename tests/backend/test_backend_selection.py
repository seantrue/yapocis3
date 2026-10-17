"""An explicit backend argument wins over YAPOCIS_BACKEND."""

from __future__ import annotations

import pytest

from yapocis.core import (
    BACKEND_ALIASES,
    canonical_backend,
    get_backend,
    get_backends,
    get_manager,
    get_test_backends,
    load_program,
)

_MODULES = {
    "ocl": "yapocis.ocl.kernel_manager",
    "metal": "yapocis.metallic.kernel_manager",
    "cuda": "yapocis.cu.kernel_manager",
    "numba": "yapocis.numbax.kernel_manager",
    "numbax": "yapocis.numbax.kernel_manager",
}


def _other_backend(backend: str) -> str:
    others = [b for b in get_test_backends() if _MODULES[b] != _MODULES[backend]]
    if not others:
        pytest.skip("needs two test backends")
    return others[0]


@pytest.mark.parametrize("backend", get_test_backends())
def test_get_manager_uses_argument(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", _other_backend(backend))
    assert get_manager(backend).__name__ == _MODULES[backend]
    assert get_manager(backend.upper()).__name__ == _MODULES[backend]


@pytest.mark.parametrize("backend", get_test_backends())
def test_get_manager_default_uses_environment(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", backend)
    assert get_manager().__name__ == _MODULES[backend]
    assert get_manager(None).__name__ == _MODULES[backend]


@pytest.mark.parametrize("backend", get_test_backends())
def test_load_program_uses_backend_argument(backend, monkeypatch):
    monkeypatch.setenv("YAPOCIS_BACKEND", _other_backend(backend))
    kmgr = load_program("demo", backend=backend)
    assert type(kmgr).__module__ == _MODULES[backend]


def test_get_manager_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        get_manager("vulkan")


@pytest.mark.parametrize("alias", sorted(BACKEND_ALIASES))
def test_environment_accepts_backend_alias(alias, monkeypatch):
    canonical = BACKEND_ALIASES[alias]
    if canonical not in get_backends():
        pytest.skip(f"{canonical} not installed")
    monkeypatch.setenv("YAPOCIS_BACKEND", alias.upper())
    assert get_backend() == canonical
    assert get_manager().__name__ == _MODULES[canonical]


def test_canonical_backend_passes_through_unknown_names():
    assert canonical_backend("Vulkan") == "vulkan"

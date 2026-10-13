"""Search order for kernels and assets: $YAPOCIS_PROJECT, then the current directory, then
the yapocis package; and how numbax kernel modules are loaded from each."""

from __future__ import annotations

import os
import sys

from yapocis.numbax.kernel_manager import _load_numba_kernel_module
from yapocis.utils.internals import get_yapocis_dir, get_yapocis_dirs

ALIAS = "yapocis_ext.kernels.numbax.gradient"


def test_project_first(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("YAPOCIS_PROJECT", str(project))
    monkeypatch.chdir(tmp_path)
    assert get_yapocis_dirs() == [str(project), str(tmp_path), get_yapocis_dir()]


def test_no_duplicates(tmp_path, monkeypatch):
    monkeypatch.setenv("YAPOCIS_PROJECT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert get_yapocis_dirs() == [str(tmp_path), get_yapocis_dir()]


def test_without_project(tmp_path, monkeypatch):
    monkeypatch.delenv("YAPOCIS_PROJECT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert get_yapocis_dirs() == [str(tmp_path), get_yapocis_dir()]


def test_builtin_numbax_kernel_uses_package_name(tmp_path, monkeypatch):
    monkeypatch.delenv("YAPOCIS_PROJECT", raising=False)
    monkeypatch.chdir(tmp_path)
    module = _load_numba_kernel_module("gradient")
    assert module.__name__ == "yapocis.kernels.numbax.gradient"
    assert module is sys.modules["yapocis.kernels.numbax.gradient"]
    # Numba caches written before October 2026 pickled this name
    assert sys.modules[ALIAS] is module


def test_project_numbax_kernel_overrides_builtin(tmp_path, monkeypatch):
    kernels = tmp_path / "kernels" / "numbax"
    kernels.mkdir(parents=True)
    (kernels / "gradient.py").write_text("FROM_PROJECT = True\n")
    monkeypatch.setenv("YAPOCIS_PROJECT", str(tmp_path))
    monkeypatch.chdir(os.path.dirname(get_yapocis_dir()))
    # The loader replaces this entry; put back whatever was there
    if ALIAS in sys.modules:
        monkeypatch.setitem(sys.modules, ALIAS, sys.modules[ALIAS])
    else:
        monkeypatch.delitem(sys.modules, ALIAS, raising=False)
    module = _load_numba_kernel_module("gradient")
    assert getattr(module, "FROM_PROJECT", False)
    assert module.__name__ == ALIAS

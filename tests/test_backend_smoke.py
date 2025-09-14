"""
Smoke-compile every interface across available backends (OCL, Metal, CUDA).

- Skips backends if their runtime is not available.
- Supplies minimal template context for parameterized interfaces.
- Only compiles; does not execute kernels.
"""

from __future__ import annotations

import os
import pytest
from yapocis.interfaces import interfaces as list_interfaces, find_interface
from yapocis.core.manager import get_backends


def _list_interface_names() -> list[str]:
    # Use registry to list available interfaces, normalize to basenames (stems).
    return [n.rsplit(".", 1)[0] for n in list_interfaces()]


def _context_for_interface(name: str) -> dict:
    """Return minimal mako variables to render parameterized interfaces."""
    if name == "operators":
        # define a couple operators
        return {"operators": [("add", "+"), ("sub", "-")]} 
    if name == "convolves":
        # define a couple 1D kernels
        return {"convs": [("conv3", [0.25, 0.5, 0.25]), ("edge3", [-1.0, 0.0, 1.0])]} 
    if name == "convolvesep":
        return {"convs": [("sep3", [0.25, 0.5, 0.25])]} 
    if name == "convolve":
        return {"name": "conv3", "conv": [0.25, 0.5, 0.25]}
    # default: no extra context needed
    return {}


@pytest.mark.parametrize("backend", get_backends())  # exercise detected backends
@pytest.mark.parametrize("iname", _list_interface_names())
def test_compile_interfaces_all_backends(backend: str, iname: str):
    os.environ["YAPOCIS_BACKEND"] = backend
    from yapocis.core.manager import get_manager

    mgr = get_manager(backend)
    ctx = _context_for_interface(iname)
    # Ensure the interface is discoverable via the registry
    assert os.path.basename(find_interface(iname)).startswith(iname)
    # load_program compiles per-backend sources
    prog = mgr.load_program(iname, **ctx)
    assert getattr(prog, "source", None) is not None or backend == "cuda"
    # CUDA path does not expose source attribute in the same way

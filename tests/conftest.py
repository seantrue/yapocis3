import importlib
from pathlib import Path


def pytest_ignore_collect(path: str, config):
    """Conditionally skip collecting CUDA PyCUDA test if pycuda is missing.

    The file yapocis/cukernels/test_convolve.py imports pycuda.autoinit unconditionally.
    When PyCUDA is not installed, let pytest skip collecting just that file.
    """
    p = Path(path)
    if p.as_posix().endswith("yapocis/cukernels/test_convolve.py"):
        try:
            importlib.import_module("pycuda.autoinit")
            return False
        except Exception:
            return True
    return False


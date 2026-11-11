"""#82: a test that sets YAPOCIS_BACKEND directly must not leak it into later tests.

Before the fix, tests/async/test_performance.py (and backend_comparison) left the last
backend they set in os.environ, so every later unparametrized test ran on it.
"""

import os

_AT_IMPORT = os.environ.get("YAPOCIS_BACKEND")


def test_a_sets_the_backend_directly():
    os.environ["YAPOCIS_BACKEND"] = "numbax" if _AT_IMPORT != "numbax" else "ocl"


def test_b_the_next_test_sees_the_original_backend():
    assert os.environ.get("YAPOCIS_BACKEND") == _AT_IMPORT

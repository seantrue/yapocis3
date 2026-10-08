"""IDL dtype names map to numpy dtypes."""

import numpy as np
import pytest

from yapocis.core.interface import ctype_to_nptype, get_interface, nptype_to_ctype


@pytest.mark.parametrize(
    "idl, expected",
    [
        ("float", np.float32),
        ("half", np.float16),
        ("f16", np.float16),
        ("double", np.float64),
    ],
)
def test_param_dtype(idl, expected):
    spec = get_interface(f"interface t {{ kernel k(in {idl} a, out_like a); }};")
    a = spec.kernels[0].parameters[0]
    assert np.dtype(getattr(np, a.np_dtype)) == expected


def test_every_mapped_type_is_a_numpy_dtype():
    for name in ctype_to_nptype.values():
        assert isinstance(getattr(np, name)(0), np.generic)


def test_reverse_map_prefers_canonical_names():
    assert nptype_to_ctype["float16"] == "half"
    assert nptype_to_ctype["float32"] == "float"

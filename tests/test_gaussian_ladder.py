"""The pinned Gaussian ladder (yapocis.lib.gaussian_ladder) is what GaussianWeights()
returns, and it agrees with a fresh computation to within the last bits."""

import numpy as np

from yapocis.lib.gaussian import GaussianWeights
from yapocis.lib.gaussian_ladder import LADDER, LADDER_PARAMS


def test_default_weights_are_the_pinned_ladder():
    gws = GaussianWeights().gaussian_weights()
    assert [len(w) for w in gws] == [9, 13, 19, 27, 39, 59, 87, 127]
    for got, pinned in zip(gws, LADDER, strict=True):
        assert got.dtype == np.float64
        assert got.tolist() == pinned


def test_pinned_ladder_matches_a_fresh_computation():
    fresh = GaussianWeights(pinned=False).gaussian_weights()
    assert len(fresh) == len(LADDER)
    for got, pinned in zip(fresh, LADDER, strict=True):
        np.testing.assert_allclose(got, pinned, rtol=1e-12, atol=0)


def test_other_parameters_are_computed():
    gws = GaussianWeights(max_width=50, scale=1.6, limit=0.001).gaussian_weights()
    assert LADDER_PARAMS != {"max_width": 50, "scale": 1.6, "limit": 0.001}
    assert [len(w) for w in gws] == [9, 13, 19, 27, 39]


def test_ladder_stops_when_the_peak_falls_below_the_limit():
    # With limit 0.01 the peak of a sigma-43 Gaussian is below the limit, so no
    # kernel can reach width 50; this used to loop forever
    gws = GaussianWeights(max_width=50, scale=1.6, limit=0.01).gaussian_weights()
    assert all(len(w) > 1 for w in gws)
    assert len(gws) == 8

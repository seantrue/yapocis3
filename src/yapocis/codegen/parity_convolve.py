#  Copyright Sean D. True and Swapwizard LLC (c) 2011-2026.
"""Library parity for runtime-weight convolutions: Convolve1D and Gaussian(gaussian_weights=...)."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np

from .parity_lib import generator


@generator("convolve")
def convolve_cases(
    out_dir: str, backend: str, log: Callable[..., None]
) -> list[dict[str, Any]]:
    from yapocis.lib.convolve import Convolve1D
    from yapocis.lib.gaussian import Gaussian, GaussianWeights

    rng = np.random.default_rng(20261019)
    cases: list[dict[str, Any]] = []

    # Convolve1D with weights that are not in the bundle
    weights = [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]
    signal = rng.uniform(-1.0, 1.0, size=4096).astype(np.float32)
    cnv = Convolve1D(name="k7", conv=weights)
    out = cnv.k7(signal.copy())
    fname = "convolve.k7.npz"
    np.savez_compressed(os.path.join(out_dir, fname), signal=signal, out=out)
    cases.append({"name": "k7", "file": fname, "weights": weights, "kernel": "k7"})
    cnv.purge()

    # Gaussian with custom sigmas: blur at each scale and the zcsdog pipeline
    gw = GaussianWeights(max_width=150, sigmas=[1.0, 2.5], limit=0.001)
    ladder = [list(map(float, w)) for w in gw.gaussian_weights()]
    g = Gaussian(gaussian_weights=gw)
    image = rng.uniform(0.0, 1.0, size=(96, 80)).astype(np.float32)
    blur0 = g.gauss_image(image, 0)
    blur1 = g.gauss_image(image, 1)
    zca, grad, theta = g.zcsdog(image, 0)
    fname = "convolve.gaussian_custom.npz"
    np.savez_compressed(
        os.path.join(out_dir, fname),
        image=image,
        blur0=blur0,
        blur1=blur1,
        zca=zca,
        grad=grad,
        theta=theta,
    )
    cases.append(
        {
            "name": "gaussian_custom",
            "file": fname,
            "weights": ladder,
            "widths": [len(w) for w in ladder],
        }
    )
    g.purge()
    return cases

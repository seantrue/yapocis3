#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import yapocis.rpc.programs

from yapocis.utils.basic import set_margin, set_frame
from yapocis.lib.gradient import gradient

from yapocis.lib.operators import sub_res
from yapocis.lib.zcs import zcs, zcs_res
from yapocis.utils.typing import *


def are_close(a: Array, b: Array) -> bool:
    eps = .00001
    return abs(a - b) <= eps


def gauss_1d(sigma: float = 1., dt: float = 1., limit: float = .01, normalize: bool = True) -> Array:
    """
    Get a Gaussian distribution that sums to 1 along 1 dimension, quantized by discrete steps.

    Args:
        sigma: width
        dt: step
        limit: smallest change
        normalize: normalize sum to 1.0

    Returns: 1D Array

    """
    tail = []
    x = 0.
    val = 1.
    two_times_sigma_squared = 2 * sigma * sigma
    scale_factor = 1. / np.sqrt(two_times_sigma_squared * np.pi)
    while val > limit:
        val = float(np.exp(-(x * x) / two_times_sigma_squared) * scale_factor)
        tail.append(val)
        x += dt

    k = tail[::-1]  # Make a copy of the reversed tail
    k.extend(tail[1:])  # Reflect

    # Normalize
    if normalize:
        k = np.divide(k, np.sum(k) / dt)

    return k


def gaussians(maxwidth: int = 100, sigma: float = 1.0, scale: float = 1.6, limit: float = .001,
              sigmas: Optional[List[float]] = None) -> List[Array]:
    """
    Generate a list of gaussian kernels with size less than maxwidth
    Args:
        maxwidth:
        sigma:
        scale:
        limit:
        sigmas:

    Returns:

    """
    sigma = 1.0
    gs = []
    if sigmas:
        for sigma in sigmas:
            gs.append(gauss_1d(sigma, limit=limit))
    else:
        while 1:
            g = gauss_1d(sigma, limit=limit)
            if len(g) > maxwidth:
                break
            gs.append(g)
            sigma *= scale
    return gs


program = None


def gaussian_kernels(gs):
    global program
    convs = [("gauss%s" % len(a), a) for a in gs]
    convsres = [("gauss%s_res" % len(a), a) for a in gs]
    program = yapocis.rpc.programs.load_program("convolvesep", convs=convs)
    krnls = [getattr(program, name) for (name, conv) in convs]
    for i, (name, conv) in enumerate(convs):
        krnls[i].res = getattr(program, convsres[i][0])
        krnls[i].width = len(conv)
    return krnls


_gaussian_basis = gaussians()


def get_gaussian(scale: int):
    assert 0 <= scale and scale < len(_gaussian_basis)
    return _gaussian_basis[scale]


def get_scales() -> int:
    return (len(_gaussian_basis))


def get_gaussian_width(scale):
    return len(get_gaussian(scale))


_gaussian_kernels = gaussian_kernels(_gaussian_basis)


def get_gaussian_kernels():
    return _gaussian_kernels


def get_gaussian_kernel(scale: int) -> Array:
    assert 0 <= scale and scale < len(_gaussian_kernels)
    return _gaussian_kernels[scale]


def gauss_image(a: Image1, scale: int):
    g = get_gaussian_kernel(scale)
    b = g(0, a)
    c = g(1, b)
    return c.copy()


def zcsdog(a: Image1, scale: int, clearmargin: bool = True, frame: bool = True, res: bool = True) -> \
        Tuple[Image1, Image1, Image1]:
    if res:
        g = get_gaussian_kernel(scale)
        g1 = get_gaussian_kernel(scale + 1)
        g.write(a)
        smaller = np.zeros_like(a)
        larger = np.zeros_like(a)
        tmp = np.zeros_like(a)
        zca = np.zeros_like(a)
        g.write(smaller)
        g.write(larger)
        g.write(tmp)
        g.write(zca)
        g.res(0, a, tmp)
        g.res(1, tmp, smaller)
        g1.res(0, a, tmp)
        g1.res(1, tmp, larger)
        sub_res(larger, smaller, tmp)
        zcs_res(tmp, zca)
        g.read(zca)
        g.read(smaller)
    else:
        smaller = gauss_image(a, scale)
        larger = gauss_image(a, scale + 1)
        dog = smaller - larger
        zca = zcs(dog)
    # Zero out of bounds and create a frame.
    grad, theta = gradient(smaller)
    margin = get_gaussian_width(scale) + 1
    if clearmargin:
        set_margin(zca, margin)
        set_margin(grad, margin)
        set_margin(theta, margin)
    if frame:
        set_frame(zca, margin)
    return zca.copy(), grad.copy(), theta.copy()

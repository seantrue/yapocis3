#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

from yapocis.lib.gradient import Gradient
from yapocis.lib.operators import Operators
from yapocis.lib.zc import ZC
from yapocis.utils.basic import set_frame, set_margin
from yapocis.utils.stereo_typing import *

from .bound_kernels import BoundKernels


def are_close(a: float, b: float) -> bool:
    eps = .00001
    return abs(a - b) <= eps


class GaussianWeights:
    def __init__(self, max_width=150, **kwargs):
        self.gws = self.compute_gaussian_weights(max_width, **kwargs)

    def gaussian_weights(self):
        return self.gws

    def compute_gaussian_weights(self, max_width: int = 100, scale: float = 1.6, limit: float = .001,
                                 sigmas: Optional[List[float]] = None) -> List[Array]:
        """
        Generate a list of gaussian kernels with size less than max_width
        Args:
            max_width:
            scale:
            limit:
            sigmas:

        Returns:
            List of numpy arrays as a 1D gaussian
        """
        sigma = 1.0
        gws = []
        if sigmas:
            for sigma in sigmas:
                gws.append(self.compute_gaussian_weight(sigma, limit=limit))
        else:
            while 1:
                gw = self.compute_gaussian_weight(sigma, limit=limit)
                if len(gw) > max_width:
                    break
                gws.append(gw)
                sigma *= scale
        return gws

    @staticmethod
    def compute_gaussian_weight(sigma: float = 1., dt: float = 1., limit: float = .01, normalize: bool = True) -> Array:
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

    def get_gaussian_weight(self, scale: int) -> np.ndarray:
        return self.gws[scale]

    def get_scales(self) -> int:
        return len(self.gws)

    def get_gaussian_width(self, scale: int) -> int:
        return len(self.get_gaussian_weight(scale))


# noinspection PyUnresolvedReferences
class Gaussian(BoundKernels):
    INTERFACE = "convolvesep"
    load_program_config =  {}

    def __init__(self, device_name: Optional[str] = None, gaussian_weights: Optional[List[np.ndarray]] = None):
        self.gaussian_weights = GaussianWeights() if gaussian_weights is None else gaussian_weights
        gws = self.gaussian_weights.gaussian_weights()
        convs = [("gauss%s" % len(a), a) for a in gws]
        convsres = [("gauss%s_res" % len(a), a) for a in gws]
        convsh = [("gauss%s_h" % len(a), a) for a in gws]
        convsv = [("gauss%s_v" % len(a), a) for a in gws]
        super(Gaussian, self).__init__(device_name=device_name, convs=convs)
        kmgr = self.kmgr
        self.zc = ZC(device_name=device_name)
        self.grad = Gradient(device_name=device_name)
        self.ops = Operators(device_name=device_name)
        # Avoid importing OpenCL-specific Kernel class here; keep backend-agnostic
        self.gaussian_kernels: List[object] = [getattr(kmgr, name) for (name, conv) in convs]
        for i, (name, conv) in enumerate(convs):
            self.gaussian_kernels[i].res = getattr(kmgr, convsres[i][0])
            self.gaussian_kernels[i].h = getattr(kmgr, convsh[i][0])
            self.gaussian_kernels[i].v = getattr(kmgr, convsv[i][0])
            self.gaussian_kernels[i].width = len(conv)
        self.threshold = 7.5e-06

    def get_scales(self):
        return len(self.gaussian_kernels)

    def get_gaussian_kernels(self):
        return self.gaussian_kernels

    def get_gaussian_kernel(self, scale: int):
        return self.gaussian_kernels[scale]

    def gauss_image(self, a: Image1, scale: int) -> Image1:
        g = self.get_gaussian_kernel(scale)
        #a = a.copy()
        b = np.empty_like(a)
        g.write(a)
        g.write(b)
        g.h(a, b)
        g.v(b, a)
        g.read(a)
        return a

    def zcs(self, image, threshold=None):
        threshold = self.threshold if threshold is None else threshold
        return self.zc.zcs(image, threshold=threshold)

    def zcs_res(self, image, zcs, threshold=None):
        threshold = self.threshold if threshold is None else threshold
        return self.zc.zcs_res(image, zcs, threshold=threshold)

    def zcsdog(self, a: Image1, scale: int, threshold=None, clear_margin: bool = True, frame: bool = True,
               res: bool = True) -> \
            Tuple[Image1, Image1, Image1]:
        threshold = self.threshold if threshold is None else threshold
        if res:
            g = self.get_gaussian_kernel(scale)
            g1 = self.get_gaussian_kernel(scale + 1)
            g.write(a)
            smaller = np.empty_like(a)
            larger = np.empty_like(a)
            tmp = np.empty_like(a)
            zca = np.empty_like(a)
            g.write(smaller)
            g.write(larger)
            g.write(tmp)
            g.write(zca)
            g.res(0, a, tmp)
            g.res(1, tmp, smaller)
            g1.res(0, a, tmp)
            g1.res(1, tmp, larger)
            self.ops.sub_res(larger, smaller, tmp)
            self.zcs_res(tmp, zca, threshold=threshold)
            g.read(zca)
            g.read(smaller)
        else:
            smaller = self.gauss_image(a, scale)
            larger = self.gauss_image(a, scale + 1)
            dog = smaller - larger
            zca = self.zcs(dog, threshold=threshold)
            del larger
            del dog
        # Zero out of bounds and create a frame.
        grad, theta = self.grad.gradient(smaller, 3)
        margin = self.gaussian_weights.get_gaussian_width(scale) + 1
        if clear_margin:
            set_margin(zca, margin)
            set_margin(grad, margin)
            set_margin(theta, margin)
        if frame:
            set_frame(zca, margin)
        zca, grad, theta =  zca.copy(), grad.copy(), theta.copy()
        self.purge()
        return zca, grad, theta

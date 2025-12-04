#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.

from yapocis.core.interface_context import load_interface_context
from yapocis.lib.gradient import Gradient
from yapocis.lib.operators import Operators
from yapocis.lib.zc import ZC
from yapocis.utils.basic import set_frame, set_margin
from yapocis.utils.stereo_typing import *

from .bound_kernels import BoundKernels


def are_close(a: float, b: float) -> bool:
    eps = 0.00001
    return abs(a - b) <= eps


def generate_convolvesep_context(context):
    """
    Context generator for convolvesep interface.

    Generates Gaussian convolution kernels from gaussian_params specification
    and adds them to the context as 'convs' parameter.

    This function is invoked by the generic context generator system when
    rendering the convolvesep interface template.

    Args:
        context: Dictionary containing YAML context with 'gaussian_params'

    Returns:
        Modified context dictionary with 'convs' parameter added

    Example:
        Input context (from convolvesep.yml):
            {
                'gaussian_params': {
                    'max_width': 150,
                    'scale': 1.6,
                    'limit': 0.001
                }
            }

        Output context:
            {
                'gaussian_params': {...},
                'convs': [
                    ('gauss5', array([...])),
                    ('gauss9', array([...])),
                    ('gauss13', array([...])),
                    ...
                ]
            }
    """
    if "gaussian_params" in context:
        gaussian_params = context["gaussian_params"]
        gw = GaussianWeights(**gaussian_params)
        gws = gw.gaussian_weights()
        convs = [("gauss%s" % len(a), a) for a in gws]
        context["convs"] = convs
    return context


class GaussianWeights:
    def __init__(self, max_width=None, **kwargs):
        # Always load YAML defaults first
        yaml_context = load_interface_context("convolvesep")
        yaml_params = yaml_context.get("gaussian_params", {})

        # Use YAML default for max_width if not provided
        if max_width is None:
            max_width = yaml_params.get("max_width")
            if max_width is None:
                raise ValueError(
                    "max_width must be provided explicitly or defined in convolvesep.yml"
                )

        # Merge YAML params with kwargs (kwargs take precedence)
        final_kwargs = {}
        for key in ["scale", "limit", "sigmas"]:
            if key in kwargs:
                # Explicit parameter takes precedence
                final_kwargs[key] = kwargs[key]
            elif key in yaml_params:
                # Use YAML default
                final_kwargs[key] = yaml_params[key]
            # If neither provided, compute_gaussian_weights will validate

        self.gws = self.compute_gaussian_weights(max_width, **final_kwargs)

    def gaussian_weights(self):
        return self.gws

    def compute_gaussian_weights(
        self,
        max_width: int,
        scale: float = None,
        limit: float = None,
        sigmas: list[float] | None = None,
    ) -> list[Array]:
        """
        Generate a list of gaussian kernels with size less than max_width.

        Parameters must come from convolvesep.yml or be passed explicitly.

        Args:
            max_width: Maximum kernel width (required)
            scale: Sigma progression factor (required if sigmas not provided)
            limit: Weight threshold for kernel truncation (required if sigmas not provided)
            sigmas: Optional predefined sigma values (if provided, scale is ignored)

        Returns:
            list of numpy arrays as 1D gaussian kernels

        Raises:
            ValueError: If required parameters are not provided
        """
        sigma = 1.0
        gws = []

        if sigmas:
            # Using predefined sigmas - only need limit
            if limit is None:
                raise ValueError(
                    "limit parameter is required when using custom sigmas. "
                    "It should come from convolvesep.yml or be passed explicitly."
                )
            for sigma in sigmas:
                gws.append(self.compute_gaussian_weight(sigma, limit=limit))
        else:
            # Auto-generating sigmas - need both scale and limit
            if scale is None or limit is None:
                raise ValueError(
                    "scale and limit parameters are required when sigmas is not provided. "
                    "They should come from convolvesep.yml or be passed explicitly."
                )
            while 1:
                gw = self.compute_gaussian_weight(sigma, limit=limit)
                if len(gw) > max_width:
                    break
                gws.append(gw)
                sigma *= scale
        return gws

    @staticmethod
    def compute_gaussian_weight(
        sigma: float = 1.0, dt: float = 1.0, limit: float = 0.01, normalize: bool = True
    ) -> Array:
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
        x = 0.0
        val = 1.0
        two_times_sigma_squared = 2 * sigma * sigma
        scale_factor = 1.0 / np.sqrt(two_times_sigma_squared * np.pi)
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
    load_program_config = {}
    use_async = True  # Enable async execution by default for Gaussian operations

    def __init__(
        self,
        device_name: str | None = None,
        gaussian_weights: list[np.ndarray] | None = None,
        use_async: bool | None = None,
        async_write: bool | None = None,
    ):
        self.gaussian_weights = (
            GaussianWeights() if gaussian_weights is None else gaussian_weights
        )
        gws = self.gaussian_weights.gaussian_weights()
        convs = [("gauss%s" % len(a), a) for a in gws]

        super(Gaussian, self).__init__(
            device_name=device_name,
            use_async=use_async,
            async_write=async_write,
            convs=convs,
        )

        # Initialize with traditional path
        self._init_with_bound_kernels(convs)

        # Rest of initialization unchanged
        self.zc = ZC(device_name=device_name)
        self.grad = Gradient(device_name=device_name)
        self.ops = Operators(device_name=device_name)
        self.threshold = 7.5e-06

    def _init_with_bound_kernels(self, convs):
        """Initialize using traditional runtime interpretation (slow path)."""
        convsres = [("gauss%s_res" % len(conv), conv) for (name, conv) in convs]
        convsh = [("gauss%s_h" % len(conv), conv) for (name, conv) in convs]
        convsv = [("gauss%s_v" % len(conv), conv) for (name, conv) in convs]

        kmgr = self.kmgr
        self.gaussian_kernels: list[object] = [
            getattr(kmgr, name) for (name, conv) in convs
        ]
        for i, (name, conv) in enumerate(convs):
            self.gaussian_kernels[i].res = getattr(kmgr, convsres[i][0])
            self.gaussian_kernels[i].h = getattr(kmgr, convsh[i][0])
            self.gaussian_kernels[i].v = getattr(kmgr, convsv[i][0])
            self.gaussian_kernels[i].width = len(conv)

    def get_scales(self):
        return len(self.gaussian_kernels)

    def get_gaussian_kernels(self):
        return self.gaussian_kernels

    def get_gaussian_kernel(self, scale: int):
        return self.gaussian_kernels[scale]

    def gauss_image(self, a: Image1, scale: int) -> Image1:
        g = self.get_gaussian_kernel(scale)
        aa = a.copy()
        b = np.empty_like(aa)
        g.write(aa)
        g.write(b)

        if self.use_async:
            # Async path: enqueue both convolutions without waiting
            with self.kmgr.async_mode():
                g.h(aa, b)
                g.v(b, aa)
        else:
            # Synchronous path (traditional)
            g.h(aa, b)
            g.v(b, aa)

        g.read(aa)
        return aa

    def zcs(self, image, threshold=None):
        threshold = self.threshold if threshold is None else threshold
        return self.zc.zcs(image, threshold=threshold)

    def zcs_res(self, image, zcs, threshold=None):
        threshold = self.threshold if threshold is None else threshold
        return self.zc.zcs_res(image, zcs, threshold=threshold)

    def zcsdog(
        self,
        a: Image1,
        scale: int,
        threshold=None,
        clear_margin: bool = True,
        frame: bool = True,
        res: bool = True,
    ) -> tuple[Image1, Image1, Image1]:
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

            if self.use_async:
                # Async path: enqueue all operations without waiting
                with self.kmgr.async_mode():
                    g.res(0, a, tmp)
                    g.res(1, tmp, smaller)
                    g1.res(0, a, tmp)
                    g1.res(1, tmp, larger)
                    self.ops.sub_res(larger, smaller, tmp)
                    self.zcs_res(tmp, zca, threshold=threshold)
            else:
                # Synchronous path (traditional)
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
        zca, grad, theta = zca.copy(), grad.copy(), theta.copy()
        self.purge()
        return zca, grad, theta

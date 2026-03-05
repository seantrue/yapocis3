#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.
"""
Wrapper for 2D gradient kernel
"""

import numpy

#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2022.
import numpy as np

from ..utils.stereo_typing import Image1
from .bound_kernels import BoundKernels


class Imageops(BoundKernels):
    INTERFACE = "imageops"

    def clamp_pixels(
        self, input: np.ndarray, low: float = 0.0, high: float = 1.0
    ) -> np.ndarray:
        """Clamp pixel values to the range [low, high].

        Args:
            input: Input array of float32 values
            low: Minimum value (default 0.0)
            high: Maximum value (default 1.0)

        Returns:
            Array with values clamped to [low, high]
        """
        return self.kmgr.clamp_pixels(input, low, high)

    def stretch(
        self,
        input: np.ndarray,
        input_min: float,
        input_max: float,
        low: float = 0.0,
        high: float = 1.0,
    ) -> np.ndarray:
        """Stretch pixel values from [input_min, input_max] to [low, high].

        Args:
            input: Input array of float32 values
            input_min: Minimum value of input range
            input_max: Maximum value of input range
            low: Minimum value of output range (default 0.0)
            high: Maximum value of output range (default 1.0)

        Returns:
            Array with values stretched to [low, high]
        """
        return self.kmgr.stretch(input, input_min, input_max, low, high)

    def binarize(self, input: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binarize pixel values using a threshold.

        Args:
            input: Input array of float32 values
            threshold: Threshold value (default 0.5)

        Returns:
            Array with values 0.0 (below threshold) or 1.0 (at or above threshold)
        """
        return self.kmgr.binarize(input, threshold)

    def histogram(self, input, n_bins=256, n_workers=32, low=0.0, high=1.0):
        hst = np.zeros(n_bins, dtype=np.int32)
        self.kmgr.histogram(input, low, high, hst)
        return hst

    def histogram_res(self, input, hst, low=0.0, high=1.0):
        self.kmgr.histogram_res(input, low, high, hst)
        return hst

    def histeq(self, im: Image1, nbr_bins: int = 2**16) -> Image1:
        """
        Histogram equalization of a 2D array

        Args:
            im: 2D Array
            nbr_bins: number of bins

        Returns:
            2d Array, dtype=np.float32
        """
        imhist = self.histogram(im, n_bins=nbr_bins)
        imhist[0] = 0

        cdf = imhist.cumsum()  # cumulative distribution function
        cdf = (nbr_bins - 1) * cdf / cdf[-1]
        cdf = cdf / nbr_bins
        bins = np.linspace(0, 1.0, nbr_bins)
        im2 = np.interp(im.flatten(), bins, cdf)
        return np.array(im2, np.float32).reshape(im.shape)

    def histmatch(
        self,
        im: Image1,
        target_hist: np.ndarray,
        nbr_bins: int | None = None,
    ) -> Image1:
        """
        Histogram matching - transform image to match a target histogram.

        Args:
            im: 2D Array to transform
            target_hist: Target histogram to match (1D array of counts/frequencies)
            nbr_bins: Number of bins for source histogram (default: len(target_hist))

        Returns:
            2D Array with histogram matched to target, dtype=np.float32
        """
        if nbr_bins is None:
            nbr_bins = len(target_hist)

        # Compute source CDF
        src_hist = self.histogram(im, n_bins=nbr_bins)
        src_hist[0] = 0
        src_cdf = src_hist.cumsum().astype(np.float64)
        src_cdf = src_cdf / src_cdf[-1]  # Normalize to [0, 1]

        # Compute target CDF
        target_hist = np.asarray(target_hist, dtype=np.float64)
        target_hist[0] = 0
        target_cdf = target_hist.cumsum()
        target_cdf = target_cdf / target_cdf[-1]  # Normalize to [0, 1]

        # Build lookup table: for each source bin, find target bin with closest CDF
        # Use searchsorted to find where each src_cdf value would fit in target_cdf
        lookup = np.searchsorted(target_cdf, src_cdf)
        lookup = np.clip(lookup, 0, nbr_bins - 1)

        # Map source bins to target intensity values
        bins = np.linspace(0, 1.0, nbr_bins)
        mapped_values = bins[lookup]

        # Apply mapping to image
        im2 = np.interp(im.flatten(), bins, mapped_values)
        return np.array(im2, np.float32).reshape(im.shape)

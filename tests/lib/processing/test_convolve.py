#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import pytest
from matplotlib import pyplot as plt

from yapocis.lib.convolve import Convolve1D
from yapocis.utils.stereo_typing import *
from . import StopWatcher


@pytest.mark.smoke
def test_convolve():
    a = np.random.uniform(-1, 1, size=1000)
    cnv = Convolve1D(name="convolve", conv=[0.01] * 100)
    with StopWatcher("convolve", kernel=cnv) as sw:
        with sw.checkpoint("convolve_fast"):
            b = cnv.convolve_fast(a, 100)
    plt.plot(b)
    plt.show()

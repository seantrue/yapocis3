#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import numpy as np

from yapocis.utils.basic import show_array, stage
from yapocis.lib.channels import split_channels, join_channels
from yapocis.utils.stereo_typing import *

from yapocis.lib.convolve import Convolve1D

from . import show_snapshots, snapshow, show, snapshot, StopWatcher
from matplotlib import pyplot as plt

def test_convolve():
    a = np.random.uniform(-1, 1, size=1000)
    cnv = Convolve1D(name="convolve",conv=[.01]*100)
    b = cnv.convolve_fast(a,100)
    plt.plot(a)
    plt.plot(b)
    plt.show()
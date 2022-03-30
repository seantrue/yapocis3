#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.
import sys

import matplotlib
import seaborn

seaborn.set_style("dark")
matplotlib.use("module://yapocis.lib.mplbackend")
try:
    from scipy.ndimage.filters import median_filter
    from matplotlib import pyplot as plot
except ImportError:
    print("Please install scipy and matplotlib to compare median filter performance")
    sys.exit(1)

import time
import numpy as np



from yapocis.ocl.device import get_available_device_names
from yapocis.ocl.device_manager import display_snapshots

from yapocis.lib.median import Median
def median3x3_scipy(input, iterations=1):
    for i in range(iterations):
        input = median_filter(input, 3)
    return input


ITERATIONS = 20
SAMPLES = 20
SMALLEST=100
LARGEST=1000

# Get just the official device name
device_names = get_available_device_names()

# Range of images widths spaced logarithmically
widths = np.logspace(np.log2(SMALLEST), np.log2(LARGEST), num=SAMPLES, base=2).astype(int)
images = []
for width in widths:
    images.append(np.float32(np.random.random_sample((width, width))))

# Get the baseline scipy performance
scipy_deltas = []
print("scipy base line")
for width, img in zip(widths, images):
    t = time.time()
    output1 = median3x3_scipy(img, ITERATIONS)
    scipy_delta = time.time() - t
    print(width, scipy_delta)
    scipy_deltas.append(scipy_delta)

# Slow (fetch result for each iteration)
# and fast (resident ping pong buffers)
by_device = {}
for device_name in device_names:
    print(device_name)
    median = Median(device_name=device_name)
    areas = []
    fast_deltas = []
    slow_deltas = []
    errors = []
    for scipy_delta, img, width in zip(scipy_deltas, images, widths):
        t = time.time()
        output2 = median.median3x3slow(img, ITERATIONS)
        slow_delta = time.time() - t

        t = time.time()
        output3 = median.median3x3fast(img, ITERATIONS)
        fast_delta = time.time() - t

        print(width, scipy_delta, slow_delta, fast_delta)
        areas.append(width * width)
        slow_deltas.append(slow_delta)
        fast_deltas.append(fast_delta)
    median.snapshot(name="median")
    print()
    by_device[device_name] = (areas, slow_deltas, fast_deltas)

# Graphic is a set of vertically stacked plots, with a common X axis
fig, axs = plot.subplots(len(device_names), 1, sharex=True)
fig.suptitle(f"Yapocis Performance Comparison:\nmedian 3x3 filter, {ITERATIONS} iterations", fontsize="large")

for iengine, device_name in enumerate(device_names):
    # Fetch the stats, and compute the performance ratios
    areas, slow_deltas, fast_deltas = by_device[device_name]
    ax1 = axs[iengine]
    areas = np.array(areas) / 1000.
    scipy_deltas = np.array(scipy_deltas)
    slow_deltas = np.array(slow_deltas)
    fast_deltas = np.array(fast_deltas)
    ratio = 1.0 / (slow_deltas / scipy_deltas)
    ratio2 = 1.0 / (fast_deltas / scipy_deltas)

    # Lines show performance improvement over scipy
    # Filled areas show total seconds per engine
    # Lines show performance improvement over scipy
    # Filled areas show total seconds per engine
    ax1.set_title(device_name, loc="left")
    ax1.set_xlabel("Pixels(k)", fontsize="medium")
    # ax1.plot(areas, scipy_deltas, color="darkgreen", label="numpy")
    ax1.plot(areas, slow_deltas, color="magenta", label="slow")
    ax1.plot(areas, fast_deltas, color="orange", label="fast")
    # ax1.fill_between(areas, scipy_deltas, slow_deltas, color="lightgreen")
    ax1.fill_between(areas, fast_deltas, slow_deltas, color="pink")
    ax1.fill_between(areas, np.zeros_like(slow_deltas), fast_deltas, color="yellow")
    ax1.set_ylabel("Seconds", fontsize="medium")
    ax2 = ax1.twinx()
    ax2.plot(areas, ratio, color="red", label="slow*")
    ax2.plot(areas, ratio2, color="black", label="fast*")
    ax2.set_ylabel("Speed-up")
    if iengine == 0:
        ax1.legend(bbox_to_anchor=(1.2, 1.3), loc='upper left')
        ax2.legend(bbox_to_anchor=(1.2, .5), loc='upper left')
plot.tight_layout()
fig.savefig("matplotlib_median_performance.png")

display_snapshots()
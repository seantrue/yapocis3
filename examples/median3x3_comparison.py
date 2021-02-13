#  Copyright Sean D. True & SwapWizard dba (c) 2011-2021.

import numpy as np
import time
from scipy.ndimage.filters import median_filter
from yapocis.lib import median


def median3x3(input, iterations=1):
    for i in range(iterations):
        input = median_filter(input,3)
    return input

areas = []
yapocisfast_deltas = []
yapocis_deltas = []
numpy_deltas = []
errors = []
ITERATIONS=10
SAMPLES=20
sizes = np.logspace(6,12,num=SAMPLES,base=2).astype(np.int)

for width in sizes:
    img = np.float32(np.random.random_sample((width,width)))
    
    t = time.time()
    output1 = median3x3(img,ITERATIONS)
    numpy_delta = time.time() - t
    
    t = time.time()
    output2 = median.median3x3slow(img, ITERATIONS)
    yapocis_delta = time.time() - t
    
    t = time.time()
    output3 = median.median3x3fast(img, ITERATIONS)
    yapocisfast_delta = time.time() - t

    print(width, numpy_delta, yapocis_delta, yapocisfast_delta)
    areas.append(width*width)
    numpy_deltas.append(numpy_delta)
    yapocis_deltas.append(yapocis_delta)
    yapocisfast_deltas.append(yapocisfast_delta)
    
from matplotlib import pyplot as plot
areas = np.array(areas)/1000.
numpy_deltas = np.array(numpy_deltas)
yapocis_deltas = np.array(yapocis_deltas)
yapocisfast_deltas = np.array(yapocisfast_deltas)
ratio = 1.0/(yapocis_deltas/numpy_deltas)
ratio2 = 1.0/(yapocisfast_deltas/numpy_deltas)
unity = numpy_deltas/numpy_deltas
fig, ax1 = plot.subplots()
ax1.set_title("Median 3x3 filter, %s iterations" % ITERATIONS,fontsize="large")
ax1.set_xlabel("Pixels(k)",fontsize="medium")
ax1.plot(areas, numpy_deltas,color="darkgreen",label="numpy")
ax1.plot(areas, yapocis_deltas,color="magenta",label="slow")
ax1.plot(areas, yapocisfast_deltas,color="orange",label="fast")
ax1.fill_between(areas, numpy_deltas, yapocis_deltas,color="lightgreen")
ax1.fill_between(areas, yapocisfast_deltas, yapocis_deltas, color="pink")
ax1.fill_between(areas, np.zeros_like(yapocis_deltas),yapocisfast_deltas, 
                  color="yellow")
ax1.set_ylabel("Seconds",fontsize="medium")
ax2 = ax1.twinx()
ax2.plot(areas,ratio,color="red",label="slow>np")
ax2.plot(areas,ratio2,color="black",label="fast>np")
#ax2.plot(areas,unity,color="blue",label="baseline")
ax2.set_ylabel("Speed-up")
#ax2.set_ylim(0,1.2)
ax1.legend(bbox_to_anchor=(1.1, 1.0), loc='upper left')
ax2.legend(bbox_to_anchor=(.15, 1.0), loc='upper left')
fig.tight_layout()
plot.savefig("docs/images/median3x3-performance.png")
plot.show()
